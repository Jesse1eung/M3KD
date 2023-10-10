"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from copy import deepcopy
import numpy as np
import os
import random
import re
import torch
import torch.nn.functional as F
from lavis.common.registry import registry
from lavis.common.utils import get_abs_path
from lavis.models.albef_models import AlbefBase
from lavis.models.xvlm_models import XVLMBase

from lavis.models.albef_models.albef_outputs import (
    AlbefIntermediateOutput,
    AlbefOutput,
    AlbefSimilarity,
)
from lavis.models.base_model import (
    MomentumDistilationMixin,
    concat_all_gather,
    all_gather_with_grad,
    SharedQueueMixin)
from lavis.models.xmed import BertForMaskedLM
from lavis.models.mix_models.mixmome import BeitModel, BeitOnlyMLMHead

from lavis.models.vit import VisionTransformerEncoder
from lavis.models.swt import SwinTransformerEncoder
from torch import nn
from transformers import BertConfig
from transformers import BeitConfig

from torch.nn import MSELoss


@registry.register_model("albef_pretrain")
class AlbefPretrain(AlbefBase, MomentumDistilationMixin, SharedQueueMixin):
    """
    ALBEF pretrain model.

    Supported model types:
        - base: ALBEF base model used for pretraining.
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "base": "configs/models/albef/albef_pretrain_base.yaml",
    }

    def __init__(
        self,
        image_encoder, text_encoder, queue_size,
        add_mlm=False,add_itc=False, add_itm=False, caption_distill=False,
        add_logits=False, add_att=False, add_hid=False, itc_distill=False,
        embed_dim=256, mlm_mask_prob=0.15,
        temp=0.07, momentum=0.995, alpha=0.4, max_txt_len=30,
        teacher=None, l_teacher=None, v_teacher=None,is_teacher=False ):

        super().__init__()

        self.tokenizer = self.init_tokenizer("./pretrained_ckpt/bert-base-uncased")

        # add obj
        self.add_mlm = add_mlm
        self.add_itc = add_itc
        self.add_itm = add_itm
        self.add_logits = add_logits
        self.add_att = add_att
        self.add_hid = add_hid
        self.itc_distill = itc_distill
        self.caption_distill = caption_distill

        self.visual_encoder = image_encoder
        self.text_encoder = text_encoder

        text_width = text_encoder.config.hidden_size
        vision_width = image_encoder.vision_width

        self.embed_dim = embed_dim

        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)

        self.itm_head = nn.Linear(text_width, 2)
        self.teacher = teacher
        self.l_teacher = l_teacher
        self.v_teacher = v_teacher
        if self.l_teacher is not None:
            self.freeze_teacher(self.l_teacher)
        if self.v_teacher is not None:
            self.freeze_teacher(self.v_teacher)


        if self.teacher:
            for n,p in self.teacher.named_parameters():
                if 'encoder.bert.encoder.layer.7.attention.self.query.bias' in n:
                    print(n, p[:5])

        # create the momentum encoder
        self.visual_encoder_m = deepcopy(self.visual_encoder)
        self.text_encoder_m = deepcopy(self.text_encoder)

        self.vision_proj_m = deepcopy(self.vision_proj)
        self.text_proj_m = deepcopy(self.text_proj)

        self.model_pairs = [
            [self.visual_encoder, self.visual_encoder_m],
            [self.text_encoder, self.text_encoder_m],
            [self.vision_proj, self.vision_proj_m],
            [self.text_proj, self.text_proj_m],
        ]
        self.copy_params()

        # create the queue
        self.register_buffer("image_queue", torch.randn(embed_dim, queue_size))
        self.register_buffer("text_queue", torch.randn(embed_dim, queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)

        self.queue_size = queue_size
        self.momentum = momentum
        self.temp = nn.Parameter(temp * torch.ones([]))

        self.alpha = alpha
        self.max_txt_len = max_txt_len

        self.mlm_probability = mlm_mask_prob

        # self.update_fusion_layer()
        # self.print_paras_require_grad()
        # if not is_teacher:
        #     self.freeze_text(self.text_encoder)
        #     self.freeze_vis(self.visual_encoder)

    def freeze_vis(self,vis_encoder):
        for name, params in vis_encoder.named_parameters():
            params.requires_grad = False
            if 'v_' in name:
                print('freezed_v:',name)
                params.requires_grad = True

    def freeze_text(self, text_encoder):
        for name, params in text_encoder.named_parameters():
            params.requires_grad = False
            if 'l_inter' in name or 'l_out' in name or 'embeddings' in name:
                print('freezed_l:',name)
                params.requires_grad = True
            # layer_str = re.findall("\d+", name)
            # if layer_str:
            #     layer = int(layer_str[0])
            #     if layer >= 3:
            #         params.requires_grad = True
            # elif 'predictions' in name:
            #     params.requires_grad = True
            # # if 'cross' not in name:
            # #     params.requires_grad = False

    def freeze_teacher(self, teacher):
        for params in teacher.parameters():
            params.requires_grad = False

    def update_fusion_layer(self):
        for name, params in self.named_parameters():

            digit_list = re.findall("\d+", name)
            if not digit_list:
                continue
            layer = int(digit_list[0])
            if layer >= 6 and 'text_encoder.' in name and 'teacher' not in name:
                params.requires_grad = True
            elif ('predictions' in name or 'itm_head' in name) and 'teacher' not in name:
                params.requires_grad = True
            else:
                params.requires_grad = False

    def update_vis_layer(self):
        for name, params in self.named_parameters():
            if ('visual_encoder' in name or 'proj' in name) and 'teacher' not in name:
                params.requires_grad = True
            else:
                params.requires_grad = False

    def update_text_layer(self):
        for name, params in self.named_parameters():
            layer = int(re.findall("\d+", name)[0])
            # if 'text_encoder' in name and 'teacher' not in name and layer < fusion_layer:
            if 'text_encoder' in name and 'teacher' not in name:
                params.requires_grad = True
            else:
                params.requires_grad = False

    def print_paras_require_grad(self):
        print('params need grad:')
        for name, params in self.named_parameters():
            if params.requires_grad:
                print(name)

    def _rampup_factor(self, epoch, iters, num_iters_per_epoch):
        return min(1, (epoch * num_iters_per_epoch + iters) / (2 * num_iters_per_epoch))

    def soft_cross_entropy(self, predicts, targets):
        student_likelihood = nn.functional.log_softmax(predicts, dim=-1)
        targets_prob = nn.functional.softmax(targets, dim=-1)
        return (- targets_prob * student_likelihood).mean()

    def forward(self, samples):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W). The input images. Default: H=224, W=224.
                - text_input (list): A list of length batch_size, each element is a string of text/caption.
                - epoch (int): The current epoch.
                - iters (int): The current iteration.
                - num_iters_per_epoch (int): The number of iterations per epoch.

        Returns:
            BlipOutput: A BlipOutput object containing loss and intermediate output. See ``lavis.models.blip_models.blip_outputs.BlipOutput`` for more details.

        Examples:
            >>> import torch
            >>> from lavis.models import load_model
            >>> model = load_model("albef_pretrain")
            >>> images = torch.randn(4, 3, 224, 224)
            >>> text_input = ["caption of image 1", "another caption of image 1", "caption of image 2", "caption of image 3"]
            >>> samples = {"image": images, "text_input": text_input, "epoch": 0, "iters": 0, "num_iters_per_epoch": 100}
            >>> output = model(samples)
            >>> output.keys()
            odict_keys(['sims', 'intermediate_output', 'loss', 'loss_itc', 'loss_itm', 'loss_mlm'])
        """

        loss = 0.

        loss_itc = 0.
        loss_itm = 0.
        loss_mlm = 0.
        loss_qk = 0.
        loss_vv = 0.
        loss_cls = 0.

        loss_mse = MSELoss()
        loss_logits = 0.
        loss_att = 0.
        loss_hid = 0.
        loss_itc_distill = 0.
        loss_it_hid = 0.
        sim_i2t_m = 0.
        sim_t2i_m = 0.
        image_embeds_m = 0.
        text_embeds_m = 0.
        sim_i2t_targets = 0.
        sim_t2i_targets = 0.

        # print(type(samples), samples.keys())

        kl = nn.KLDivLoss(reduction='none')

        if "image" not in samples:
            caption = samples["text_input"]
            text = self.tokenizer(
                caption,
                padding="max_length",
                truncation=True,
                max_length=128,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():

                output_l = self.l_teacher.bert(
                    text.input_ids,
                    attention_mask=text.attention_mask,
                    return_dict=True,
                    mode="mlmnovisual",

                )
            output = self.text_encoder.bert(
                text.input_ids,
                attention_mask=text.attention_mask,
                return_dict=True,
                mode="mlmnovisual",
            )
            # print('txt onlyyyyyyyyyyyyyy')
            # stu_atts = ()
            # stu_hids = ()
            output_hid_fit = ()
            for i, hid in enumerate(output.hidden_states):
                hid_fit = self.text_encoder.fit_denses[i](hid)
                output_hid_fit += (hid_fit,)
            # rd = random.random()
            rd=0
            if rd > 0.6:
                stu_atts = output.attentions[:3]
                stu_hids= output_hid_fit[0::3]
            elif rd > 0.2:
                stu_atts = output.attentions[:4]
                stu_hids = output_hid_fit[0::4]
            else:
                # stu_atts = output.attentions[2::3]
                # stu_hids = output_hid_fit[0::3]
                stu_atts = output.attentions
                stu_hids = output_hid_fit

            loss_att = self.get_att_loss(output_l.attentions, stu_atts)
            loss_hid = self.get_hid_loss(output_l.hidden_states, stu_hids)
            #取 3 6层
            # loss_att = self.get_att_loss(output_l.attentions[11::6], output.attentions[5::3])
            # loss_hid = self.get_hid_loss(output_l.hidden_states[12::3], output.hidden_states[6::3])
            loss = loss_att + loss_hid
            # loss = loss_hid
            return AlbefOutput(loss=loss,
                               loss_itc=loss_itc,
                               loss_itm=loss_itm,)

        elif "text_input" not in samples:
            image = samples["image"]
            # kl = nn.KLDivLoss(reduction='none')
            with torch.no_grad():
                output_v = self.v_teacher.forward_features(image, return_relation=True)
            output = self.visual_encoder.forward_features(image, return_relation=True, use_moe=False)

            # loss_qk = kl(output[0].log(),output_v[0]).sum(-1).mean()
            # loss_vv = kl(output[1].log(),output_v[1]).sum(-1).mean()

            # print(output[0].shape,output[1].shape,output_v[0].shape, output_v[1].shape)
            # loss_cls = kl(output[2].log(),output_v[2]).sum(-1).mean()
            loss_cls  = 0.1 * loss_mse(self.visual_encoder.fit_denses[-2](output[2]),output_v[2])
            # print()
            return AlbefOutput(loss=loss_cls,
                               loss_itc=loss_itc,
                               loss_itm=loss_itm,)

        # print('pair rrrr')
        image = samples["image"]
        caption = samples["text_input"]

        alpha = self.alpha * self._rampup_factor(
            epoch=samples["epoch"],
            iters=samples["iters"],
            num_iters_per_epoch=samples["num_iters_per_epoch"],
        )

        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)
        output = self.visual_encoder.forward_features(image,return_relation=True)
        image_embeds = output[2]
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(self.device)

        s=False
        if self.v_teacher is not None and s:
            with torch.no_grad():
                output_v = self.v_teacher.forward_features(image, return_relation=True)
            # output = self.visual_encoder.forward_features(image, return_relation=True)
            # loss_qk = kl(output[0].log(),output_v[0]).sum(-1).mean()
            # loss_vv = kl(output[1].log(),output_v[1]).sum(-1).mean()
            # loss_img = loss_qk+loss_vv
            # print(output[0].shape,kl(output[0].log(),output_v[0]).shape, output[2].shape)
            loss_cls = loss_mse(output[2],output_v[2])
            # print('loss img',loss_cls.shape, loss_cls)
            loss += loss_cls
        # print(caption)
        text = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(self.device)


        if self.caption_distill:
            with torch.no_grad():

                output_l = self.l_teacher.bert(
                    text.input_ids,
                    attention_mask=text.attention_mask,
                    return_dict=True,
                    mode="mlmnovisual",
                )
            output = self.text_encoder.bert(
                text.input_ids,
                attention_mask=text.attention_mask,
                return_dict=True,
                mode="mlmnovisual",
            )

            # print("caption distill att:", len(output_l.attentions), len(output.attentions))
            # print("caption distill hid:", len(output_l.hidden_states), len(output.hidden_states))
            mlm_att = self.get_att_loss(output_l.attentions, output.attentions)
            mlm_hid = self.get_hid_loss(output_l.hidden_states, output.hidden_states)
            loss += 0.5 * (mlm_att + mlm_hid)

        image_feat = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)

        text_output = self.text_encoder.bert(
            text.input_ids,
            attention_mask=text.attention_mask,
            return_dict=True,
            mode="text",
        )
        text_embeds = text_output.last_hidden_state
        text_feat = F.normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)
        a = True
        # get momentum features
        if self.add_itc:
            with torch.no_grad():
                self._momentum_update()
                image_embeds_m = self.visual_encoder_m(image)
                image_feat_m = F.normalize(
                    self.vision_proj_m(image_embeds_m[:, 0, :]), dim=-1)
                image_feat_all = torch.cat(
                    [image_feat_m.t(), self.image_queue.clone().detach()], dim=1)
                text_output_m = self.text_encoder_m.bert(
                    text.input_ids,
                    attention_mask=text.attention_mask,
                    return_dict=True,
                    mode="text",
                )
                text_embeds_m = text_output_m.last_hidden_state
                text_feat_m = F.normalize(self.text_proj_m(text_embeds_m[:, 0, :]), dim=-1)
                text_feat_all = torch.cat(
                    [text_feat_m.t(), self.text_queue.clone().detach()], dim=1)

                sim_i2t_m = image_feat_m @ text_feat_all / self.temp
                sim_t2i_m = text_feat_m @ image_feat_all / self.temp
                
                sim_targets = torch.zeros(sim_i2t_m.size()).to(image.device)
                sim_targets.fill_diagonal_(1)

                sim_i2t_targets = (alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets)
                sim_t2i_targets = (alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets)
        elif self.itc_distill:
            with torch.no_grad():
                image_embeds_teacher = self.teacher.visual_encoder(image)
                image_feat_teacher = F.normalize(
                    self.teacher.vision_proj(image_embeds_teacher[:, 0, :]), dim=-1)
                image_feat_all = torch.cat([image_feat_teacher.t(), self.image_queue.clone().detach()], dim=1)
                # image_feat_all = concat_all_gather(image_feat_teacher)


                text_output_teacher = self.teacher.text_encoder.bert(
                    text.input_ids,
                    attention_mask=text.attention_mask,
                    return_dict=True,
                    mode="text",
                )
                text_embeds_teacher = text_output_teacher.last_hidden_state
                text_feat_teacher = F.normalize(self.teacher.text_proj(text_embeds_teacher[:, 0, :]), dim=-1)
                text_feat_all = torch.cat([text_feat_teacher.t(), self.text_queue.clone().detach()], dim=1)
                # text_feat_all = concat_all_gather(text_feat_teacher)

                # if queue, there is not a t.(), while concat with t.()
                sim_i2t_teacher = image_feat_teacher @ text_feat_all/ self.temp
                sim_t2i_teacher = text_feat_teacher @ image_feat_all / self.temp

                sim_targets = torch.zeros(sim_i2t_teacher.size()).to(image.device)
                # sim_targets[:, sim_i2t_teacher.size()[0]* int(os.environ["RANK"]): ].fill_diagonal_(1)
                sim_targets.fill_diagonal_(1)

                sim_i2t_t = F.softmax(sim_i2t_teacher, dim=1)
                sim_t2i_t = F.softmax(sim_t2i_teacher, dim=1)

                sim_i2t_targets = (alpha * sim_i2t_t + (1 - alpha) * sim_targets)
                sim_t2i_targets = (alpha * sim_t2i_t + (1 - alpha) * sim_targets)


            # bs *n_gpu
            # image_feat_world = concat_all_gather(image_feat)
            # text_feat_world = concat_all_gather(text_feat)
            # 1 x n
            # sim_i2t = image_feat @ text_feat_world.t() /self.temp.float()
            # sim_t2i = text_feat @ image_feat_world.t() /self.temp.float()


            sim_i2t = image_feat @ text_feat_all / self.temp
            sim_t2i = text_feat @ image_feat_all / self.temp

            loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1).mean()
            loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1).mean()
            loss_itc_distill = (loss_i2t + loss_t2i) / 2
            loss += loss_itc_distill

            # loss_i_hid = loss_mse(image_embeds, image_embeds_teacher)
            # loss_t_hid = loss_mse(text_embeds, text_embeds_teacher)
            # loss_it_hid = (loss_i_hid + loss_t_hid) / 2
            # loss += loss_i_hid

        if not (self.add_itc or self.itc_distill):
            with torch.no_grad():
                image_feat_world = concat_all_gather(image_feat)
                text_feat_world = concat_all_gather(text_feat)
                # print(text_feat_world.shape,'heloollllll')
                sim_i2t = image_feat @ text_feat_world.t() /self.temp
                sim_t2i = text_feat @ image_feat_world.t() /self.temp
                # sim_i2t = image_feat @ text_feat_all / self.temp.float()
                # sim_t2i = text_feat @ image_feat_all / self.temp.float()
        #
        if self.add_itc:
            # image_feat_all = concat_all_gather(image_feat)
            # text_feat_all = concat_all_gather(text_feat)
            # sim_i2t = image_feat_all @ text_feat_all.t() / self.temp.float()
            # sim_t2i = text_feat_all @ image_feat_all.t() / self.temp.float()
            # bsz = image_feat_all.shape[0]
            # labels = torch.arange(bsz, device=image_feat.device)
            # loss_i2t = F.cross_entropy(sim_i2t, labels)
            # loss_t2i = F.cross_entropy(sim_t2i, labels)

            sim_i2t = image_feat @ text_feat_all / self.temp
            sim_t2i = text_feat @ image_feat_all / self.temp
            loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1).mean()
            loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1).mean()
            loss_itc = (loss_i2t + loss_t2i) / 2




        if self.add_itc:
            self._dequeue_and_enqueue(image_feat_m, text_feat_m)
        elif self.itc_distill:
            self._dequeue_and_enqueue(image_feat_teacher, text_feat_teacher)
        # forward the positve image-text pair

        encoder_output_pos = self.text_encoder.bert(
            encoder_embeds=text_embeds,
            attention_mask=text.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
            mode="fusion",
        )
        with torch.no_grad():
            bs = image.size(0)

            # image_feat_world = concat_all_gather(image_feat)
            # text_feat_world = concat_all_gather(text_feat)
            #
            # sim_i2t = image_feat @ text_feat_world.t() / self.temp.float()
            # sim_t2i = text_feat @ image_feat_world.t() / self.temp.float()
            # weights_i2t = F.softmax(sim_i2t, dim=1)
            # weights_i2t.masked_fill_(mask, 0)
            #
            # weights_t2i = F.softmax(sim_t2i, dim=1)
            # weights_t2i.masked_fill_(mask, 0)

            # select hard negative sample based on the teacher's output
            weights_i2t = sim_i2t[:, :bs].clone()
            weights_t2i = sim_t2i[:, :bs].clone()
            weights_i2t[:,:bs].fill_diagonal_(-np.Inf)
            weights_t2i[:,:bs].fill_diagonal_(-np.Inf)

            # weights_i2t = sim_i2t_teacher[:, :].clone()
            # weights_t2i = sim_t2i_teacher[:, :].clone()
            # weights_i2t[:,sim_i2t_teacher.size()[0]* int(os.environ["RANK"]): ].fill_diagonal_(-np.Inf)
            # weights_t2i[:,sim_i2t_teacher.size()[0]* int(os.environ["RANK"]): ].fill_diagonal_(-np.Inf)

            weights_i2t = F.softmax(weights_i2t, dim=1) + 1e-5
            weights_t2i = F.softmax(weights_t2i, dim=1) + 1e-5

        # image_embeds_world = all_gather_with_grad(image_embeds)
        # select a negative image for each text
        image_embeds_neg = []
        image_negids = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_negids.append(neg_idx)
            image_embeds_neg.append(image_embeds[neg_idx])
            # image_embeds_neg.append(image_embeds_world[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg, dim=0)

        # select a negative text for each image
        text_embeds_neg = []
        text_atts_neg = []
        # text_embeds_world = all_gather_with_grad(text_embeds)
        # att_mask_world = concat_all_gather(text.attention_mask)
        text_nedids = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_nedids.append(neg_idx)
            text_embeds_neg.append(text_embeds[neg_idx])
            text_atts_neg.append(text.attention_mask[neg_idx])
            # text_embeds_neg.append(text_embeds_world[neg_idx])
            # text_atts_neg.append(att_mask_world[neg_idx])
        text_embeds_neg = torch.stack(text_embeds_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        text_embeds_all = torch.cat([text_embeds, text_embeds_neg], dim=0)
        text_atts_all = torch.cat([text.attention_mask, text_atts_neg], dim=0)

        image_embeds_all = torch.cat([image_embeds_neg, image_embeds], dim=0)
        image_atts_all = torch.cat([image_atts, image_atts], dim=0)
        # print('s neg')
        encoder_output_neg = self.text_encoder.bert(
            encoder_embeds=text_embeds_all,
            attention_mask=text_atts_all,
            encoder_hidden_states=image_embeds_all,
            encoder_attention_mask=image_atts_all,
            return_dict=True,
            mode="fusion",
        )

        vl_embeddings = torch.cat([encoder_output_pos.last_hidden_state[:, 0, :],
                                       encoder_output_neg.last_hidden_state[:, 0, :],],dim=0,)

        itm_logits = self.itm_head(vl_embeddings)
        if self.add_itm:
            itm_labels = torch.cat([torch.ones(bs, dtype=torch.long),
                                    torch.zeros(2 * bs, dtype=torch.long)],dim=0,).to(self.device)

            loss_itm = F.cross_entropy(itm_logits, itm_labels)
        else:
            itm_labels =0.


        # MLM
        if self.add_mlm:
            input_ids = text.input_ids.clone()
            labels = input_ids.clone()

            probability_matrix = torch.full(labels.shape, self.mlm_probability)
            input_ids, labels = self.mask(
                input_ids,
                self.text_encoder.config.vocab_size,
                self.device,
                targets=labels,
                probability_matrix=probability_matrix,
            )

            with torch.no_grad():
                if self.add_itc:
                    logits_m = self.text_encoder_m(
                        input_ids,
                        attention_mask=text.attention_mask,
                        encoder_hidden_states=image_embeds_m,
                        encoder_attention_mask=image_atts,
                        return_dict=True,
                        return_logits=True,)
                # elif self.itc_distill:
                #     logits_m = self.teacher.text_encoder(
                #         input_ids,
                #         attention_mask=text.attention_mask,
                #         encoder_hidden_states=image_embeds_teacher,
                #         encoder_attention_mask=image_atts,
                #         return_dict=True,
                #         return_logits=True,)

            mlm_output = self.text_encoder(
                input_ids,
                attention_mask=text.attention_mask,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
                labels=labels,
                # soft_labels=F.softmax(logits_m, dim=-1),
                alpha=alpha,
                mode='multimodal',
            )
            loss_mlm = mlm_output.loss
            loss += loss_mlm



        if self.teacher :
            with torch.no_grad():
                image_embeds_t = self.teacher.visual_encoder.forward_features(image)
                text_embeds_t = self.teacher.text_encoder.bert(
                    text.input_ids,
                    attention_mask=text.attention_mask,
                    return_dict=True,
                    mode="text",

                ).last_hidden_state
                # self.text_encoder.fit_denses[i](hid)
                # loss_i_hid = loss_mse(image_embeds_t, self.visual_encoder.fit_denses[-1](image_embeds))
                loss_t_hid = loss_mse(text_embeds_t, self.text_encoder.fit_denses[6](text_embeds))
                # loss_it_hid = (loss_i_hid + loss_t_hid) /2
                # 不除以2
                # print('loss_i_hid', loss_i_hid, image_embeds_t.shape)
                # loss_it_hid = (loss_i_hid + loss_t_hid)
                loss_it_hid = loss_t_hid
                loss += loss_it_hid


                encoder_output_pos_t = self.teacher.text_encoder.bert(
                    encoder_embeds=text_embeds_t,
                    attention_mask=text.attention_mask,
                    encoder_hidden_states=image_embeds_t,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                    mode="fusion",
                )
                # image_embeds_t_world = concat_all_gather(image_embeds_t)
                image_embeds_neg_t = []
                for b in range(bs):
                    # neg_idx = torch.multinomial(weights_t2i[b], 1).item()
                    neg_idx = image_negids[b]
                    image_embeds_neg_t.append(image_embeds_t[neg_idx])
                    # image_embeds_neg_t.append(image_embeds_t_world[neg_idx])
                image_embeds_neg_t = torch.stack(image_embeds_neg_t, dim=0)

                # text_embeds_t_world = concat_all_gather(text_embeds_t)
                # text_mask_world = concat_all_gather(text.attention_mask)
                text_embeds_neg_t = []
                text_atts_neg_t = []
                for b in range(bs):
                    # neg_idx = torch.multinomial(weights_i2t[b], 1).item()
                    neg_idx = text_nedids[b]
                    text_embeds_neg_t.append(text_embeds_t[neg_idx])
                    text_atts_neg_t.append(text.attention_mask[neg_idx])
                    # text_embeds_neg_t.append(text_embeds_t_world[neg_idx])
                    # text_atts_neg_t.append(text_mask_world[neg_idx])

                text_embeds_neg_t = torch.stack(text_embeds_neg_t, dim=0)
                text_atts_neg_t = torch.stack(text_atts_neg_t, dim=0)

                text_embeds_all_t = torch.cat([text_embeds_t, text_embeds_neg_t], dim=0)
                text_atts_all_t = torch.cat([text.attention_mask, text_atts_neg_t], dim=0)

                image_embeds_all_t = torch.cat([image_embeds_neg_t, image_embeds_t], dim=0)
                image_atts_all_t = torch.cat([image_atts, image_atts], dim=0)
                encoder_output_neg_t = self.teacher.text_encoder.bert(
                    encoder_embeds=text_embeds_all_t,
                    attention_mask=text_atts_all_t,
                    encoder_hidden_states=image_embeds_all_t,
                    encoder_attention_mask=image_atts_all_t,
                    return_dict=True,
                    mode="fusion",
                )

                vl_embeddings_t = torch.cat(
                    [
                        encoder_output_pos_t.last_hidden_state[:, 0, :],
                        encoder_output_neg_t.last_hidden_state[:, 0, :],
                    ],
                    dim=0,
                )

                itm_logits_t = self.teacher.itm_head(vl_embeddings_t)

            # loss += loss_logits

            # print("encoder_output_pos_t", len(encoder_output_pos_t.cross_attentions))

            # speedup 1.5x
        if self.add_att:
            student_xatts = ()
            teacher_xatts = ()
            for p, n in zip(encoder_output_pos.cross_attentions, encoder_output_neg.cross_attentions):
                xatt = torch.cat([p,n], dim=0)
                student_xatts = student_xatts + (xatt,)
            for p, n in zip(encoder_output_pos_t.cross_attentions, encoder_output_neg_t.cross_attentions):
                xatt = torch.cat([p,n], dim=0)
                teacher_xatts = teacher_xatts + (xatt,)
            teacher_xatts = [teacher_xatt.detach() for teacher_xatt in teacher_xatts]

            # loss_att =self.get_att_loss(teacher_xatts, student_xatts)
            # 取最后一层x-att
            loss_att =self.get_att_loss(teacher_xatts[-1:], student_xatts[-1:])
            loss += loss_att
        if self.add_hid:
            student_reps = ()
            teacher_reps = ()
            for p, n in zip(encoder_output_pos.hidden_states, encoder_output_neg.hidden_states):
                rep = torch.cat([p,n], dim=0)
                student_reps = student_reps + (rep,)
            for p, n in zip(encoder_output_pos_t.hidden_states, encoder_output_neg_t.hidden_states):
                rep = torch.cat([p,n], dim=0)
                teacher_reps = teacher_reps + (rep,)
            teacher_reps = [teacher_rep.detach() for teacher_rep in teacher_reps]
            # loss_hid = self.get_hid_loss(teacher_reps, student_reps)
            # 取最后一层hid
            loss_hid = self.get_hid_loss(teacher_reps[-1:], (self.text_encoder.fit_denses[-1](student_reps[-1]),))
            loss += loss_hid

        if self.add_itc:

            loss += loss_itc
        if self.add_itm:

            loss += loss_itm
        if self.add_logits:
            loss_logits = self.soft_cross_entropy(itm_logits/ 1.0,
                                                  itm_logits_t/ 1.0)
            loss += loss_logits

        # print(loss_att+loss_hid, loss_att, loss_hid,)
        # print('loss of logits:', loss_logits,itm_logits[0],itm_logits_t[0])
        return AlbefOutput(
            loss=loss,
            loss_img=loss_cls,
            loss_logits=loss_logits,
            loss_reps = loss_att+loss_hid,
            loss_itc_distill = loss_itc_distill,
            loss_it_hid = loss_it_hid,
            loss_itc=loss_itc,
            loss_itm=loss_itm,
            loss_mlm=loss_mlm,
            sims=AlbefSimilarity(
                sim_i2t=sim_i2t,
                sim_t2i=sim_t2i,
                sim_i2t_m=sim_i2t_m,
                sim_t2i_m=sim_t2i_m,
                sim_i2t_targets=sim_i2t_targets,
                sim_t2i_targets=sim_t2i_targets,
            ),
            intermediate_output=AlbefIntermediateOutput(
                image_embeds=image_embeds,
                image_embeds_m=image_embeds_m,
                text_embeds=text_embeds,
                text_embeds_m=text_embeds_m,
                encoder_output=encoder_output_pos,
                encoder_output_neg=encoder_output_neg,
                itm_logits=itm_logits,
                itm_labels=itm_labels,
            ),
        )

    def mask(self, input_ids, vocab_size, device, targets=None, masked_indices=None, probability_matrix=None,):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        if masked_indices is None:
            masked_indices = torch.bernoulli(probability_matrix).bool()

        masked_indices[input_ids == self.tokenizer.pad_token_id] = False
        masked_indices[input_ids == self.tokenizer.cls_token_id] = False

        if targets is not None:
            targets[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = (
            torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        )
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = (
            torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(vocab_size, input_ids.shape, dtype=torch.long).to(
            device
        )
        input_ids[indices_random] = random_words[indices_random]
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged

        if targets is not None:
            return input_ids, targets
        else:
            return input_ids


    @classmethod
    def from_config(cls, cfg=None):

        embed_dim = cfg.get("embed_dim", 256)
        momentum = cfg.get("momentum", 0.995)
        alpha = cfg.get("alpha", 0.4)
        print('alpha,',alpha)
        mlm_mask_prob = cfg.get("mlm_mask_prob", 0.15)
        temp = cfg.get("temp", 0.07)
        max_txt_len = cfg.get("max_txt_len", 30)
        queue_size = cfg.get("queue_size", 65536)

        teacher_model = None
        has_teacher = cfg.get("has_teacher",False)
        has_l_teacher = cfg.get("has_l_teacher",False)
        has_v_teacher = cfg.get("has_v_teacher",False)

        output_attentions = cfg.get("output_attentions", False)
        output_hidden_states = cfg.get("output_hidden_states", False)

        caption_distill = cfg.get("caption_distill", False)
        add_mlm = cfg.get("add_mlm", False)
        add_itc = cfg.get("add_itc", False)
        add_itm = cfg.get("add_itm", False)
        add_logits = cfg.get("add_logits", False)
        add_att = cfg.get("add_att", False)
        add_hid = cfg.get("add_hid", False)
        itc_distill = cfg.get("itc_distill", False)


        config_text_t = BertConfig.from_json_file(get_abs_path(cfg["med_config_path"]))
        config_text_t.output_attentions = output_attentions
        config_text_t.output_hidden_states = output_hidden_states
        if has_teacher:

            cfg["vit_depth"] = 12
            # image_encoder = VisionTransformerEncoder.from_config(cfg, from_pretrained=False)
            image_encoder, vis_width = SwinTransformerEncoder.from_config(cfg, load_params=False)
            config_text_t.fusion_layer = 6
            config_text_t.num_hidden_layers = 12
            config_text_t.add_moe = False
            config_text_t.hidden_size = 1024
            config_text_t.num_attention_heads = 16

            # ck_path = '/share2/liangjx/pro'
            # text_encoder = BertForMaskedLM.from_pretrained("./pretrained_ckpt/bert-base-uncased", config=config_text_t)
            text_encoder = BertForMaskedLM(config=config_text_t)

            teacher_model = cls(
                image_encoder=image_encoder,
                text_encoder=text_encoder,
                queue_size=queue_size,
                embed_dim=embed_dim,
                mlm_mask_prob=mlm_mask_prob,
                temp=temp,
                momentum=momentum,
                alpha=alpha,
                max_txt_len=max_txt_len,
                is_teacher=True,
            )
            pretrained_ckpt = cfg.get("pretrained")
            print('path of ck:', pretrained_ckpt)
            teacher_model.load_from_pretrained(pretrained_ckpt,
                                               rename_text_keys=False)
        # has_l_teacher = True
        # has_v_teacher = False
        if has_l_teacher:
            l_config= BertConfig.from_json_file(get_abs_path(cfg["med_config_path"]))
            l_config.add_cross_attention = False
            l_config.use_adapter =False
            l_config.output_attentions = output_attentions
            l_config.output_hidden_states = output_hidden_states
            l_config.add_moe = False
            l_config.num_hidden_layers = 12
            l_teacher = BertForMaskedLM.from_pretrained("bert-base-uncased", config=l_config)
        if has_v_teacher:
            # v_config = BeitConfig.from_json_file(get_abs_path(cfg["student_config_path"]))
            # v_config.num_hidden_layers = 12
            # v_teacher = BeitModel(config=v_config)
            # beit_ckpt = cfg['beit_ckpt']
            # v_teacher.load_pretrained(ckpt_rpath=beit_ckpt,load_beit_by_sep=False)
            cfg["vit_depth"] = 12
            v_teacher = VisionTransformerEncoder.from_config(cfg, from_pretrained=True)

        cfg["vit_depth"] = 6
        image_encoder = VisionTransformerEncoder.from_config(cfg, from_pretrained=True,
                                                             vision_width=768,use_moe=False)
        config_text_encoder = BertConfig.from_json_file(
            get_abs_path(cfg["med_config_path"])
        )
        config_text_encoder.output_attentions = output_attentions
        config_text_encoder.output_hidden_states = output_hidden_states
        config_text_encoder.fusion_layer = 3
        config_text_encoder.num_hidden_layers = 6
        config_text_encoder.hidden_size = 768
        config_text_encoder.encoder_width = 768
        config_text_encoder.add_moe = False
        ck_path = "bert-base-uncased"
        ck_path = './pretrained_ckpt/tinybertv2'
        # text_encoder = BertForMaskedLM.from_pretrained(
        #     ck_path, config=config_text_encoder, add_fit=True)
        text_encoder = BertForMaskedLM(config=config_text_encoder, add_fit=True)
        for name in text_encoder.state_dict():
            if 'fit_denses' in name:
                    print(name)
        print(text_encoder.state_dict()["fit_denses.6.weight"][0,:5], 'fit_dense.6')
        model = cls(
            image_encoder=image_encoder,
            text_encoder=text_encoder,
            add_mlm=add_mlm,
            add_itc=add_itc,
            add_itm=add_itm,
            add_logits=add_logits,
            add_hid=add_hid,
            add_att=add_att,
            itc_distill=itc_distill,
            caption_distill=caption_distill,
            queue_size=queue_size,
            embed_dim=embed_dim,
            mlm_mask_prob=mlm_mask_prob,
            temp=temp,
            momentum=momentum,
            alpha=alpha,
            max_txt_len=max_txt_len,
            teacher=teacher_model,
            l_teacher = l_teacher if has_l_teacher else None,
            v_teacher = v_teacher if has_v_teacher else None,

        )
        # print('tea fusion:', model.teacher.text_encoder.bert.config)
        pretrained_ckpt_stu = cfg.get("pretrained_stu")
        if not pretrained_ckpt_stu:
            pretrained_ckpt_stu = cfg.get("pretrained")
        print('preckpt:', pretrained_ckpt_stu)
        model.load_from_pretrained(pretrained_ckpt_stu, rename_text_keys=False,load_albef_by_sep=False,covertiny=True)
        # 12 384
        # model.load_from_pretrained(pretrained_ckpt_stu, rename_text_keys=False,load_albef_by_sep=False,covertiny=True)





        return model
