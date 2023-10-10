"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import os,sys
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from lavis.common.registry import registry
from lavis.common.utils import get_abs_path
from lavis.models.vilt_models import ViltBase
from lavis.models.xvlm_models import XVLMBase

from lavis.models.base_model import BaseModel

from lavis.models.mix_models.mix_outputs import (
    # AlbefIntermediateOutput,
    MixDistillOutput,
    # AlbefSimilarity,
)

from lavis.models.base_model import MomentumDistilationMixin, SharedQueueMixin
from lavis.models.med import BertForMaskedLM
from lavis.models.vit import VisionTransformerEncoder
from torch import nn
base_dir = os.path.dirname('/home1/liangjx/projects/mm/')
sys.path.append(base_dir)
# from trans.models.vilt import ViltConfig, ViltModel,  ViltForMaskedLM
from transformers import BertModel, BertConfig, ViltForMaskedLM,ViltConfig,ViltModel
from lavis.models.mix_models.mixmome import BeitModel, BeitOnlyMLMHead


from transformers import BeitConfig

class AllGather(torch.autograd.Function):
    """An autograd function that performs allgather on a tensor."""

    @staticmethod
    def forward(ctx, tensor, rank, world_size):
        output = [torch.empty_like(tensor) for _ in range(world_size)]
        torch.distributed.all_gather(output, tensor)
        ctx.rank = rank
        ctx.batch_size = tensor.shape[0]
        return torch.cat(output, 0)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            grad_output[ctx.batch_size * ctx.rank: ctx.batch_size * (ctx.rank + 1)],
            None,
            None
        )


allgather = AllGather.apply


# @registry.register_model("xvlm_pretrain")
class XVLM(XVLMBase):
    def __init__(self, config):
        super().__init__(config, load_vision_params=False, load_text_params=False,
                         use_contrastive_loss=True, use_matching_loss=True, use_mlm_loss=False, use_bbox_loss=False, config_text=None)

    @classmethod
    def from_config(cls, cfg=None):

        return True
    # def forward(self, batch):
    def forward(self, image, text_ids, text_atts, text_ids_masked=None, masked_pos=None, masked_ids=None,
                image_atts=None, idx_to_group_img=None, target_bbox=None, is_image=None, ret_bbox_loss=False,
                image_neg_ids=None, text_neg_ids=None):

        # if ret_bbox_loss:
        #     image_embeds, image_atts, image_embeds_fullatts = \
        #         self.get_vision_embeds(image, image_atts=image_atts, idx_to_group_img=idx_to_group_img)
        # else:
        image_embeds, image_atts = self.get_vision_embeds(image)

        text_embeds = self.get_text_embeds(text_ids, text_atts)

        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)

        image_feat, text_feat = self.get_features(image_embeds, text_embeds)

        loss_itc = self.get_contrastive_loss(image_feat, text_feat)
        logits_itm = self.get_matching_loss(image_embeds, image_atts, image_feat,
                                            text_embeds, text_atts, text_feat,
                                            image_neg_ids=image_neg_ids, text_neg_ids=text_neg_ids)
        loss_mlm = self.get_mlm_loss(text_ids_masked, text_atts, image_embeds, image_atts, masked_pos, masked_ids)

        # if ret_bbox_loss:
        #     output_coord = self.predict_bbox(image_embeds_fullatts, text_embeds, text_atts)
        #     loss_bbox, loss_giou = self.get_bbox_loss(output_coord, target_bbox, is_image=is_image)
        #
        #     return loss_itc, loss_itm, loss_mlm, loss_bbox, loss_giou

        # return loss_itc, loss_itm, loss_mlm
        return logits_itm




class MixDistill(ViltBase, MomentumDistilationMixin,SharedQueueMixin):
    """
    ALBEF pretrain model.

    Supported model types:
        - base: ALBEF base model used for pretraining.
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "distill": "configs/models/mix_distill.yaml",
    }

    def __init__(self,text_encoder, image_encoder, text_lm_head,
        student, teacher, student_config,
        add_itm, add_itc, add_mlm,
        pretrained_teacher=None,
        embed_dim=None,
        encoder_width=768,
        temp=0.07, max_txt_len=30, mlm_mask_prob=0.15, fit_size=768,
    ):
        super().__init__()

        student_hidden_size = student_config.hidden_size
        self.tokenizer = self.init_tokenizer(pretrained_ckpt=pretrained_teacher)

        self.add_itm=add_itm
        self.add_itc=add_itc
        self.add_mlm=add_mlm
        self.temp = temp
        self.max_txt_len = max_txt_len

        self.image_encoder = image_encoder
        self.text_encoder = text_encoder

        self.student_model = student
        self.teacher_model = teacher
        self.loss_mse = torch.nn.MSELoss()
        self.fit_dense = nn.Linear(student_hidden_size, fit_size)

        # if self.add_itc:
        self.embed_dim = embed_dim
        self.encoder_width = encoder_width
        self.vision_proj = nn.Linear(self.encoder_width, self.embed_dim)
        self.text_proj = nn.Linear(self.encoder_width, self.embed_dim)

        if self.add_itm:
            self.itm_head = nn.Linear(student_hidden_size, 2)
        if self.add_mlm:
            self.mlm_probability = mlm_mask_prob
            self.text_lm_head = text_lm_head
        # self.get_cross_embeds=None

        # create the momentum encoder
        self.visual_encoder_m = deepcopy(self.image_encoder)
        self.text_encoder_m = deepcopy(self.text_encoder)

        self.vision_proj_m = deepcopy(self.vision_proj)
        self.text_proj_m = deepcopy(self.text_proj)

        self.model_pairs = [
            [self.image_encoder, self.visual_encoder_m],
            [self.text_encoder, self.text_encoder_m],
            [self.vision_proj, self.vision_proj_m],
            [self.text_proj, self.text_proj_m],
        ]
        self.copy_params()

        # create the queue
        queue_size = 65536
        self.register_buffer("image_queue", torch.randn(embed_dim, queue_size))
        self.register_buffer("text_queue", torch.randn(embed_dim, queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)

        self.queue_size = queue_size
        self.momentum = 0.995



    def _rampup_factor(self, epoch, iters, num_iters_per_epoch):
        return min(1, (epoch * num_iters_per_epoch + iters) / (2 * num_iters_per_epoch))

    def get_contrastive_loss(self, text_feat, image_feat, idx=None):
        """
        Args:
            image_feat, text_feat: normalized
        Returns: contrastive loss
        """
        # bs = batch['input_ids'].size(0)
        text_feat = F.normalize(self.text_proj(text_feat[:,0,:].clone()),dim=-1)
        image_feat = F.normalize(self.vision_proj(image_feat[:,0,:].clone()), dim=-1)

        image_feat_all =image_feat
        text_feat_all = text_feat
        # image_feat_all = allgather(image_feat, torch.distributed.get_rank(), torch.distributed.get_world_size())
        # text_feat_all = allgather(text_feat, torch.distributed.get_rank(), torch.distributed.get_world_size())
        logits = image_feat_all @ text_feat_all.t() / self.temp

        bsz = image_feat_all.shape[0]

        if idx is None:
            labels = torch.arange(bsz, device=image_feat.device)
            loss_i2t = F.cross_entropy(logits, labels)
            loss_t2i = F.cross_entropy(logits.t(), labels)

        else:
            idx = idx.view(-1, 1)
            assert idx.size(0) == image_feat.size(0)
            idx_all = allgather(idx, torch.distributed.get_rank(), torch.distributed.get_world_size())
            pos_idx = torch.eq(idx_all, idx_all.t()).float()
            labels = pos_idx / pos_idx.sum(1, keepdim=True)

            loss_i2t = -torch.sum(F.log_softmax(logits, dim=1) * labels, dim=1).mean()
            loss_t2i = -torch.sum(F.log_softmax(logits.t(), dim=1) * labels, dim=1).mean()

        return (loss_i2t + loss_t2i) / 2

    def get_matching_loss(self, text_hidden_state, text_attention_mask,image_hidden_state,
                          image_attention_mask, x_outputs_pos,sim_i2t=None,sim_t2i=None,idx=None):
        """
        Matching Loss with hard negatives
        """
        # batch = None
        bs = text_hidden_state.size(0)

        text_feat = F.normalize(self.text_proj(text_hidden_state[:,0,:].clone()),dim=-1)
        image_feat = F.normalize(self.vision_proj(image_hidden_state[:,0,:].clone()), dim=-1)
        with torch.no_grad():
            if sim_i2t is not None and sim_t2i is not None:
                sim_i2t = sim_i2t[:, :bs].clone()
                sim_t2i = sim_t2i[:, :bs].clone()
            else:
                sim_i2t = image_feat @ text_feat.t() / self.temp
                sim_t2i = text_feat @ image_feat.t() / self.temp



            weights_i2t = F.softmax(sim_i2t, dim=1) + 1e-5
            weights_t2i = F.softmax(sim_t2i, dim=1) + 1e-5

            if idx is None:
                weights_i2t.fill_diagonal_(0)
                weights_t2i.fill_diagonal_(0)
            else:
                idx = idx.view(-1, 1)
                assert idx.size(0) == bs
                mask = torch.eq(idx, idx.t())
                weights_i2t.masked_fill_(mask, 0)
                weights_t2i.masked_fill_(mask, 0)

        image_hidden_state_neg, image_attention_mask_neg = [], []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_hidden_state_neg.append(image_hidden_state[neg_idx])
            image_attention_mask_neg.append(image_attention_mask[neg_idx])
        image_hidden_state_neg = torch.stack(image_hidden_state_neg, dim=0)
        image_attention_mask_neg = torch.stack(image_attention_mask_neg, dim=0)

        x_hidden_states_neg1 = torch.cat((text_hidden_state, image_hidden_state_neg), dim=1)
        x_attention_mask_neg1 = torch.cat((text_attention_mask,
                                           image_attention_mask_neg), dim=-1)
        x_outputs_neg1 = self.image_encoder.encoder(x_hidden_states_neg1,
                                                     x_attention_mask_neg1,
                                                     mode='fusion')

        text_hidden_state_neg, text_attention_mask_neg = [], []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_hidden_state_neg.append(text_hidden_state[neg_idx])
            text_attention_mask_neg.append(text_attention_mask[neg_idx])
        text_hidden_state_neg = torch.stack(text_hidden_state_neg, dim=0)
        text_attention_mask_neg = torch.stack(text_attention_mask_neg, dim=0)

        x_hidden_states_neg2 = torch.cat((text_hidden_state_neg, image_hidden_state), dim=1)
        x_attention_mask_neg2 = torch.cat((text_attention_mask_neg,
                                           image_attention_mask), dim=-1)
        x_outputs_neg2 = self.image_encoder.encoder(x_hidden_states_neg2,
                                                     x_attention_mask_neg2,
                                                     mode='fusion')

        output = self.itm_head(torch.cat([x_outputs_pos.last_hidden_state[:, 0, :],
                                          x_outputs_neg1.last_hidden_state[:, 0, :],
                                          x_outputs_neg2.last_hidden_state[:, 0, :]], dim=0))
        itm_labels = torch.cat([torch.ones(bs, dtype=torch.long),
                                torch.zeros(2 * bs, dtype=torch.long)], dim=0).to(self.device)

        return F.cross_entropy(output, itm_labels)

    def get_mlm_loss(self,text_batch, image_hidden_state, cross_attention_mask):
        text_batch_mlm = text_batch.copy()
        input_ids = text_batch_mlm['input_ids'].clone()
        labels = input_ids.clone()
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        input_ids, labels = self.mask(
            input_ids,
            self.text_encoder.config.vocab_size,
            self.device,
            targets=labels,
            probability_matrix=probability_matrix,
        )
        # print(labels)
        text_batch_mlm['input_ids'] = input_ids
        # text_batch_mlm['labels'] = labels
        text_hs_mlm = self.text_encoder(**text_batch_mlm)

        t_hidden_state_mlm = text_hs_mlm.last_hidden_state
        x_hidden_states_mlm = torch.cat((t_hidden_state_mlm, image_hidden_state), dim=1)

        x_outputs = self.image_encoder.encoder(x_hidden_states_mlm,
                                               cross_attention_mask,
                                               output_attentions=True,
                                               output_hidden_states=True,
                                               mode='fusion')
        hs = x_outputs.last_hidden_state
        text_output_mlm = self.text_lm_head(hs[:, :self.max_txt_len], labels=labels)
        return text_output_mlm.loss

    def mask(
            self,
            input_ids,
            vocab_size,
            device,
            targets=None,
            masked_indices=None,
            probability_matrix=None,
    ):
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

    def forward(self, batch):

        if batch["text_input"] is not None:
            text_encoding = self.tokenizer(batch["text_input"],
                                      padding="max_length",
                                      truncation=True,
                                      max_length=self.max_txt_len,
                                      return_tensors="pt",)
        # pixel_values + pixel_mask
        # text_batch =
        if batch['image'] is not None:
            image_encoding = batch['image']
            for k, v in image_encoding.items():
                image_encoding[k] = v.squeeze()
            # encoding.update(encoding_feature_extractor)
        loss_itm = None
        loss_itc = None
        loss_att = None
        loss_rep = None
        loss = 0.
        # outputs = self.student_model(**batch)
        # student_atts, student_reps = outputs.attentions, outputs.hidden_states
        assert batch["text_input"] is not None or batch['image'] is not None, \
            " There shoule be at least image or text input. "

        if batch["image"] is None:
            batch = {k: v.to(self.device) for k, v in text_encoding.items()}
            text_output = self.text_encoder(**batch)
            text_hidden_state = text_output.last_hidden_state
            student_outputs = self.image_encoder.beit.encoder(text_hidden_state,
                                                              text_encoding.attention_mask,
                                                              mode='text')
            # outputs_m = self.teacher_model(**batch)
        elif batch["text_input"] is None:
            batch = {k: v.to(self.device) for k, v in image_encoding.items()}
            batch['mode'] = 'image'
            img_output = self.image_encoder(**batch)
            student_outputs = img_output

        elif batch["text_input"] is not None and batch['image'] is not None:
            text_batch = {k: v.to(self.device) for k, v in text_encoding.items()}
            image_batch = {k: v.to(self.device) for k, v in image_encoding.items()}

            assert  text_batch['input_ids'].shape[0] == image_batch['pixel_values'].shape[0], "batch of text and image should be identied"

            image_batch['mode'] = 'image'
            image_batch['return_dict'] = True
            img_output = self.image_encoder(**image_batch)
            text_output = self.text_encoder(**text_batch)

            t_hidden_state = text_output.last_hidden_state
            i_hidden_state = img_output.last_hidden_state

            text_mask = text_encoding.attention_mask[:,None,None,:]
            t_attention_mask = (1.0 - text_mask) * \
                                  torch.finfo(self.image_encoder.img_attention_mask.dtype).min
            t_attention_mask = t_attention_mask.to(self.device)
            i_attention_mask = self.image_encoder.img_attention_mask
            x_hidden_states = torch.cat((t_hidden_state, i_hidden_state), dim=1)
            x_attention_mask = torch.cat((t_attention_mask,
                                          i_attention_mask), dim=-1)
            x_outputs_pos = self.image_encoder.encoder(x_hidden_states,
                                                       x_attention_mask,
                                                       output_attentions=True,
                                                       output_hidden_states=True,
                                                       mode='fusion')
            # get momentum features
            moco=False
            if moco:
                with torch.no_grad():
                    text_feat = F.normalize(self.text_proj(t_hidden_state[:, 0, :]), dim=-1)
                    image_feat = F.normalize(self.vision_proj(i_hidden_state[:, 0, :]), dim=-1)
                    self._momentum_update()
                    # image_embeds_m = self.visual_encoder_m(image)
                    image_embeds_m = i_hidden_state.clone()
                    image_feat_m = F.normalize(
                        self.vision_proj_m(image_embeds_m[:, 0, :]), dim=-1
                    )
                    image_feat_all = torch.cat(
                        [image_feat_m.t(), self.image_queue.clone().detach()], dim=1
                    )
                    # text_output_m = self.text_encoder_m.bert(
                    #     text.input_ids,
                    #     attention_mask=text.attention_mask,
                    #     return_dict=True,
                    #     mode="text",
                    # )
                    text_embeds_m = t_hidden_state.clone()
                    text_feat_m = F.normalize(self.text_proj_m(text_embeds_m[:, 0, :]), dim=-1)
                    text_feat_all = torch.cat(
                        [text_feat_m.t(), self.text_queue.clone().detach()], dim=1
                    )

                    sim_i2t_m = image_feat_m @ text_feat_all / self.temp
                    sim_t2i_m = text_feat_m @ image_feat_all / self.temp

                    sim_targets = torch.zeros(sim_i2t_m.size()).to(self.device)
                    sim_targets.fill_diagonal_(1)
                    alpha = 0.4
                    sim_i2t_targets = (
                            alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
                    )
                    sim_t2i_targets = (
                            alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets
                    )

                sim_i2t = image_feat @ text_feat_all / self.temp
                sim_t2i = text_feat @ image_feat_all / self.temp

                loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1).mean()
                loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1).mean()
                loss_itc = (loss_i2t + loss_t2i) / 2

                self._dequeue_and_enqueue(image_feat_m, text_feat_m)
            itm=False
            if itm:
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

                    weights_i2t = sim_i2t[:, :bs].clone()
                    weights_t2i = sim_t2i[:, :bs].clone()

                    weights_i2t.fill_diagonal_(-np.Inf)
                    weights_t2i.fill_diagonal_(-np.Inf)

                    weights_i2t = F.softmax(weights_i2t, dim=1)
                    weights_t2i = F.softmax(weights_t2i, dim=1)

                # select a negative image for each text
                image_embeds_neg = []
                for b in range(bs):
                    neg_idx = torch.multinomial(weights_t2i[b], 1).item()
                    image_embeds_neg.append(image_embeds[neg_idx])
                image_embeds_neg = torch.stack(image_embeds_neg, dim=0)

                # select a negative text for each image
                text_embeds_neg = []
                text_atts_neg = []
                for b in range(bs):
                    neg_idx = torch.multinomial(weights_i2t[b], 1).item()
                    text_embeds_neg.append(text_embeds[neg_idx])
                    text_atts_neg.append(text.attention_mask[neg_idx])
                text_embeds_neg = torch.stack(text_embeds_neg, dim=0)
                text_atts_neg = torch.stack(text_atts_neg, dim=0)

                text_embeds_all = torch.cat([text_embeds, text_embeds_neg], dim=0)
                text_atts_all = torch.cat([text.attention_mask, text_atts_neg], dim=0)

                image_embeds_all = torch.cat([image_embeds_neg, image_embeds], dim=0)
                image_atts_all = torch.cat([image_atts, image_atts], dim=0)

                encoder_output_neg = self.text_encoder.bert(
                    encoder_embeds=text_embeds_all,
                    attention_mask=text_atts_all,
                    encoder_hidden_states=image_embeds_all,
                    encoder_attention_mask=image_atts_all,
                    return_dict=True,
                    mode="fusion",
                )

                vl_embeddings = torch.cat(
                    [
                        encoder_output_pos.last_hidden_state[:, 0, :],
                        encoder_output_neg.last_hidden_state[:, 0, :],
                    ],
                    dim=0,
                )
                itm_logits = self.itm_head(vl_embeddings)

                itm_labels = torch.cat(
                    [torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
                    dim=0,
                ).to(self.device)
                loss_itm = F.cross_entropy(itm_logits, itm_labels)
            if self.add_itc:
                loss_itc = self.get_contrastive_loss(t_hidden_state, i_hidden_state)
                loss += loss_itc

            if self.add_itm:
                loss_itm = self.get_matching_loss(t_hidden_state, t_attention_mask,
                                                  i_hidden_state, i_attention_mask,x_outputs_pos,
                                                  # sim_i2t,sim_t2i,
                                                  )
                loss += loss_itm

            if self.add_mlm:
                loss_mlm = self.get_mlm_loss(text_batch, i_hidden_state, x_attention_mask)
                # print(loss_mlm)
                loss += loss_mlm
            # self.teacher_model=1
            if self.teacher_model is not None:
                student_atts, student_reps = x_outputs_pos.attentions, x_outputs_pos.hidden_states[1:]
                # with torch.no_grad():
                #     teacher_outputs = self.teacher_model(**batch)
                #     teacher_logits, teacher_atts, teacher_reps = teacher_outputs.logits, teacher_outputs.attentions, teacher_outputs.hidden_states
                #
                # teacher_layer_num = len(teacher_atts)
                teacher_layer_num = 12
                student_layer_num = len(student_atts)
                print(len(student_reps))
                assert teacher_layer_num % student_layer_num == 0
                layers_per_block = int(teacher_layer_num / student_layer_num)
                new_teacher_atts = [teacher_atts[i * layers_per_block + layers_per_block - 1]
                                    for i in range(student_layer_num)]

                for student_att, teacher_att in zip(student_atts, new_teacher_atts):
                    student_att = torch.where(student_att <= -1e2, torch.zeros_like(student_att).to(self.device),
                                              student_att)
                    teacher_att = torch.where(teacher_att <= -1e2, torch.zeros_like(teacher_att).to(self.device),
                                              teacher_att)

                    tmp_loss = self.loss_mse(student_att, teacher_att)
                    loss_att += tmp_loss

                new_teacher_reps = [teacher_reps[i * layers_per_block] for i in range(student_layer_num + 1)]
                new_student_reps = student_reps
                for student_rep, teacher_rep in zip(new_student_reps, new_teacher_reps):
                    tmp_loss = self.loss_mse(self.fit_dense(student_rep), teacher_rep)
                    loss_rep += tmp_loss

                # loss = loss_rep + loss_att + outputs.loss

                loss += loss_rep + loss_att

        else:
            raise ValueError("The input is invalid")

            # if self.add_itm:
                #     loss_itm = self.get_matching_loss(batch, last_hidden_state=outputs.last_hidden_state)
                #     loss += loss_itm
                # if self.add_itc:
                #     loss_itc = self.get_contrastive_loss(last_hidden_state=outputs.last_hidden_state)
                #     loss += loss_itc

        return MixDistillOutput(
            loss=loss,
            loss_rep=loss_rep,
            loss_att=loss_att,
            loss_itm=loss_itm,
            loss_itc=loss_itc,
        )

    @classmethod
    def from_config(cls, cfg=None):
        #TODO
        # print(cfg)
        add_itm = cfg.get("add_itm", False)
        add_itc = cfg.get("add_itc", False)
        add_mlm = cfg.get("add_mlm", False)
        embed_dim = cfg.get("embed_dim", 256)
        encoder_width = cfg.get("encoder_width", 768)
        fusion_layer = cfg.get("fusion_layer", 3)
        num_hidden_layers = cfg.get("num_hidden_layers", 6)
        # print(fusion_layer, num_hidden_layers)
        max_txt_len = cfg.get("max_txt_len", 30)
        temp = cfg.get("temp", 0.07)
        image_size = cfg.get("image_size", 256)


        student_config = BeitConfig.from_json_file(
            get_abs_path(cfg["student_config_path"]))
        student_config.fusion_layer = fusion_layer
        student_config.image_size = image_size
        student_config.num_hidden_layers = num_hidden_layers

        fit_size = student_config.hidden_size
        # student_config.is_student=True

        student = None

        image_encoder = BeitModel(config=student_config)
        # text_encoder = BertModel(config=student_config)
        med_config_path = get_abs_path(cfg.get("med_config_path"))
        med_config = BertConfig.from_json_file(med_config_path)
        text_encoder = BertModel(config=med_config)

        student_config.text_vocab_size = med_config.vocab_size
        text_lm_head_inbeit = BeitOnlyMLMHead(config=student_config)

        teacher = None
        # teacher_text = BertForMaskedLM.from_pretrained("bert-base-uncased", config=config_text_encoder)
        # teacher_img = BeitModel(config=)
        # teacher_x =


        # student = ViltModel(config=student_config)
        # teacher = ViltForMaskedLM.from_config(cfg, from_pretrained=True)

        teacher_ckpt = cfg.get("teacher_ckpt","vilt-b32-mlm")
        # teacher = ViltForMaskedLM.from_pretrained(teacher_ckpt)
        # cfg = ViltConfig.from_pretrained('dandelin/vilt-b32-mlm')
        # teacher = BeitModel(config=student_config)

        # teacher = ViltModel.from_pretrained(teacher_ckpt)
        # mlm_score = ViltMLMHead.from_config()
        # itm_score = ViltITMHead.from_config()


        # print('model_type::1',type(i_encoder))

        model = cls(
            text_encoder=text_encoder,
            image_encoder=image_encoder,
            text_lm_head=text_lm_head_inbeit,
            student=student,
            teacher=teacher,
            student_config=student_config,
            add_itm=add_itm,
            add_itc=add_itc,
            add_mlm=add_mlm,
            pretrained_teacher=teacher_ckpt,
            embed_dim=embed_dim,
            encoder_width=encoder_width,
            temp=temp,
            max_txt_len=max_txt_len,
        )
        return model
