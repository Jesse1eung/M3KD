"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from copy import deepcopy

import torch
from torch import nn
import torch.nn.functional as F

from lavis.common.registry import registry
from lavis.common.utils import get_abs_path

from lavis.models.vilt_models import ViltBase, compute_sim_matrix

from lavis.models.base_model import MomentumDistilationMixin, SharedQueueMixin
from lavis.models.med import XBertEncoder
from lavis.models.vit import VisionTransformerEncoder
from transformers import BertModel, BertConfig
from lavis.models.mix_models.mixmome import BeitModel, BeitOnlyMLMHead
from transformers import BeitConfig
from transformers import BertTokenizerFast
from lavis.models.mix_models.mix_outputs import MixDistillOutput


@registry.register_model("mix_retrieval")
class MixRetrieval(ViltBase, MomentumDistilationMixin, SharedQueueMixin):

    PRETRAINED_MODEL_CONFIG_DICT = {
        "coco": "configs/models/mix/retrieval_coco.yaml",
        "flickr": "configs/models/mix/retrieval_flickr.yaml",
    }

    def __init__(self,text_encoder, image_encoder, text_lm_head,
                 student, beit_config,
                 add_itm, add_itc, use_moco,
                 tokenizer_path=None, embed_dim=None, encoder_width=768,
                 temp=0.07, max_txt_len=30,  queue_size=65536,fit_size=768,
                 ):
        super().__init__()

        _hidden_size = beit_config.hidden_size
        # self.tokenizer = self.init_tokenizer(pretrained_ckpt=pretrained_teacher)
        self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)

        self.add_itm=add_itm
        self.add_itc=add_itc
        self.use_moco=use_moco

        self.temp = temp
        self.max_txt_len = max_txt_len

        self.image_encoder = image_encoder
        self.text_encoder = text_encoder

        # self.student_model = student
        self.loss_mse = nn.MSELoss()
        self.fit_dense = nn.Linear(_hidden_size, fit_size)

        # if self.add_itc:
        self.embed_dim = embed_dim
        self.encoder_width = encoder_width
        self.vision_proj = nn.Linear(self.encoder_width, self.embed_dim)
        self.text_proj = nn.Linear(self.encoder_width, self.embed_dim)

        if self.add_itm:
            self.itm_head = nn.Linear(_hidden_size, 2)
        # self.get_cross_embeds=None

        # create the momentum encoder
        self.image_encoder_m = deepcopy(self.image_encoder)
        self.text_encoder_m = deepcopy(self.text_encoder)

        self.vision_proj_m = deepcopy(self.vision_proj)
        self.text_proj_m = deepcopy(self.text_proj)

        self.model_pairs = [
            [self.image_encoder, self.image_encoder_m],
            [self.text_encoder, self.text_encoder_m],
            [self.vision_proj, self.vision_proj_m],
            [self.text_proj, self.text_proj_m],
        ]
        self.copy_params()

        # create the queue
        # queue_size = queue_size
        self.register_buffer("image_queue", torch.randn(embed_dim, queue_size))
        self.register_buffer("text_queue", torch.randn(embed_dim, queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)

        self.queue_size = queue_size
        self.momentum = 0.995



    def _rampup_factor(self, epoch, iters, num_iters_per_epoch):
        return min(1, (epoch * num_iters_per_epoch + iters) / (2 * num_iters_per_epoch))

    def get_contrastive_loss(self, text_embed, image_embed, idx=None):
        """
        Args:
            image_feat, text_feat: normalized
        Returns: contrastive loss
        """
        # bs = batch['input_ids'].size(0)
        text_feat = F.normalize(self.text_proj(text_embed[:,0,:].clone()),dim=-1)
        image_feat = F.normalize(self.vision_proj(image_embed[:,0,:].clone()), dim=-1)

        image_feat_all =image_feat
        text_feat_all = text_feat
        # image_feat_all = allgather(image_feat, torch.distributed.get_rank(), torch.distributed.get_world_size())
        # text_feat_all = allgather(text_feat, torch.distributed.get_rank(), torch.distributed.get_world_size())
        logits = image_feat_all @ text_feat_all.t() / self.temp
        # print(torch.isnan(logits).any(),torch.isnan(text_embed[:,0,:]).any(),'logits', image_feat_all.dtype)
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

        return (loss_i2t + loss_t2i) / 2, logits

    def get_matching_loss(self, text_hidden_state, text_attention_mask,image_hidden_state,
                          image_attention_mask, x_outputs_pos,sim_i2t=None,sim_t2i=None,idx=None):
        """
        Matching Loss with hard negatives
        """
        # batch = None
        bs = text_hidden_state.size(0)

        # text_feat = F.normalize(self.text_proj(text_hidden_state[:,0,:].clone()),dim=-1)
        # image_feat = F.normalize(self.vision_proj(image_hidden_state[:,0,:].clone()), dim=-1)
        with torch.no_grad():
            # if sim_i2t is not None and sim_t2i is not None:
            sim_i2t = sim_i2t[:, :bs].clone()
            sim_t2i = sim_t2i[:, :bs].clone()
            # else:
            #     sim_i2t = image_feat @ text_feat.t() / self.temp
            #     sim_t2i = text_feat @ image_feat.t() / self.temp
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
        image_neg_ids = []
        # print(torch.isnan(weights_t2i).any(),torch.isinf(weights_t2i).any())
        for b in range(bs):

            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_neg_ids.append(neg_idx)
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
        text_neg_ids = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_neg_ids.append(neg_idx)
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
        # print(output.isnan().any(),'itm')
        return F.cross_entropy(output.float(), itm_labels), output, image_neg_ids, text_neg_ids

    def forward(self, batch):
        if batch["text_input"] is not None:
            text_encoding = self.tokenizer(batch["text_input"],
                                           padding="max_length",
                                           truncation=True,
                                           max_length=self.max_txt_len,
                                           return_tensors="pt",)
        # pixel_values + pixel_mask
        if batch['image'] is not None:
            image_encoding = batch['image']
            if type(image_encoding) == torch.Tensor:
                image_encoding = {'pixel_values': image_encoding}
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

            # for k in image_batch:
            #     print(k)

            assert  text_batch['input_ids'].shape[0] == image_batch['pixel_values'].shape[0], "batch of text and image should be identied"

            image_batch['mode'] = 'image'
            image_batch['output_hidden_states'] = True
            image_batch['return_dict'] = True
            img_output = self.image_encoder(**image_batch)

            text_output = self.text_encoder(**text_batch)
            # print(type(text_output))

            text_mask = text_encoding.attention_mask[:,None,None,:]
            t_attention_mask = (1.0 - text_mask) * \
                               torch.finfo(self.image_encoder.img_attention_mask.dtype).min
            t_attention_mask = t_attention_mask.to(self.device)

            # pass text output to mome share layer
            x_text_embed  = self.image_encoder.encoder(text_output.last_hidden_state,
                                                       t_attention_mask,
                                                       output_attentions=True,
                                                       output_hidden_states=True,
                                                       mode='text')
            t_hidden_state_share = x_text_embed.last_hidden_state

            i_hidden_state_share = img_output.hidden_states[3]
            i_hidden_state = img_output.last_hidden_state
            # print(i_hidden_state.isnan().any())
            i_attention_mask = self.image_encoder.img_attention_mask

            x_text_embed  = self.image_encoder.encoder(t_hidden_state_share,
                                                       t_attention_mask,
                                                       output_attentions=True,
                                                       output_hidden_states=True,
                                                       mode='text')
            t_hidden_state = x_text_embed.last_hidden_state

            x_hidden_states = torch.cat((t_hidden_state, i_hidden_state), dim=1)
            x_attention_mask = torch.cat((t_attention_mask,
                                          i_attention_mask), dim=-1)

            x_outputs_pos = self.image_encoder.encoder(x_hidden_states,
                                                       x_attention_mask,
                                                       output_attentions=True,
                                                       output_hidden_states=True,
                                                       mode='fusion')

            image_feat = F.normalize(self.vision_proj(i_hidden_state[:, 0, :]), dim=-1)
            text_feat = F.normalize(self.text_proj(t_hidden_state[:, 0, :]), dim=-1)




            # get momentum features
            if self.use_moco:
                with torch.no_grad():
                    self._momentum_update()
                    img_output_m = self.image_encoder_m(**image_batch)
                    i_hidden_state_m = img_output_m.last_hidden_state

                    image_feat_m = F.normalize(self.vision_proj_m(i_hidden_state_m[:, 0, :]), dim=-1)

                    text_output_m = self.text_encoder_m(**text_batch)
                    x_text_embed_m  = self.image_encoder_m.encoder(text_output_m.last_hidden_state,
                                                               t_attention_mask,
                                                               output_attentions=True,
                                                               output_hidden_states=True,
                                                               mode='text')
                    t_hidden_state_m = x_text_embed_m.last_hidden_state
                    text_feat_m = F.normalize(self.text_proj_m(t_hidden_state_m[:, 0, :]), dim=-1)

                    image_feat_all = torch.cat(
                        [image_feat_m.t(), self.image_queue.clone().detach()], dim=1
                    )
                    # text_output_m = self.text_encoder_m.bert(
                    #     text.input_ids,
                    #     attention_mask=text.attention_mask,
                    #     return_dict=True,
                    #     mode="text",
                    # )
                    # text_embeds_m = t_hidden_state.clone()
                    # text_feat_m = F.normalize(self.text_proj_m(text_embeds_m[:, 0, :]), dim=-1)
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
                loss += loss_itc
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
                loss_itc, sim_i2t = self.get_contrastive_loss(t_hidden_state, i_hidden_state)
                loss += loss_itc

            if self.add_itm:
                loss_itm, itm_logits_s, image_neg_ids, text_neg_ids = self.get_matching_loss(t_hidden_state, t_attention_mask,
                                                                                             i_hidden_state, i_attention_mask,x_outputs_pos,
                                                                                             sim_i2t,sim_i2t.t(), #if moco
                                                                                             )
                loss += loss_itm

        else:
            raise ValueError("The input is invalid")

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
        use_moco = cfg.get("use_moco", False)

        embed_dim = cfg.get("embed_dim", 256)
        encoder_width = cfg.get("encoder_width", 768)
        share_layer = cfg.get("share_layer", -1)
        fusion_layer = cfg.get("fusion_layer", 3)
        # fusion_layer = 5
        num_hidden_layers = cfg.get("num_hidden_layers", 6)

        max_txt_len = cfg.get("max_txt_len", 30)
        temp = cfg.get("temp", 0.07)
        image_size = cfg.get("image_size", 256)
        patch_size = cfg.get("patch_size", 16)
        queue_size = cfg.get("queue_size", 65536)

        beit_config = BeitConfig.from_json_file(
            get_abs_path(cfg["student_config_path"]))
        beit_config.share_layer = share_layer
        beit_config.fusion_layer = fusion_layer
        beit_config.patch_size = patch_size
        beit_config.image_size = image_size
        beit_config.num_hidden_layers = num_hidden_layers

        student = None
        image_encoder = BeitModel(config=beit_config)

        med_config_path = get_abs_path(cfg.get("med_config_path"))
        med_config = BertConfig.from_json_file(med_config_path)
        text_encoder = BertModel(config=med_config)

        beit_config.text_vocab_size = med_config.vocab_size
        text_lm_head_inbeit = BeitOnlyMLMHead(config=beit_config)

        tokenizer_path = cfg.get("teacher_ckpt","vilt-b32-mlm")

        model = cls(
            text_encoder=text_encoder,
            image_encoder=image_encoder,
            text_lm_head=text_lm_head_inbeit,
            student=student,
            beit_config=beit_config,
            add_itm=add_itm,
            add_itc=add_itc,
            use_moco=use_moco,
            tokenizer_path=tokenizer_path,
            embed_dim=embed_dim,
            encoder_width=encoder_width,
            temp=temp,
            max_txt_len=max_txt_len,
            queue_size=queue_size,
        )
        model.load_checkpoint_from_config(cfg)
        return model

    def compute_sim_matrix(self, data_loader, task_cfg):
        """
        Compute similarity i2t, t2i matrix for the given data loader.
        """
        k_test = task_cfg.k_test

        return compute_sim_matrix(model=self, data_loader=data_loader, k_test=k_test)
