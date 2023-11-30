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

from lavis.models.vilt_models.vilt_outputs import (
    # AlbefIntermediateOutput,
    ViltDistillOutput,
    # AlbefSimilarity,
)

# from lavis.models.base_model import MomentumDistilationMixin, SharedQueueMixin
from lavis.models.med import BertForMaskedLM
from lavis.models.vit import VisionTransformerEncoder
from torch import nn
base_dir = os.path.dirname('/home1/liangjx/projects/mm/')
sys.path.append(base_dir)
# from trans.models.vilt import ViltConfig, ViltModel,  ViltForMaskedLM
from transformers import ViltConfig, ViltModel,  ViltForMaskedLM

from transformers import BertConfig

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


@registry.register_model("vilt_distill")
class ViltDistill(ViltBase):
    """
    ALBEF pretrain model.

    Supported model types:
        - base: ALBEF base model used for pretraining.
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "distill": "configs/models/vilt_distill.yaml",
    }

    def __init__(
        self,
        student,
        teacher,
        student_config,
        add_itm,
        add_itc,
        pretrained_teacher=None,
        temp=0.07,
        max_txt_len=30,
        fit_size=768,

    ):
        super().__init__()

        student_hidden_size = student_config.hidden_size
        self.tokenizer = self.init_tokenizer(pretrained_ckpt=pretrained_teacher)

        self.add_itm=add_itm
        self.add_itc=add_itc
        self.temp = temp
        self.max_txt_len = max_txt_len
        self.student_model = student
        self.teacher_model = teacher
        self.loss_mse = torch.nn.MSELoss()
        self.fit_dense = nn.Linear(student_hidden_size, fit_size)
        self.itm_head = nn.Linear(student_hidden_size, 2)

        # self.get_cross_embeds=None
    def _rampup_factor(self, epoch, iters, num_iters_per_epoch):
        return min(1, (epoch * num_iters_per_epoch + iters) / (2 * num_iters_per_epoch))

    def get_contrastive_loss(self, last_hidden_state, idx=None):
        """
        Args:
            image_feat, text_feat: normalized

        Returns: contrastive loss

        """

        # bs = batch['input_ids'].size(0)
        image_feat = last_hidden_state[:,30,:].clone()
        text_feat = last_hidden_state[:,0,:].clone()

        # assert image_feat.size(-1) == self.embed_dim
        # assert text_feat.size(-1) == self.embed_dim

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

    def get_matching_loss(self, batch, last_hidden_state, idx=None):
        """
        Matching Loss with hard negatives
        """
        # batch = None
        bs = batch['input_ids'].size(0)
        image_feat = last_hidden_state[:,30,:]
        text_feat = last_hidden_state[:,0,:]
        with torch.no_grad():
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

        batch_neg1 = batch.copy()

        pixel_values_neg = []
        pixel_mask_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            pixel_values_neg.append(batch['pixel_values'][neg_idx])
            pixel_mask_neg.append(batch['pixel_mask'][neg_idx])

        batch_neg1.update({'pixel_values': torch.stack(pixel_values_neg, dim=0)})
        batch_neg1.update({'pixel_mask': torch.stack(pixel_mask_neg, dim=0)})

        batch_neg2 = batch.copy()
        input_ids_neg = []
        token_type_ids_neg = []
        attention_mask_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            input_ids_neg.append(batch['input_ids'][neg_idx])
            token_type_ids_neg.append(batch['token_type_ids'][neg_idx])
            attention_mask_neg.append(batch['attention_mask'][neg_idx])


        batch_neg2.update({'input_ids': torch.stack( input_ids_neg, dim=0)})
        batch_neg2.update({'token_type_ids': torch.stack( token_type_ids_neg, dim=0)})
        batch_neg2.update({'attention_mask': torch.stack( attention_mask_neg, dim=0)})

        cross_pos = self.student_model(**batch).last_hidden_state[:, 0, :]

        batch_neg1 = {k: v.to(self.device) for k, v in batch_neg1.items()}
        cross_neg_1 = self.student_model(**batch_neg1).last_hidden_state[:, 0, :]

        batch_neg2 = {k: v.to(self.device) for k, v in batch_neg2.items()}
        cross_neg_2 = self.student_model(**batch_neg2).last_hidden_state[:, 0, :]
        # cross_neg = self.student_model(image_embeds_all, image_atts_all, text_embeds=text_embeds_all,
        #                                   text_atts=text_atts_all).last_hidden_state[:, 0, :]

        output = self.itm_head(torch.cat([cross_pos, cross_neg_1, cross_neg_2], dim=0))
        itm_labels = torch.cat([torch.ones(bs, dtype=torch.long),
                                torch.zeros(2 * bs, dtype=torch.long)], dim=0).to(self.device)

        return F.cross_entropy(output, itm_labels)


    def forward(self, batch):


        encoding = self.tokenizer(batch["text_input"],
                                  padding="max_length",
                                  truncation=True,
                                  max_length=self.max_txt_len,
                                  return_tensors="pt",)
        # pixel_values + pixel_mask
        encoding_feature_extractor = batch['image']
        for k, v in encoding_feature_extractor.items():
            encoding_feature_extractor[k] = v.squeeze()
        encoding.update(encoding_feature_extractor)
        # for k, v in encoding.items():
        #     print(k, v.shape)
            # if 'input_id' in k:
            #     print(k, v.shape, v)
        # print(type(batch))
        batch = encoding
        loss_itm = 0.
        loss_itc = 0.
        loss_att = 0.
        loss_rep = 0.

        # batch = {k: v.to(self.device) for k, v in batch.items() if k != 'question_id'}
        batch = {k: v.to(self.device) for k, v in batch.items()}
        # print('batch',batch['pixel_mask'].shape)
        outputs = self.student_model(**batch)
        student_atts, student_reps = outputs.attentions, outputs.hidden_states

        with torch.no_grad():
            teacher_outputs = self.teacher_model(**batch)
            teacher_logits, teacher_atts, teacher_reps = teacher_outputs.logits, teacher_outputs.attentions, teacher_outputs.hidden_states

        teacher_layer_num = len(teacher_atts)
        student_layer_num = len(student_atts)
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

        loss = loss_rep + loss_att

        # do_itm = True
        # do_itc = True
        # print(self.add_itc, self.add_itm)
        if self.add_itm:
            loss_itm = self.get_matching_loss(batch, last_hidden_state=outputs.last_hidden_state)
            loss += loss_itm
        if self.add_itc:
            loss_itc = self.get_contrastive_loss(last_hidden_state=outputs.last_hidden_state)
            loss += loss_itc

        return ViltDistillOutput(
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
        add_itm=cfg.get("add_itm", False)
        add_itc=cfg.get("add_itc", False)

        max_txt_len = cfg.get("max_txt_len", 30)
        temp = cfg.get("temp", 0.07)
        student_config = ViltConfig.from_json_file(
            get_abs_path(cfg["student_config_path"])
        )
        fit_size = student_config.hidden_size
        # student_config.is_student=True
        student = ViltModel(config=student_config)
        # teacher = ViltForMaskedLM.from_config(cfg, from_pretrained=True)
        teacher_ckpt = cfg.get("teacher_ckpt","vilt-b32-mlm")
        teacher = ViltForMaskedLM.from_pretrained(teacher_ckpt)
        # teacher = ViltModel.from_pretrained(teacher_ckpt)
        # mlm_score = ViltMLMHead.from_config()
        # itm_score = ViltITMHead.from_config()



        model = cls(
            student=student,
            teacher=teacher,
            student_config=student_config,
            pretrained_teacher=teacher_ckpt,
            temp=temp,
            max_txt_len=max_txt_len,
            add_itm=add_itm,
            add_itc=add_itc,
        )
            # image_encoder=image_encoder,
            # text_encoder=text_encoder,
            # queue_size=queue_size,
            # embed_dim=embed_dim,
            # mlm_mask_prob=mlm_mask_prob,
            # temp=temp,
            # momentum=momentum,
            # alpha=alpha,
            # max_txt_len=max_txt_len,
        # )
        return model