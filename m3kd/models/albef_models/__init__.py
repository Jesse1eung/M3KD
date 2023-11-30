"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import datetime
import logging
import os
import time, re

import lavis.common.dist_utils as dist_utils
import torch
import torch.distributed as dist
import torch.nn.functional as F
from lavis.common.dist_utils import download_cached_file
from lavis.common.logger import MetricLogger
from lavis.common.utils import is_url
from lavis.models.base_model import BaseModel
from lavis.models.vit import interpolate_pos_embed
from transformers import BertTokenizer

from torch.nn import MSELoss

class AlbefBase(BaseModel):
    @classmethod
    def init_tokenizer(cls,cls_name="bert-base-uncased"):
        return BertTokenizer.from_pretrained(cls_name)

    def load_from_pretrained(self, url_or_filename, rename_text_keys=True,load_albef_by_sep=False,covertiny=True):
        if is_url(url_or_filename):
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            checkpoint = torch.load(cached_file, map_location="cpu")
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location="cpu")
        else:
            print('error:', url_or_filename)
            raise RuntimeError("checkpoint url or path is invalid")

        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
        temp = {}
        for k in state_dict:
            if 'module' in k:
                temp[k[7:]] = state_dict[k]
                # del state_dict[k]
            # if 'text_encoder' in k:
            #     del state_dict[k]
        if temp:
            state_dict = temp
        if (
                "visual_encoder.pos_embed" in self.state_dict().keys()
                and "visual_encoder.pos_embed" in state_dict and self.visual_encoder
        ):
            state_dict["visual_encoder.pos_embed"] = interpolate_pos_embed(
                state_dict["visual_encoder.pos_embed"], self.visual_encoder
            )

        if (
            "visual_encoder_m.pos_embed" in self.state_dict().keys()
            and "visual_encoder_m.pos_embed" in state_dict
        ):
            state_dict["visual_encoder_m.pos_embed"] = interpolate_pos_embed(
                state_dict["visual_encoder_m.pos_embed"], self.visual_encoder_m
            )

        if rename_text_keys:
            for key in list(state_dict.keys()):
                if "bert" in key:
                    new_key = key.replace("bert.", "")
                    state_dict[new_key] = state_dict[key]
                    del state_dict[key]
        for key in self.state_dict().keys():
            if key in state_dict.keys():
                if state_dict[key].shape != self.state_dict()[key].shape:
                    del state_dict[key]


        for key in self.state_dict():
            if ('v_mlp' in key or 'v_norm' in key) and 'teacher' not in key and '_m.' not in key:
                if key not in state_dict:
                    state_dict[key] = state_dict[key.replace('v_', '')]
            if ('l_intermediate' in key or 'l_output' in key) and 'teacher' not in key and '_m.' not in key:
                if key not in state_dict:
                    state_dict[key] = state_dict[key.replace('l_', '')]

        if load_albef_by_sep:
            # for name, param in self.named_parameters():
            #     print(name)


            for key in self.state_dict():
                if  f'queue' in key or f'_m' in key or 'teacher' in key:
                # if f'_m' in key or f'queue' in key:
                    if key in state_dict:
                        del state_dict[key]
                    continue
                # for i in range(6):
                if 'l_intermediate' in key or 'l_output' in key:
                    if key not in state_dict:
                        state_dict[key] = state_dict[key.replace('l_', '')]

                # if f'blocks.{i}' in key:
                if re.findall('blocks.[0-5]\.', key) and 'v_mlp' not in key and 'v_norm' not in key:
                    if key in state_dict:
                        layer = int(re.findall('[0-5]', key)[0])
                        new_key = key.replace(f'blocks.{layer}', f'blocks.{2*layer+1}')
                        state_dict[key] = state_dict[new_key]

                if 'v_mlp' in key or 'v_norm' in key:
                    if key not in state_dict:
                        state_dict[key] = state_dict[key.replace('v_', '')]
                # if re.findall('layer.[0-5]\.', key):
                #     layer = int(re.findall('[0-5]', key)[0])
                #     new_key = key.replace(f'layer.{layer}', f'layer.{2*layer+1}')
                #     try:
                #         state_dict[key] = state_dict[new_key]
                #     except KeyError:
                #         # print(key, new_key)
                #         continue
        if not covertiny:
            for key in list(state_dict.keys()):
                if 'l_intermediate' in key or 'l_output' in key:
                    self.state_dict()[key] = self.state_dict()[key.replace('l_', '')]
                if 'text_encoder' in key:
                    del state_dict[key]
        # else:
        msg = self.load_state_dict(state_dict, strict=False)

        logging.info("Missing keys {}".format(msg.missing_keys))
        logging.info("load checkpoint from %s" % url_or_filename)
        return msg

    def get_hid_loss(self, teacher_reps, student_reps):
        loss_hid = 0.
        loss_mse = MSELoss()
        student_layer_num = len(student_reps)
        teacher_layer_num = len(teacher_reps)
        # print('teacher_reps: ', len(teacher_reps))
        # print('student_reps: ', len(student_reps))
        # layers_per_block = int(teacher_layer_num/student_layer_num)
        assert teacher_layer_num % student_layer_num == 0 or (teacher_layer_num-1) % (student_layer_num-1) == 0
        layers_per_block = int((teacher_layer_num-1) / (student_layer_num-1)) if student_layer_num>1 else 1
        teacher_reps = [teacher_rep.detach() for teacher_rep in teacher_reps]
        new_teacher_reps = [teacher_reps[i * layers_per_block] for i in range(student_layer_num)]
        new_student_reps = student_reps

        for student_rep, teacher_rep in zip(new_student_reps, new_teacher_reps):
            loss_hid += loss_mse(student_rep, teacher_rep)

        return loss_hid

    def get_att_loss(self, teacher_xatts, student_xatts):
        loss_att = 0.
        loss_mse = MSELoss()
        teacher_layer_num = len(teacher_xatts)
        student_layer_num = len(student_xatts)
        # print('teacher_xatts: ',type(teacher_xatts), len(teacher_xatts))
        # print('student_xatts: ',len(student_xatts))
        assert teacher_layer_num % student_layer_num == 0
        layers_per_block = int(teacher_layer_num / student_layer_num)
        new_teacher_xatts = [teacher_xatts[i * layers_per_block + layers_per_block - 1]
                             for i in range(student_layer_num)]

        for student_xatt, teacher_xatt in zip(student_xatts, new_teacher_xatts):
            student_xatt = torch.where(student_xatt <= -1e2, torch.zeros_like(student_xatt).to(self.device),
                                       student_xatt)
            teacher_xatt = torch.where(teacher_xatt <= -1e2, torch.zeros_like(teacher_xatt).to(self.device),
                                       teacher_xatt)
        loss_att += loss_mse(student_xatt, teacher_xatt)
        return  loss_att



def compute_sim_matrix(model, data_loader, **kwargs):
    k_test = kwargs.pop("k_test")

    metric_logger = MetricLogger(delimiter="  ")
    header = "Evaluation:"

    logging.info("Computing features for evaluation...")
    start_time = time.time()

    texts = data_loader.dataset.text
    num_text = len(texts)
    text_bs = 256
    text_ids = []
    text_feats = []
    text_embeds = []
    text_atts = []
    for i in range(0, num_text, text_bs):
        text = texts[i : min(num_text, i + text_bs)]
        text_input = model.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=35,
            return_tensors="pt",
        ).to(model.device)
        text_output = model.text_encoder.forward_text(text_input)
        text_feat = text_output.last_hidden_state
        text_embed = F.normalize(
            model.text_proj(text_output.last_hidden_state[:, 0, :])
        )
        text_embeds.append(text_embed)
        text_feats.append(text_feat)
        text_ids.append(text_input.input_ids)
        text_atts.append(text_input.attention_mask)

    text_embeds = torch.cat(text_embeds, dim=0)
    text_feats = torch.cat(text_feats, dim=0)
    text_ids = torch.cat(text_ids, dim=0)
    text_atts = torch.cat(text_atts, dim=0)
    if hasattr(model.tokenizer, "enc_token_id"):
        text_ids[:, 0] = model.tokenizer.enc_token_id

    image_feats = []
    image_embeds = []
    for samples in data_loader:
        image = samples["image"]

        image = image.to(model.device)
        image_feat = model.visual_encoder.forward_features(image)
        image_embed = model.vision_proj(image_feat[:, 0, :])
        image_embed = F.normalize(image_embed, dim=-1)

        image_feats.append(image_feat.cpu())
        image_embeds.append(image_embed)

    image_feats = torch.cat(image_feats, dim=0)
    image_embeds = torch.cat(image_embeds, dim=0)

    # sims_matrix = image_embeds @ text_embeds.t()
    sims_matrix = text_embeds @ image_embeds.t()


    score_matrix_t2i = torch.full(
        (len(texts), len(data_loader.dataset.image)), -100.0
    ).to(model.device)
    num_tasks = dist_utils.get_world_size()
    rank = dist_utils.get_rank()
    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    for i, sims in enumerate(
            metric_logger.log_every(sims_matrix[start:end], 50, header)
    ):

        topk_sim, topk_idx = sims.topk(k=k_test, dim=0)
        encoder_output = image_feats[topk_idx.cpu()].to(model.device)
        encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(
            model.device
        )

        output = model.text_encoder(
            # text_ids[start + i].repeat(k_test, 1),
            encoder_embeds=text_feats[start + i].repeat(k_test, 1,1),
            attention_mask=text_atts[start + i].repeat(k_test, 1),
            encoder_hidden_states=encoder_output,
            encoder_attention_mask=encoder_att,
            return_dict=True,
            mode='fusion',
        )
        score = model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
        score_matrix_t2i[start + i, topk_idx] = score
        # score_matrix_t2i[start + i, topk_idx] = topk_sim


    sims_matrix = sims_matrix.t()
    score_matrix_i2t = torch.full(
        (len(data_loader.dataset.image), len(texts)), -100.0
    ).to(model.device)

    # num_tasks = dist_utils.get_world_size()
    # rank = dist_utils.get_rank()
    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    for i, sims in enumerate(
        metric_logger.log_every(sims_matrix[start:end], 50, header)
    ):
        # # topk_sim, topk_idx = sims.topk(k=config["k_test"], dim=0)
        topk_sim, topk_idx = sims.topk(k=k_test, dim=0)
        # print(topk_sim, topk_idx)
        encoder_output = image_feats[start + i].repeat(k_test, 1, 1).to(model.device)
        encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(
            model.device
        )
        output = model.text_encoder(
            # text_ids[topk_idx],
            encoder_embeds=text_feats[topk_idx],
            attention_mask=text_atts[topk_idx],
            encoder_hidden_states=encoder_output,
            encoder_attention_mask=encoder_att,
            return_dict=True,
            mode='fusion',
        )
        score = model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
        score_matrix_i2t[start + i, topk_idx] = score
        # score_matrix_i2t[start + i, topk_idx] = topk_sim


    if dist_utils.is_dist_avail_and_initialized():
        dist.barrier()
        torch.distributed.all_reduce(
            score_matrix_i2t, op=torch.distributed.ReduceOp.SUM
        )
        torch.distributed.all_reduce(
            score_matrix_t2i, op=torch.distributed.ReduceOp.SUM
        )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info("Evaluation time {}".format(total_time_str))

    return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()
