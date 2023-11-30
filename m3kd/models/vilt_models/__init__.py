"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import datetime
import logging
import os
import time

import lavis.common.dist_utils as dist_utils
import torch
import torch.distributed as dist
import torch.nn.functional as F
from lavis.common.dist_utils import download_cached_file
from lavis.common.logger import MetricLogger
from lavis.common.utils import is_url
from lavis.models.base_model import BaseModel
# from lavis.models.vit import interpolate_pos_embed
from transformers import BertTokenizerFast


class ViltBase(BaseModel):
    @classmethod
    def init_tokenizer(cls,pretrained_ckpt):
        return BertTokenizerFast.from_pretrained(pretrained_ckpt)

    def load_from_pretrained(self, url_or_filename, rename_text_keys=True):
        if is_url(url_or_filename):
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            checkpoint = torch.load(cached_file, map_location="cpu")
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location="cpu")
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
        num_patches = self.image_encoder.embeddings.patch_embeddings.num_patches
        state_dict["image_encoder.embeddings.position_embeddings"] = interpolate_pos_embed(
            state_dict["image_encoder.embeddings.position_embeddings"], num_patches)
        # if (
        #     "visual_encoder_m.pos_embed" in self.state_dict().keys()
        #     and "visual_encoder_m.pos_embed" in state_dict
        # ):
        #     state_dict["visual_encoder_m.pos_embed"] = interpolate_pos_embed(
        #         state_dict["visual_encoder_m.pos_embed"], self.visual_encoder_m
        #     )

        # if rename_text_keys:
        #     for key in list(state_dict.keys()):
        #         if "bert" in key:
        #             new_key = key.replace("bert.", "")
        #             state_dict[new_key] = state_dict[key]
        #             del state_dict[key]
        # for key in list(state_dict.keys()):
        #     if '_m.' in key:
        #         del state_dict[key]
        for key in self.state_dict().keys():
            if key in state_dict.keys():
                if state_dict[key].shape != self.state_dict()[key].shape:
                    del state_dict[key]

        msg = self.load_state_dict(state_dict,strict=False)

        logging.info("Missing keys {}".format(msg.missing_keys))
        logging.info("load checkpoint from %s" % url_or_filename)
        return msg

def interpolate_pos_embed(pos_embed_checkpoint, num_patches, num_extra_tokens=1):
    # interpolate position embedding
    embedding_size = pos_embed_checkpoint.shape[-1]
    # height (== width) for the checkpoint position embedding
    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
    # height (== width) for the new position embedding
    new_size = int(num_patches**0.5)

    if orig_size != new_size:
        # class_token and dist_token are kept unchanged
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(
            -1, orig_size, orig_size, embedding_size
        ).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode="bicubic", align_corners=False
        )
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        print(
            "reshape position embedding from %d to %d" % (orig_size**2, new_size**2)
        )

        return new_pos_embed
    else:
        return pos_embed_checkpoint



def compute_sim_matrix(model, data_loader, **kwargs):
    k_test = kwargs.pop("k_test")

    metric_logger = MetricLogger(delimiter="  ")
    header = "Evaluation:"

    logging.info("Computing features for evaluation...")
    start_time = time.time()



    image_feats = []
    image_embeds = []
    image_masks = []
    for samples in data_loader:
        image_batch = samples["image"]
        if type(image_batch) == torch.Tensor:
            image_batch = {'pixel_values': image_batch}
        for k, v in image_batch.items():
            image_batch[k] = v.squeeze().to(model.device)
        image_batch['mode'] = 'image'
        image_batch['output_hidden_states'] = True
        image_batch['return_dict'] = True

        image_output = model.image_encoder(**image_batch)
        image_feat = image_output.last_hidden_state
        # image_feat_for_itm = image_output.hidden_states[3]
        # image_feat = model.visual_encoder.forward_features(image)
        image_embed = model.vision_proj(image_feat[:, 0, :])
        image_embed = F.normalize(image_embed, dim=-1)

        image_masks.append(model.image_encoder.img_attention_mask)
        image_feats.append(image_feat.cpu())
        image_embeds.append(image_embed)

    image_masks = torch.cat(image_masks, dim=0)
    image_feats = torch.cat(image_feats, dim=0)
    image_embeds = torch.cat(image_embeds, dim=0)


    texts = data_loader.dataset.text
    num_text = len(texts)
    text_bs = 256
    text_feats = []
    text_ids = []
    text_embeds = []
    text_att_masks = []
    for i in range(0, num_text, text_bs):
        text = texts[i : min(num_text, i + text_bs)]
        text_input = model.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=35,
            return_tensors="pt",
        ).to(model.device)

        text_feat = model.text_encoder(**text_input)
        # pass text feat to share
        text_mask = text_input.attention_mask[:,None,None,:]
        t_attention_mask = (1.0 - text_mask) * \
                           torch.finfo(model.image_encoder.img_attention_mask.dtype).min
        t_attention_mask = t_attention_mask.to(model.device)
        text_feat  = model.image_encoder.encoder(text_feat.last_hidden_state,
                                                   t_attention_mask,
                                                   output_attentions=True,
                                                   output_hidden_states=True,
                                                   mode='text')
        # print(type(text_feat))
        # text_output = model.text_encoder.forward_text(text_input)
        text_embed = F.normalize(
            model.text_proj(text_feat.last_hidden_state[:, 0, :])
        )

        # text_feats.append(text_feat.last_hidden_state)
        text_feats.append(text_feat.last_hidden_state)
        text_embeds.append(text_embed)
        text_ids.append(text_input.input_ids)
        text_mask = text_input.attention_mask[:,None,None,:]
        t_attention_mask = (1.0 - text_mask) * \
                           torch.finfo(model.image_encoder.img_attention_mask.dtype).min
        text_att_masks.append(t_attention_mask)

    text_feats = torch.cat(text_feats, dim=0)
    text_embeds = torch.cat(text_embeds, dim=0)
    # text_ids = torch.cat(text_ids, dim=0)
    text_att_masks = torch.cat(text_att_masks, dim=0)
    # if hasattr(model.tokenizer, "enc_token_id"):
    #     text_ids[:, 0] = model.tokenizer.enc_token_id



    sims_matrix = image_embeds @ text_embeds.t()
    score_matrix_i2t = torch.full(
        (len(data_loader.dataset.image), len(texts)), -100.0
    ).to(model.device)

    num_tasks = dist_utils.get_world_size()
    rank = dist_utils.get_rank()
    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    for i, sims in enumerate(
        metric_logger.log_every(sims_matrix[start:end], 50, header)
    ):
        # topk_sim, topk_idx = sims.topk(k=config["k_test"], dim=0)
        topk_sim, topk_idx = sims.topk(k=k_test, dim=0)

        encoder_output = image_feats[start + i].repeat(k_test, 1, 1).to(model.device)
        i_attention_mask = image_masks[start + i].repeat(k_test,1, 1, 1).to(model.device)

        x_hidden_states = torch.cat((text_feats[topk_idx], encoder_output), dim=1)
        # print(text_atts[topk_idx].shape, i_attention_mask.shape)
        x_attention_mask = torch.cat((text_att_masks[topk_idx],
                                      i_attention_mask), dim=-1)

        # encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(model.device)
        output = model.image_encoder.encoder(
            x_hidden_states,
            x_attention_mask,
            # attention_mask=text_atts[topk_idx],
            output_attentions=True,
            output_hidden_states=True,
            mode='fusion')

        score = model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
        score_matrix_i2t[start + i, topk_idx] = score + topk_sim


    sims_matrix = sims_matrix.t()
    score_matrix_t2i = torch.full(
        (len(texts), len(data_loader.dataset.image)), -100.0
    ).to(model.device)

    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    for i, sims in enumerate(
            metric_logger.log_every(sims_matrix[start:end], 50, header)
    ):

        topk_sim, topk_idx = sims.topk(k=k_test, dim=0)
        encoder_output = image_feats[topk_idx].to(model.device)
        i_attention_mask = image_masks[topk_idx].to(model.device)
        x_hidden_states = torch.cat((text_feats[start + i].repeat(k_test, 1, 1), encoder_output), dim=1)
        x_attention_mask = torch.cat((text_att_masks[start + i].repeat(k_test, 1, 1,1),
                                      i_attention_mask), dim=-1)
        # encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(model.device)
        output = model.image_encoder.encoder(
            x_hidden_states,
            x_attention_mask,
            # attention_mask=text_atts[start + i].repeat(k_test, 1),
            # encoder_hidden_states=encoder_output,
            # encoder_attention_mask=encoder_att,
            # return_dict=True,)
            output_attentions=True,
            output_hidden_states=True,
            mode='fusion')

        score = model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
        score_matrix_t2i[start + i, topk_idx] = score + topk_sim
        # score_matrix_t2i[start + i, topk_idx] = score


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

