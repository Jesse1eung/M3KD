"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from copy import deepcopy
import re

import torch
import torch.nn.functional as F
from lavis.common.registry import registry
from lavis.models.albef_models import AlbefBase, compute_sim_matrix
from lavis.models.albef_models.albef_outputs import (
    AlbefIntermediateOutput,
    AlbefOutput,
    AlbefSimilarity,
)
from lavis.models.base_model import MomentumDistilationMixin, SharedQueueMixin
from lavis.models.med import XBertEncoder
from lavis.models.vit import VisionTransformerEncoder
from torch import nn


@registry.register_model("albef_retrieval")
class AlbefRetrieval(AlbefBase, MomentumDistilationMixin, SharedQueueMixin):
    """
    ALBEF retrieval model.

    Supported model types:
        - coco: fine-tuned ALBEF base model on COCO dataset (Karparthy split).
        - flickr: fine-tuned ALBEF base model on Flickr30k dataset.

    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("albef_retrieval", "coco")
        >>> model = load_model("albef_retrieval", "flickr")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "coco": "configs/models/albef/albef_retrieval_coco.yaml",
        "flickr": "configs/models/albef/albef_retrieval_flickr.yaml",
    }

    def __init__(
        self,
        image_encoder,
        text_encoder,
        queue_size,
        embed_dim=256,
        temp=0.07,
        use_distill=True,
        momentum=0.995,
        alpha=0.4,
        max_txt_len=30,
    ):
        super().__init__()

        self.tokenizer = self.init_tokenizer()

        self.visual_encoder = image_encoder
        self.text_encoder = text_encoder

        text_width = text_encoder.config.hidden_size
        vision_width = image_encoder.vision_width

        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)

        self.itm_head = nn.Linear(text_width, 2)

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
        # self.register_buffer("text_attention_mask", torch.ones(queue_size, max_txt_len))
        # self.register_buffer("image_emd", torch.randn(queue_size, 577, vision_width))
        # self.register_buffer("text_emd", torch.randn(queue_size, max_txt_len, text_width))

        self.register_buffer("idx_queue", torch.full((1, queue_size), -100))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)
        # self.t_attention_mask = nn.functional.normalize(self.attention_mask, dim=0)

        self.queue_size = queue_size
        self.momentum = momentum
        self.temp = nn.Parameter(temp * torch.ones([]))

        self.alpha = alpha
        self.max_txt_len = max_txt_len
        self.use_distill = use_distill
        # self.freeze_vm()
        # self.freeze_lm()

        # self.freeze_text(self.text_encoder)
        # self.freeze_vis(self.visual_encoder)

    def freeze_text(self, text_encoder):
        for name, params in text_encoder.named_parameters():
            params.requires_grad = False
            layer_str = re.findall("\d+", name)
            if layer_str:
                layer = int(layer_str[0])
                if layer >= 3:
                    params.requires_grad = True
            # elif 'predictions' in name:
            #     params.requires_grad = True
            # # if 'cross' not in name:
            # #     params.requires_grad = False

    def freeze_vis(self,vis_encoder):
        for name, params in vis_encoder.named_parameters():
            params.requires_grad = False

    def _rampup_factor(self, epoch, iters, num_iters_per_epoch):
        return min(1, (epoch * num_iters_per_epoch + iters) / (2 * num_iters_per_epoch))

    def freeze_vm(self):
        for name, param in self.visual_encoder.named_parameters():
            param.requires_grad = False
            if 'adapter' in name:
                print(name)
                param.requires_grad = True

    def freeze_lm(self):
        for name, param in self.text_encoder.named_parameters():
            param.requires_grad = False
            if 'cross' in name or 'adapter' in name:
                print(name)
                param.requires_grad = True

    def forward(self, samples):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W). The input images.
                - text_input (list): A list of length batch_size, each element is a string of text/caption.
                - image_id (torch.Tensor): A tensor of shape (batch_size, ). The image ids, used to identify same images in batch.
                - epoch (int): The current epoch.
                - iters (int): The current iteration.
                - num_iters_per_epoch (int): The number of iterations per epoch.

        Returns:
            BlipOutput: A BlipOutput object. See ``lavis.models.blip_models.blip_outputs.BlipOutput`` for more details.

        Examples:
            >>> import torch
            >>> from lavis.models import load_model
            >>> model = load_model("albef_retrieval", "coco")
            >>> images = torch.randn(4, 3, 384, 384)
            >>> text_input = ["caption of image 1", "another caption of image 1", "caption of image 2", "caption of image 3"]
            >>> image_id = torch.tensor([1, 1, 2, 3])
            >>> samples = {"image": images, "text_input": text_input, "image_id": image_id, "epoch": 0, "iters": 0, "num_iters_per_epoch": 100}
            >>> output = model(samples)
            >>> output.keys()
            odict_keys(['sims', 'intermediate_output', 'loss', 'loss_itc', 'loss_itm'])
        """


        image = samples["image"]
        caption = samples["text_input"]
        idx = samples["image_id"]

        alpha = self.alpha * self._rampup_factor(
            epoch=samples["epoch"],
            iters=samples["iters"],
            num_iters_per_epoch=samples["num_iters_per_epoch"],
        )

        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)

        image_embeds = self.visual_encoder.forward_features(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            self.device
        )

        image_feat = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)

        text = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(self.device)
        # print(text.attention_mask.shape)
        text_output = self.text_encoder.forward_text(text)

        text_embeds = text_output.last_hidden_state
        text_feat = F.normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)

        idx = idx.view(-1, 1)
        idx_all = torch.cat([idx.t(), self.idx_queue.clone().detach()], dim=1)
        pos_idx = torch.eq(idx, idx_all).float()
        sim_targets = pos_idx / pos_idx.sum(1, keepdim=True)

        with torch.no_grad():
            self._momentum_update()
            image_embeds_m = self.visual_encoder_m(image)
            image_feat_m = F.normalize(
                self.vision_proj_m(image_embeds_m[:, 0, :]), dim=-1
            )
            image_feat_all = torch.cat(
                [image_feat_m.t(), self.image_queue.clone().detach()], dim=1
            )
            text_output_m = self.text_encoder_m.forward_text(text)
            text_embeds_m = text_output_m.last_hidden_state
            text_feat_m = F.normalize(self.text_proj_m(text_embeds_m[:, 0, :]), dim=-1)
            text_feat_all = torch.cat([text_feat_m.t(), self.text_queue.clone().detach()], dim=1)

            if self.use_distill:
                sim_i2t_m = image_feat_m @ text_feat_all / self.temp
                sim_t2i_m = text_feat_m @ image_feat_all / self.temp

                sim_i2t_targets = (
                    alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
                )
                sim_t2i_targets = (
                    alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets
                )

        sim_i2t = image_feat @ text_feat_all / self.temp
        sim_t2i = text_feat @ image_feat_all / self.temp

        if self.use_distill:
            loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1).mean()
            loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1).mean()
        else:
            loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_targets, dim=1).mean()
            loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_targets, dim=1).mean()

        loss_itc = (loss_i2t + loss_t2i) / 2

        self._dequeue_and_enqueue(image_feat_m, text_feat_m, idx)
        # self._de_enqueue_emd(image_embeds_m, text_embeds_m, text_mask=text.attention_mask)

        encoder_output_pos = self.text_encoder(
            encoder_embeds=text_embeds,
            attention_mask=text.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
            mode="fusion",
        )

        with torch.no_grad():
            bs = image.size(0)
            weights_i2t = F.softmax(sim_i2t[:, :bs] + 1e-4, dim=1)
            weights_t2i = F.softmax(sim_t2i[:, :bs] + 1e-4, dim=1)

            mask = torch.eq(idx, idx.T)
            weights_i2t[:,:bs].masked_fill_(mask, 0)
            weights_t2i[:,:bs].masked_fill_(mask, 0)

        # select a negative image for each text
        image_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            if neg_idx >= bs:
                image_embeds_neg.append(self.image_emd[neg_idx-bs])
            else:
                image_embeds_neg.append(image_embeds[neg_idx])

        image_embeds_neg = torch.stack(image_embeds_neg, dim=0)

        # select a negative text for each image
        text_embeds_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            if neg_idx >=bs:
                text_embeds_neg.append(self.text_emd[neg_idx-bs])
                text_atts_neg.append(self.text_attention_mask[neg_idx-bs])
            else:
                text_embeds_neg.append(text_embeds[neg_idx])
                text_atts_neg.append(text.attention_mask[neg_idx])
        text_embeds_neg = torch.stack(text_embeds_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        text_embeds_all = torch.cat([text_embeds, text_embeds_neg], dim=0)
        text_atts_all = torch.cat([text.attention_mask, text_atts_neg], dim=0)

        image_embeds_all = torch.cat([image_embeds_neg, image_embeds], dim=0)
        image_atts_all = torch.cat([image_atts, image_atts], dim=0)

        encoder_output_neg = self.text_encoder(
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

        return AlbefOutput(
            loss=loss_itc + loss_itm,
            loss_itc=loss_itc,
            loss_itm=loss_itm,
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

    @classmethod
    def from_config(cls, cfg=None):
        cfg["vit_depth"] = 6
        cfg["use_adapter"] = False
        add_moe = cfg.get("add_moe", False)
        image_encoder = VisionTransformerEncoder.from_config(cfg, from_pretrained=False,
                                                             vision_width=768)
        fusion_layer=3
        num_layers=6
        text_encoder = XBertEncoder.from_config(cfg, from_pretrained=False,
                                                add_moe=add_moe,
                                                fusion_layer=fusion_layer,
                                                num_layers=num_layers,
                                                hidden_size=768)

        embed_dim = cfg.get("embed_dim", 256)
        momentum = cfg.get("momentum", 0.995)
        alpha = cfg.get("alpha", 0.4)
        temp = cfg.get("temp", 0.07)
        max_txt_len = cfg.get("max_txt_len", 30)
        queue_size = cfg.get("queue_size", 0)
        use_distill = cfg.get("use_distill", True)

        model = cls(
            image_encoder=image_encoder,
            text_encoder=text_encoder,
            queue_size=queue_size,
            embed_dim=embed_dim,
            temp=temp,
            momentum=momentum,
            alpha=alpha,
            max_txt_len=max_txt_len,
            use_distill=use_distill,
        )
        print('hidden layers:', text_encoder.config.num_hidden_layers)
        # model.load_checkpoint_from_config(cfg, load_albef_by_sep=(text_encoder.config.num_hidden_layers==12))
        model.load_checkpoint_from_config(cfg, load_albef_by_sep=False)

        return model

    def compute_sim_matrix(self, data_loader, task_cfg):
        """
        Compute similarity i2t, t2i matrix for the given data loader.
        """
        k_test = task_cfg.k_test

        return compute_sim_matrix(model=self, data_loader=data_loader, k_test=k_test)
