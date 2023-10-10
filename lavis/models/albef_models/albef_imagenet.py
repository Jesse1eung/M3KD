"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import warnings
import logging
from tqdm import tqdm

from copy import deepcopy

import torch
import torch.nn.functional as F
from lavis.common.registry import registry
from lavis.models.albef_models import AlbefBase
from lavis.models.albef_models.albef_outputs import (
    AlbefIntermediateOutput,
    AlbefOutputWithLogits,
)
from lavis.models.base_model import MomentumDistilationMixin
from lavis.models.med import XBertEncoder
from lavis.models.vit import VisionTransformerEncoder
from lavis.tasks.multimodal_classification import MultimodalClassificationTask

from torch import nn


@registry.register_model("albef_imagenet")
class AlbefImg(AlbefBase, MomentumDistilationMixin):
    PRETRAINED_MODEL_CONFIG_DICT = {
        "imagenet": "configs/models/albef/albef_imagenet.yaml",
    }

    def __init__(
        self,
        image_encoder,
        text_encoder,
        num_classes,
        momentum=0.995,
        alpha=0.4,
        use_distill=False,
        max_txt_len=100,
    ):
        super().__init__()

        self.tokenizer = self.init_tokenizer()
        self.max_txt_len = max_txt_len

        # self.use_distill = use_distill
        embed_dim = 256
        self.visual_encoder = image_encoder
        self.text_encoder = text_encoder

        # text_width = text_encoder.config.hidden_size
        vision_width = image_encoder.vision_width

        self.vision_proj = nn.Linear(vision_width, embed_dim)
        # self.text_proj = nn.Linear(text_width, embed_dim)
        # self.itm_head = nn.Linear(text_width, 2)

        num_features = image_encoder.num_features

        if num_classes > 0:
            self.cls_head = nn.Sequential(
                nn.Linear(num_features, num_features),
                nn.ReLU(),
                nn.Linear(num_features, num_classes),
            )
        else:
            warnings.warn(
                f"Found num_classes=0, initializing {type(self)} without classifier."
            )

        self.prompt_templates = openai_imagenet_template

    def _rampup_factor(self, epoch, iters, num_iters_per_epoch):
        return min(1, (epoch * num_iters_per_epoch + iters) / num_iters_per_epoch)

    def forward(self, samples, is_train=True):
        image = samples["image"]
        prediction = self.visual_encoder.forward_head(image,use_moe=False)
        # sentences = samples["text_input"]

        targets = samples["label"]

        # image_embeds = self.visual_encoder.forward_features(samples["image"])
        # encoder_output = self.text_encoder.forward_automask(
        #     samples["tokenized_text"], image_embeds
        # )

        # encoder_output = self.text_encoder.forward_text(
        #     samples["tokenized_text"]
        # )
        # prediction = self.cls_head(image_embeds[:, 0, :])

        if is_train:

            loss = F.cross_entropy(prediction, targets)

            image_embeds, image_embeds_m, encoder_output, encoder_output_m, prediction_m = \
                None, None, None, None, None

            # return {"loss": loss}
            return AlbefOutputWithLogits(
                loss=loss,
                intermediate_output=AlbefIntermediateOutput(
                    image_embeds=image_embeds,
                    image_embeds_m=image_embeds_m,
                    encoder_output=encoder_output,
                    encoder_output_m=encoder_output_m,
                ),
                logits=prediction,
                logits_m=prediction_m,
            )
        else:
            return {"predictions": prediction, "targets": targets}

    def predict(self, samples):
        image = samples["image"]
        prediction = self.visual_encoder.forward_head(image,use_moe=False)
        # sentences = samples["text_input"]

        targets = samples["label"]
        return {"predictions": prediction, "targets": targets}

    def predict_alb(self, samples):
        # output = self.forward(samples, is_train=False)
        # return output
        image = samples["image"]
        targets = samples["label"]

        image_embeds = self.visual_encoder.forward_features(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(self.device)
        logits_list = []
        text_embeds_list=[]
        text_atts = []
        print(type(self.classnames),len(self.classnames[0]), 'lenname')

            # text_embeds_list.append(text_embeds.last_hidden_state.cpu())
            # text_atts.append(texts.attention_mask.cpu())

        for i in range(image_embeds.size()[0]): #128
            img_emd = image_embeds[i].repeat(len(self.prompt_templates), 1, 1).to(self.device)
            img_att = image_atts[i].repeat(len(self.prompt_templates), 1, 1).to(self.device)
            # for txt_emd, txt_att in zip(text_embeds_list,text_atts):
            for cls_name in self.classnames[0]:
                texts = [template(cls_name) for template in self.prompt_templates]  # format with class
                texts = self.tokenizer(texts, padding="max_length",truncation=True,return_tensors="pt").to(self.device)  # tokenize
                text_embeds = self.text_encoder.forward_text(texts)
                encoder_output_pos = self.text_encoder(
                    encoder_embeds=text_embeds.last_hidden_state,
                    attention_mask=texts.attention_mask,
                    encoder_hidden_states=img_emd,
                    encoder_attention_mask=img_att,
                    return_dict=True,
                    mode="fusion",)
                logits = self.itm_head(encoder_output_pos.last_hidden_state[:,0,:])
                # print(logits.shape, ' 1')

                logits_list.append(logits.mean(dim=0))
            print(len(logits_list))
        print(logits_list[0].shape, ' 2')
        print(len(logits_list),' 3')
        logits = torch.stack(logits_list, dim=0).to(self.device)
        print(logits.shape,' 4')
        # image_features = F.normalize(self.vision_proj(image_embeds[:, 0]), dim=-1)

        # logits = 100.0 * image_features @ self.classifier

        return {"predictions": logits[:,1].reshape(image_embeds.size()[0], len(self.classnames[0])), "targets": targets}

    def before_evaluation(self, dataset, task_type, **kwargs):
        if task_type == MultimodalClassificationTask:
            self.classnames = dataset.classnames,
            # self.classifier = self.zero_shot_classifier(
            #     classnames=dataset.classnames,
            #     templates=self.prompt_templates,
            # )


    def zero_shot_classifier(self, classnames, templates):
        with torch.no_grad():
            zeroshot_weights = []
            for classname in classnames:
                texts = [
                    template(classname) for template in templates
                ]  # format with class
                # print(texts)
                texts = self.tokenizer(texts, padding="max_length",truncation=True,return_tensors="pt").to(self.device)  # tokenize

                class_embeddings = self.text_encoder.forward_text(texts)
                class_embedding = F.normalize(self.text_proj(class_embeddings.last_hidden_state[:, 0]), dim=-1).mean(dim=0)
                # class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
            zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(self.device)
        return zeroshot_weights

    def compute_logits(self, dataloader):

        return compute_logits(model=self,dataloader=dataloader)

    @classmethod
    def from_config(cls, cfg=None):
        cfg["vit_depth"] = 6
        cfg["use_adapter"] = False
        add_moe = cfg.get("add_moe", False)
        image_encoder = VisionTransformerEncoder.from_config(cfg, from_pretrained=False,
                                                             vision_width=768,use_moe=False)
        # image_encoder = VisionTransformerEncoder.from_config(cfg)

        # text encoder + multimodal encoder
        fusion_layer=cfg.get("fusion_layer", 3)
        num_layers = cfg.get("num_layers", 6)
        add_moe = cfg.get("add_moe", False)
        text_encoder = XBertEncoder.from_config(cfg,
                                                from_pretrained=False,
                                                fusion_layer=fusion_layer,
                                                num_layers=num_layers,
                                                hidden_size=768,
                                                add_moe=add_moe,)

        alpha = cfg.get("alpha", 0.4)
        momentum = cfg.get("momentum", 0.995)
        use_distill = cfg.get("use_distill", True)
        num_classes = cfg.get("num_classes", -1)
        max_txt_len = cfg.get("max_txt_len", 128)

        assert num_classes > 1, "Invalid number of classes provided, found {}".format(
            num_classes
        )

        model = cls(
            image_encoder=image_encoder,
            text_encoder=text_encoder,
            use_distill=use_distill,
            alpha=alpha,
            num_classes=num_classes,
            momentum=momentum,
            max_txt_len=max_txt_len,
        )

        model.load_checkpoint_from_config(cfg)
        # print('#'*30)
        # model.text_encoder.from_config(cfg,
        #                                 from_pretrained=True,
        #                                 fusion_layer=fusion_layer,
        #                                 num_layers=num_layers,
        #                                 add_moe=add_moe,)
        # print('#'*30)
        return model

def compute_logits(model, dataloader):
    from lavis.common.logger import MetricLogger
    metric_logger = MetricLogger(delimiter="  ")
    header = "Evaluation:"

    logging.info("Computing features for evaluation...")

    text_embeds = []
    text_atts = []
    for cls_name in model.classnames[0]:
        texts = [template(cls_name) for template in model.prompt_templates]  # format with class
        texts = model.tokenizer(texts, padding="max_length",truncation=True,return_tensors="pt") # tokenize
        text_atts.append(texts.attention_mask)
        texts.to(model.device)
        text_embed = model.text_encoder.forward_text(texts).last_hidden_state
        text_embeds.append(text_embed.cpu()) #[80xTxD, 80xTxD,... ,]^{1000}
    text_atts = torch.cat(text_atts, dim=0)
    text_embeds = torch.cat(text_embeds, dim=0) #80k x T x D

    logging.info("Fusing features for evaluation...")
    targets = []
    logits_all_img = []
    results = []
    for samples in dataloader:
        target = samples["label"]
        # targets.append(target)

        image = samples["image"]
        image_embed = model.visual_encoder.forward_features(image)
        # image_atts = torch.ones(image_embed.size()[:-1], dtype=torch.long).to(model.device)
        image_embed = F.normalize(image_embed, dim=-1)
        # image_embeds.append(image_embed)
        # for i, sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, header)):
        for i in tqdm(range(image_embed.size(0))): # bs 128
            img_emd_bs = image_embed[i].repeat(500, 1, 1)
            image_atts = torch.ones(img_emd_bs.size()[:-1], dtype=torch.long).to(model.device)
            logits_per_img = []
            for s in range(0, text_embeds.size(0), 500):
                encoder_output_pos = model.text_encoder(
                    encoder_embeds=text_embeds[s: s+500].to(model.device),
                    attention_mask=text_atts[s: s+500].to(model.device),
                    encoder_hidden_states=img_emd_bs,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                    mode="fusion",)
                logits = model.itm_head(encoder_output_pos.last_hidden_state[:,0,:]) # 250 x 2
                logits_per_img.append(logits)
            logits_per_img = torch.cat(logits_per_img,dim=0).reshape(len(model.prompt_templates),
                                                                     1000,2).mean(dim=0) # 1k x 2
            logits_all_img.append(logits_per_img)
        logits = torch.stack(logits_all_img,dim=0)[:,:,1]

        predictions = logits.max(1)[1].cpu().numpy()
        targets = target.cpu().numpy()

        indices = samples["instance_id"]

        for pred, tgt, index in zip(predictions, targets, indices):
            if isinstance(index, torch.Tensor):
                index = index.item()

            results.append(
                {
                    "instance_id": index,
                    "prediction": pred.item(),
                    "target": tgt.item(),
                }
            )

        return results


openai_imagenet_template = [lambda c: f"a photo of a cool {c}.",]

# openai_imagenet_template = [
#     lambda c: f"a bad photo of a {c}.",
#     lambda c: f"a photo of many {c}.",
#     lambda c: f"a sculpture of a {c}.",
#     lambda c: f"a photo of the hard to see {c}.",
#     lambda c: f"a low resolution photo of the {c}.",
#     lambda c: f"a rendering of a {c}.",
#     lambda c: f"graffiti of a {c}.",
#     lambda c: f"a bad photo of the {c}.",
#     lambda c: f"a cropped photo of the {c}.",
#     lambda c: f"a tattoo of a {c}.",
#     lambda c: f"the embroidered {c}.",
#     lambda c: f"a photo of a hard to see {c}.",
#     lambda c: f"a bright photo of a {c}.",
#     lambda c: f"a photo of a clean {c}.",
#     lambda c: f"a photo of a dirty {c}.",
#     lambda c: f"a dark photo of the {c}.",
#     lambda c: f"a drawing of a {c}.",
#     lambda c: f"a photo of my {c}.",
#     lambda c: f"the plastic {c}.",
#     lambda c: f"a photo of the cool {c}.",
#     lambda c: f"a close-up photo of a {c}.",
#     lambda c: f"a black and white photo of the {c}.",
#     lambda c: f"a painting of the {c}.",
#     lambda c: f"a painting of a {c}.",
#     lambda c: f"a pixelated photo of the {c}.",
#     lambda c: f"a sculpture of the {c}.",
#     lambda c: f"a bright photo of the {c}.",
#     lambda c: f"a cropped photo of a {c}.",
#     lambda c: f"a plastic {c}.",
#     lambda c: f"a photo of the dirty {c}.",
#     lambda c: f"a jpeg corrupted photo of a {c}.",
#     lambda c: f"a blurry photo of the {c}.",
#     lambda c: f"a photo of the {c}.",
#     lambda c: f"a good photo of the {c}.",
#     lambda c: f"a rendering of the {c}.",
#     lambda c: f"a {c} in a video game.",
#     lambda c: f"a photo of one {c}.",
#     lambda c: f"a doodle of a {c}.",
#     lambda c: f"a close-up photo of the {c}.",
#     lambda c: f"a photo of a {c}.",
#     lambda c: f"the origami {c}.",
#     lambda c: f"the {c} in a video game.",
#     lambda c: f"a sketch of a {c}.",
#     lambda c: f"a doodle of the {c}.",
#     lambda c: f"a origami {c}.",
#     lambda c: f"a low resolution photo of a {c}.",
#     lambda c: f"the toy {c}.",
#     lambda c: f"a rendition of the {c}.",
#     lambda c: f"a photo of the clean {c}.",
#     lambda c: f"a photo of a large {c}.",
#     lambda c: f"a rendition of a {c}.",
#     lambda c: f"a photo of a nice {c}.",
#     lambda c: f"a photo of a weird {c}.",
#     lambda c: f"a blurry photo of a {c}.",
#     lambda c: f"a cartoon {c}.",
#     lambda c: f"art of a {c}.",
#     lambda c: f"a sketch of the {c}.",
#     lambda c: f"a embroidered {c}.",
#     lambda c: f"a pixelated photo of a {c}.",
#     lambda c: f"itap of the {c}.",
#     lambda c: f"a jpeg corrupted photo of the {c}.",
#     lambda c: f"a good photo of a {c}.",
#     lambda c: f"a plushie {c}.",
#     lambda c: f"a photo of the nice {c}.",
#     lambda c: f"a photo of the small {c}.",
#     lambda c: f"a photo of the weird {c}.",
#     lambda c: f"the cartoon {c}.",
#     lambda c: f"art of the {c}.",
#     lambda c: f"a drawing of the {c}.",
#     lambda c: f"a photo of the large {c}.",
#     lambda c: f"a black and white photo of a {c}.",
#     lambda c: f"the plushie {c}.",
#     lambda c: f"a dark photo of a {c}.",
#     lambda c: f"itap of a {c}.",
#     lambda c: f"graffiti of the {c}.",
#     lambda c: f"a toy {c}.",
#     lambda c: f"itap of my {c}.",
#     lambda c: f"a photo of a cool {c}.",
#     lambda c: f"a photo of a small {c}.",
#     lambda c: f"a tattoo of the {c}.",
# ]
