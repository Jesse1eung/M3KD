"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import re
import sys
import os


from lavis.common.registry import registry
from lavis.processors.base_processor import BaseProcessor
from lavis.processors.randaugment import RandomAugment
from omegaconf import OmegaConf
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

base_dir = os.path.dirname('/home1/liangjx/projects/mm/')
sys.path.append(base_dir)
# from trans.models.vilt import ViltFeatureExtractor
from transformers import ViltFeatureExtractor

@registry.register_processor("vilt_image_train")
class ViltImageTrainProcessor(BaseProcessor):
    def __init__(self,ckpt_file=None, image_size=None):
        self.transform = ViltFeatureExtractor.from_pretrained(ckpt_file, size=image_size)
        # self.transform = transforms.Compose([
        #     transforms.RandomResizedCrop(image_size,scale=(min_scale, max_scale),
        #                                  interpolation=InterpolationMode.BICUBIC,),
        #     transforms.RandomHorizontalFlip(),
        #     RandomAugment(2, 5, isPIL=True,augs=["Identity","AutoContrast","Brightness","Sharpness",
        #                 "Equalize","ShearX","ShearY","TranslateX","TranslateY","Rotate",],),
        #     transforms.ToTensor(),
        #     self.normalize,
        # ])


    def __call__(self, item):
        return self.transform(item, return_tensors='pt')
        # return self.transform(item)

    @classmethod
    def from_config(cls, cfg=None):
        print(cfg,'what is this cfg')
        # print(type(cls))
        # ckpt_file = cfg.get("teacher_ckpt","vilt-b32-mlm")
        image_size = cfg.get('image_size', 224)
        # image_size = 2
        ckpt_file = "./pretrained_ckpt/vilt-b32-mlm"
        return cls(ckpt_file=ckpt_file, image_size=image_size)

class BlipImageBaseProcessor(BaseProcessor):
    def __init__(self, mean=None, std=None):
        if mean is None:
            mean = (0.48145466, 0.4578275, 0.40821073)
        if std is None:
            std = (0.26862954, 0.26130258, 0.27577711)

        self.normalize = transforms.Normalize(mean, std)

# @registry.register_processor("blip_image_train")
# class BlipImageTrainProcessor(BlipImageBaseProcessor):
#     def __init__(
#         self, image_size=384, mean=None, std=None, min_scale=0.5, max_scale=1.0
#     ):
#         super().__init__(mean=mean, std=std)
#
#         self.transform = transforms.Compose(
#             [
#                 transforms.RandomResizedCrop(
#                     image_size,
#                     scale=(min_scale, max_scale),
#                     interpolation=InterpolationMode.BICUBIC,
#                 ),
#                 transforms.RandomHorizontalFlip(),
#                 RandomAugment(
#                     2,
#                     5,
#                     isPIL=True,
#                     augs=[
#                         "Identity",
#                         "AutoContrast",
#                         "Brightness",
#                         "Sharpness",
#                         "Equalize",
#                         "ShearX",
#                         "ShearY",
#                         "TranslateX",
#                         "TranslateY",
#                         "Rotate",
#                     ],
#                 ),
#                 transforms.ToTensor(),
#                 self.normalize,
#             ]
#         )
#
#     def __call__(self, item):
#         return self.transform(item)
#
#     @classmethod
#     def from_config(cls, cfg=None):
#         if cfg is None:
#             cfg = OmegaConf.create()
#
#         image_size = cfg.get("image_size", 384)
#
#         mean = cfg.get("mean", None)
#         std = cfg.get("std", None)
#
#         min_scale = cfg.get("min_scale", 0.5)
#         max_scale = cfg.get("max_scale", 1.0)
#
#         return cls(
#             image_size=image_size,
#             mean=mean,
#             std=std,
#             min_scale=min_scale,
#             max_scale=max_scale,
#         )
#
#
# @registry.register_processor("blip_image_eval")
# class BlipImageEvalProcessor(BlipImageBaseProcessor):
#     def __init__(self, image_size=384, mean=None, std=None):
#         super().__init__(mean=mean, std=std)
#
#         self.transform = transforms.Compose(
#             [
#                 transforms.Resize(
#                     (image_size, image_size), interpolation=InterpolationMode.BICUBIC
#                 ),
#                 transforms.ToTensor(),
#                 self.normalize,
#             ]
#         )
#
#     def __call__(self, item):
#         return self.transform(item)
#
#     @classmethod
#     def from_config(cls, cfg=None):
#         if cfg is None:
#             cfg = OmegaConf.create()
#
#         image_size = cfg.get("image_size", 384)
#
#         mean = cfg.get("mean", None)
#         std = cfg.get("std", None)
#
#         return cls(image_size=image_size, mean=mean, std=std)
