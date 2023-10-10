"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os, json
from collections import OrderedDict

from lavis.datasets.datasets.base_dataset import BaseDataset
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from lavis.datasets.datasets.multimodal_classification_datasets import (
    MultimodalClassificationDataset,
)
from PIL import Image


class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": os.path.basename(ann["image"]),
                "sentence": ann["sentence"],
                "label": ann["label"],
                "image": sample["image"],
            }
        )


class BookCorpusDataset(Dataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        # super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        self.vis_root = vis_root

        self.annotation = []
        for ann_path in ann_paths:
            self.annotation.extend(json.load(open(ann_path, "r")))

        self.vis_processor = vis_processor
        self.text_processor = text_processor
        # self.class_labels = self._build_class_labels()

    def __len__(self):
        return len(self.annotation)

    def collater(self, samples):
        # print(samples[0],'hellds')
        return default_collate(samples)

    def _build_class_labels(self):
        return {"negative": 0, "positive": 1}

    def __getitem__(self, index):
        ann = self.annotation[index]

        # image_id = ann["image"]
        # image_path = os.path.join(self.vis_root, "%s.jpg" % image_id)
        # image = Image.open(image_path).convert("RGB")
        # image = self.vis_processor(image)
        sentence = self.text_processor(ann)

        return {
            "text_input": sentence,
        }


class EnglishWikiDataset(Dataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        # super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        self.vis_root = vis_root

        self.annotation = []
        for ann_path in ann_paths:
            self.annotation.extend(json.load(open(ann_path, "r")))

        self.vis_processor = vis_processor
        self.text_processor = text_processor
        # self.class_labels = self._build_class_labels()

    def __len__(self):
        return len(self.annotation)

    def collater(self, samples):
        # print(samples[0],'hellds')
        return default_collate(samples)

    def _build_class_labels(self):
        return {"contradiction": 0, "neutral": 1, "entailment": 2}

    def __getitem__(self, index):
        ann = self.annotation[index]

        # image_id = ann["image"]
        # image_path = os.path.join(self.vis_root, "%s.jpg" % image_id)
        # image = Image.open(image_path).convert("RGB")

        # image = self.vis_processor(image)
        # print(self.text_processor,"text_processor")
        sentence1 = self.text_processor(ann)

        return {
            "text_input": sentence1,
        }