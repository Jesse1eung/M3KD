"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
from collections import OrderedDict

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


class SST2Dataset(MultimodalClassificationDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.class_labels = self._build_class_labels()

    def _build_class_labels(self):
        return {"negative": 0, "positive": 1}

    def __getitem__(self, index):
        ann = self.annotation[index]

        # image_id = ann["image"]
        # image_path = os.path.join(self.vis_root, "%s.jpg" % image_id)
        # image = Image.open(image_path).convert("RGB")

        # image = self.vis_processor(image)
        sentence = self.text_processor(ann["sentence"])

        return {
            # "image": image,
            "text_input": sentence,
            "label": ann["label"],
            # "image_id": image_id,
            "instance_id": ann["instance_id"],
        }

class CoLADataset(MultimodalClassificationDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.class_labels = self._build_class_labels()

    def _build_class_labels(self):
        return {"negative": 0, "positive": 1}

    def __getitem__(self, index):
        ann = self.annotation[index]

        # image_id = ann["image"]
        # image_path = os.path.join(self.vis_root, "%s.jpg" % image_id)
        # image = Image.open(image_path).convert("RGB")

        # image = self.vis_processor(image)
        sentence = self.text_processor(ann["sentence"])

        return {
            # "image": image,
            "text_input": sentence,
            "label": ann["label"],
            # "image_id": image_id,
            "instance_id": ann["instance_id"],
        }



class MNLIDataset(MultimodalClassificationDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.class_labels = self._build_class_labels()

    def _build_class_labels(self):
        return {"contradiction": 0, "neutral": 1, "entailment": 2}

    def __getitem__(self, index):
        ann = self.annotation[index]

        # image_id = ann["image"]
        # image_path = os.path.join(self.vis_root, "%s.jpg" % image_id)
        # image = Image.open(image_path).convert("RGB")

        # image = self.vis_processor(image)
        sentence1 = self.text_processor(ann["sentence1"])
        sentence2 = self.text_processor(ann["sentence2"])

        return {
            # "image": image,
            "text_input": sentence1,
            "text_input2": sentence2,
            "label": self.class_labels[ann["label"]],
            # "image_id": image_id,
            "instance_id": ann["instance_id"],
        }

class QNLIDataset(MultimodalClassificationDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.class_labels = self._build_class_labels()

    def _build_class_labels(self):
        return {"not_entailment": 0,  "entailment": 1}

    def __getitem__(self, index):
        ann = self.annotation[index]

        # image_id = ann["image"]
        # image_path = os.path.join(self.vis_root, "%s.jpg" % image_id)
        # image = Image.open(image_path).convert("RGB")

        # image = self.vis_processor(image)
        sentence1 = self.text_processor(ann["sentence1"])
        sentence2 = self.text_processor(ann["sentence2"])

        return {
            # "image": image,
            "text_input": sentence1,
            "text_input2": sentence2,
            "label": self.class_labels[ann["label"]],
            # "image_id": image_id,
            "instance_id": ann["instance_id"],
        }

class MRPCDataset(MultimodalClassificationDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.class_labels = self._build_class_labels()

    def _build_class_labels(self):
        return {"not_entailment": 0,  "entailment": 1}

    def __getitem__(self, index):
        ann = self.annotation[index]

        # image_id = ann["image"]
        # image_path = os.path.join(self.vis_root, "%s.jpg" % image_id)
        # image = Image.open(image_path).convert("RGB")

        # image = self.vis_processor(image)
        sentence1 = self.text_processor(ann["sentence1"])
        sentence2 = self.text_processor(ann["sentence2"])

        return {
            # "image": image,
            "text_input": sentence1,
            "text_input2": sentence2,
            "label": ann["label"],
            # "image_id": image_id,
            "instance_id": ann["instance_id"],
        }

class QQPDataset(MultimodalClassificationDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.class_labels = self._build_class_labels()

    def _build_class_labels(self):
        return {"not_entailment": 0,  "entailment": 1}

    def __getitem__(self, index):
        ann = self.annotation[index]

        # image_id = ann["image"]
        # image_path = os.path.join(self.vis_root, "%s.jpg" % image_id)
        # image = Image.open(image_path).convert("RGB")

        # image = self.vis_processor(image)
        sentence1 = self.text_processor(ann["sentence1"])
        sentence2 = self.text_processor(ann["sentence2"])

        return {
            # "image": image,
            "text_input": sentence1,
            "text_input2": sentence2,
            "label": ann["label"],
            # "image_id": image_id,
            "instance_id": ann["instance_id"],
        }

class RTEDataset(MultimodalClassificationDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.class_labels = self._build_class_labels()

    def _build_class_labels(self):
        return {"not_entailment": 0,  "entailment": 1}

    def __getitem__(self, index):
        ann = self.annotation[index]

        # image_id = ann["image"]
        # image_path = os.path.join(self.vis_root, "%s.jpg" % image_id)
        # image = Image.open(image_path).convert("RGB")

        # image = self.vis_processor(image)
        sentence1 = self.text_processor(ann["sentence1"])
        sentence2 = self.text_processor(ann["sentence2"])

        return {
            # "image": image,
            "text_input": sentence1,
            "text_input2": sentence2,
            "label": self.class_labels[ann["label"]],
            # "image_id": image_id,
            "instance_id": ann["instance_id"],
        }

