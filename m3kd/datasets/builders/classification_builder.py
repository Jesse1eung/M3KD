"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.common.registry import registry
from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.datasets.nlvr_datasets import NLVRDataset, NLVREvalDataset
from lavis.datasets.datasets.snli_ve_datasets import SNLIVisualEntialmentDataset

from lavis.datasets.datasets.glue_datasets import SST2Dataset, MNLIDataset, CoLADataset, \
    QNLIDataset, QQPDataset, RTEDataset, MRPCDataset


@registry.register_builder("nlvr")
class NLVRBuilder(BaseDatasetBuilder):
    train_dataset_cls = NLVRDataset
    eval_dataset_cls = NLVREvalDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/nlvr/defaults.yaml"}


@registry.register_builder("snli_ve")
class SNLIVisualEntailmentBuilder(BaseDatasetBuilder):
    train_dataset_cls = SNLIVisualEntialmentDataset
    eval_dataset_cls = SNLIVisualEntialmentDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/snli_ve/defaults.yaml"}

@registry.register_builder("sst2")
class SST2Builder(BaseDatasetBuilder):
    train_dataset_cls = SST2Dataset
    eval_dataset_cls = SST2Dataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/glue/sst2.yaml"}

@registry.register_builder("mnli")
class MNLIBuilder(BaseDatasetBuilder):
    train_dataset_cls = MNLIDataset
    eval_dataset_cls = MNLIDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/glue/mnli.yaml"}

@registry.register_builder("cola")
class CoLABuilder(BaseDatasetBuilder):
    train_dataset_cls = CoLADataset
    eval_dataset_cls = CoLADataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/glue/cola.yaml"}

@registry.register_builder("qnli")
class QNLIBuilder(BaseDatasetBuilder):
    train_dataset_cls = QNLIDataset
    eval_dataset_cls = QNLIDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/glue/qnli.yaml"}

@registry.register_builder("mrpc")
class MRPCBuilder(BaseDatasetBuilder):
    train_dataset_cls = MRPCDataset
    eval_dataset_cls = MRPCDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/glue/mrpc.yaml"}

@registry.register_builder("qqp")
class QQPBuilder(BaseDatasetBuilder):
    train_dataset_cls = QQPDataset
    eval_dataset_cls = QQPDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/glue/qqp.yaml"}

@registry.register_builder("rte")
class RTEBuilder(BaseDatasetBuilder):
    train_dataset_cls = RTEDataset
    eval_dataset_cls = RTEDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/glue/rte.yaml"}

