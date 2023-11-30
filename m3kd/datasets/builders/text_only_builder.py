"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.common.registry import registry
from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.datasets.text_only_datasets import BookCorpusDataset, EnglishWikiDataset


@registry.register_builder("book")
class BookCorpusBuilder(BaseDatasetBuilder):
    train_dataset_cls = BookCorpusDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/text_corpus/book.yaml"}

@registry.register_builder("wiki")
class EnglishWikiBuilder(BaseDatasetBuilder):
    train_dataset_cls = EnglishWikiDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/text_corpus/wiki.yaml"}

