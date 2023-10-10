"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 
 Based on huggingface code base
 https://github.com/huggingface/transformers/blob/v4.15.0/src/transformers/models/bert
"""

import math
import os
import collections
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor, device
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from transformers import BatchEncoding, PreTrainedTokenizer

from transformers.activations import ACT2FN
from transformers.file_utils import (
    ModelOutput,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPooling,
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    NextSentencePredictorOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers.modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,

)
from transformers.utils import logging
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.beit.configuration_beit import BeitConfig
from lavis.common.utils import get_abs_path

from lavis.models.base_model import BaseEncoder

logging.set_verbosity_error()
logger = logging.get_logger(__name__)


class BeitModelOutputWithPooling(BaseModelOutputWithPooling):
    """
    Class for outputs of [`BeitModel`].
    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
            Average of the last layer hidden states of the patch tokens (excluding the *[CLS]* token) if
            *config.use_mean_pooling* is set to True. If set to False, then the final hidden state of the *[CLS]* token
            will be returned.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """


def drop_path(input: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    Comment by Ross Wightman: This is the same as the DropConnect impl I created for EfficientNet, etc networks,
    however, the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for changing the
    layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use 'survival rate' as the
    argument.
    """
    if drop_prob == 0.0 or not training:
        return input
    keep_prob = 1 - drop_prob
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    random_tensor.floor_()  # binarize
    output = input.div(keep_prob) * random_tensor
    return output


class BeitDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


class BeitEmbeddings(nn.Module):
    """
    Construct the CLS token, position and patch embeddings. Optionally, also the mask token.
    """

    def __init__(self, config: BeitConfig) -> None:
        super().__init__()
        self.config = config
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        if config.use_mask_token:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        else:
            self.mask_token = None
        self.patch_embeddings = BeitPatchEmbeddings(config)
        num_patches = self.patch_embeddings.num_patches
        if config.use_absolute_position_embeddings:
            self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 1, config.hidden_size))
        else:
            self.position_embeddings = None
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, pixel_values: torch.Tensor, pixel_mask: torch.Tensor, bool_masked_pos: Optional[torch.BoolTensor] = None) -> torch.Tensor:
        _, _, ph, pw = self.patch_embeddings.projection.weight.shape

        x = self.patch_embeddings(pixel_values)
        x_mask = pixel_mask[:, None, :, :].float()
        x_mask = nn.functional.interpolate(x_mask, size=(x.shape[2], x.shape[3])).long()
        x_h = x_mask[:, 0].sum(dim=1)[:, 0]
        x_w = x_mask[:, 0].sum(dim=2)[:, 0]

        batch_size, num_channels, height, width = x.shape
        patch_dim = self.config.image_size // self.config.patch_size
        spatial_pos = self.position_embeddings[:, 1:, :].transpose(1, 2).view(1, num_channels, patch_dim, patch_dim)
        pos_embed = torch.cat(
            [
                nn.functional.pad(
                    nn.functional.interpolate(
                        spatial_pos,
                        size=(h, w),
                        mode="bilinear",
                        align_corners=True,
                    ),
                    (0, width - w, 0, height - h),
                )
                for h, w in zip(x_h, x_w)
            ],
            dim=0,
        )

        pos_embed = pos_embed.flatten(2).transpose(1, 2)
        x = x.flatten(2).transpose(1, 2)
        # Set `device` here, otherwise `patch_index` will always be on `CPU` and will fail near the end for torch>=1.13
        patch_index = torch.stack(
            torch.meshgrid(torch.arange(x_mask.shape[-2]), torch.arange(x_mask.shape[-1])), dim=-1
        ).to(device=x_mask.device)
        patch_index = patch_index[None, None, :, :, :]
        patch_index = patch_index.expand(x_mask.shape[0], x_mask.shape[1], -1, -1, -1)
        patch_index = patch_index.flatten(1, 3)
        x_mask = x_mask.flatten(1)
        max_image_length = -1

        if max_image_length < 0 or max_image_length is None or not isinstance(max_image_length, int):
            # suppose aug is 800 x 1333, then, maximum effective res is 800 x 1333 (if one side gets bigger, the other will be constrained and be shrinked)
            # (800 // self.patch_size) * (1333 // self.patch_size) is the maximum number of patches that single image can get.
            # if self.patch_size = 32, 25 * 41 = 1025
            # if res is 384 x 640, 12 * 20 = 240
            effective_resolution = x_h * x_w
            max_image_length = effective_resolution.max()
        else:
            effective_resolution = x_h * x_w
            max_image_length = min(effective_resolution.max(), max_image_length)

        valid_idx = x_mask.nonzero(as_tuple=False)
        non_valid_idx = (1 - x_mask).nonzero(as_tuple=False)
        unique_rows = valid_idx[:, 0].unique()
        valid_row_idx = [valid_idx[valid_idx[:, 0] == u] for u in unique_rows]
        non_valid_row_idx = [non_valid_idx[non_valid_idx[:, 0] == u] for u in unique_rows]

        valid_nums = [v.size(0) for v in valid_row_idx]
        non_valid_nums = [v.size(0) for v in non_valid_row_idx]
        pad_nums = [max_image_length - v for v in valid_nums]

        select = list()
        for i, (v, nv, p) in enumerate(zip(valid_nums, non_valid_nums, pad_nums)):
            if p <= 0:
                valid_choice = torch.multinomial(torch.ones(v).float(), max_image_length)
                select.append(valid_row_idx[i][valid_choice])
            else:
                pad_choice = torch.multinomial(torch.ones(nv).float(), p, replacement=True)
                select.append(torch.cat([valid_row_idx[i], non_valid_row_idx[i][pad_choice]], dim=0))

        select = torch.cat(select, dim=0)
        x = x[select[:, 0], select[:, 1]].view(batch_size, -1, num_channels)
        x_mask = x_mask[select[:, 0], select[:, 1]].view(batch_size, -1)
        # `patch_index` should be on the same device as `select` (for torch>=1.13), which is ensured at definition time.
        patch_index = patch_index[select[:, 0], select[:, 1]].view(batch_size, -1, 2)
        pos_embed = pos_embed[select[:, 0], select[:, 1]].view(batch_size, -1, num_channels)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        if bool_masked_pos is not None:
            mask_tokens = self.mask_token.expand(batch_size, seq_len, -1)
            # replace the masked visual tokens by mask_tokens
            w = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            x = x * (1 - w) + mask_tokens * w
        x = torch.cat((cls_tokens, x), dim=1)
        pos_embed = torch.cat(
            (self.position_embeddings[:, 0, :][:, None, :].expand(batch_size, -1, -1), pos_embed), dim=1
        )
        x = x + pos_embed
        x = self.dropout(x)

        x_mask = torch.cat([torch.ones(x_mask.shape[0], 1).to(x_mask), x_mask], dim=1)

        return x, x_mask, (patch_index, (height, width))
        # embeddings = self.patch_embeddings(pixel_values)
        # batch_size, seq_len, _ = embeddings.size()
        #
        # cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        # if bool_masked_pos is not None:
        #     mask_tokens = self.mask_token.expand(batch_size, seq_len, -1)
        #     # replace the masked visual tokens by mask_tokens
        #     w = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
        #     embeddings = embeddings * (1 - w) + mask_tokens * w

        # embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        # if self.position_embeddings is not None:
        #     embeddings = embeddings + self.position_embeddings
        # embeddings = self.dropout(embeddings)

        # return embeddings


class BeitPatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config):
        super().__init__()
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_size

        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        patch_shape = (image_size[0] // patch_size[0], image_size[1] // patch_size[1])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches
        self.patch_shape = patch_shape

        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        if height != self.image_size[0] or width != self.image_size[1]:
            raise ValueError(
                f"Input image size ({height}*{width}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."
            )
        embeddings = self.projection(pixel_values)

        return embeddings



class BeitSelfAttention(nn.Module):
    def __init__(self, config: BeitConfig, window_size: Optional[tuple] = None) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=False)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        if window_size:
            self.relative_position_bias = BeitRelativePositionBias(config, window_size=window_size)
        else:
            self.relative_position_bias = None

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask=None,
            head_mask: Optional[torch.Tensor] = None,
            output_attentions: bool = False,
            relative_position_bias: Optional["BeitRelativePositionBias"] = None,
    ) -> Union[Tuple[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Add relative position bias if present.
        if self.relative_position_bias is not None:
            attention_scores = attention_scores + self.relative_position_bias().unsqueeze(0)

        # Add shared relative position bias if provided.
        if relative_position_bias is not None:
            print(attention_scores.shape, relative_position_bias.shape)
            attention_scores = attention_scores + relative_position_bias

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


class BeitSelfOutput(nn.Module):
    """
    The residual connection is defined in BeitLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, config: BeitConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor, gamma=None) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


class BeitAttention(nn.Module):
    def __init__(self, config: BeitConfig, window_size: Optional[tuple] = None) -> None:
        super().__init__()
        self.attention = BeitSelfAttention(config, window_size=window_size)
        self.output = BeitSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)
        self.attention.all_head_size = self.attention.attention_head_size * self.attention.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask = None,
            head_mask: Optional[torch.Tensor] = None,
            output_attentions: bool = False,
            relative_position_bias: Optional["BeitRelativePositionBias"] = None,
    ) -> Union[Tuple[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        self_outputs = self.attention(hidden_states, attention_mask, head_mask, output_attentions, relative_position_bias)

        attention_output = self.output(self_outputs[0], hidden_states)

        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class BeitIntermediate(nn.Module):
    def __init__(self, config: BeitConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states


class BeitOutput(nn.Module):
    def __init__(self, config: BeitConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


class BeitLayer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config: BeitConfig, layer_num, window_size: Optional[tuple] = None, drop_path_rate: float = 0.0) -> None:
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BeitAttention(config, window_size=window_size)

        self.intermediate = BeitIntermediate(config)
        self.output = BeitOutput(config)
        if layer_num >= config.fusion_layer:
            self.l_intermediate = BeitIntermediate(config)
            self.l_output = BeitOutput(config)

            self.vl_intermediate = BeitIntermediate(config)
            self.vl_output = BeitOutput(config)



        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.drop_path = BeitDropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        init_values = config.layer_scale_init_value
        if init_values > 0:
            self.lambda_1 = nn.Parameter(init_values * torch.ones((config.hidden_size)), requires_grad=True)
            self.lambda_2 = nn.Parameter(init_values * torch.ones((config.hidden_size)), requires_grad=True)
        else:
            self.lambda_1, self.lambda_2 = None, None

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask=None,
            head_mask: Optional[torch.Tensor] = None,
            output_attentions: bool = False,
            relative_position_bias: Optional["BeitRelativePositionBias"] = None,
            mode='image',
    ) -> Union[Tuple[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:

        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),  # in BEiT, layernorm is applied before self-attention
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            relative_position_bias=relative_position_bias,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # apply lambda_1 if present
        if self.lambda_1 is not None:
            attention_output = self.lambda_1 * attention_output

        # first residual connection
        hidden_states = self.drop_path(attention_output) + hidden_states

        # in BEiT, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(hidden_states)



        if mode == 'image':
            layer_output = self.intermediate(layer_output)
            layer_output = self.output(layer_output)
        elif mode == 'text':
            layer_output = self.l_intermediate(layer_output)
            layer_output = self.l_output(layer_output)
        elif mode == 'fusion':
            layer_output = self.vl_intermediate(layer_output)
            layer_output = self.vl_output(layer_output)

        if self.lambda_2 is not None:
            layer_output = self.lambda_2 * layer_output

        # second residual connection
        layer_output = self.drop_path(layer_output) + hidden_states

        outputs = (layer_output,) + outputs

        return outputs

class BeitRelativePositionBias(nn.Module):
    def __init__(self, config: BeitConfig, window_size: tuple) -> None:
        super().__init__()
        self.window_size = window_size
        self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(self.num_relative_distance, config.num_attention_heads)
        )  # 2*Wh-1 * 2*Ww-1, nH
        # cls to token & token 2 cls & cls to cls

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = torch.zeros(
            size=(window_size[0] * window_size[1] + 1,) * 2, dtype=relative_coords.dtype
        )
        relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        relative_position_index[0, 0:] = self.num_relative_distance - 3
        relative_position_index[0:, 0] = self.num_relative_distance - 2
        relative_position_index[0, 0] = self.num_relative_distance - 1

        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self) -> torch.Tensor:
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1] + 1, self.window_size[0] * self.window_size[1] + 1, -1
        )  # Wh*Ww,Wh*Ww,nH

        return relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww

class BeitEncoder(nn.Module):
    def __init__(self, config: BeitConfig, window_size: Optional[tuple] = None) -> None:
        super().__init__()
        self.config = config
        if config.use_shared_relative_position_bias:
            self.relative_position_bias = BeitRelativePositionBias(config, window_size=window_size)
        else:
            self.relative_position_bias = None

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.num_hidden_layers)]
        self.layer = nn.ModuleList(
            [
                BeitLayer(
                    config,
                    layer_num=i,
                    window_size=window_size if config.use_relative_position_bias else None,
                    drop_path_rate=dpr[i],
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.gradient_checkpointing = False

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask = None,
            head_mask: Optional[torch.Tensor] = None,
            output_attentions: bool = False,
            output_hidden_states: bool = False,
            return_dict: bool = True,
            mode='multimodal',
    ) -> Union[tuple, BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        # for i, layer_module in enumerate(self.layer):

        try:
            # ALBEF
            fusion_layer = self.config.fusion_layer
        except AttributeError:
            # BLIP
            fusion_layer = self.config.num_hidden_layers

        if mode == "text" or mode == "fusion":
            start_layer = fusion_layer
            # output_layer = self.config.fusion_layer
            output_layer = self.config.num_hidden_layers

        elif mode == 'image':
            start_layer = 0
            output_layer = fusion_layer if fusion_layer < self.config.num_hidden_layers else self.config.num_hidden_layers

        # elif mode == "fusion":
        #     # start_layer = self.config.fusion_layer
        #     start_layer = fusion_layer
        #     output_layer = self.config.num_hidden_layers

        # elif mode == "multimodal":
        #     start_layer = 0
        #     output_layer = self.config.num_hidden_layers
        for i in range(start_layer, output_layer):
            layer_module = self.layer[i]
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    layer_head_mask,
                )
            else:
                relative_position_bias = (
                    self.relative_position_bias() if self.relative_position_bias is not None else None
                )
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    output_attentions,
                    relative_position_bias,
                    mode=mode,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )



class BeitPooler(nn.Module):
    def __init__(self, config: BeitConfig) -> None:
        super().__init__()
        self.layernorm = (
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps) if config.use_mean_pooling else None
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.layernorm is not None:
            # Mean pool the final hidden states of the patch tokens
            patch_tokens = hidden_states[:, 1:, :]
            pooled_output = self.layernorm(patch_tokens.mean(1))
        else:
            # Pool by simply taking the final hidden state of the [CLS] token
            pooled_output = hidden_states[:, 0]

        return pooled_output

class BeitPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BeitLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BeitPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.text_vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.text_vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BeitOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BeitLMPredictionHead(config)
        self.config = config

    def forward(self, sequence_output, labels=None):
        prediction_scores = self.predictions(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.text_vocab_size), labels.view(-1)
            )
        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
        )


class BertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = BertConfig
    base_model_prefix = "bert"
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

class BeitPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = BeitConfig
    base_model_prefix = "beit"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, BeitEncoder):
            module.gradient_checkpointing = value

class BeitModel(BeitPreTrainedModel):
    def __init__(self, config: BeitConfig, add_pooling_layer: bool = True) -> None:
        super().__init__(config)
        self.config = config

        self.embeddings = BeitEmbeddings(config)
        self.encoder = BeitEncoder(config, window_size=self.embeddings.patch_embeddings.patch_shape)

        self.layernorm = (
            nn.Identity() if config.use_mean_pooling else nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        )
        self.pooler = BeitPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()
        self.img_attention_mask = None
    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)


    def forward(
            self,
            pixel_values: Optional[torch.Tensor] = None,
            pixel_mask: Optional[torch.Tensor] = None,
            bool_masked_pos: Optional[torch.BoolTensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            mode='multimodal',
    ) -> Union[tuple, BeitModelOutputWithPooling]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")
        image_batch_size = pixel_values.shape[0]
        device = pixel_values.device
        if pixel_mask is None:
            pixel_mask = torch.ones((image_batch_size, self.config.image_size, self.config.image_size), device=device)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)



        image_embeds, image_masks, patch_index  = self.embeddings(pixel_values, pixel_mask,bool_masked_pos)

        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(image_masks, (image_batch_size,image_embeds.shape[1]))
        self.img_attention_mask = extended_attention_mask

        encoder_outputs = self.encoder(
            image_embeds,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            mode=mode,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            head_outputs = (sequence_output, pooled_output) if pooled_output is not None else (sequence_output,)
            return head_outputs + encoder_outputs[1:]

        return BeitModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

class BeitPooler(nn.Module):
    def __init__(self, config: BeitConfig) -> None:
        super().__init__()
        self.layernorm = (
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps) if config.use_mean_pooling else None
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.layernorm is not None:
            # Mean pool the final hidden states of the patch tokens
            patch_tokens = hidden_states[:, 1:, :]
            pooled_output = self.layernorm(patch_tokens.mean(1))
        else:
            # Pool by simply taking the final hidden state of the [CLS] token
            pooled_output = hidden_states[:, 0]

        return pooled_output




class BeitForMaskedImageModeling(BeitPreTrainedModel):
    def __init__(self, config: BeitConfig) -> None:
        super().__init__(config)

        self.num_labels = config.num_labels
        self.beit = BeitModel(config, add_pooling_layer=False)

        # Classifier head
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            pixel_values: Optional[torch.Tensor] = None,
            bool_masked_pos: Optional[torch.BoolTensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            mode='multimodal',
    ) -> Union[tuple, MaskedLMOutput]:
        r"""
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        Returns:
        Examples:
        ```python
        >>> from transformers import BeitImageProcessor, BeitForMaskedImageModeling
        >>> import torch
        >>> from PIL import Image
        >>> import requests
        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> image_processor = BeitImageProcessor.from_pretrained("microsoft/beit-base-patch16-224-pt22k")
        >>> model = BeitForMaskedImageModeling.from_pretrained("microsoft/beit-base-patch16-224-pt22k")
        >>> num_patches = (model.config.image_size // model.config.patch_size) ** 2
        >>> pixel_values = image_processor(images=image, return_tensors="pt").pixel_values
        >>> # create random boolean mask of shape (batch_size, num_patches)
        >>> bool_masked_pos = torch.randint(low=0, high=2, size=(1, num_patches)).bool()
        >>> outputs = model(pixel_values, bool_masked_pos=bool_masked_pos)
        >>> loss, logits = outputs.loss, outputs.logits
        >>> list(logits.shape)
        [1, 196, 8192]
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.beit(
            pixel_values,
            bool_masked_pos=bool_masked_pos,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        sequence_output = self.layernorm(sequence_output)
        prediction_scores = self.lm_head(sequence_output[:, 1:])

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores[bool_masked_pos], labels)

        if not return_dict:
            output = (prediction_scores,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

# class BertLMHeadModel(BertPreTrainedModel):
#
#     _keys_to_ignore_on_load_unexpected = [r"pooler"]
#     _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]
#
#     def __init__(self, config):
#         super().__init__(config)
#
#         self.bert = BertModel(config, add_pooling_layer=False)
#         self.cls = BertOnlyMLMHead(config)
#
#         self.init_weights()
#
#     def get_output_embeddings(self):
#         return self.cls.predictions.decoder
#
#     def set_output_embeddings(self, new_embeddings):
#         self.cls.predictions.decoder = new_embeddings
#
#     def forward(
#         self,
#         input_ids=None,
#         attention_mask=None,
#         position_ids=None,
#         head_mask=None,
#         inputs_embeds=None,
#         encoder_hidden_states=None,
#         encoder_attention_mask=None,
#         labels=None,
#         past_key_values=None,
#         use_cache=None,
#         output_attentions=None,
#         output_hidden_states=None,
#         return_dict=None,
#         return_logits=False,
#         is_decoder=True,
#         reduction="mean",
#         mode="multimodal",
#         soft_labels=None,
#         alpha=0,
#     ):
#         r"""
#         encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
#             Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
#             the model is configured as a decoder.
#         encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
#             Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
#             the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:
#             - 1 for tokens that are **not masked**,
#             - 0 for tokens that are **masked**.
#         labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
#             Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
#             ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are
#             ignored (masked), the loss is only computed for the tokens with labels n ``[0, ..., config.vocab_size]``
#         past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
#             Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
#             If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
#             (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
#             instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
#         use_cache (:obj:`bool`, `optional`):
#             If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
#             decoding (see :obj:`past_key_values`).
#         Returns:
#         Example::
#             >>> from transformers import BertTokenizer, BertLMHeadModel, BertConfig
#             >>> import torch
#             >>> tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
#             >>> config = BertConfig.from_pretrained("bert-base-cased")
#             >>> model = BertLMHeadModel.from_pretrained('bert-base-cased', config=config)
#             >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
#             >>> outputs = model(**inputs)
#             >>> prediction_logits = outputs.logits
#         """
#         return_dict = (
#             return_dict if return_dict is not None else self.config.use_return_dict
#         )
#         if labels is not None:
#             use_cache = False
#
#         outputs = self.bert(
#             input_ids,
#             attention_mask=attention_mask,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             encoder_hidden_states=encoder_hidden_states,
#             encoder_attention_mask=encoder_attention_mask,
#             past_key_values=past_key_values,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#             is_decoder=is_decoder,
#             mode=mode,
#         )
#
#         sequence_output = outputs[0]
#         prediction_scores = self.cls(sequence_output)
#
#         if return_logits:
#             return prediction_scores[:, :-1, :].contiguous()
#
#         lm_loss = None
#         if labels is not None:
#             # we are doing next-token prediction; shift prediction scores and input ids by one
#             shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous()
#             labels = labels[:, 1:].contiguous()
#             loss_fct = CrossEntropyLoss(reduction=reduction, label_smoothing=0.1)
#             lm_loss = loss_fct(
#                 shifted_prediction_scores.view(-1, self.config.vocab_size),
#                 labels.view(-1),
#             )
#             if reduction == "none":
#                 lm_loss = lm_loss.view(prediction_scores.size(0), -1).sum(1)
#
#         if soft_labels is not None:
#             loss_distill = -torch.sum(
#                 F.log_softmax(shifted_prediction_scores, dim=-1) * soft_labels, dim=-1
#             )
#             loss_distill = (loss_distill * (labels != -100)).sum(1)
#             lm_loss = (1 - alpha) * lm_loss + alpha * loss_distill
#
#         if not return_dict:
#             output = (prediction_scores,) + outputs[2:]
#             return ((lm_loss,) + output) if lm_loss is not None else output
#
#         return CausalLMOutputWithCrossAttentions(
#             loss=lm_loss,
#             logits=prediction_scores,
#             past_key_values=outputs.past_key_values,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#             cross_attentions=outputs.cross_attentions,
#         )
#
#     def prepare_inputs_for_generation(
#         self, input_ids, past=None, attention_mask=None, **model_kwargs
#     ):
#         input_shape = input_ids.shape
#         # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
#         if attention_mask is None:
#             attention_mask = input_ids.new_ones(input_shape)
#
#         # cut decoder_input_ids if past is used
#         if past is not None:
#             input_ids = input_ids[:, -1:]
#
#         return {
#             "input_ids": input_ids,
#             "attention_mask": attention_mask,
#             "past_key_values": past,
#             "encoder_hidden_states": model_kwargs.get("encoder_hidden_states", None),
#             "encoder_attention_mask": model_kwargs.get("encoder_attention_mask", None),
#             "is_decoder": True,
#         }
#
#     def _reorder_cache(self, past, beam_idx):
#         reordered_past = ()
#         for layer_past in past:
#             reordered_past += (
#                 tuple(
#                     past_state.index_select(0, beam_idx) for past_state in layer_past
#                 ),
#             )
#         return reordered_past


# class XBertLMHeadDecoder(BertLMHeadModel):
#     """
#     This class decouples the decoder forward logic from the VL model.
#     In this way, different VL models can share this decoder as long as
#     they feed encoder_embeds as required.
#     """
#
#     @classmethod
#     def from_config(cls, cfg, from_pretrained=False):
#
#         med_config_path = get_abs_path(cfg.get("med_config_path"))
#         med_config = BeitConfig.from_json_file(med_config_path)
#
#         if from_pretrained:
#             return cls.from_pretrained("bert-base-uncased", config=med_config)
#         else:
#             return cls(config=med_config)
#
#     def generate_from_encoder(
#         self,
#         tokenized_prompt,
#         visual_embeds,
#         sep_token_id,
#         pad_token_id,
#         use_nucleus_sampling=False,
#         num_beams=3,
#         max_length=30,
#         min_length=10,
#         top_p=0.9,
#         repetition_penalty=1.0,
#         **kwargs
#     ):
#
#         if not use_nucleus_sampling:
#             num_beams = num_beams
#             visual_embeds = visual_embeds.repeat_interleave(num_beams, dim=0)
#
#         image_atts = torch.ones(visual_embeds.size()[:-1], dtype=torch.long).to(
#             self.device
#         )
#
#         model_kwargs = {
#             "encoder_hidden_states": visual_embeds,
#             "encoder_attention_mask": image_atts,
#         }
#
#         if use_nucleus_sampling:
#             # nucleus sampling
#             outputs = self.generate(
#                 input_ids=tokenized_prompt.input_ids,
#                 max_length=max_length,
#                 min_length=min_length,
#                 do_sample=True,
#                 top_p=top_p,
#                 num_return_sequences=1,
#                 eos_token_id=sep_token_id,
#                 pad_token_id=pad_token_id,
#                 repetition_penalty=1.1,
#                 **model_kwargs
#             )
#         else:
#             # beam search
#             outputs = self.generate(
#                 input_ids=tokenized_prompt.input_ids,
#                 max_length=max_length,
#                 min_length=min_length,
#                 num_beams=num_beams,
#                 eos_token_id=sep_token_id,
#                 pad_token_id=pad_token_id,
#                 repetition_penalty=repetition_penalty,
#                 **model_kwargs
#             )
#
#         return outputs


# class XBertEncoder(BertModel, BaseEncoder):
#     @classmethod
#     def from_config(cls, cfg, from_pretrained=False):
#
#         med_config_path = get_abs_path(cfg.get("med_config_path"))
#         med_config = BertConfig.from_json_file(med_config_path)
#         print(med_config.fusion_layer, med_config.num_hidden_layers)
#         if from_pretrained:
#             return cls.from_pretrained(
#                 "bert-base-uncased", config=med_config, add_pooling_layer=False
#             )
#         else:
#             return cls(config=med_config, add_pooling_layer=False)
#
#     def forward_automask(self, tokenized_text, visual_embeds, **kwargs):
#         image_atts = torch.ones(visual_embeds.size()[:-1], dtype=torch.long).to(
#             self.device
#         )
#
#         text = tokenized_text
#         text_output = super().forward(
#             text.input_ids,
#             attention_mask=text.attention_mask,
#             encoder_hidden_states=visual_embeds,
#             encoder_attention_mask=image_atts,
#             return_dict=True,
#         )
#
#         return text_output
#
#     def forward_text(self, tokenized_text, **kwargs):
#         text = tokenized_text
#         token_type_ids = kwargs.get("token_type_ids", None)
#
#         text_output = super().forward(
#             text.input_ids,
#             attention_mask=text.attention_mask,
#             token_type_ids=token_type_ids,
#             return_dict=True,
#             mode="text",
#         )
#
#         return text_output
