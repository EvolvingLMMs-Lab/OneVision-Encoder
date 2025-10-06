# # coding=utf-8
# # Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
# #
# # This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# # and OPT implementations in this library. It has been modified from its
# # original forms to accommodate minor architectural differences compared
# # to GPT-NeoX and OPT used by the Meta AI team that trained the model.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# """PyTorch Qwen2-VL model."""

# import math
# from dataclasses import dataclass
# from typing import Any, Dict, List, Optional, Tuple, Union

# import time
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.utils.checkpoint
# from torch.nn import LayerNorm

# from transformers.activations import ACT2FN
# from transformers.cache_utils import Cache, DynamicCache, SlidingWindowCache, StaticCache
# from transformers.generation import GenerationMixin
# from transformers.modeling_attn_mask_utils import AttentionMaskConverter
# from transformers.modeling_flash_attention_utils import flash_attn_supports_top_left_mask, is_flash_attn_available
# from transformers.modeling_outputs import BaseModelOutputWithPast, ModelOutput
# from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
# from transformers.modeling_utils import PreTrainedModel
# from transformers.utils import auto_docstring, can_return_tuple, is_torch_flex_attn_available, is_torchdynamo_compiling, logging
# # from .configuration_qwen2_vl import Qwen2VLConfig, Qwen2VLTextConfig, Qwen2VLVisionConfig
# from transformers.integrations import use_kernel_forward_from_hub
# from transformers.processing_utils import Unpack
# from transformers.modeling_flash_attention_utils import FlashAttentionKwargs


# # if is_flash_attn_available():
# #     from transformers.modeling_flash_attention_utils import  flash_attn_varlen_func

# # if is_torch_flex_attn_available():
# #     from torch.nn.attention.flex_attention import BlockMask

#     # from transformers.integrations.flex_attention import make_flex_block_causal_mask

# """Qwen2VL model configuration"""

# from transformers.configuration_utils import PretrainedConfig
# from transformers.modeling_rope_utils import rope_config_validation
# from transformers.utils import logging


# logger = logging.get_logger(__name__)

# # bigG
# # class Qwen2VLVisionConfig(PretrainedConfig):
# #     model_type = "qwen2_vl"
# #     base_config_key = "vision_config"

# #     def __init__(
# #         self,
# #         depth=48,
# #         embed_dim=1024,
# #         hidden_size=1664,
# #         hidden_act="gelu",
# #         intermediate_size=8192,
# #         num_heads=16,
# #         in_channels=3,
# #         patch_size=14,
# #         spatial_merge_size=2,
# #         temporal_patch_size=1,
# #         initializer_range=0.02,
# #         layer_norm_eps=1e-05,
# #         text_hidden_size=3584,
# #         **kwargs,
# #     ):
# #         super().__init__(**kwargs)

# #         self.depth = depth
# #         self.embed_dim = embed_dim
# #         self.hidden_size = hidden_size
# #         self.hidden_act = hidden_act
# #         self.intermediate_size = intermediate_size
# #         self.num_heads = num_heads
# #         self.in_channels = in_channels
# #         self.patch_size = patch_size
# #         self.spatial_merge_size = spatial_merge_size
# #         self.temporal_patch_size = temporal_patch_size
# #         self.initializer_range = initializer_range
# #         self.layer_norm_eps = layer_norm_eps
# #         self.text_hidden_size = text_hidden_size

# class Qwen2VLVisionConfig(PretrainedConfig):
#     model_type = "qwen2_vl"
#     base_config_key = "vision_config"

#     def __init__(
#         self,
#         depth=24,
#         embed_dim=1024,
#         hidden_size=1024,
#         hidden_act="gelu",
#         intermediate_size=4096,
#         num_heads=16,
#         in_channels=3,
#         patch_size=14,
#         spatial_merge_size=2,
#         temporal_patch_size=1,
#         initializer_range=0.02,
#         layer_norm_eps=1e-05,
#         text_hidden_size=4096,
#         **kwargs,
#     ):
#         super().__init__(**kwargs)

#         self.depth = depth
#         self.embed_dim = embed_dim
#         self.hidden_size = hidden_size
#         self.hidden_act = hidden_act
#         self.intermediate_size = intermediate_size
#         self.num_heads = num_heads
#         self.in_channels = in_channels
#         self.patch_size = patch_size
#         self.spatial_merge_size = spatial_merge_size
#         self.temporal_patch_size = temporal_patch_size
#         self.initializer_range = initializer_range
#         self.layer_norm_eps = layer_norm_eps
#         self.text_hidden_size = text_hidden_size

# class Qwen2VLTextConfig(PretrainedConfig):
#     r"""
#     This is the configuration class to store the configuration of a [`Qwen2VLTextModel`]. It is used to instantiate a
#     Qwen2-VL model according to the specified arguments, defining the model architecture. Instantiating a configuration
#     with the defaults will yield a similar configuration to that of
#     Qwen2-VL-7B-Instruct [Qwen/Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct).

#     Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
#     documentation from [`PretrainedConfig`] for more information.

#     Args:
#         vocab_size (`int`, *optional*, defaults to 152064):
#             Vocabulary size of the Qwen2VL model. Defines the number of different tokens that can be represented by the
#             `inputs_ids` passed when calling [`Qwen2VLModel`]
#         hidden_size (`int`, *optional*, defaults to 8192):
#             Dimension of the hidden representations.
#         intermediate_size (`int`, *optional*, defaults to 29568):
#             Dimension of the MLP representations.
#         num_hidden_layers (`int`, *optional*, defaults to 80):
#             Number of hidden layers in the Transformer encoder.
#         num_attention_heads (`int`, *optional*, defaults to 64):
#             Number of attention heads for each attention layer in the Transformer encoder.
#         num_key_value_heads (`int`, *optional*, defaults to 8):
#             This is the number of key_value heads that should be used to implement Grouped Query Attention. If
#             `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
#             `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
#             converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
#             by meanpooling all the original heads within that group. For more details checkout [this
#             paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to `32`.
#         hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
#             The non-linear activation function (function or string) in the decoder.
#         max_position_embeddings (`int`, *optional*, defaults to 32768):
#             The maximum sequence length that this model might ever be used with.
#         initializer_range (`float`, *optional*, defaults to 0.02):
#             The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
#         rms_norm_eps (`float`, *optional*, defaults to 1e-05):
#             The epsilon used by the rms normalization layers.
#         use_cache (`bool`, *optional*, defaults to `True`):
#             Whether or not the model should return the last key/values attentions (not used by all models). Only
#             relevant if `config.is_decoder=True`.
#         tie_word_embeddings (`bool`, *optional*, defaults to `False`):
#             Whether the model's input and output word embeddings should be tied.
#         rope_theta (`float`, *optional*, defaults to 1000000.0):
#             The base period of the RoPE embeddings.
#         use_sliding_window (`bool`, *optional*, defaults to `False`):
#             Whether to use sliding window attention.
#         sliding_window (`int`, *optional*, defaults to 4096):
#             Sliding window attention (SWA) window size. If not specified, will default to `4096`.
#         max_window_layers (`int`, *optional*, defaults to 80):
#             The number of layers that use SWA (Sliding Window Attention). The bottom layers use SWA while the top use full attention.
#         attention_dropout (`float`, *optional*, defaults to 0.0):
#             The dropout ratio for the attention probabilities.
#         rope_scaling (`Dict`, *optional*):
#             Dictionary containing the scaling configuration for the RoPE embeddings. NOTE: if you apply new rope type
#             and you expect the model to work on longer `max_position_embeddings`, we recommend you to update this value
#             accordingly.
#             Expected contents:
#                 `rope_type` (`str`):
#                     The sub-variant of RoPE to use. Can be one of ['default', 'linear', 'dynamic', 'yarn', 'longrope',
#                     'llama3'], with 'default' being the original RoPE implementation.
#                 `factor` (`float`, *optional*):
#                     Used with all rope types except 'default'. The scaling factor to apply to the RoPE embeddings. In
#                     most scaling types, a `factor` of x will enable the model to handle sequences of length x *
#                     original maximum pre-trained length.
#                 `original_max_position_embeddings` (`int`, *optional*):
#                     Used with 'dynamic', 'longrope' and 'llama3'. The original max position embeddings used during
#                     pretraining.
#                 `attention_factor` (`float`, *optional*):
#                     Used with 'yarn' and 'longrope'. The scaling factor to be applied on the attention
#                     computation. If unspecified, it defaults to value recommended by the implementation, using the
#                     `factor` field to infer the suggested value.
#                 `beta_fast` (`float`, *optional*):
#                     Only used with 'yarn'. Parameter to set the boundary for extrapolation (only) in the linear
#                     ramp function. If unspecified, it defaults to 32.
#                 `beta_slow` (`float`, *optional*):
#                     Only used with 'yarn'. Parameter to set the boundary for interpolation (only) in the linear
#                     ramp function. If unspecified, it defaults to 1.
#                 `short_factor` (`List[float]`, *optional*):
#                     Only used with 'longrope'. The scaling factor to be applied to short contexts (<
#                     `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
#                     size divided by the number of attention heads divided by 2
#                 `long_factor` (`List[float]`, *optional*):
#                     Only used with 'longrope'. The scaling factor to be applied to long contexts (<
#                     `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
#                     size divided by the number of attention heads divided by 2
#                 `low_freq_factor` (`float`, *optional*):
#                     Only used with 'llama3'. Scaling factor applied to low frequency components of the RoPE
#                 `high_freq_factor` (`float`, *optional*):
#                     Only used with 'llama3'. Scaling factor applied to high frequency components of the RoPE
#         image_token_id (`int`, *optional*):
#             Token index used as placeholder for image embeddings.
#         video_token_id (`int`, *optional*):
#             Token index used as placeholder for video embeddings.

#     ```python
#     >>> from transformers import Qwen2VLTextModel, Qwen2VLConfig

#     >>> # Initializing a Qwen2VL style configuration
#     >>> configuration = Qwen2VLConfig()

#     >>> # Initializing a model from the Qwen2-VL-7B style configuration
#     >>> model = Qwen2VLTextModel(configuration)

#     >>> # Accessing the model configuration
#     >>> configuration = model.config
#     ```"""

#     model_type = "qwen2_vl_text"
#     base_config_key = "text_config"
#     keys_to_ignore_at_inference = ["past_key_values"]
#     # Default tensor parallel plan for base model `Qwen2VL`
#     base_model_tp_plan = {
#         "layers.*.self_attn.q_proj": "colwise",
#         "layers.*.self_attn.k_proj": "colwise",
#         "layers.*.self_attn.v_proj": "colwise",
#         "layers.*.self_attn.o_proj": "rowwise",
#         "layers.*.mlp.gate_proj": "colwise",
#         "layers.*.mlp.up_proj": "colwise",
#         "layers.*.mlp.down_proj": "rowwise",
#     }
#     base_model_pp_plan = {
#         "embed_tokens": (["input_ids"], ["inputs_embeds"]),
#         "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
#         "norm": (["hidden_states"], ["hidden_states"]),
#     }

#     def __init__(
#         self,
#         vocab_size=151936,
#         hidden_size=4096,
#         intermediate_size=12288,
#         num_hidden_layers=36,
#         num_attention_heads=32,
#         num_key_value_heads=8,
#         head_dim=128,
#         hidden_act="silu",
#         max_position_embeddings=32768,
#         initializer_range=0.02,
#         rms_norm_eps=1e-06,
#         use_cache=True,
#         tie_word_embeddings=False,
#         rope_theta=1000000.0,
#         attention_bias=False,
#         use_sliding_window=False,
#         sliding_window=None,
#         max_window_layers=36,
#         attention_dropout=0.0,
#         rope_scaling=None,
#         layer_types=None,
#         image_token_id=None,
#         video_token_id=None,
#         **kwargs,
#     ):
#         self.vocab_size = vocab_size
#         self.max_position_embeddings = max_position_embeddings
#         self.hidden_size = hidden_size
#         self.intermediate_size = intermediate_size
#         self.num_hidden_layers = num_hidden_layers
#         self.num_attention_heads = num_attention_heads
#         self.use_sliding_window = use_sliding_window
#         self.sliding_window = sliding_window
#         self.max_window_layers = max_window_layers

#         # for backward compatibility
#         if num_key_value_heads is None:
#             num_key_value_heads = num_attention_heads

#         self.num_key_value_heads = num_key_value_heads
#         self.head_dim = head_dim
#         self.hidden_act = hidden_act
#         self.initializer_range = initializer_range
#         self.rms_norm_eps = rms_norm_eps
#         self.use_cache = use_cache
#         self.rope_theta = rope_theta
#         self.attention_dropout = attention_dropout
#         self.rope_scaling = rope_scaling
#         self.attention_bias = attention_bias
#         self.tie_word_embeddings = tie_word_embeddings

#         # Validate the correctness of rotary position embeddings parameters
#         # BC: if there is a 'type' field, move it to 'rope_type'.
#         # and change type from 'mrope' to 'default' because `mrope` does default RoPE calculations
#         # one can set it to "linear"/"dynamic" etc. to have scaled RoPE
#         # TODO: @raushan update config in the hub
#         if self.rope_scaling is not None and "type" in self.rope_scaling:
#             if self.rope_scaling["type"] == "mrope":
#                 self.rope_scaling["type"] = "default"
#             self.rope_scaling["rope_type"] = self.rope_scaling["type"]
#         rope_config_validation(self, ignore_keys={"mrope_section"})
#         self.image_token_id = image_token_id
#         self.video_token_id = video_token_id

#         self.layer_types = layer_types
#         if self.layer_types is None:
#             self.layer_types = [
#                 "sliding_attention"
#                 if self.sliding_window is not None and i >= self.max_window_layers
#                 else "full_attention"
#                 for i in range(self.num_hidden_layers)
#             ]
#         # layer_type_validation(self.layer_types)

#         super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


# class Qwen2VLConfig(PretrainedConfig):
#     r"""
#     This is the configuration class to store the configuration of a [`Qwen2VLModel`]. It is used to instantiate a
#     Qwen2-VL model according to the specified arguments, defining the model architecture. Instantiating a configuration
#     with the defaults will yield a similar configuration to that of
#     Qwen2-VL-7B-Instruct [Qwen/Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct).

#     Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
#     documentation from [`PretrainedConfig`] for more information.


#     Args:
#         text_config (`Union[PreTrainedConfig, dict]`, *optional*, defaults to `Qwen2_5_VLTextConfig`):
#             The config object or dictionary of the text backbone.
#         vision_config (`Union[PreTrainedConfig, dict]`,  *optional*, defaults to `Qwen2_5_VLVisionConfig`):
#             The config object or dictionary of the vision backbone.
#         image_token_id (`int`, *optional*, defaults to 151655):
#             The image token index to encode the image prompt.
#         video_token_id (`int`, *optional*, defaults to 151656):
#             The video token index to encode the image prompt.

#     ```python
#     >>> from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLConfig

#     >>> # Initializing a Qwen2_5_VL style configuration
#     >>> configuration = Qwen2_5_VLConfig()

#     >>> # Initializing a model from the Qwen2-VL-7B style configuration
#     >>> model = Qwen2_5_VLForConditionalGeneration(configuration)

#     >>> # Accessing the model configuration
#     >>> configuration = model.config
#     ```"""

#     model_type = "qwen2_vl"
#     sub_configs = {"vision_config": Qwen2VLVisionConfig, "text_config": Qwen2VLTextConfig}
#     keys_to_ignore_at_inference = ["past_key_values"]

#     def __init__(
#         self,
#         text_config=None,
#         vision_config=None,
#         image_token_id=151655,
#         video_token_id=151656,
#         vocab_size=151936,
#         **kwargs,
#     ):
#         if isinstance(vision_config, dict):
#             self.vision_config = self.sub_configs["vision_config"](**vision_config)
#         elif vision_config is None:
#             self.vision_config = self.sub_configs["vision_config"]()

#         if isinstance(text_config, dict):
#             self.text_config = self.sub_configs["text_config"](**text_config)
#         elif text_config is None:
#             # For BC use all kwargs to init `TextConfig`
#             self.text_config = self.sub_configs["text_config"](**kwargs)

#         self.image_token_id = image_token_id
#         self.video_token_id = video_token_id
#         self.vocab_size = vocab_size

#         super().__init__(**kwargs)


# # logger = logging.get_logger(__name__)


# @dataclass
# class Qwen2VLModelOutputWithPast(ModelOutput):
#     """
#     Base class for Llava outputs, with hidden states and attentions.

#     Args:
#         last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
#             Sequence of hidden-states at the output of the last layer of the model.
#         past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
#             Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
#             `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

#             Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
#             `past_key_values` input) to speed up sequential decoding.
#         hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
#             Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
#             one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

#             Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
#         attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
#             Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
#             sequence_length)`.

#             Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
#             heads.
#         rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
#             The rope index difference between sequence length and multimodal rope.
#     """

#     last_hidden_state: torch.FloatTensor = None
#     past_key_values: Optional[List[torch.FloatTensor]] = None
#     hidden_states: Optional[Tuple[torch.FloatTensor]] = None
#     attentions: Optional[Tuple[torch.FloatTensor]] = None
#     rope_deltas: Optional[torch.LongTensor] = None


# @dataclass
# class Qwen2VLCausalLMOutputWithPast(ModelOutput):
#     """
#     Base class for Qwen2VL causal language model (or autoregressive) outputs.

#     Args:
#         loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
#             Language modeling loss (for next-token prediction).
#         logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
#             Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
#         past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
#             Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
#             `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

#             Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
#             `past_key_values` input) to speed up sequential decoding.
#         hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
#             Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
#             one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

#             Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
#         attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
#             Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
#             sequence_length)`.

#             Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
#             heads.
#         rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
#             The rope index difference between sequence length and multimodal rope.
#     """

#     loss: Optional[torch.FloatTensor] = None
#     logits: Optional[torch.FloatTensor] = None
#     past_key_values: Optional[List[torch.FloatTensor]] = None
#     hidden_states: Optional[Tuple[torch.FloatTensor]] = None
#     attentions: Optional[Tuple[torch.FloatTensor]] = None
#     rope_deltas: Optional[torch.LongTensor] = None


# class Qwen2VLRotaryEmbedding(nn.Module):
#     def __init__(self, config: Qwen2VLTextConfig, device=None):
#         super().__init__()
#         # BC: "rope_type" was originally "type"
#         if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
#             self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
#         else:
#             self.rope_type = "default"
#         self.max_seq_len_cached = config.max_position_embeddings
#         self.original_max_seq_len = config.max_position_embeddings

#         self.config = config
#         self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

#         inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
#         self.register_buffer("inv_freq", inv_freq, persistent=False)
#         self.original_inv_freq = self.inv_freq

#     @torch.no_grad()
#     @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
#     def forward(self, x, position_ids):
#         # In contrast to other models, Qwen2_VL has different position ids for the grids
#         # So we expand the inv_freq to shape (3, ...)
#         inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
#         position_ids_expanded = position_ids[:, :, None, :].float()  # shape (3, bs, 1, positions)

#         device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
#         with torch.autocast(device_type=device_type, enabled=False):  # Force float32
#             freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
#             emb = torch.cat((freqs, freqs), dim=-1)
#             cos = emb.cos() * self.attention_scaling
#             sin = emb.sin() * self.attention_scaling

#         return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

# class Qwen2RotaryEmbedding(nn.Module):
#     def __init__(self, config: Qwen2VLTextConfig, device=None):
#         super().__init__()
#         # BC: "rope_type" was originally "type"
#         if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
#             self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
#         else:
#             self.rope_type = "default"
#         self.max_seq_len_cached = config.max_position_embeddings
#         self.original_max_seq_len = config.max_position_embeddings

#         self.config = config
#         self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

#         inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
#         self.register_buffer("inv_freq", inv_freq, persistent=False)
#         self.original_inv_freq = self.inv_freq

#     @torch.no_grad()
#     @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
#     def forward(self, x, position_ids):
#         inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
#         position_ids_expanded = position_ids[:, None, :].float()

#         device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
#         with torch.autocast(device_type=device_type, enabled=False):  # Force float32
#             freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
#             emb = torch.cat((freqs, freqs), dim=-1)
#             cos = emb.cos() * self.attention_scaling
#             sin = emb.sin() * self.attention_scaling

#         return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# # Copied from transformers.models.llama.modeling_llama.rotate_half
# def rotate_half(x):
#     """Rotates half the hidden dims of the input."""
#     x1 = x[..., : x.shape[-1] // 2]
#     x2 = x[..., x.shape[-1] // 2 :]
#     return torch.cat((-x2, x1), dim=-1)


# def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
#     """Applies Rotary Position Embedding with Multimodal Sections to the query and key tensors (https://qwenlm.github.io/blog/qwen2-vl/).

#     Explanation:
#         Multimodal 3D rotary position embedding is an extension to 1D rotary position embedding. The input embedding
#         sequence contains vision (images / videos) embedding and text embedding or just contains text embedding. For
#         vision embedding part, we apply rotary position embedding on temporal, height and width dimension separately.
#         Here we split the channel dimension to 3 chunks for the temporal, height and width rotary position embedding.
#         For text embedding part, we just apply 1D rotary position embedding. The three rotary position index (temporal,
#         height and width) of text embedding is always the same, so the text embedding rotary position embedding has no
#         difference with modern LLMs.

#     Args:
#         q (`torch.Tensor`): The query tensor.
#         k (`torch.Tensor`): The key tensor.
#         cos (`torch.Tensor`): The cosine part of the rotary embedding.
#         sin (`torch.Tensor`): The sine part of the rotary embedding.
#         position_ids (`torch.Tensor`):
#             The position indices of the tokens corresponding to the query and key tensors. For example, this can be
#             used to pass offsetted position ids when working with a KV-cache.
#         mrope_section(`List(int)`):
#             Multimodal rope section is for channel dimension of temporal, height and width in rope calculation.
#         unsqueeze_dim (`int`, *optional*, defaults to 1):
#             The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
#             sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
#             that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
#             k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
#             cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
#             the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
#     Returns:
#         `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
#     """
#     mrope_section = mrope_section * 2
#     cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
#         unsqueeze_dim
#     )
#     sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
#         unsqueeze_dim
#     )

#     q_embed = (q * cos) + (rotate_half(q) * sin)
#     k_embed = (k * cos) + (rotate_half(k) * sin)
#     return q_embed, k_embed


# def apply_rotary_pos_emb_vision(
#     q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     orig_q_dtype = q.dtype
#     orig_k_dtype = k.dtype
#     q, k = q.float(), k.float()
#     cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
#     q_embed = (q * cos) + (rotate_half(q) * sin)
#     k_embed = (k * cos) + (rotate_half(k) * sin)
#     q_embed = q_embed.to(orig_q_dtype)
#     k_embed = k_embed.to(orig_k_dtype)
#     return q_embed, k_embed

# def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
#     """Applies Rotary Position Embedding to the query and key tensors.

#     Args:
#         q (`torch.Tensor`): The query tensor.
#         k (`torch.Tensor`): The key tensor.
#         cos (`torch.Tensor`): The cosine part of the rotary embedding.
#         sin (`torch.Tensor`): The sine part of the rotary embedding.
#         position_ids (`torch.Tensor`, *optional*):
#             Deprecated and unused.
#         unsqueeze_dim (`int`, *optional*, defaults to 1):
#             The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
#             sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
#             that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
#             k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
#             cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
#             the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
#     Returns:
#         `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
#     """
#     cos = cos.unsqueeze(unsqueeze_dim)
#     sin = sin.unsqueeze(unsqueeze_dim)
#     q_embed = (q * cos) + (rotate_half(q) * sin)
#     k_embed = (k * cos) + (rotate_half(k) * sin)
#     return q_embed, k_embed


# class VisionRotaryEmbedding(nn.Module):
#     def __init__(self, dim: int, theta: float = 10000.0) -> None:
#         super().__init__()
#         inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
#         self.register_buffer("inv_freq", inv_freq, persistent=False)

#     def forward(self, seqlen: int) -> torch.Tensor:
#         seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
#         freqs = torch.outer(seq, self.inv_freq)
#         return freqs


# class PatchEmbed(nn.Module):
#     def __init__(
#         self,
#         patch_size: int = 14,
#         temporal_patch_size: int = 2,
#         in_channels: int = 3,
#         embed_dim: int = 1152,
#     ) -> None:
#         super().__init__()
#         self.patch_size = patch_size
#         self.temporal_patch_size = 1 # FIXME
#         self.in_channels = in_channels
#         self.embed_dim = embed_dim

#         kernel_size = [patch_size, patch_size]
#         self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=False)

#     def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
#         target_dtype = self.proj.weight.dtype
#         hidden_states = hidden_states.reshape(
#             -1, self.in_channels, self.patch_size, self.patch_size
#         )
#         hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
#         return hidden_states


# class PatchMerger(nn.Module):
#     def __init__(self, dim: int, context_dim: int, spatial_merge_size: int = 2, layer_norm_eps: float = 1e-05) -> None:
#         super().__init__()
#         self.hidden_size = context_dim * (spatial_merge_size**2)
#         self.ln_q = LayerNorm(context_dim, eps=layer_norm_eps)
#         self.mlp = nn.Sequential(
#             nn.Linear(self.hidden_size, self.hidden_size),
#             nn.GELU(),
#             nn.Linear(self.hidden_size, dim),
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.mlp(self.ln_q(x).view(-1, self.hidden_size))
#         return x


# class VisionMlp(nn.Module):
#     def __init__(self, dim: int, hidden_dim: int, hidden_act: str) -> None:
#         super().__init__()
#         self.fc1 = nn.Linear(dim, hidden_dim)
#         self.act = ACT2FN[hidden_act]
#         self.fc2 = nn.Linear(hidden_dim, dim)

#     def forward(self, x) -> torch.Tensor:
#         return self.fc2(self.act(self.fc1(x)))


# class VisionAttention(nn.Module):
#     def __init__(self, dim: int, num_heads: int = 16) -> None:
#         super().__init__()
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.qkv = nn.Linear(dim, dim * 3, bias=True)
#         self.proj = nn.Linear(dim, dim)

#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         cu_seqlens: torch.Tensor,
#         rotary_pos_emb: Optional[torch.Tensor] = None,
#         position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
#     ) -> torch.Tensor:
#         seq_length = hidden_states.shape[0]
#         q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
#         if position_embeddings is None:
#             logger.warning_once(
#                 "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
#                 "through `rotary_pos_emb` (2D tensor of RoPE theta values), to using externally computed "
#                 "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.54 `rotary_pos_emb` will be "
#                 "removed and `position_embeddings` will be mandatory."
#             )
#             emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
#             cos = emb.cos()
#             sin = emb.sin()
#         else:
#             cos, sin = position_embeddings
#         q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

#         attention_mask = torch.full(
#             [1, seq_length, seq_length], torch.finfo(q.dtype).min, device=q.device, dtype=q.dtype
#         )
#         for i in range(1, len(cu_seqlens)):
#             attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = 0

#         q = q.transpose(0, 1)
#         k = k.transpose(0, 1)
#         v = v.transpose(0, 1)
#         attn_weights = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.head_dim)
#         attn_weights = attn_weights + attention_mask
#         attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
#         attn_output = torch.matmul(attn_weights, v)
#         attn_output = attn_output.transpose(0, 1)
#         attn_output = attn_output.reshape(seq_length, -1)
#         attn_output = self.proj(attn_output)
#         return attn_output


# # class VisionFlashAttention2(nn.Module):
# #     def __init__(self, dim: int, num_heads: int = 16) -> None:
# #         super().__init__()
# #         self.num_heads = num_heads
# #         self.qkv = nn.Linear(dim, dim * 3, bias=True)
# #         self.proj = nn.Linear(dim, dim)

# #     def forward(
# #         self,
# #         hidden_states: torch.Tensor,
# #         cu_seqlens: torch.Tensor,
# #         rotary_pos_emb: Optional[torch.Tensor] = None,
# #         position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
# #     ) -> torch.Tensor:
# #         seq_length = hidden_states.shape[0]
# #         q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
# #         if position_embeddings is None:
# #             logger.warning_once(
# #                 "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
# #                 "through `rotary_pos_emb` (2D tensor of RoPE theta values), to using externally computed "
# #                 "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.54 `rotary_pos_emb` will be "
# #                 "removed and `position_embeddings` will be mandatory."
# #             )
# #             emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
# #             cos = emb.cos()
# #             sin = emb.sin()
# #         else:
# #             cos, sin = position_embeddings
# #         q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

# #         max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
# #         attn_output = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen).reshape(
# #             seq_length, -1
# #         )
# #         attn_output = self.proj(attn_output)
# #         return attn_output


# class VisionSdpaAttention(nn.Module):
#     def __init__(self, dim: int, num_heads: int = 16) -> None:
#         super().__init__()
#         self.num_heads = num_heads
#         self.qkv = nn.Linear(dim, dim * 3, bias=True)
#         self.proj = nn.Linear(dim, dim)

#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         cu_seqlens: torch.Tensor,
#         rotary_pos_emb: Optional[torch.Tensor] = None,
#         position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
#     ) -> torch.Tensor:
#         seq_length = hidden_states.shape[0]
#         q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
#         if position_embeddings is None:
#             logger.warning_once(
#                 "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
#                 "through `rotary_pos_emb` (2D tensor of RoPE theta values), to using externally computed "
#                 "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.54 `rotary_pos_emb` will be "
#                 "removed and `position_embeddings` will be mandatory."
#             )
#             emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
#             cos = emb.cos()
#             sin = emb.sin()
#         else:
#             cos, sin = position_embeddings
#         q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

#         attention_mask = torch.zeros([1, seq_length, seq_length], device=q.device, dtype=torch.bool)
#         for i in range(1, len(cu_seqlens)):
#             attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = True
#         q = q.transpose(0, 1)
#         k = k.transpose(0, 1)
#         v = v.transpose(0, 1)
#         attn_output = F.scaled_dot_product_attention(
#             q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), attention_mask, dropout_p=0.0
#         )
#         attn_output = attn_output.squeeze(0).transpose(0, 1)
#         attn_output = attn_output.reshape(seq_length, -1)
#         attn_output = self.proj(attn_output)
#         return attn_output


# QWEN2_VL_VISION_ATTENTION_CLASSES = {
#     "eager": VisionAttention,
#     # "flash_attention_2": VisionFlashAttention2,
#     "sdpa": VisionSdpaAttention,
# }


# class Qwen2VLVisionBlock(nn.Module):
#     def __init__(self, config, attn_implementation: str = "sdpa") -> None:
#         super().__init__()
#         self.norm1 = LayerNorm(config.hidden_size, eps=1e-5)
#         self.norm2 = LayerNorm(config.hidden_size, eps=1e-5)
#         mlp_hidden_dim = int(config.intermediate_size)
        
#         self.attn = QWEN2_VL_VISION_ATTENTION_CLASSES[attn_implementation](
#             config.hidden_size, num_heads=config.num_heads
#         )        
#         # self.attn = VisionSdpaAttention(
#         #     config.hidden_size, num_heads=config.num_heads
#         # )
#         self.mlp = VisionMlp(dim=config.hidden_size, hidden_dim=mlp_hidden_dim, hidden_act=config.hidden_act)

#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         cu_seqlens: torch.Tensor,
#         rotary_pos_emb: Optional[torch.Tensor] = None,
#         position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
#     ) -> torch.Tensor:
#         hidden_states = hidden_states + self.attn(
#             self.norm1(hidden_states),
#             cu_seqlens=cu_seqlens,
#             rotary_pos_emb=rotary_pos_emb,
#             position_embeddings=position_embeddings,
#         )
#         hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
#         return hidden_states

# @auto_docstring
# class Qwen2VLPreTrainedModel(PreTrainedModel):
#     config_class = Qwen2VLConfig
#     base_model_prefix = "model"
#     supports_gradient_checkpointing = True
#     _no_split_modules = ["Qwen2VLDecoderLayer", "Qwen2VLVisionBlock"]
#     _skip_keys_device_placement = "past_key_values"
#     _supports_flash_attn_2 = True
#     _supports_sdpa = True
#     _supports_cache_class = True
#     _supports_static_cache = True

#     def _init_weights(self, module):
#         std = self.config.get_text_config().initializer_range
#         if isinstance(module, (nn.Linear, nn.Conv3d)):
#             module.weight.data.normal_(mean=0.0, std=std)
#             if module.bias is not None:
#                 module.bias.data.zero_()
#         elif isinstance(module, nn.Embedding):
#             module.weight.data.normal_(mean=0.0, std=std)
#             if module.padding_idx is not None:
#                 module.weight.data[module.padding_idx].zero_()
#         elif isinstance(module, nn.LayerNorm):
#             module.weight.data.fill_(1.0)
#             module.bias.data.zero_()
#         elif isinstance(module, Qwen2RMSNorm):
#             module.weight.data.fill_(1.0)


# @auto_docstring
# class Qwen2VisionTransformerPretrainedModel(Qwen2VLPreTrainedModel):
#     config_class = Qwen2VLVisionConfig
#     _no_split_modules = ["Qwen2VLVisionBlock"]

#     def __init__(self, config) -> None:
#         super().__init__(config)
#         self.spatial_merge_size = config.spatial_merge_size
#         self.patch_size = config.patch_size
#         self.patch_embed = PatchEmbed(
#             patch_size=config.patch_size,
#             temporal_patch_size=config.temporal_patch_size,
#             in_channels=config.in_channels,
#             embed_dim=config.hidden_size,
#         )

#         head_dim = config.hidden_size // config.num_heads
#         self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)

#         scale = config.hidden_size ** -0.5
#         self.class_embedding = nn.Parameter(scale * torch.randn(config.hidden_size))
#         self.class_pos_emb = nn.Parameter(torch.randn(1, head_dim // 2))
#         # self.window_size = config.window_size
#         self.window_size = None

#         self.pre_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
#         self.blocks = nn.ModuleList(
#             [Qwen2VLVisionBlock(config, config._attn_implementation) for _ in range(config.depth)]
#         )
#         self.merger = PatchMerger(
#             dim=config.text_hidden_size, context_dim=config.hidden_size, spatial_merge_size=config.spatial_merge_size, layer_norm_eps = config.layer_norm_eps
#         )
#         self.gradient_checkpointing = False

#     def get_dtype(self) -> torch.dtype:
#         return self.blocks[0].mlp.fc2.weight.dtype

#     def get_device(self) -> torch.device:
#         return self.blocks[0].mlp.fc2.weight.device

#     def rot_pos_emb(self, grid_thw):
#         pos_ids = []
#         for t, h, w in grid_thw:
#             hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
#             hpos_ids = hpos_ids.reshape(
#                 h // self.spatial_merge_size,
#                 self.spatial_merge_size,
#                 w // self.spatial_merge_size,
#                 self.spatial_merge_size,
#             )
#             hpos_ids = hpos_ids.permute(0, 2, 1, 3)
#             hpos_ids = hpos_ids.flatten()

#             wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
#             wpos_ids = wpos_ids.reshape(
#                 h // self.spatial_merge_size,
#                 self.spatial_merge_size,
#                 w // self.spatial_merge_size,
#                 self.spatial_merge_size,
#             )
#             wpos_ids = wpos_ids.permute(0, 2, 1, 3)
#             wpos_ids = wpos_ids.flatten()
#             pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
#         pos_ids = torch.cat(pos_ids, dim=0)
#         max_grid_size = grid_thw[:, 1:].max()
#         rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
#         rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
#         return rotary_pos_emb
    
#     def get_window_index(self, grid_thw):
#         window_index: list = []
#         cu_window_seqlens: list = [0]
#         window_index_id = 0
#         vit_window_size = self.window_size // self.patch_size

#         for grid_t, grid_h, grid_w in grid_thw:
#             llm_grid_h, llm_grid_w = (
#                 grid_h,
#                 grid_w,
#             )
#             index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(grid_t, llm_grid_h, llm_grid_w)
#             pad_h = vit_window_size - llm_grid_h % vit_window_size
#             pad_w = vit_window_size - llm_grid_w % vit_window_size
#             num_windows_h = (llm_grid_h + pad_h) // vit_window_size
#             num_windows_w = (llm_grid_w + pad_w) // vit_window_size
#             index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
#             index_padded = index_padded.reshape(
#                 grid_t,
#                 num_windows_h,
#                 vit_window_size,
#                 num_windows_w,
#                 vit_window_size,
#             )
#             index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
#                 grid_t,
#                 num_windows_h * num_windows_w,
#                 vit_window_size,
#                 vit_window_size,
#             )
#             seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
#             index_padded = index_padded.reshape(-1)
#             index_new = index_padded[index_padded != -100]
#             window_index.append(index_new + window_index_id)
#             cu_seqlens_tmp = seqlens.cumsum(0) + cu_window_seqlens[-1]
#             cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
#             window_index_id += (grid_t * llm_grid_h * llm_grid_w).item()
#         window_index = torch.cat(window_index, dim=0)

#         return window_index, cu_window_seqlens

#     @auto_docstring
#     def forward(self, hidden_states: torch.Tensor, list_I = None, list_P = None) -> torch.Tensor:
#         """
#         grid_thw (`torch.LongTensor` of shape `(num_images, 3)`):
#             The temporal, height and width dimensions of feature shape for each image. Each row contains [t, h, w] values.
#         """
#         # import pdb; pdb.set_trace()
#         B = hidden_states.size(0)
#         if len(hidden_states.shape) == 4:  # Image Input, (B, C, H, W)
#             t = 1
#             h = hidden_states.size(2) // self.patch_size
#             w = hidden_states.size(3) // self.patch_size
#         else:  # Video Input, (B, C, T, H, W)
#             t = hidden_states.size(2)
#             h = hidden_states.size(3) // self.patch_size
#             w = hidden_states.size(4) // self.patch_size

#         # 构造 (B, 3) 的 grid_thw
#         grid_thw = hidden_states.new_tensor([t, h, w], dtype=torch.long).unsqueeze(0).repeat(B, 1)
#         # print(grid_thw.shape)
#         # print(grid_thw)
#         patches_per_frame = h * w // (self.spatial_merge_size**2)
#         hidden_states = self.patch_embed(hidden_states)
#         rotary_pos_emb = self.rot_pos_emb(grid_thw)
#         img_feats = hidden_states.shape[0]
#         # import pdb; pdb.set_trace()
#         cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
#             dim=0,
#             # Select dtype based on the following factors:
#             #  - FA2 requires that cu_seqlens_q must have dtype int32
#             #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
#             # See https://github.com/huggingface/transformers/pull/34852 for more information
#             dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
#         )
        
#         cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
#         cu = cu_seqlens.to(torch.long)
#         num_segments = cu.numel() - 1
#         cls_token = self.class_embedding.to(hidden_states.dtype).unsqueeze(0)

#         # 计算新总长度
#         total_patches = cu[-1].item()
#         new_total = total_patches + num_segments
#         D = hidden_states.size(-1)
#         new_hidden = hidden_states.new_empty((new_total, D))
#         new_rotary_pos_emb = rotary_pos_emb.new_empty((new_total, rotary_pos_emb.shape[-1]))

#         write_ptr = 0
#         new_cu = [0]
#         for i in range(1, num_segments + 1):
#             seg_start = cu[i-1].item()
#             seg_end = cu[i].item()
#             seg_len = seg_end - seg_start
#             # 写入 class
#             new_hidden[write_ptr] = cls_token
#             new_rotary_pos_emb[write_ptr] = self.class_pos_emb
#             # 写入该段原 patch
#             new_hidden[write_ptr + 1: write_ptr + 1 + seg_len] = hidden_states[seg_start:seg_end]
#             new_rotary_pos_emb[write_ptr + 1: write_ptr + 1 + seg_len] = rotary_pos_emb[seg_start:seg_end]
#             write_ptr += 1 + seg_len
#             new_cu.append(write_ptr)

#         # import pdb; pdb.set_trace()

#         hidden_states = new_hidden
#         cu_seqlens = torch.tensor(new_cu, device=hidden_states.device, dtype=torch.int32) 
#         rotary_pos_emb = new_rotary_pos_emb

#         # hidden_states = hidden_states.reshape(hidden_states.shape[0], -1, hidden_states.shape[-1])
#         # hidden_states = torch.cat(
#         #     [self.class_embedding.to(hidden_states.dtype).unsqueeze(0),
#         #      hidden_states], dim=0
#         # )
#         hidden_states = self.pre_layernorm(hidden_states)

#         emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
#         position_embeddings = (emb.cos(), emb.sin())

#         # import pdb; pdb.set_trace()
#         for blk in self.blocks:
#             if self.gradient_checkpointing and self.training:
#                 hidden_states = self._gradient_checkpointing_func(
#                     blk.__call__, hidden_states, cu_seqlens, None, position_embeddings
#                 )
#             else:
#                 hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens, position_embeddings=position_embeddings)
        
#         new_hidden = hidden_states.new_empty((img_feats, D))

#         for i in range(1, num_segments + 1):
#             seg_start = cu[i-1].item()
#             seg_end = cu[i].item()
#             new_hidden[seg_start:seg_end] = hidden_states[seg_start+1:seg_end+1]
#         hidden_states = new_hidden

#         hidden_states = self.merger(hidden_states)
#         hidden_states = hidden_states.reshape(B, -1, D*(self.spatial_merge_size**2))
#         # print("hidden_states.shape", hidden_states.shape)
#         # return hidden_states
#         # import pdb; pdb.set_trace()
#         # start_time = time.time()
#         if t != 1:
#             all_I, all_P = [], []

#             for b in range(B):
#                 I_patch_idx = []
#                 for i in list_I[b].tolist():   # 当前 batch 的 I 帧索引
#                     start = i * patches_per_frame
#                     end = (i + 1) * patches_per_frame
#                     I_patch_idx.extend(range(start, end))

#                 P_patch_idx = []
#                 for p in list_P[b].tolist():   # 当前 batch 的 P 帧索引
#                     start = p * patches_per_frame
#                     end = (p + 1) * patches_per_frame
#                     P_patch_idx.extend(range(start, end))

#                 I_patch_idx = torch.tensor(I_patch_idx, dtype=torch.long, device=hidden_states.device)
#                 P_patch_idx = torch.tensor(P_patch_idx, dtype=torch.long, device=hidden_states.device)

#                 all_I.append(hidden_states[b, I_patch_idx, :])  # (num_I_patches, D)
#                 all_P.append(hidden_states[b, P_patch_idx, :])  # (num_P_patches, D)

#             # 按 batch 打包回来
#             unmask_hidden_states = torch.stack(all_I, dim=0)  # (B, num_I_patches, D)
#             mask_hidden_states   = torch.stack(all_P, dim=0)  # (B, num_P_patches, D)
#             # end_time = time.time()
#             # print("end_time", end_time-start_time)
#             return {
#                 "unmasked_embeddings": unmask_hidden_states,
#                 "masked_embeddings": mask_hidden_states,
#             }
#         # return self.merger(hidden_states)
#         return hidden_states



# # import os
# # import json
# # from safetensors.torch import load_file

# # # model_path = "/video_vit/huggingface/pretrian_LLM/Qwen2.5-VL-3B-Instruct"

# # def RiceEncoder():
# #     # 1. 初始化
# #     model_dir = "/video_vit/huggingface/pretrian_LLM/LLaVA-OneVision-1.5-8B-Instruct"
# #     config = Qwen2VLVisionConfig.from_pretrained(model_dir)
# #     # config = Qwen2VLVisionConfig.from_pretrained(model_path)
# #     print(config)
# #     # print("这里")
# #     encoder = Qwen2VisionTransformerPretrainedModel._from_config(config)

    
# #     # 1. 读取 index.json
# #     index_file = os.path.join(model_dir, "model.safetensors.index.json")
# #     with open(index_file, "r") as f:
# #         index = json.load(f)

# #     # 2. 找到所有分片文件名
# #     weight_map = index["weight_map"]  # dict: {param_name: filename}
# #     shards = sorted(set(weight_map.values()))  # 去重+排序
# #     print("找到的分片文件：", shards)

# #     # 3. 依次加载分片
# #     state_dict = {}
# #     for shard in shards:
# #         shard_path = os.path.join(model_dir, shard)
# #         state_dict.update(load_file(shard_path))

# #     # 3. 只保留 vision 模块的参数
# #     vision_state_dict = {k.replace("visual.", ""): v for k, v in state_dict.items() if k.startswith("visual.")}

# #     # print("示例原始 key:", list(state_dict.keys())[-20:])
# #     # print("示例 vision key:", list(vision_state_dict.keys())[:20])
# #     # print("模型参数名:", list(encoder.state_dict().keys())[:20])

# #     # 4. 加载权重
# #     encoder.load_state_dict(vision_state_dict, strict=True)

# #     return encoder

# # model = RiceEncoder()


# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# import torchvision.transforms as transforms
# import torch.nn.functional as F



# def compute_patch_similarity_heatmap(patch_features, H, W, target_patch_coord):
#     """
#     计算指定patch与其他所有patch的余弦相似性，并生成热力图
    
#     Args:
#         patch_features: patch特征张量, shape (1, num_patches, feature_dim)
#         H: patch网格高度
#         W: patch网格宽度  
#         target_patch_coord: 目标patch坐标 (h_idx, w_idx)
    
#     Returns:
#         heatmap: 相似性热力图, shape (H, W)
#     """
#     # print(patch_features.shape[1])
#     assert patch_features.shape[1] == H * W, f"特征数量{H*W}与网格大小{H}x{W}不匹配"
    
#     # 提取目标patch的特征
#     target_idx = target_patch_coord[0] * W + target_patch_coord[1]
#     target_feature = patch_features[0, target_idx]  # shape (feature_dim,)
    
#     # 计算余弦相似性
#     similarities = F.cosine_similarity(
#         target_feature.unsqueeze(0),  # shape (1, feature_dim)
#         patch_features[0],            # shape (num_patches, feature_dim)
#         dim=1
#     )
    
#     # 重塑为2D热力图
#     heatmap = similarities.reshape(H, W).cpu().numpy()
#     return heatmap

# def plot_similarity_heatmap(heatmap, target_patch_coord, save_path=None):
#     """
#     绘制相似性热力图，并在目标patch位置显示红点
    
#     Args:
#         heatmap: 相似性热力图, shape (H, W)
#         target_patch_coord: 目标patch坐标 (h_idx, w_idx)
#         original_img_size: 原始图像尺寸 (可选，用于调整显示比例)
#         patch_size: 每个patch的像素大小
#     """
#     H, W = heatmap.shape
    
#     fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    
#     # 显示热力图
#     im = ax.imshow(heatmap, cmap='viridis', aspect='equal')
    
#     # 在目标patch位置添加红点
#     target_h, target_w = target_patch_coord
#     ax.plot(target_w, target_h, 'ro', markersize=10, markeredgecolor='white', markeredgewidth=2)
    
#     # 添加颜色条
#     plt.colorbar(im, ax=ax, label='Cosine Similarity')
    
#     # 设置坐标轴标签
#     ax.set_xlabel('Width (patch index)')
#     ax.set_ylabel('Height (patch index)')
#     ax.set_title(f'Cosine Similarity to Patch at ({target_h}, {target_w})')
    
#     # 设置网格线
#     ax.set_xticks(np.arange(-0.5, W, 1), minor=True)
#     ax.set_yticks(np.arange(-0.5, H, 1), minor=True)
#     ax.grid(which="minor", color="white", linestyle='-', linewidth=0.5)
#     ax.tick_params(which="minor", size=0)
    
#     # 设置主刻度
#     ax.set_xticks(np.arange(0, W, max(1, W//10)))
#     ax.set_yticks(np.arange(0, H, max(1, H//10)))
    
#     plt.tight_layout()

#     if save_path is not None:
#         fig.savefig(save_path, dpi=300, bbox_inches="tight")
#         print(f"Heatmap saved to {save_path}")

#     plt.show()
    
#     return fig, ax


















# import os
# import json
# from safetensors.torch import load_file

# # model_path = "/video_vit/huggingface/pretrian_LLM/Qwen2.5-VL-3B-Instruct"

# def RiceEncoder():
#     # 1. 初始化
#     model_dir = "/video_vit/huggingface/pretrian_LLM/LLaVA-OneVision-1.5-8B-Instruct"
#     config = Qwen2VLVisionConfig.from_pretrained(model_dir)
#     # config = Qwen2VLVisionConfig.from_pretrained(model_path)
#     # print(config)
#     # print("这里")
#     encoder = Qwen2VisionTransformerPretrainedModel._from_config(config)
#     # model.forward(model_input)

    
#     # 1. 读取 index.json
#     index_file = os.path.join(model_dir, "model.safetensors.index.json")
#     with open(index_file, "r") as f:
#         index = json.load(f)

#     # 2. 找到所有分片文件名
#     weight_map = index["weight_map"]  # dict: {param_name: filename}
#     shards = sorted(set(weight_map.values()))  # 去重+排序
#     print("找到的分片文件：", shards)

#     # 3. 依次加载分片
#     state_dict = {}
#     for shard in shards:
#         shard_path = os.path.join(model_dir, shard)
#         state_dict.update(load_file(shard_path))

#     # 3. 只保留 vision 模块的参数
#     vision_state_dict = {k.replace("visual.", ""): v for k, v in state_dict.items() if k.startswith("visual.")}

#     # print("示例原始 key:", list(state_dict.keys())[-20:])
#     # print("示例 vision key:", list(vision_state_dict.keys())[:20])
#     # print("模型参数名:", list(encoder.state_dict().keys())[:20])

#     # 4. 加载权重
#     encoder.load_state_dict(vision_state_dict, strict=True)

#     return encoder

# model = RiceEncoder()




# from transformers import Qwen2_5_VLModel, AutoProcessor

# model_path = "/video_vit/huggingface/pretrian_LLM/Qwen2-VL-7B-Instruct"

# # model = Qwen2_5_VLModel.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
# processor = AutoProcessor.from_pretrained(model_path)

# from qwen_vl_utils import process_vision_info
# url = "/video_vit/utils_haolin/deduplication/cifar10_images/dog.jpg"
# prompt = "1"
# messages = [
#     {
#         "role": "user",
#         "content": [
#             {
#                 "type": "images",
#                 "image": url,
#             },
#             {
#                 "type": "text",
#                 "text": prompt,
#             }
#         ]
#     }
# ]
# text = processor.apply_chat_template(
#         messages, tokenize=False, add_generation_prompt=True
#     )
# image_inputs, video_inputs, _ = process_vision_info([messages], return_video_kwargs=True)

# inputs = processor(
#     text=[text],
#     images=image_inputs,
#     videos=video_inputs,
#     padding=True,
#     return_tensors="pt",
# )

# inputs = inputs.to(torch.device(model.device))


# # # print(inputs)
# # # with torch.no_grad():
# model_input = inputs['pixel_values'].reshape(1, 3, 86*14, 114*14)
# model_input = model_input.repeat(2, 1, 1, 1)
# # print("到这里了")
# # model_input = torch.randn(2, 3, 16, 448, 448)
# model.cuda()
# model_input = model_input.to(model.device)
# # import pdb; pdb.set_trace()
# list_I = torch.tensor([[0]])
# list_P = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]])
# # for i in range(10):
# #     start_time = time.time()
# with torch.no_grad():
#     # with torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda"):
#     img_feature = model.forward(model_input, list_I, list_P)
#     # end_time = time.time()
#     # print("time : ", end_time-start_time)

# print(img_feature.shape)

# # qwen_feature = img_feature.reshape(1, -1, 1024)

# qwen_feature1 = img_feature[:1, :, :]
# qwen_feature2 = img_feature[1:, :, :]
# # # # img_size=224

# with torch.inference_mode():
#     # patch_size = 14
#     # H = W = img_size // patch_size
#     # print(f"Patch grid: H={H}, W={W}")
#     # 你选中的目标patch的坐标，注意要在[0,H-1]和[0,W-1]范围内。
#     target_patch_coord = (20, 20)
#     # print(outputs.shape)
#     # heatmap_qwen = compute_patch_similarity_heatmap(qwen_feature, 86, 114, target_patch_coord)

#     # plot_similarity_heatmap(heatmap_qwen, target_patch_coord, "/video_vit/llava-onevision/Llava-vit-v0/model_factory/rice_vit55_without_merger.png")
#     heatmap_qwen1 = compute_patch_similarity_heatmap(qwen_feature1, 43, 57, target_patch_coord)

#     plot_similarity_heatmap(heatmap_qwen1, target_patch_coord, "/video_vit/llava-onevision/Llava-vit-v0/model_factory/rice_vit1.png")
    
#     heatmap_qwen2 = compute_patch_similarity_heatmap(qwen_feature2, 43, 57, target_patch_coord)
#     plot_similarity_heatmap(heatmap_qwen2, target_patch_coord, "/video_vit/llava-onevision/Llava-vit-v0/model_factory/rice_vit2.png")