# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch OneVision Encoder model."""

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoImageProcessor, AutoModel
from transformers.utils import logging

from llava.utils import rank0_print

logger = logging.get_logger(__name__)


class OneVisionEncoderTower(nn.Module):
    """
    Vision Tower wrapper for LlavaViT model, compatible with LLaVA framework.
    """
    def __init__(self, vision_tower, vision_tower_cfg=None, delay_load=False):
        super().__init__()

        self.is_loaded = False
        self.vision_tower_name = vision_tower
        self.select_layer = vision_tower_cfg.mm_vision_select_layer if vision_tower_cfg is not None else None
        
        # Default config - will be updated after loading
        self.config = AutoConfig.from_pretrained(self.vision_tower_name, trust_remote_code=True)
        
        self.image_processor = AutoImageProcessor.from_pretrained(self.vision_tower_name, trust_remote_code=True)

        if not delay_load:
            rank0_print(f"Loading vision tower: {vision_tower}")
            self.load_model()
        elif getattr(vision_tower_cfg, "unfreeze_mm_vision_tower", False):
            rank0_print(f"The checkpoint seems to contain `vision_tower` weights: `unfreeze_mm_vision_tower`: True.")
            self.load_model()
        elif hasattr(vision_tower_cfg, "mm_tunable_parts") and "mm_vision_tower" in vision_tower_cfg.mm_tunable_parts:
            rank0_print(f"The checkpoint seems to contain `vision_tower` weights: `mm_tunable_parts` contains `mm_vision_tower`.")
            self.load_model()
        else:
            self.cfg_only = self.config

    def load_model(self, device_map=None):
        if self.is_loaded:
            rank0_print("{} is already loaded, `load_model` called again, skipping.".format(self.vision_tower_name))
            return

        self.vision_tower = AutoModel.from_pretrained(self.vision_tower_name, trust_remote_code=True)
        
        # Update config from loaded model
        self.config = self.vision_tower.config
        
        self.is_loaded = True

    def forward(self, images, grid_thw=None, visible_indices=None):
        """
        Forward pass for the vision tower.
        
        Args:
            images: Can be:
                - Tensor of shape (B, C, H, W) for single images
                - Tensor of shape (B, C, T, H, W) for video
                - List of tensors
            grid_thw: Optional grid info for variable resolution
            
        Returns:
            image_features: Tensor of shape (B, num_patches, hidden_size)
        """
        if isinstance(images, list):
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(
                    image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                    output_hidden_states=True,
                    visible_indices=visible_indices
                )
                image_feature = image_forward_out.hidden_states[-1].to(image.dtype)
                image_features.append(image_feature)
            image_features = torch.cat(image_features, dim=0)
        else:
            # Handle tensor input
            pixel_values = images.to(device=self.device, dtype=self.dtype)
            
            # Ensure correct shape: (B, C, H, W) or (B, C, T, H, W)
            if pixel_values.dim() == 3:
                # (C, H, W) -> (1, C, H, W)
                pixel_values = pixel_values.unsqueeze(0)
            bs = pixel_values.shape[0]
            if bs == 8: # FIXME hardcoded for 8 images input as video sample
                # (B, C, T, H, W) -> (1, C, B*T, H, W)
                pixel_values = pixel_values.unsqueeze(0).permute(0, 2, 1, 3, 4)
            image_forward_outs = self.vision_tower(
                pixel_values,
                output_hidden_states=True,
                visible_indices=visible_indices
            )
            
        # Get last hidden state (before head if exists)
        if self.select_layer is not None:
            image_features = image_forward_outs.hidden_states[self.select_layer]
        else:
            image_features = image_forward_outs.hidden_states[-2]
        if bs == 8: # FIXME hardcoded for 8 images input as video sample
            image_features = image_features.squeeze(0).reshape(8, -1, self.hidden_size)
        
        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        for p in self.vision_tower.parameters():
            return p.dtype

    @property
    def device(self):
        for p in self.vision_tower.parameters():
            return p.device

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def image_size(self):
        return self.config.image_size
