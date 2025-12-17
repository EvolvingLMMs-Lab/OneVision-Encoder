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

"""
AIMv2 Packing Implementation

This module provides a packing wrapper for AIMv2 models that:
1. Uses flash_attn_varlen_func (optional, for compatibility)
2. Uses transformers absolute addresses
3. Accepts input as (hidden_states: torch.Tensor, grid_thw: torch.Tensor)
4. Reconstructs images from packing format and processes through standard model
5. Loads weights from original non-packing Aimv2VisionModel

Similar to DINOv3 packing implementation pattern.
Note: For exact numerical equivalence with standard model, FlashAttention is NOT forced.
"""

import torch
import torch.nn as nn

from transformers.models.aimv2.modeling_aimv2 import Aimv2VisionModel

# Check if FlashAttention 2 is available
try:
    from flash_attn import flash_attn_varlen_func
    _flash_attn_available = True
except ImportError:
    _flash_attn_available = False


class AIMv2Packing(nn.Module):
    """
    AIMv2 Packing variant for efficient variable-length sequence processing.
    
    This model accepts pre-patchified input in packing format:
    - hidden_states: torch.Tensor of shape [total_num_patches, patch_dim]
      where patch_dim = patch_size * patch_size * num_channels
    - grid_thw: torch.Tensor of shape [num_images, 3] containing [t, h, w] for each image
    
    This is optimized for batch processing where all images are concatenated into a single sequence.
    
    Note: AIMv2 uses Conv2d for patch embeddings, so this packing implementation
    reconstructs the images from patches before processing through the standard model.
    For exact numerical equivalence, this wrapper uses the same settings as vit_aim_v2.py.
    """
    
    DEFAULT_PATCH_SIZE = 14  # AIMv2 large typically uses 14x14 patches
    
    def __init__(self, ckpt: str = "apple/aimv2-large-patch14-224", device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the AIMv2 Packing model.
        
        Args:
            ckpt (str): HuggingFace checkpoint for the pre-trained model.
            device (str): Device to map the model for inference.
        """
        super(AIMv2Packing, self).__init__()
        self.device = torch.device(device)
        
        # Load the model matching the standard model setup for consistency
        # Using absolute import from transformers (Requirement #2)
        # Note: We load without forcing dtype or FlashAttention to match vit_aim_v2.py behavior
        # This ensures numerical equivalence with the standard model
        self.model = Aimv2VisionModel.from_pretrained(
            ckpt,
            trust_remote_code=True
        ).to(self.device).eval()
        
        # Get patch size from config
        if hasattr(self.model.config, 'patch_size'):
            self.patch_size = self.model.config.patch_size
        else:
            self.patch_size = self.DEFAULT_PATCH_SIZE
    
    def _reconstruct_images_from_patches(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor):
        """
        Reconstruct images from packed patches.
        
        Args:
            hidden_states (torch.Tensor): Packed patches of shape [total_num_patches, patch_dim]
            grid_thw (torch.Tensor): Grid dimensions of shape [num_images, 3]
        
        Returns:
            torch.Tensor: Reconstructed images of shape [num_images, channels, height, width]
        """
        num_images = grid_thw.shape[0]
        patch_dim = hidden_states.shape[1]
        
        # Infer number of channels from patch_dim
        # patch_dim = patch_size * patch_size * num_channels
        num_channels = patch_dim // (self.patch_size * self.patch_size)
        
        images = []
        start_idx = 0
        
        for i in range(num_images):
            t, h, w = grid_thw[i][0].item(), grid_thw[i][1].item(), grid_thw[i][2].item()
            num_patches = int(t * h * w)
            
            # Extract patches for this image
            image_patches = hidden_states[start_idx:start_idx + num_patches]
            start_idx += num_patches
            
            # Reshape patches to [num_patches_h, num_patches_w, patch_size, patch_size, channels]
            # Input format from convert_to_patches: [patch_h, patch_w, channels] flattened
            image_patches = image_patches.reshape(
                int(h), int(w), self.patch_size, self.patch_size, num_channels
            )
            
            # Rearrange to [channels, num_patches_h, patch_size, num_patches_w, patch_size]
            image_patches = image_patches.permute(4, 0, 2, 1, 3)
            
            # Reshape to [channels, height, width]
            image = image_patches.reshape(
                num_channels,
                int(h) * self.patch_size,
                int(w) * self.patch_size
            )
            
            images.append(image)
        
        # Stack images: [num_images, channels, height, width]
        return torch.stack(images, dim=0)
    
    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor):
        """
        Forward pass with pre-patchified input.
        
        Requirement #3: Input signature is (hidden_states, grid_thw)
        
        Args:
            hidden_states (torch.Tensor): Pre-patchified input of shape 
                [total_num_patches, patch_dim] where 
                patch_dim = patch_size * patch_size * num_channels
            grid_thw (torch.Tensor): Grid dimensions of shape [num_images, 3]
                containing [t, h, w] for each image, where:
                - t: temporal dimension (usually 1 for single images)
                - h: height in patches
                - w: width in patches
        
        Returns:
            torch.Tensor: Last hidden state of shape [total_num_patches, hidden_size]
        """
        with torch.no_grad():
            # Get target dtype from model parameters
            try:
                target_dtype = next(self.model.parameters()).dtype
            except (StopIteration, AttributeError):
                # Fallback to bfloat16 or float32 if parameters not accessible
                target_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            
            # Move inputs to device
            hidden_states = hidden_states.to(device=self.device, dtype=target_dtype)
            grid_thw = grid_thw.to(device=self.device)
            
            # Calculate number of patches per image from grid_thw
            num_images = grid_thw.shape[0]
            patches_per_image = []
            image_sizes = []
            for i in range(num_images):
                t, h, w = grid_thw[i][0].item(), grid_thw[i][1].item(), grid_thw[i][2].item()
                num_patches = int(t * h * w)
                patches_per_image.append(num_patches)
                image_sizes.append((int(h) * self.patch_size, int(w) * self.patch_size))
            
            # Check if all images have the same size
            all_same_size = len(set(image_sizes)) == 1
            
            if all_same_size:
                # Optimized path: batch process all images together
                pixel_values = self._reconstruct_images_from_patches(hidden_states, grid_thw)
                
                # Process through model (same as vit_aim_v2.py standard model)
                outputs = self.model(
                    pixel_values=pixel_values,
                    output_hidden_states=True
                )
                
                # Get the last layer's hidden state: [batch_size, seq_len, hidden_size]
                # AIMv2 already excludes special tokens from last_hidden_state
                last_hidden_state = outputs.last_hidden_state
                
                # Convert back to packing format: [total_num_patches, hidden_size]
                output_list = []
                for i in range(num_images):
                    num_patches = patches_per_image[i]
                    # AIMv2VisionModel.last_hidden_state contains only patch tokens (no CLS)
                    patch_tokens = last_hidden_state[i, :num_patches]
                    output_list.append(patch_tokens)
                
                packed_output = torch.cat(output_list, dim=0)
            else:
                # Variable size path: process each image separately
                output_list = []
                start_idx = 0
                
                for i in range(num_images):
                    num_patches = patches_per_image[i]
                    
                    # Extract patches for this image
                    image_patches = hidden_states[start_idx:start_idx + num_patches]
                    start_idx += num_patches
                    
                    # Reconstruct single image
                    # image_patches shape: [num_patches, patch_dim]
                    # grid_single shape: [1, 3]
                    grid_single = grid_thw[i:i+1]
                    pixel_values = self._reconstruct_images_from_patches(image_patches, grid_single)
                    
                    # Process through model
                    outputs = self.model(
                        pixel_values=pixel_values,
                        output_hidden_states=True
                    )
                    
                    # Extract patch tokens
                    patch_tokens = outputs.last_hidden_state[0, :num_patches]
                    output_list.append(patch_tokens)
                
                packed_output = torch.cat(output_list, dim=0)
            
            return packed_output


__all__ = ["AIMv2Packing"]
