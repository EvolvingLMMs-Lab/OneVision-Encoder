# coding=utf-8
# Copyright 2025 Apple Inc. and The HuggingFace Inc. team.
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

import torch
from torch import nn
from transformers import AutoModel

# =============================================================================
# CUSTOM EXTENSION: AIMv2 Packing implementation for LLaVA-ViT
# =============================================================================


class AIMv2Packing(nn.Module):
    """
    AIMv2 Packing variant for efficient variable-length sequence processing using FlashAttention.
    
    This model accepts pre-patchified input in packing format:
    - hidden_states: torch.Tensor of shape [total_num_patches, patch_dim]
      where patch_dim = patch_size * patch_size * num_channels
    - grid_thw: torch.Tensor of shape [num_images, 3] containing [t, h, w] for each image
    
    This is optimized for batch processing where all images are concatenated into a single sequence.
    Uses FlashAttention for efficient processing without explicit attention masks.
    
    Note: AIMv2 uses Conv2d for patch embeddings, so this packing implementation
    reconstructs the images from patches before processing.
    """
    
    DEFAULT_PATCH_SIZE = 14  # AIMv2 typically uses 14x14 patches for large model
    
    def __init__(
        self, 
        ckpt: str = "apple/aimv2-large-patch14-224",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        revision: str = "ac764a25c832c7dc5e11871daa588e98e3cdbfb7"
    ):
        """
        Initialize the AIMv2 Packing model with FlashAttention.
        
        Args:
            ckpt (str): HuggingFace checkpoint for the pre-trained model.
            device (str): Device to map the model for inference.
            revision (str): Specific git revision to use. Default is the verified working version.
        """
        super(AIMv2Packing, self).__init__()
        self.device = torch.device(device)
        
        # Load the full model with FlashAttention enabled
        # Note: trust_remote_code is required for AIMv2
        # Using specific revision for stability and reproducibility
        self.model = AutoModel.from_pretrained(
            ckpt,
            revision=revision,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
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
        Forward pass with pre-patchified input using FlashAttention.
        
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
                # FlashAttention handles this efficiently without needing explicit masks
                pixel_values = self._reconstruct_images_from_patches(hidden_states, grid_thw)
                
                # Process through model - no attention mask needed with FlashAttention
                outputs = self.model(
                    pixel_values=pixel_values,
                    output_hidden_states=True
                )
                
                # Get the last layer's hidden state: [batch_size, seq_len, hidden_size]
                if hasattr(outputs, 'last_hidden_state'):
                    last_hidden_state = outputs.last_hidden_state
                else:
                    # If output is a tuple, first element is usually last_hidden_state
                    last_hidden_state = outputs[0]
                
                # AIMv2 typically has a CLS token at position 0
                # Extract patch tokens (excluding CLS token)
                prefix_length = 1  # CLS token
                
                # Convert back to packing format: [total_num_patches, hidden_size]
                output_list = []
                for i in range(num_images):
                    num_patches = patches_per_image[i]
                    # Extract patch tokens starting after CLS token
                    patch_tokens = last_hidden_state[i, prefix_length:prefix_length + num_patches]
                    output_list.append(patch_tokens)
                
                packed_output = torch.cat(output_list, dim=0)
            else:
                # Variable size path: process each image separately
                # FlashAttention automatically handles variable-length sequences efficiently
                output_list = []
                start_idx = 0
                
                for i in range(num_images):
                    num_patches = patches_per_image[i]
                    
                    # Extract patches for this image
                    image_hidden_states = hidden_states[start_idx:start_idx + num_patches]
                    image_grid_thw = grid_thw[i:i+1]
                    
                    # Reconstruct and process this image
                    pixel_values = self._reconstruct_images_from_patches(
                        image_hidden_states, image_grid_thw
                    )
                    
                    # Process through model - no attention mask needed
                    outputs = self.model(
                        pixel_values=pixel_values,
                        output_hidden_states=True
                    )
                    
                    # Get the last layer's hidden state
                    if hasattr(outputs, 'last_hidden_state'):
                        last_hidden_state = outputs.last_hidden_state
                    else:
                        last_hidden_state = outputs[0]
                    
                    # Extract patch tokens (excluding CLS token)
                    prefix_length = 1  # CLS token
                    
                    patch_tokens = last_hidden_state[0, prefix_length:prefix_length + num_patches]
                    output_list.append(patch_tokens)
                    
                    start_idx += num_patches
                
                packed_output = torch.cat(output_list, dim=0)
        
        return packed_output


__all__ = ["AIMv2Packing"]
