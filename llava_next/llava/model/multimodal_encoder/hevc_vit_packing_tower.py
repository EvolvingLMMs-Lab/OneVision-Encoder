import torch
import torch.nn as nn
from llava.utils import rank0_print
from transformers import CLIPImageProcessor

from model_factory.vit_preview_v0_packing_hf import LlavaViTPackingConfig as HEVCViTPackingConfig
from model_factory.vit_preview_v0_packing_hf import LlavaViTPackingModel as HEVCViTPackingModel


class HEVCViTPackingVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False
        self.projector_type = getattr(args, "mm_projector_type", "patch_merger")

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, "mm_vision_select_feature", "patch")

        if not delay_load:
            rank0_print(f"Loading vision tower (packing mode): {vision_tower}")
            self.load_model()
        elif getattr(args, "unfreeze_mm_vision_tower", False):
            rank0_print(f"The checkpoint seems to contain `vision_tower` weights: `unfreeze_mm_vision_tower`: True.")
            self.load_model()
        elif hasattr(args, "mm_tunable_parts") and "mm_vision_tower" in args.mm_tunable_parts:
            rank0_print(f"The checkpoint seems to contain `vision_tower` weights: `mm_tunable_parts` contains `mm_vision_tower`.")
            self.load_model()
        else:
            self.cfg_only = HEVCViTPackingConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            rank0_print("{} is already loaded, `load_model` called again, skipping.".format(self.vision_tower_name))
            return

        # Load CLIPImageProcessor (saved in conversion script)
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)

        # Load HEVCViTPackingModel
        self.vision_tower = HEVCViTPackingModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        select_feature_type = self.select_feature

        if self.select_feature in ["slicefour_patch", "slicefour_cls_patch"]:
            select_every_k_layer = len(image_forward_outs.hidden_states) // 4
            image_features = torch.cat([image_forward_outs.hidden_states[i] for i in range(select_every_k_layer + self.select_layer, len(image_forward_outs.hidden_states), select_every_k_layer)], dim=-1)
            select_feature_type = select_feature_type.replace("slicefour_", "")
        elif self.select_feature in ["slice_m25811_f6_patch", "slice_m25811_f6_cls_patch"]:
            select_layers = [-2, -5, -8, -11, 6]
            image_features = torch.cat([image_forward_outs.hidden_states[i] for i in select_layers], dim=-1)
            select_feature_type = select_feature_type.replace("slice_m25811_f6_", "")
        else:
            # Use select_layer to pick a specific hidden state
            # hidden_states is a tuple where the last element is the final layer output
            # For select_layer=-1, we get last_hidden_state; for -2, we get second-to-last, etc.
            image_features = image_forward_outs.hidden_states[self.select_layer]

        # Note: HEVC ViT does not have a cls token, so we just return all patch features
        # Both "patch" and "cls_patch" return the same features for HEVC ViT
        if select_feature_type not in ["patch", "cls_patch"]:
            raise ValueError(f"Unexpected select feature: {select_feature_type}")
        return image_features

    def forward(self, images, return_spatial_dims=False):
        """
        Args:
            images: Input images in standard format [B, C, H, W] or list of images
            return_spatial_dims: If True, return (features, h, w) tuple for spatial_merge projector
        """
        # Calculate spatial dimensions from input images
        if type(images) is list:
            # For list of images, process each separately and combine
            sample_image = images[0]
            height, width = sample_image.shape[-2:]
            
            # ============================================================
            # 【INPUT CONVERSION】: Convert list of images to packing format
            # Standard format: List of [C, H, W] tensors
            # Packing format: [total_num_patches, patch_dim] where 
            #                 patch_dim = patch_size * patch_size * in_channels
            # ============================================================
            all_hidden_states = []
            all_grid_thw = []
            
            for image in images:
                # Convert single image [C, H, W] to packing format
                hidden_states, grid_thw = self._image_to_packing_input(
                    image.to(device=self.device, dtype=self.dtype)
                )
                all_hidden_states.append(hidden_states)
                all_grid_thw.append(grid_thw)
            
            # Concatenate all patches along sequence dimension
            packed_hidden_states = torch.cat(all_hidden_states, dim=0)  # [total_seq_len, patch_dim]
            packed_grid_thw = torch.cat(all_grid_thw, dim=0)  # [num_images, 3]
            # ============================================================
            # 【END INPUT CONVERSION】
            # ============================================================
            
            # Forward pass through packing model
            image_forward_outs = self.vision_tower(
                hidden_states=packed_hidden_states,
                grid_thw=packed_grid_thw,
                output_hidden_states=True
            )
            
            # ============================================================
            # 【OUTPUT CONVERSION】: Convert packing output back to feature format
            # Packing output: [total_seq_len, hidden_size] - all patches concatenated
            # Target format: List of [num_patches, hidden_size] per image
            # ============================================================
            image_features = self.feature_select(image_forward_outs)
            
            # Split the packed output back to individual images
            image_features_list = []
            start_idx = 0
            for grid in packed_grid_thw:
                t, h, w = grid[0].item(), grid[1].item(), grid[2].item()
                seq_len = t * h * w
                image_features_list.append(image_features[start_idx:start_idx + seq_len])
                start_idx += seq_len
            
            image_features = image_features_list
            # ============================================================
            # 【END OUTPUT CONVERSION】
            # ============================================================
            
            # For list processing, use the first image's dimensions for spatial dims
            # (Note: in list mode, all images should ideally have the same size)
            # height and width already set from sample_image at line 80
        else:
            # Extract height and width from batch of images
            if images.ndim == 5:  # (B, C, T, H, W) - video batch
                height, width = images.shape[-2:]
            else:  # (B, C, H, W) - image batch
                height, width = images.shape[-2:]
            
            # ============================================================
            # 【INPUT CONVERSION】: Convert batch images to packing format
            # Standard format: [B, C, H, W]
            # Packing format: [total_num_patches, patch_dim] where
            #                 total_num_patches = B * h_patches * w_patches
            #                 patch_dim = patch_size * patch_size * in_channels
            # ============================================================
            images = images.to(device=self.device, dtype=self.dtype)
            batch_size = images.shape[0]
            
            # Convert batch to packing format
            packed_hidden_states, packed_grid_thw = self._batch_images_to_packing_input(images)
            # ============================================================
            # 【END INPUT CONVERSION】
            # ============================================================
            
            # Forward pass through packing model
            image_forward_outs = self.vision_tower(
                hidden_states=packed_hidden_states,
                grid_thw=packed_grid_thw,
                output_hidden_states=True
            )
            
            # ============================================================
            # 【OUTPUT CONVERSION】: Convert packing output back to feature format
            # Packing output: [total_seq_len, hidden_size] - all patches from all images concatenated
            # Target format: [B, num_patches, hidden_size]
            # ============================================================
            raw_features = self.feature_select(image_forward_outs)
            
            # Split the packed output back to batch format
            # Calculate num_patches per image
            t, h_patches, w_patches = packed_grid_thw[0][0].item(), packed_grid_thw[0][1].item(), packed_grid_thw[0][2].item()
            num_patches_per_image = t * h_patches * w_patches
            
            # Reshape from [total_seq_len, hidden_size] to [B, num_patches, hidden_size]
            image_features = raw_features.view(batch_size, num_patches_per_image, -1)
            # ============================================================
            # 【END OUTPUT CONVERSION】
            # ============================================================

        # Calculate h and w in patch coordinates
        h = height // self.config.patch_size
        w = width // self.config.patch_size
        
        if return_spatial_dims:
            return image_features, h, w
        return image_features

    def _image_to_packing_input(self, image_tensor):
        """
        Convert a single image tensor to packing model input format.
        
        Args:
            image_tensor: [C, H, W] tensor
            
        Returns:
            hidden_states: [seq_len, patch_dim] tensor
            grid_thw: [1, 3] tensor with [t, h, w] patches
        """
        patch_size = self.config.patch_size
        channels, height, width = image_tensor.shape
        
        # Calculate patch dimensions
        h_patches = height // patch_size
        w_patches = width // patch_size
        t_frames = 1  # Images have t=1
        
        # Reshape to patches: (C, H, W) -> (h_patches, w_patches, C, patch_size, patch_size)
        patches = image_tensor.view(
            channels, h_patches, patch_size, w_patches, patch_size
        )
        patches = patches.permute(1, 3, 0, 2, 4).contiguous()  # (h, w, C, pH, pW)
        
        # Flatten to (seq_len, patch_dim)
        seq_len = t_frames * h_patches * w_patches
        patch_dim = patch_size * patch_size * channels
        hidden_states = patches.view(seq_len, patch_dim)
        
        # Create grid_thw: [t, h, w]
        grid_thw = torch.tensor(
            [[t_frames, h_patches, w_patches]], 
            dtype=torch.long, 
            device=image_tensor.device
        )
        
        return hidden_states, grid_thw

    def _batch_images_to_packing_input(self, images):
        """
        Convert a batch of images to packing model input format.
        
        Args:
            images: [B, C, H, W] tensor
            
        Returns:
            hidden_states: [total_seq_len, patch_dim] tensor
            grid_thw: [B, 3] tensor with [t, h, w] patches for each image
        """
        batch_size, channels, height, width = images.shape
        patch_size = self.config.patch_size
        
        # Calculate patch dimensions
        h_patches = height // patch_size
        w_patches = width // patch_size
        t_frames = 1  # Images have t=1
        
        # Reshape batch to patches
        # [B, C, H, W] -> [B, C, h_patches, patch_size, w_patches, patch_size]
        patches = images.view(
            batch_size, channels, h_patches, patch_size, w_patches, patch_size
        )
        # [B, C, h_patches, patch_size, w_patches, patch_size] -> [B, h_patches, w_patches, C, patch_size, patch_size]
        patches = patches.permute(0, 2, 4, 1, 3, 5).contiguous()
        
        # Flatten to (total_seq_len, patch_dim)
        seq_len_per_image = t_frames * h_patches * w_patches
        patch_dim = patch_size * patch_size * channels
        hidden_states = patches.view(batch_size * seq_len_per_image, patch_dim)
        
        # Create grid_thw for each image in batch
        grid_thw = torch.full(
            (batch_size, 3),
            0,
            dtype=torch.long,
            device=images.device
        )
        grid_thw[:, 0] = t_frames
        grid_thw[:, 1] = h_patches
        grid_thw[:, 2] = w_patches
        
        return hidden_states, grid_thw

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        if self.is_loaded:
            return self.vision_tower.dtype
        else:
            return torch.float32  # Default dtype when not loaded

    @property
    def device(self):
        if self.is_loaded:
            return self.vision_tower.device
        else:
            return torch.device("cpu")  # Default device when not loaded

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        _hidden_size = self.config.hidden_size
        if "slicefour" in self.select_feature:
            _hidden_size *= 4
        if "slice_m25811_f6" in self.select_feature:
            _hidden_size *= 5
        return _hidden_size

    @property
    def num_patches_per_side(self):
        # Base patch count per side
        base_patches_per_side = self.config.image_size // self.config.patch_size
        # If using spatial_merge projector, reduce each side by 2x (merge_size)
        if self.projector_type == "spatial_merge":
            return base_patches_per_side // 2
        return base_patches_per_side

    @property
    def num_patches(self):
        # Base total patch count
        base_patches = (self.config.image_size // self.config.patch_size) ** 2
        # If using spatial_merge projector, reduce total patches by 4x (merge_size^2 = 2^2)
        if self.projector_type == "spatial_merge":
            return base_patches // 4
        return base_patches

    @property
    def image_size(self):
        return self.config.image_size
