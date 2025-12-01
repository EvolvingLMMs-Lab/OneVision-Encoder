import torch
import torch.nn as nn
from llava.utils import rank0_print
from transformers import CLIPImageProcessor

from model_factory.vit_preview_v0_hf import LlavaViTConfig as HEVCViTConfig
from model_factory.vit_preview_v0_hf import LlavaViTModel as HEVCViTModel


class HEVCViTVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False
        self.projector_type = getattr(args, "mm_projector_type", "patch_merger")

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, "mm_vision_select_feature", "patch")

        if not delay_load:
            rank0_print(f"Loading vision tower: {vision_tower}")
            self.load_model()
        elif getattr(args, "unfreeze_mm_vision_tower", False):
            rank0_print(f"The checkpoint seems to contain `vision_tower` weights: `unfreeze_mm_vision_tower`: True.")
            self.load_model()
        elif hasattr(args, "mm_tunable_parts") and "mm_vision_tower" in args.mm_tunable_parts:
            rank0_print(f"The checkpoint seems to contain `vision_tower` weights: `mm_tunable_parts` contains `mm_vision_tower`.")
            self.load_model()
        else:
            self.cfg_only = HEVCViTConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            rank0_print("{} is already loaded, `load_model` called again, skipping.".format(self.vision_tower_name))
            return

        # 加载 CLIPImageProcessor (我们在转换脚本里保存了这个配置)
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)

        # 加载 HEVCViTModel
        self.vision_tower = HEVCViTModel.from_pretrained(self.vision_tower_name, device_map=device_map)
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
            images: Input images
            return_spatial_dims: If True, return (features, h, w) tuple for spatial_merge projector
        """
        # Calculate spatial dimensions from input images
        if type(images) is list:
            # For list of images, use the first image to determine dimensions
            sample_image = images[0]
            # Extract height and width (works for both image and video)
            height, width = sample_image.shape[-2:]
                
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(
                    image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                    output_hidden_states=True
                )
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            # Extract height and width from batch of images
            if images.ndim == 5:  # (B, C, T, H, W) - video batch
                height, width = images.shape[-2:]
            else:  # (B, C, H, W) - image batch
                height, width = images.shape[-2:]
                
            image_forward_outs = self.vision_tower(
                images.to(device=self.device, dtype=self.dtype),
                output_hidden_states=True
            )
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        # Calculate h and w in patch coordinates
        h = height // self.config.patch_size
        w = width // self.config.patch_size
        
        if return_spatial_dims:
            return image_features, h, w
        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

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
