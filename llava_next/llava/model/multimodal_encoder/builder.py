import os
from .clip_encoder import CLIPVisionTower, CLIPVisionTowerS2
from .imagebind import ImageBindWrapper
from .open_clip_encoder import OpenCLIPVisionTower
from .hf_vision import HFVisionTower
from .siglip_encoder import SigLipVisionTower
from .mlcd_encoder import MLCDVisionTower, MLCDVisionTowerS2
from .internViT_300M_448px_encoder import InternViT_300M_448px_VisionTower, InternViT_300M_448px_VisionTowerS2
from .eva_8b_448px_encoder import EVA_8B_448px_VisionTower, EVA_8B_448px_VisionTowerS2
from .hevc_vit_tower import HEVCViTVisionTower
from .hevc_vit_packing_tower import HEVCViTPackingVisionTower

def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, "mm_vision_tower", getattr(vision_tower_cfg, "vision_tower", None))
    use_s2 = getattr(vision_tower_cfg, "s2", False)

    # 1. HEVC-ViT (Your New Model) - priority match
    # Check for packing mode first (more specific match)
    if "hevc_vit_packing" in vision_tower.lower() or "packing" in vision_tower.lower():
        return HEVCViTPackingVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif "hevc_vit" in vision_tower.lower():
        return HEVCViTVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    # 2. MLCD Vision Towers
    list_mlcd_vision_towers = [
        "rice-vit-huge-patch14",
        "mlcd-vit-bigG-patch14",
        "rice-vit-bigG-patch14",
        "rice-vit-large-patch14"
    ]
    for _mlcd_tower_name in list_mlcd_vision_towers:
        if _mlcd_tower_name in vision_tower:
            if use_s2:
                return MLCDVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
            else:
                return MLCDVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    # 3. Specific CLIP variants
    if "rice-vit-large-patch14-378" in vision_tower:
        if use_s2:
            return CLIPVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
        else:
            return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    # 4. EVA-8B
    elif "EVA_8B_448px" in vision_tower:
        if use_s2:
            return EVA_8B_448px_VisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
        else:
            return EVA_8B_448px_VisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    # 5. InternViT
    elif "InternViT-300M-448px" in vision_tower:
        if use_s2:
            return InternViT_300M_448px_VisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
        else:
            return InternViT_300M_448px_VisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    # 6. General CLIP / DFN / ShareGPT4V / DeepGlint / MLCD (generic match)
    # 这里把比较通用的匹配放在特定模型之后
    elif any(x in vision_tower for x in ["clip", "mlcd", "unicom", "ShareGPT4V"]) or \
         any(vision_tower.lower().startswith(x) for x in ["dfn", "openai", "laion", "deepglint"]):
        if use_s2:
            return CLIPVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
        else:
            return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    # 7. Other Specific Architectures
    elif "siglip" in vision_tower:
        return SigLipVisionTower(vision_tower, vision_tower_cfg=vision_tower_cfg, **kwargs)

    elif vision_tower.startswith("hf:"):
        return HFVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    elif vision_tower in ["imagebind_huge"]:
        return ImageBindWrapper(vision_tower, args=vision_tower_cfg, **kwargs)

    elif vision_tower.startswith("open_clip_hub"):
        return OpenCLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f"Unknown vision tower: {vision_tower}")
