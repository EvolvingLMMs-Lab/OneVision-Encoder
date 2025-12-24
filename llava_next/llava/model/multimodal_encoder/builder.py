import os
from .clip_encoder import CLIPVisionTower, CLIPVisionTowerS2
from .siglip2_naflex import SigLip2NaflexVisionTower
from .onevision_encoder import OneVisionEncoderTower

def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, "mm_vision_tower", getattr(vision_tower_cfg, "vision_tower", None))
    use_s2 = getattr(vision_tower_cfg, "s2", False)

    if "siglip2" in vision_tower:
        return SigLip2NaflexVisionTower(vision_tower, vision_tower_cfg=vision_tower_cfg, **kwargs)

    elif "onevision" in vision_tower:
        return OneVisionEncoderTower(vision_tower, vision_tower_cfg=vision_tower_cfg, **kwargs)

    raise ValueError(f"Unknown vision tower: {vision_tower}")
