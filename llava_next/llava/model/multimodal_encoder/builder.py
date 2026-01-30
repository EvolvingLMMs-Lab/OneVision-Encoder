from .siglip2_naflex import SigLip2NaflexVisionTower
from .onevision_encoder import OneVisionEncoderTower
from .qwen3_vl_encoder import Qwen3VLVisionTower


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, "mm_vision_tower", getattr(vision_tower_cfg, "vision_tower", None))

    if "siglip2" in vision_tower.lower():
        return SigLip2NaflexVisionTower(vision_tower, vision_tower_cfg=vision_tower_cfg, **kwargs)

    if "onevision" in vision_tower.lower():
        return OneVisionEncoderTower(vision_tower, vision_tower_cfg=vision_tower_cfg, **kwargs)

    if "qwen3" in vision_tower.lower():
        return Qwen3VLVisionTower(vision_tower, vision_tower_cfg=vision_tower_cfg, **kwargs)

    raise ValueError(f"Unknown vision tower: {vision_tower}")
