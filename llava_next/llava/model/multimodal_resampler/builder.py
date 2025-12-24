import torch
import warnings

from .masked_drop import MaskedDrop
from .spatial_pool import SpatialPool
from .perceiver import PerceiverResampler
try:
    from .qformer import Qformer
except ImportError as e:
    warnings.warn(f"Failed to import Qformer: {e}. Qformer resampler will not be available.")
    Qformer = None


class IdentityMap(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_resampler_type": None}


def build_vision_resampler(model_args, delay_load=False, **kwargs):
    resampler_type = getattr(model_args, "mm_resampler_type", None)
    if resampler_type == "masked_drop":
        return MaskedDrop(model_args)
    elif resampler_type == "spatial_pool":
        return SpatialPool(model_args, **kwargs)
    elif resampler_type == "perceiver":
        return PerceiverResampler(model_args, **kwargs)
    elif resampler_type == "qformer":
        if Qformer is None:
            raise ImportError("Qformer is not available due to import error. Please check the warning message.")
        return Qformer(model_args, **kwargs)
    elif resampler_type is None:
        return IdentityMap()

    raise ValueError(f"Unknown resampler type: {resampler_type}")
