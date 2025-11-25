from collections import namedtuple
from typing import Optional, Union, Tuple

import torch
from torch import nn

from timm.models import VisionTransformer, create_model, checkpoint_seq 

from transformers import PretrainedConfig, PreTrainedModel

import timm
from timm.models.vision_transformer import _create_vision_transformer
from timm.models.registry import register_model

from .input_conditioner import get_default_conditioner, InputConditioner
from .vit_patch_generator import ViTPatchGenerator

@register_model
def vit_huge_patch14_224_1B(pretrained: bool = False, **kwargs) -> VisionTransformer:
    model_args = dict(patch_size=14, embed_dim=1280, depth=50, num_heads=16)
    model = _create_vision_transformer('vit_huge_patch14_224_1B', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


class UNITConfig(PretrainedConfig):
    """Pretrained Hugging Face configuration for RADIO models."""

    def __init__(
        self,
        args: Optional[dict] = None,
        version: Optional[str] = "v1",
        **kwargs,
    ):
        self.args = args
        self.version = version
        super().__init__(**kwargs)

class UNITModelEncoder(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        max_img_size: Union[int, Tuple[int, int]] = 1024,
        num_cls_tokens: int = 1,
        register_multiple: int = 0,
        pos_dropout: float = 0.1,
    ):
        super().__init__()

        self.model = model

        input_conditioner: InputConditioner = get_default_conditioner()
        self.input_conditioner = input_conditioner

        patch_size = model.patch_embed.patch_size[0]
        embed_dim = model.embed_dim
        input_dims = model.patch_embed.img_size
        normalize_patches = not isinstance(model.patch_embed.norm, nn.Identity)
        cls_token = model.cls_token is not None

        max_img_size = int(round(max_img_size / patch_size) * patch_size)

        patch_generator = ViTPatchGenerator(
            patch_size=patch_size,
            embed_dim=embed_dim,
            input_dims=input_dims,
            normalize_patches=normalize_patches,
            cls_token=cls_token,
            max_input_dims=max_img_size,
            pos_dropout=pos_dropout,
            num_cls_tokens=num_cls_tokens,
            register_multiple=register_multiple,

        )

        model.patch_generator = patch_generator
        model.patch_embed = None
        model.cls_token = None
        model.pos_embed = None
        model.pos_drop = None
        model.num_cls_tokens = num_cls_tokens
        model.num_registers = patch_generator.num_registers
        self.ln_final = torch.nn.LayerNorm(embed_dim, eps=1e-6)



    def forward(self, x: torch.Tensor):
        x = self.input_conditioner(x)
        x = self.model.patch_generator(x)

        if self.model.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.model.blocks, x)
        else:
            x = self.model.blocks(x)
        x = self.model.norm(x)
        
        cls_tokens = x[:, 0]
        spatial_tokens = x[:, self.model.patch_generator.num_skip:]

        # Apply layer normalization to the cls_tokens and spatial_tokens
        spatial_tokens = self.ln_final(spatial_tokens)
        return cls_tokens, spatial_tokens


class UNITModel(PreTrainedModel):
    """Pretrained Hugging Face model for UNIT.

    This class inherits from PreTrainedModel, which provides
    HuggingFace's functionality for loading and saving models.
    """

    config_class = UNITConfig

    def __init__(self, config: UNITConfig):
        super().__init__(config)

        UNITArgs = namedtuple("UNITArgs", config.args.keys())
        args = UNITArgs(**config.args)
        self.config = config

        weight_init = args.model_kwargs.pop("weight_init", "skip")

        model = create_model(
            args.model,
            pretrained=args.pretrained,
            in_chans=args.in_chans,
            num_classes=args.num_classes,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            drop_block_rate=args.drop_block,
            global_pool=args.gp,
            bn_momentum=args.bn_momentum,
            bn_eps=args.bn_eps,
            scriptable=args.torchscript,
            checkpoint_path=args.initial_checkpoint,
            weight_init=weight_init,
            **args.model_kwargs,
        )

        self.unit_model = UNITModelEncoder(model,
            args.cpe_max_size,
            args.num_cls_tokens,
            args.register_multiple
        )

    def forward(self, x: torch.Tensor):
        return self.unit_model.forward(x)
