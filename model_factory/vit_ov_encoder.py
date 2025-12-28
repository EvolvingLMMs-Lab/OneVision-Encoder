from onevision_encoder import OneVisionEncoderModel, OneVisionEncoderConfig
from timm.models.registry import register_model


@register_model
def ov_encoder_base(pretrained: bool = False, **kwargs):
    config = OneVisionEncoderConfig(
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        image_size=224,
        patch_size=14,
        num_channels=3,
    )
    model = OneVisionEncoderModel(config)
    return model


@register_model
def ov_encoder_large(pretrained: bool = False, **kwargs):
    config = OneVisionEncoderConfig(
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        intermediate_size=4096,
        image_size=224,
        patch_size=14,
        num_channels=3,
    )
    model = OneVisionEncoderModel(config)
    return model


@register_model
def ov_encoder_huge(pretrained: bool = False, **kwargs):
    config = OneVisionEncoderConfig(
        hidden_size=1280,
        num_hidden_layers=32,
        num_attention_heads=20,
        intermediate_size=5120,
        image_size=224,
        patch_size=14,
        num_channels=3,
    )
    model = OneVisionEncoderModel(config)
    return model
