import torch
from torch import nn
from transformers import CLIPModel
from timm.models.registry import register_model

class MetaClip(nn.Module):
    def __init__(
        self,
        ckpt: str = "meta-clip/MetaCLIP-ViT-B-16",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super(MetaClip, self).__init__()
        self.device = torch.device(device)
        base_model = CLIPModel.from_pretrained(ckpt)
        self.model = base_model.vision_model.to(self.device).eval()

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        pixel_values = pixel_values.to(self.device)
        with torch.no_grad():
            outputs = self.model(pixel_values=pixel_values, output_hidden_states=True)
            last_hidden_state = outputs.last_hidden_state

        return last_hidden_state

@register_model
def metaclip_base16_fullcc(pretrained: bool = False, **kwargs):
    model = MetaClip(
        ckpt=kwargs.get("ckpt", "/video_vit/pretrain_models/metaclip-b16-fullcc2.5b/"),
        device=kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
    )
    return model

@register_model
def metaclip_large14_fullcc(pretrained: bool = False, **kwargs):
    model = MetaClip(
        ckpt=kwargs.get("ckpt", "/video_vit/pretrain_models/metaclip-l14-fullcc2.5b/"),
        device=kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
    )
    return model


@register_model
def metaclip2_large14(pretrained: bool = False, **kwargs):
    model = MetaClip(
        ckpt=kwargs.get("ckpt", "/video_vit/pretrain_models/metaclip-2-worldwide-l14"),
        device=kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
    )
    return model

if __name__ == "__main__":
    import timm

    model = timm.create_model("metaclip_base", pretrained=False)

    bs = 4
    test_input = torch.randn(bs, 3, 224, 224, device=model.device)
    last_hidden_state = model(test_input)

    print(f"Input shape: {test_input.shape}")
    print(f"Last hidden state shape: {last_hidden_state.shape}")
