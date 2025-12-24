import torch
from torch import nn
from transformers import Dinov2Model
from timm.models.registry import register_model


class Dinov2(nn.Module):
    """DINOv2 Vision Transformer wrapper (returns patch tokens without CLS)."""

    def __init__(
        self,
        ckpt: str = "facebook/dinov2-base",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        local_files_only: bool = False,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.model = Dinov2Model.from_pretrained(
            ckpt, local_files_only=local_files_only
        ).to(self.device).eval()

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Forward pass returning patch tokens without CLS token."""
        pixel_values = pixel_values.to(self.device)
        with torch.no_grad():
            outputs = self.model(pixel_values=pixel_values)
            patch_tokens = outputs.last_hidden_state[:, 1:, :]
        return patch_tokens


@register_model
def dinov2_base(pretrained: bool = False, **kwargs):
    """Register DINOv2 Base with timm."""
    model = Dinov2(
        ckpt=kwargs.get("ckpt", "facebook/dinov2-base"),
        device=kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
        local_files_only=kwargs.get("local_files_only", False),
    )
    return model

@register_model
def dinov2_large(pretrained: bool = False, **kwargs):
    model = Dinov2(
        ckpt=kwargs.get("ckpt", "facebook/dinov2-large"),
        device=kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
        local_files_only=kwargs.get("local_files_only", False),
    )
    return model


if __name__ == "__main__":
    import timm

    model = timm.create_model("dinov2_base", pretrained=False)
    bs = 4
    test_input = torch.randn(bs, 3, 224, 224, device=model.device)

    patch_tokens = model(test_input)

    print(f"Input shape: {test_input.shape}")
    print(f"Patch tokens shape (no CLS): {patch_tokens.shape}")
