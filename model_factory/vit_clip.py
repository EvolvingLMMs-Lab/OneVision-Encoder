import torch
from torch import nn
from transformers import CLIPModel
from timm.models.registry import register_model


class CLIP(nn.Module):
    def __init__(
        self,
        ckpt: str = "openai/clip-vit-base-patch16",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize the CLIP vision encoder to retrieve hidden states.

        Args:
            ckpt (str): HuggingFace checkpoint for the pre-trained CLIP model.
                        e.g. "openai/clip-vit-base-patch16"
            device (str): Device to map the model for inference.
        """
        super(CLIP, self).__init__()
        self.device = torch.device(device)
        # Import CLIPModel directly from transformers, then get vision_model
        base_model = CLIPModel.from_pretrained(ckpt)
        self.model = base_model.vision_model.to(self.device).eval()

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to get the last hidden state.

        Args:
            pixel_values (torch.Tensor): Input tensor of shape [bs, 3, h, w]

        Returns:
            torch.Tensor: Last hidden state of shape [bs, seq_len, hidden_size]
        """
        # pixel_values: [bs, 3, h, w]
        pixel_values = pixel_values.to(self.device)
        with torch.no_grad():
            outputs = self.model(pixel_values=pixel_values, output_hidden_states=True)
            # Last layer's hidden state: [bs, seq_len, hidden_size]
            last_hidden_state = outputs.last_hidden_state

        return last_hidden_state


@register_model
def clip_vit_base_patch16(pretrained: bool = False, **kwargs):
    """
    Register the CLIP Base Vision Transformer (ViT-B/16, 224x224) model for timm.

    Args:
        pretrained (bool): If True, load pretrained weights (from the HuggingFace ckpt path).
                           The pretrained flag here is only for interface compatibility; weight loading is done in CLIP.
        **kwargs: Additional arguments passed to CLIP.

    Returns:
        CLIP: An instance of CLIP.
    """
    model = CLIP(
        # To use a local checkpoint, set to local path; otherwise pass the default/custom HF path
        ckpt=kwargs.get("ckpt", "openai/clip-vit-base-patch16"),
        device=kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
    )
    return model


@register_model
def clip_vit_large_patch14(pretrained: bool = False, **kwargs):
    """
    Register the CLIP Base Vision Transformer (ViT-B/16, 224x224) model for timm.

    Args:
        pretrained (bool): If True, load pretrained weights (from the HuggingFace ckpt path).
                           The pretrained flag here is only for interface compatibility; weight loading is done in CLIP.
        **kwargs: Additional arguments passed to CLIP.

    Returns:
        CLIP: An instance of CLIP.
    """
    model = CLIP(
        # To use a local checkpoint, set to local path; otherwise pass the default/custom HF path
        ckpt=kwargs.get("ckpt", "openai/clip-vit-large-patch14"),
        device=kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
    )
    return model

if __name__ == "__main__":
    import timm

    # Create model using timm (name matches the registered function name)
    model = timm.create_model("clip_base", pretrained=False)

    # Test input: [bs, 3, 224, 224]
    bs = 4
    # Align with model device to avoid .cuda() error in CPU-only environment
    test_input = torch.randn(bs, 3, 224, 224, device=model.device)

    # Get the last hidden state
    last_hidden_state = model(test_input)

    # Print shapes
    print(f"Input shape: {test_input.shape}")
    print(f"Last hidden state shape: {last_hidden_state.shape}")
    # Expected: [4, seq_len, hidden_size]
    # For CLIP ViT-B/16, 224x224 it is typically [4, 197, 768]
