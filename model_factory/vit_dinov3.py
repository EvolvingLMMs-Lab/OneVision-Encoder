import torch
from torch import nn
from transformers import AutoModel
from timm.models.registry import register_model


class Dinov3(nn.Module):
    def __init__(self, ckpt: str = "facebook/dinov3-base", device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the DINOv3 model to retrieve hidden states.

        Args:
            ckpt (str): HuggingFace checkpoint for the pre-trained model.
            device (str): Device to map the model for inference.
        """
        super(Dinov3, self).__init__()
        self.device = torch.device(device)
        # Load the model
        self.model = AutoModel.from_pretrained(ckpt).to(self.device).eval()

    def forward(self, pixel_values):
        """
        Forward pass to get the last hidden state.

        Args:
            pixel_values (torch.Tensor): Input tensor of shape [bs, 3, h, w]

        Returns:
            torch.Tensor: Last hidden state of shape [bs, seq_len, hidden_size]
        """
        # pixel_values: [bs, 3, h, w]
        with torch.no_grad():
            outputs = self.model(pixel_values=pixel_values, output_hidden_states=True)
            last_hidden_state = outputs.last_hidden_state
        return last_hidden_state


@register_model
def dinov3_base(pretrained=False, **kwargs):
    """
    Register the DINOv3 model for timm.

    Args:
        pretrained (bool): If True, load pretrained weights (default: False).
        **kwargs: Additional arguments passed to Dinov3.

    Returns:
        Dinov3: An instance of Dinov3.
    """
    ckpt = kwargs.get("ckpt")
    if ckpt is None:
        raise ValueError("DINOv3 requires a checkpoint path via ckpt=... argument")
    model = Dinov3(
        ckpt=ckpt,
        device=kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
    )
    return model


@register_model
def dinov3_large(pretrained=False, **kwargs):
    ckpt = kwargs.get("ckpt")
    if ckpt is None:
        raise ValueError("DINOv3 requires a checkpoint path via ckpt=... argument")
    model = Dinov3(
        ckpt=ckpt,
        device=kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
    )
    return model


@register_model
def dinov3_giant(pretrained=False, **kwargs):
    """
    Register the DINOv3 Giant model for timm.

    Args:
        pretrained (bool): If True, load pretrained weights (default: False).
        **kwargs: Additional arguments passed to Dinov3.

    Returns:
        Dinov3: An instance of Dinov3 with giant variant.
    """
    ckpt = kwargs.get("ckpt")
    if ckpt is None:
        raise ValueError("DINOv3 requires a checkpoint path via ckpt=... argument")
    model = Dinov3(
        ckpt=ckpt,
        device=kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
    )
    return model


if __name__ == "__main__":
    import timm

    model = timm.create_model("dinov3_base", pretrained=False)

    bs = 4
    test_input = torch.randn(bs, 3, 224, 224).cuda()
    last_hidden_state = model(test_input)

    print(f"Input shape: {test_input.shape}")
    print(f"Last hidden state shape: {last_hidden_state.shape}")

    print("\nTesting all variants:")
    for variant in ["dinov3_base", "dinov3_large", "dinov3_giant"]:
        model = timm.create_model(variant, pretrained=False)
        output = model(test_input)
        print(f"{variant}: {output.shape}")