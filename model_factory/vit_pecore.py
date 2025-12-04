import torch
from torch import nn
from transformers import AutoModel
from timm.models.registry import register_model


class PECore(nn.Module):
    def __init__(
        self,
        ckpt: str = "facebook/PE-Core-B16-224",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        local_files_only: bool = True,
    ):
        """
        Initialize PE-Core vision transformer model.
        
        PE-Core (Position Encoded Core) is a vision transformer model from Meta/Facebook
        that uses position encoding for better spatial understanding.

        Args:
            ckpt (str): HuggingFace checkpoint path for PE-Core model.
                       Default: "facebook/PE-Core-B16-224"
            device (str): Device to map the model for inference.
            local_files_only (bool): Whether to only use local files (offline mode).
        """
        super(PECore, self).__init__()
        self.device = torch.device(device)
        
        # Load the model from HuggingFace
        # PE-Core models have a vision_model attribute similar to CLIP/SigLIP
        try:
            base_model = AutoModel.from_pretrained(
                ckpt, 
                local_files_only=local_files_only,
                trust_remote_code=True
            )
            # Try to access vision_model if it exists
            if hasattr(base_model, 'vision_model'):
                self.model = base_model.vision_model.to(self.device).eval()
            else:
                # If no vision_model attribute, use the base model directly
                self.model = base_model.to(self.device).eval()
        except Exception as e:
            print(f"Warning: Error loading model: {e}")
            # Fallback to loading the entire model
            self.model = AutoModel.from_pretrained(
                ckpt,
                local_files_only=local_files_only,
                trust_remote_code=True
            ).to(self.device).eval()

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to get the last hidden state.

        Args:
            pixel_values (torch.Tensor): Input tensor of shape [bs, 3, h, w]

        Returns:
            torch.Tensor: Last hidden state of shape [bs, seq_len, hidden_size]
                         For PE-Core-B16-224: [bs, 197, 768] (196 patches + 1 CLS token)
        """
        pixel_values = pixel_values.to(self.device)
        with torch.no_grad():
            outputs = self.model(pixel_values=pixel_values, output_hidden_states=True)
            # Return the last hidden state which includes all tokens
            last_hidden_state = outputs.last_hidden_state

        return last_hidden_state


@register_model
def pecore_base_patch16_224(pretrained: bool = False, **kwargs):
    """
    Register the PE-Core Base Vision Transformer (B16, 224x224) model for timm.

    Args:
        pretrained (bool): If True, load pretrained weights (from the HuggingFace ckpt path).
                          This flag is for interface compatibility; weight loading happens in PECore.
        **kwargs: Additional arguments passed to PECore.

    Returns:
        PECore: An instance of PECore.
    """
    model = PECore(
        ckpt=kwargs.get("ckpt", "/video_vit/pretrain_models/facebook/PE-Core-B16-224"),
        device=kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
        local_files_only=kwargs.get("local_files_only", True),
    )
    return model


@register_model
def pecore_base(pretrained: bool = False, **kwargs):
    """
    Alias for pecore_base_patch16_224.
    Register the PE-Core Base Vision Transformer model for timm.

    Args:
        pretrained (bool): If True, load pretrained weights.
        **kwargs: Additional arguments passed to PECore.

    Returns:
        PECore: An instance of PECore.
    """
    return pecore_base_patch16_224(pretrained=pretrained, **kwargs)


if __name__ == "__main__":
    import timm

    # Test the model creation through timm
    print("Creating PE-Core model...")
    model = timm.create_model("pecore_base", pretrained=False)

    # Test input: [bs, 3, 224, 224]
    bs = 2
    # Get device from the model's internal device attribute
    device = model.device if hasattr(model, 'device') else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_input = torch.randn(bs, 3, 224, 224, device=device)

    # Get the last hidden state
    print("Running forward pass...")
    last_hidden_state = model(test_input)

    # Print shapes
    print(f"Input shape: {test_input.shape}")
    print(f"Last hidden state shape: {last_hidden_state.shape}")
    # Expected: [2, 197, 768] for PE-Core-B16-224
    # (196 patches from 14x14 grid + 1 CLS token, 768 hidden dimensions)
