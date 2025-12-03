import torch
from torch import nn
from transformers import AutoModel
from timm.models.registry import register_model

class Siglip(nn.Module):
    def __init__(self, ckpt: str = "google/siglip-base-patch16-224", device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize Siglip model for hidden states (without CLS token).
        """
        super(Siglip, self).__init__()
        self.device = torch.device(device)
        self.model = AutoModel.from_pretrained(ckpt).vision_model.to(self.device).eval()

    def forward(self, pixel_values):
        """
        Get last hidden state, removing the CLS token.

        Args:
            pixel_values (torch.Tensor): [bs, 3, h, w]
        Returns:
            torch.Tensor: [bs, seq_len-1, hidden_size]
        """
        with torch.no_grad():
            outputs = self.model(pixel_values=pixel_values, output_hidden_states=True)
            last_hidden_state = outputs.last_hidden_state  # [bs, seq_len, hidden_size]
            patch_tokens = last_hidden_state[:, :, :]     # Remove CLS token at index 0
        return patch_tokens

@register_model
def siglip_base(pretrained=False, **kwargs):
    """
    Register Siglip without CLS token for timm.
    """
    model = Siglip(
        ckpt="/video_vit/pretrain_models/siglip-base-patch16-224",
        device=kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
    )
    return model


@register_model
def siglip_large_patch16_256(pretrained=False, **kwargs):
    model = Siglip(
        ckpt="/video_vit/pretrain_models/siglip-large-patch16-256",
        device=kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
    )
    return model

if __name__ == "__main__":
    import timm

    model = timm.create_model("siglip_base", pretrained=False)
    bs = 4
    test_input = torch.randn(bs, 3, 224, 224).cuda()
    patch_tokens = model(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Patch tokens shape (no CLS): {patch_tokens.shape}")
    # Expect: [bs, 196, hidden_size]
