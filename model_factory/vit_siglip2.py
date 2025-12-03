import torch
from torch import nn
from transformers import AutoModel
from timm.models.registry import register_model


class Siglip2Base(nn.Module):
    def __init__(self, ckpt: str = "google/siglip2-base-patch16-224", device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the Siglip2 Base model to retrieve hidden states.

        Args:
            ckpt (str): HuggingFace checkpoint for the pre-trained model.
            device (str): Device to map the model for inference.
        """
        super(Siglip2Base, self).__init__()
        self.device = torch.device(device)
        # Load the model (only vision model)
        self.model = AutoModel.from_pretrained(ckpt).vision_model.to(self.device).eval()

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
            # 获取最后一层的 hidden state
            last_hidden_state = outputs.last_hidden_state  # [bs, seq_len, hidden_size]

        return last_hidden_state


@register_model
def siglip2_base(pretrained=False, **kwargs):
    """
    Register the Siglip2 Base model for timm.

    Args:
        pretrained (bool): If True, load pretrained weights (default: False).
        **kwargs: Additional arguments passed to Siglip2Base.

    Returns:
        Siglip2Base: An instance of Siglip2Base.
    """
    model = Siglip2Base(
        ckpt="/video_vit/pretrain_models/siglip2-base-patch16-224",
        device=kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
    )
    return model


@register_model
def siglip2_large_patch16_256(pretrained=False, **kwargs):
    """
    Register the Siglip2 Base model for timm.

    Args:
        pretrained (bool): If True, load pretrained weights (default: False).
        **kwargs: Additional arguments passed to Siglip2Base.

    Returns:
        Siglip2Base: An instance of Siglip2Base.
    """
    model = Siglip2Base(
        ckpt="/video_vit/pretrain_models/siglip2-large-patch16-256",
        device=kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
    )
    return model


@register_model
def siglip2_so400m_patch16_naflex(pretrained=False, **kwargs):
    """
    Register the Siglip2 so400m-patch16-naflex model for timm.

    Args:
        pretrained (bool): If True, load pretrained weights (default: False).
        **kwargs: Additional arguments passed to Siglip2Base.

    Returns:
        Siglip2Base: An instance of Siglip2Base.
    """
    model = Siglip2Base(
        ckpt=kwargs.get("ckpt", "/video_vit/pretrain_models/siglip2-so400m-patch16-naflex"),
        device=kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
    )
    return model

if __name__ == "__main__":
    # Test the registered model with timm
    import timm

    # Create the model using timm framework
    model = timm.create_model("siglip2_base", pretrained=False)

    # 创建测试输入: [bs, 3, 224, 224]
    bs = 4
    test_input = torch.randn(bs, 3, 224, 224).cuda()

    # 获取最后的 hidden state
    last_hidden_state = model(test_input)

    # 打印形状信息
    print(f"Input shape: {test_input.shape}")
    print(f"Last hidden state shape: {last_hidden_state.shape}")
    # 预期输出: [bs, seq_len, hidden_size]
    # 例如: [4, 197, 768] (196 patches + 1 CLS token)
