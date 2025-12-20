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
            # 获取最后一层的 hidden state
            last_hidden_state = outputs.last_hidden_state  # [bs, seq_len, hidden_size]

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
    model = Dinov3(
        ckpt="/video_vit/pretrain_models/dinov3-vitb16-pretrain-lvd1689m",
        device=kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
    )
    return model


@register_model
def dinov3_large(pretrained=False, **kwargs):
    model = Dinov3(
        ckpt="/video_vit/pretrain_models/dinov3-vitl16-pretrain-lvd1689m",
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
    model = Dinov3(
        ckpt="/video_vit/pretrain_models/dinov3-giant",
        device=kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
    )
    return model


if __name__ == "__main__":
    # Test the registered model with timm
    import timm

    # Create the model using timm framework
    model = timm.create_model("dinov3_base", pretrained=False)

    # 创建测试输入: [bs, 3, 224, 224]
    bs = 4
    test_input = torch.randn(bs, 3, 224, 224).cuda()

    # 获取最后的 hidden state
    last_hidden_state = model(test_input)

    # 打印形状信息
    print(f"Input shape: {test_input.shape}")
    print(f"Last hidden state shape: {last_hidden_state.shape}")
    # 预期输出: [bs, seq_len, hidden_size]
    # 例如 dinov3-base: [4, 257, 768] (256 patches + 1 CLS token for 224x224)
    # 例如 dinov3-large: [4, 257, 1024]
    # 例如 dinov3-giant: [4, 257, 1536]

    print("\nTesting all variants:")
    for variant in ["dinov3_base", "dinov3_large", "dinov3_giant"]:
        model = timm.create_model(variant, pretrained=False)
        output = model(test_input)
        print(f"{variant}: {output.shape}")