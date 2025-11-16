import torch
from torch import nn
from transformers import CLIPModel
from timm.models.registry import register_model


class CLIPBase(nn.Module):
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
        super(CLIPBase, self).__init__()
        self.device = torch.device(device)
        # 直接从 transformers 导入 CLIPModel，然后取 vision_model
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
            # 最后一层的 hidden state: [bs, seq_len, hidden_size]
            last_hidden_state = outputs.last_hidden_state

        return last_hidden_state


@register_model
def clip_base(pretrained: bool = False, **kwargs):
    """
    Register the CLIP Base Vision Transformer (ViT-B/16, 224x224) model for timm.

    Args:
        pretrained (bool): If True, load pretrained weights (from the HuggingFace ckpt path).
                           这里的 pretrained 标志仅用于接口兼容，权重加载在 CLIPBase 中完成。
        **kwargs: Additional arguments passed to CLIPBase.

    Returns:
        CLIPBase: An instance of CLIPBase.
    """
    model = CLIPBase(
        # 如需使用本地 ckpt，设置为本地路径；否则传入默认/自定义的 HF 路径
        ckpt=kwargs.get("ckpt", "/video_vit/pretrain_models/openai/clip-vit-base-patch16"),
        device=kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
    )
    return model


if __name__ == "__main__":
    import timm

    # 通过 timm 创建模型（名称与 register 的函数名一致）
    model = timm.create_model("clip_base", pretrained=False)

    # 测试输入: [bs, 3, 224, 224]
    bs = 4
    # 与模型 device 对齐，避免 CPU-only 环境下 .cuda() 报错
    test_input = torch.randn(bs, 3, 224, 224, device=model.device)

    # 获取最后的 hidden state
    last_hidden_state = model(test_input)

    # 打印形状
    print(f"Input shape: {test_input.shape}")
    print(f"Last hidden state shape: {last_hidden_state.shape}")
    # 预期: [4, seq_len, hidden_size]
    # 对 CLIP ViT-B/16, 224x224 通常是 [4, 197, 768]
