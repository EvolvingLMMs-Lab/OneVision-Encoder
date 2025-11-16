```python
import torch
from torch import nn
from transformers import Dinov2Model
from timm.models.registry import register_model


class DINOv2Base(nn.Module):
    def __init__(
        self,
        ckpt: str = "facebook/dinov2-base",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        初始化 DINOv2 视觉 Transformer，用于获取最后一层 hidden states。

        Args:
            ckpt (str): HuggingFace 上的预训练模型名称或本地路径。
                        常用: "facebook/dinov2-base", "facebook/dinov2-large",
                             "facebook/dinov2-giant", "facebook/dinov2-small"
            device (str): 推理所用设备。
        """
        super(DINOv2Base, self).__init__()
        self.device = torch.device(device)
        base_model = Dinov2Model.from_pretrained(ckpt)
        # Dinov2Model 本身就是纯视觉 backbone
        self.model = base_model.to(self.device).eval()

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        前向：返回最后一层 hidden state。

        Args:
            pixel_values (torch.Tensor): 输入图像张量，形状 [bs, 3, H, W]，通常为 [bs, 3, 224, 224]

        Returns:
            torch.Tensor: 最后一层 hidden state，形状 [bs, seq_len, hidden_size]
                          对 dinov2-base：seq_len=257，hidden_size=768
        """
        pixel_values = pixel_values.to(self.device)
        with torch.no_grad():
            outputs = self.model(pixel_values=pixel_values, output_hidden_states=True)
            last_hidden_state = outputs.last_hidden_state  # [bs, seq_len, hidden_size]
        return last_hidden_state


@register_model
def dinov2_base(pretrained: bool = False, **kwargs):
    """
    注册 DINOv2 Base (facebook/dinov2-base) 到 timm。

    Args:
        pretrained (bool): 与 timm 接口兼容；实际权重加载在 DINOv2Base 内完成。
        **kwargs:
            ckpt (str): 自定义 HuggingFace 名称或本地路径。
            device (str): 设备。

    Returns:
        DINOv2Base: 模型实例。
    """
    model = DINOv2Base(
        ckpt=kwargs.get("ckpt", "facebook/dinov2-base"),
        device=kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
    )
    return model


if __name__ == "__main__":
    import timm

    # 通过 timm 创建模型
    model = timm.create_model("dinov2_base", pretrained=False)

    bs = 4
    test_input = torch.randn(bs, 3, 224, 224, device=model.device)

    last_hidden_state = model(test_input)

    print(f"Input shape: {test_input.shape}")
    print(f"Last hidden state shape: {last_hidden_state.shape}")
    # 预期: [4, 257, 768] 对 facebook/dinov2-base (Patch14, 224)
