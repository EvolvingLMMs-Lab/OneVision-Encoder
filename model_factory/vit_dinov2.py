import torch
from torch import nn
from transformers import Dinov2Model
from timm.models.registry import register_model


class Dinov2(nn.Module):
    def __init__(
        self,
        ckpt: str = "/video_vit/pretrain_models/dinov2-base",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        local_files_only: bool = True,
    ):
        """
        DINOv2 视觉 Transformer 封装（forward 返回去掉 CLS 的 patch tokens）

        Args:
            ckpt (str): 本地或 HuggingFace 上的模型路径/名称
            device (str): 推理设备
            local_files_only (bool): 是否仅使用本地文件（离线推荐 True）
        """
        super().__init__()
        self.device = torch.device(device)
        self.model = Dinov2Model.from_pretrained(
            ckpt, local_files_only=local_files_only
        ).to(self.device).eval()

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values (torch.Tensor): [bs, 3, H, W]，常用 H=W=224

        Returns:
            torch.Tensor: 去掉 CLS 后的 hidden states，形状 [bs, seq_len-1, hidden]
                          dinov2-base(224, p14)：[bs, 256, 768]
        """
        pixel_values = pixel_values.to(self.device)
        with torch.no_grad():
            outputs = self.model(pixel_values=pixel_values)
            # 去掉 CLS token
            patch_tokens = outputs.last_hidden_state[:, 1:, :]
        return patch_tokens


@register_model
def dinov2_base(pretrained: bool = False, **kwargs):
    """
    timm 注册：DINOv2 Base（forward 返回去 CLS 的 patch tokens）
    Args:
        pretrained (bool): 仅为接口兼容，实际加载在 Dinov2 内部完成
        **kwargs: 透传给 Dinov2（ckpt, device, local_files_only）
    """
    model = Dinov2(
        ckpt=kwargs.get("ckpt", "/video_vit/pretrain_models/dinov2-base"),
        device=kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
        local_files_only=kwargs.get("local_files_only", True),
    )
    return model

@register_model
def dinov2_large(pretrained: bool = False, **kwargs):
    model = Dinov2(
        ckpt=kwargs.get("ckpt", "/video_vit/pretrain_models/dinov2-large"),
        device=kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
        local_files_only=kwargs.get("local_files_only", True),
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
    # 预期：dinov2-base(224, p14) -> [4, 256, 768]
