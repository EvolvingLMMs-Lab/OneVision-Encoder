import torch
from torch import nn
from timm.models.registry import register_model

from transformers import Aimv2VisionModel
class AIMv2(nn.Module):
    def __init__(
        self,
        ckpt: str = "apple/aimv2-large-patch14-native",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super(AIMv2, self).__init__()
        self.device = torch.device(device)
        # Note: trust_remote_code is required for AIMv2 models
        model = Aimv2VisionModel.from_pretrained(ckpt, trust_remote_code=True)
        self.model = model.to(self.device)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        pixel_values = pixel_values.to(self.device)
        with torch.no_grad():
            # AIMv2 的 forward 通常接受 pixel_values
            # output_hidden_states=True 确保我们可以获取隐藏层状态
            outputs = self.model(pixel_values=pixel_values, output_hidden_states=True)
            # print(outputs)
            # 获取最后一层隐藏状态
            if hasattr(outputs, "last_hidden_state"):
                last_hidden_state = outputs.last_hidden_state
            else:
                # 如果返回的是 tuple，通常第一个元素是 last_hidden_state
                last_hidden_state = outputs[0]

        return last_hidden_state

@register_model
def aimv2_large_patch14_native_ap(pretrained: bool = False, **kwargs):
    model = AIMv2("/video_vit/pretrain_models/apple/aimv2-large-patch14-native")
    return model

if __name__ == "__main__":
    import timm

    # 创建模型
    model = timm.create_model("aimv2_large_patch14_native_ap")
    # model.model.save_pretrained("/video_vit/pretrain_models/apple/aimv2-large-patch14-native")

    bs = 4
    # AIMv2 Large Patch14 通常输入大小为 224x224
    test_input = torch.randn(bs, 3, 224, 224, device=model.device)

    # 前向传播
    last_hidden_state = model(test_input)

    print(f"Model: {type(model)}")
    print(f"Input shape: {test_input.shape}")
    print(f"Last hidden state shape: {last_hidden_state.shape}")
