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
            # AIMv2's forward typically accepts pixel_values
            # output_hidden_states=True ensures we can get hidden layer states
            outputs = self.model(pixel_values=pixel_values, output_hidden_states=True)
            # print(outputs)
            # Get the last hidden state
            if hasattr(outputs, "last_hidden_state"):
                last_hidden_state = outputs.last_hidden_state
            else:
                # If the return is a tuple, the first element is typically last_hidden_state
                last_hidden_state = outputs[0]

        return last_hidden_state

@register_model
def aimv2_large_patch14_native_ap(pretrained: bool = False, **kwargs):
    model = AIMv2("/video_vit/apple/aimv2-large-patch14-native")
    return model

if __name__ == "__main__":
    import timm

    # Create model
    model = timm.create_model("aimv2_large_patch14_native_ap")
    # /path/to/data/...

    bs = 4
    # AIMv2 Large Patch14 typically uses input size of 224x224
    test_input = torch.randn(bs, 3, 224, 224, device=model.device)

    # Forward pass
    last_hidden_state = model(test_input)

    print(f"Model: {type(model)}")
    print(f"Input shape: {test_input.shape}")
    print(f"Last hidden state shape: {last_hidden_state.shape}")
