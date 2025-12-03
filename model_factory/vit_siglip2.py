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

    def _convert_to_patches(self, pixel_values, patch_size):
        """
        Convert image tensor to patches.
        
        Args:
            pixel_values (torch.Tensor): Input tensor of shape [bs, channels, height, width]
            patch_size (int): Size of each patch
            
        Returns:
            torch.Tensor: Patches of shape [bs, num_patches, channels * patch_size * patch_size]
        """
        batch_size, channels, height, width = pixel_values.shape
        num_patches_height = height // patch_size
        num_patches_width = width // patch_size
        
        # Reshape to patches: [bs, channels, num_patches_h, patch_size, num_patches_w, patch_size]
        patches = pixel_values.reshape(
            batch_size, channels,
            num_patches_height, patch_size,
            num_patches_width, patch_size
        )
        
        # Rearrange to: [bs, num_patches_h, num_patches_w, patch_size, patch_size, channels]
        patches = patches.permute(0, 2, 4, 3, 5, 1)
        
        # Flatten patches: [bs, num_patches_h * num_patches_w, patch_size * patch_size * channels]
        patches = patches.reshape(
            batch_size,
            num_patches_height * num_patches_width,
            patch_size * patch_size * channels
        )
        
        return patches
    
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
            batch_size = pixel_values.shape[0]
            height = pixel_values.shape[2]
            width = pixel_values.shape[3]
            
            # Get patch_size from model config if available, otherwise infer from model name
            if hasattr(self.model.config, 'patch_size'):
                patch_size = self.model.config.patch_size
            else:
                # Default to 16 for siglip2 models
                patch_size = 16
            
            # Calculate spatial shapes (number of patches in height and width)
            num_patches_height = height // patch_size
            num_patches_width = width // patch_size
            num_patches = num_patches_height * num_patches_width
            
            # Create spatial_shapes tensor: [batch_size, 2]
            spatial_shapes = torch.tensor(
                [[num_patches_height, num_patches_width]] * batch_size,
                dtype=torch.long,
                device=pixel_values.device
            )
            
            # Create attention_mask: all ones for non-masked (no padding)
            # Shape: [batch_size, num_patches]
            attention_mask = torch.ones(
                batch_size, num_patches,
                dtype=torch.long,
                device=pixel_values.device
            )
            
            # Check if the model expects patchified input (naflex-style models)
            # This is determined by checking if the embeddings layer has patch_embedding (Linear layer)
            # instead of conv_projection (Conv2d layer)
            needs_patchified_input = False
            if hasattr(self.model, 'embeddings'):
                if hasattr(self.model.embeddings, 'patch_embedding'):
                    # Linear layer expects pre-patchified input
                    needs_patchified_input = True
                elif hasattr(self.model.embeddings, 'patch_projection'):
                    # Conv2d layer expects regular [B, C, H, W] input
                    needs_patchified_input = False
            
            # Prepare pixel_values in the appropriate format
            if needs_patchified_input:
                pixel_values = self._convert_to_patches(pixel_values, patch_size)
            
            # Call model with required parameters
            # Pass as positional arguments to match the required signature
            outputs = self.model(
                pixel_values,
                attention_mask,
                spatial_shapes,
                output_hidden_states=True
            )
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
        ckpt=kwargs.get("ckpt", "/video_vit/pretrain_models/siglip2-base-patch16-224"),
        device=kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
    )
    return model


@register_model
def siglip2_large_patch16_256(pretrained=False, **kwargs):
    """
    Register the Siglip2 Large model for timm.

    Args:
        pretrained (bool): If True, load pretrained weights (default: False).
        **kwargs: Additional arguments passed to Siglip2Base.

    Returns:
        Siglip2Base: An instance of Siglip2Base.
    """
    model = Siglip2Base(
        ckpt=kwargs.get("ckpt", "/video_vit/pretrain_models/siglip2-large-patch16-256"),
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
