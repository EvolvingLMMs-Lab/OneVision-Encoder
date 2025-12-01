# LlavaViT Packing Model

This document describes how to use the `LlavaViTPackingModel`, a vision transformer designed for efficient variable-length sequence processing with FlashAttention support.

## Overview

The packing model is similar to Qwen2VL's vision encoder, accepting inputs in `[seq_len, patch_dim]` format instead of traditional `[batch, channels, height, width]`. This allows for efficient batch processing of images with different sizes through sequence packing.

## Requirements

- **FlashAttention 2** is **mandatory** for this model:
  ```bash
  pip install flash-attn --no-build-isolation
  ```
- CUDA-compatible GPU
- PyTorch with CUDA support

## Quick Start

### Basic Usage

```python
import torch
from model_factory.vit_preview_v0_packing_hf import (
    LlavaViTPackingModel,
    LlavaViTPackingConfig,
)

# Create model configuration
config = LlavaViTPackingConfig(
    patch_size=16,           # Size of each patch
    hidden_size=768,         # Hidden dimension
    num_attention_heads=12,  # Number of attention heads
    num_hidden_layers=12,    # Number of transformer layers
    num_channels=3,          # RGB channels
)

# Create model
model = LlavaViTPackingModel(config)
model = model.cuda().bfloat16()
model.eval()

# Prepare input
# For a single 224x224 image with patch_size=16:
# - h_patches = w_patches = 224 // 16 = 14
# - seq_len = 1 * 14 * 14 = 196
# - patch_dim = 16 * 16 * 3 = 768

patch_size = 16
in_channels = 3
h_patches, w_patches = 14, 14
t_frames = 1  # 1 for images, >1 for videos

patch_dim = patch_size * patch_size * in_channels  # 768
seq_len = t_frames * h_patches * w_patches  # 196

# Input format: (seq_len, patch_dim)
hidden_states = torch.randn(seq_len, patch_dim, dtype=torch.bfloat16, device='cuda')

# grid_thw specifies [temporal, height, width] patches for each image
grid_thw = torch.tensor([[t_frames, h_patches, w_patches]], dtype=torch.long, device='cuda')

# Forward pass
with torch.no_grad():
    outputs = model(hidden_states=hidden_states, grid_thw=grid_thw)
    
# outputs.last_hidden_state: (seq_len, hidden_size)
# outputs.pooler_output: (num_images, hidden_size) if use_head=True
print(f"Output shape: {outputs.last_hidden_state.shape}")
```

### Processing Real Images

```python
import torch
from PIL import Image
import requests

def image_to_packing_input(image: Image.Image, patch_size: int = 16):
    """Convert a PIL Image to packing model input format.
    
    Args:
        image: PIL Image (will be resized to patch-aligned size)
        patch_size: Size of each patch
        
    Returns:
        hidden_states: (seq_len, patch_dim) tensor
        grid_thw: (1, 3) tensor with [t, h, w] patches
    """
    # Resize to patch-aligned size
    w, h = image.size
    new_h = (h // patch_size) * patch_size
    new_w = (w // patch_size) * patch_size
    image = image.resize((new_w, new_h))
    
    # Convert to tensor: (C, H, W)
    import torchvision.transforms as T
    transform = T.Compose([
        T.ToTensor(),  # (C, H, W), values in [0, 1]
        T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                   std=[0.26862954, 0.26130258, 0.27577711]),
    ])
    pixel_tensor = transform(image)  # (3, H, W)
    
    # Calculate patch dimensions
    channels = pixel_tensor.shape[0]
    h_patches = new_h // patch_size
    w_patches = new_w // patch_size
    t_frames = 1
    
    # Reshape to patches: (C, H, W) -> (h_patches, w_patches, C, patch_size, patch_size)
    patches = pixel_tensor.view(
        channels, h_patches, patch_size, w_patches, patch_size
    )
    patches = patches.permute(1, 3, 0, 2, 4).contiguous()  # (h, w, C, pH, pW)
    
    # Flatten to (seq_len, patch_dim)
    seq_len = t_frames * h_patches * w_patches
    patch_dim = patch_size * patch_size * channels
    hidden_states = patches.view(seq_len, patch_dim)
    
    # Create grid_thw
    grid_thw = torch.tensor([[t_frames, h_patches, w_patches]], dtype=torch.long)
    
    return hidden_states, grid_thw


# Example: Process a real image
url = "https://images.cocodataset.org/val2017/000000039769.jpg"
response = requests.get(url, stream=True, timeout=10)
image = Image.open(response.raw)

hidden_states, grid_thw = image_to_packing_input(image, patch_size=16)
hidden_states = hidden_states.cuda().bfloat16()
grid_thw = grid_thw.cuda()

with torch.no_grad():
    outputs = model(hidden_states=hidden_states, grid_thw=grid_thw)
print(f"Output shape: {outputs.last_hidden_state.shape}")
```

### Batch Processing Multiple Images

```python
def batch_images_to_packing_input(images: list, patch_size: int = 16):
    """Convert multiple images to a packed batch.
    
    Args:
        images: List of PIL Images
        patch_size: Size of each patch
        
    Returns:
        hidden_states: (total_seq_len, patch_dim) tensor
        grid_thw: (num_images, 3) tensor
    """
    all_hidden_states = []
    all_grid_thw = []
    
    for img in images:
        hs, grid = image_to_packing_input(img, patch_size)
        all_hidden_states.append(hs)
        all_grid_thw.append(grid)
    
    # Concatenate all patches along sequence dimension
    hidden_states = torch.cat(all_hidden_states, dim=0)  # (total_seq_len, patch_dim)
    grid_thw = torch.cat(all_grid_thw, dim=0)  # (num_images, 3)
    
    return hidden_states, grid_thw


# Example: Process multiple images
images = [
    Image.new('RGB', (224, 224), color='red'),
    Image.new('RGB', (224, 224), color='blue'),
    Image.new('RGB', (448, 448), color='green'),  # Different size!
]

hidden_states, grid_thw = batch_images_to_packing_input(images, patch_size=16)
hidden_states = hidden_states.cuda().bfloat16()
grid_thw = grid_thw.cuda()

print(f"Batch input shape: {hidden_states.shape}")
print(f"grid_thw:\n{grid_thw}")
# grid_thw will be:
# [[1, 14, 14],   # 224x224 image: 14*14 = 196 patches
#  [1, 14, 14],   # 224x224 image: 14*14 = 196 patches
#  [1, 28, 28]]   # 448x448 image: 28*28 = 784 patches
# Total seq_len = 196 + 196 + 784 = 1176

with torch.no_grad():
    outputs = model(hidden_states=hidden_states, grid_thw=grid_thw)
    
# outputs.last_hidden_state: (1176, hidden_size)
# outputs.pooler_output: (3, hidden_size) - one pooled output per image
```

## Model Variants

Available model variants through `timm`:

```python
import timm

# Small model (6 layers, 384 hidden size, patch_size=16)
model = timm.create_model("hf_llava_vit_packing_small_ln", pretrained=False)

# Base model (12 layers, 768 hidden size, patch_size=16)
model = timm.create_model("hf_llava_vit_packing_base_ln", pretrained=False)

# Large model (24 layers, 1024 hidden size, patch_size=14)
model = timm.create_model("hf_llava_vit_packing_large_ln", pretrained=False)

# Huge model (32 layers, 1280 hidden size, patch_size=14)
model = timm.create_model("hf_llava_vit_packing_huge_ln", pretrained=False)

# Giant model (40 layers, 1536 hidden size, patch_size=14)
model = timm.create_model("hf_llava_vit_packing_giant_ln", pretrained=False)
```

## Weight Conversion

To convert weights from `vit_preview_v0_hf` to the packing model:

```bash
python model_factory/convert_llava_vit_packing_to_hf.py \
    llava_vit_large_ln \
    /path/to/backbone.pt \
    --output_dir /path/to/output
```

## Input/Output Format

### Input

| Parameter | Shape | Description |
|-----------|-------|-------------|
| `hidden_states` | `(seq_len, patch_dim)` | Flattened pixel patches. `patch_dim = patch_size * patch_size * in_channels` |
| `grid_thw` | `(num_images, 3)` | Grid dimensions `[t, h, w]` for each image. `t` is temporal (1 for images), `h` and `w` are patch counts |

### Output

| Field | Shape | Description |
|-------|-------|-------------|
| `last_hidden_state` | `(seq_len, hidden_size)` | Encoded patch representations |
| `pooler_output` | `(num_images, hidden_size)` | Pooled representation per image (if `use_head=True`) |
| `hidden_states` | tuple | All layer hidden states (if `output_hidden_states=True`) |

## Key Differences from Standard ViT

1. **Input Format**: Uses `[seq_len, patch_dim]` instead of `[B, C, H, W]`
2. **FlashAttention Required**: Mandatory for efficient variable-length attention
3. **Packing**: Multiple images can be packed into a single sequence
4. **RoPE**: Uses 3D rotary position embeddings (temporal + 2D spatial)
5. **grid_thw**: Explicit patch grid specification allows variable-size inputs

## Notes

- The model uses LayerNorm (not RMSNorm) by default
- FlashAttention 2 is **required** - model will raise `ImportError` if not available
- For best performance, use `bfloat16` precision
- The pooler uses multi-head attention pooling (similar to SigLIP2)
