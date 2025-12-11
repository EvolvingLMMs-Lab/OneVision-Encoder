# Vision Transformer Packing Format Implementation

## Overview

This document describes the packing format implementation for multiple Vision Transformer models in LLaVA-ViT. The packing format enables efficient batch processing of variable-length image sequences by concatenating all patches into a single sequence without padding.

### Supported Models

| Model | Class | Patch Size | Prefix Tokens | Implementation |
|-------|-------|------------|---------------|----------------|
| **AIMv2** | `AIMv2Packing` | 14 | None (excluded in output) | `vit_aim_v2_packing_hf.py` |
| **DINOv3** | `DINOv3ViTPacking` | 14 | CLS + registers (excluded in output) | `vit_dinov3_packing_hf.py` |
| **SigLIP2** | `Siglip2NaflexPacking` | 16 | None (Naflex variant) | `vit_siglip2_packing_hf.py` |

## Packing Format Specification

### Input Format

**`hidden_states`**: `[total_num_patches, patch_dim]`
- Pre-patchified input containing all image patches concatenated
- `total_num_patches = sum(t_i × h_i × w_i)` across all images
- `patch_dim = patch_size × patch_size × channels` (typically 588 for size 14, 768 for size 16)

**`grid_thw`**: `[num_images, 3]`
- Dimensions for each image as `[t, h, w]`
- `t`: temporal dimension (usually 1 for single images)
- `h`: height in patches
- `w`: width in patches

### Output Format

**`last_hidden_state`**: `[total_num_patches, hidden_size]`
- Packed output with prefix tokens (CLS, registers) excluded
- Maintains same order as input patches

## Quick Start

### Basic Usage

**AIMv2:**
```python
import torch
from vit_aim_v2_packing_hf import AIMv2Packing

# Initialize model (Hub or local path)
model = AIMv2Packing(ckpt="apple/aimv2-large-patch14-native", device="cuda")
# Or use local checkpoint
model = AIMv2Packing(ckpt="/video_vit/pretrain_models/apple/aimv2-large-patch14-native/", device="cuda")

# Prepare input (2 images of 224x224, patch_size=14)
hidden_states = torch.randn(512, 588).cuda()  # 2×256 patches, 14×14×3
grid_thw = torch.tensor([[1, 16, 16], [1, 16, 16]]).cuda()

# Forward pass
output = model(hidden_states, grid_thw)  # [512, 1024]
```

**DINOv3:**
```python
from vit_dinov3_packing_hf import DINOv3ViTPacking

# Initialize model (Hub or local path)
model = DINOv3ViTPacking(ckpt="facebook/dinov3-large", device="cuda")
# Or use local checkpoint
model = DINOv3ViTPacking(ckpt="/video_vit/pretrain_models/dinov3-vitl16-pretrain-lvd1689m/", device="cuda")

# Prepare input (2 images of 224x224, patch_size=14)
hidden_states = torch.randn(512, 588).cuda()  # 2×256 patches, 14×14×3
grid_thw = torch.tensor([[1, 16, 16], [1, 16, 16]]).cuda()

# Forward pass
output = model(hidden_states, grid_thw)  # [512, 1024]
```

**SigLIP2:**
```python
from vit_siglip2_packing_hf import Siglip2NaflexPacking

# Initialize model (Hub or local path)
model = Siglip2NaflexPacking(ckpt="google/siglip2-so400m-patch16-naflex", device="cuda")
# Or use local checkpoint
model = Siglip2NaflexPacking(ckpt="/video_vit/pretrain_models/siglip2-so400m-patch16-naflex/", device="cuda")

# Prepare input (2 images of 224x224, patch_size=16)
hidden_states = torch.randn(392, 768).cuda()  # 2×196 patches, 16×16×3
grid_thw = torch.tensor([[1, 14, 14], [1, 14, 14]]).cuda()

# Forward pass
output = model(hidden_states, grid_thw)  # [392, 1152]
```

### Converting Standard Images to Packing Format

```python
def convert_to_packing_format(images, patch_size):
    """Convert standard images to packing format."""
    batch_size, channels, height, width = images.shape
    h_patches = height // patch_size
    w_patches = width // patch_size
    
    # Reshape to patches
    patches = images.reshape(
        batch_size, channels,
        h_patches, patch_size,
        w_patches, patch_size
    ).permute(0, 2, 4, 3, 5, 1).reshape(
        batch_size, h_patches * w_patches,
        patch_size * patch_size * channels
    )
    
    # Concatenate all batches
    hidden_states = patches.reshape(-1, patch_size * patch_size * channels)
    
    # Create grid_thw
    grid_thw = torch.tensor(
        [[1, h_patches, w_patches]] * batch_size,
        dtype=torch.long, device=images.device
    )
    
    return hidden_states, grid_thw

# Example
images = torch.randn(2, 3, 224, 224).cuda()
hidden_states, grid_thw = convert_to_packing_format(images, patch_size=14)
```

## Model-Specific Details

### AIMv2

- **Checkpoint**: `apple/aimv2-large-patch14-native` (local or HuggingFace Hub)
- **Patch Size**: 14
- **Hidden Size**: 1024
- **Note**: `Aimv2VisionModel.last_hidden_state` already excludes CLS token
- **Patch Embedding**: Conv2d (reconstructs images from patches)

### DINOv3

- **Checkpoint**: `facebook/dinov3-base`, `dinov3-large`, `dinov3-giant`
- **Patch Size**: 14
- **Hidden Size**: 768 (base), 1024 (large), 1536 (giant)
- **Prefix Tokens**: CLS + 4 register tokens (automatically excluded in output)
- **Patch Embedding**: Conv2d (reconstructs images from patches)

### SigLIP2 Naflex

- **Checkpoint**: `google/siglip2-so400m-patch16-naflex`
- **Patch Size**: 16
- **Hidden Size**: 1152
- **Note**: Naflex variant has no prefix tokens
- **Processing**: Uses attention masks for variable-length sequences

## Alignment Testing

Each model includes an alignment verification script to ensure the packing implementation produces outputs consistent with the standard model.

### Running Alignment Tests

**With HuggingFace Hub:**
```bash
# AIMv2
python align_aim_v2_packing.py --ckpt apple/aimv2-large-patch14-native

# DINOv3
python align_dinov3_packing.py --ckpt facebook/dinov3-large

# SigLIP2
python align_siglip2_packing.py --ckpt google/siglip2-so400m-patch16-naflex
```

**With Local Checkpoints:**
```bash
# AIMv2
python align_aim_v2_packing.py --ckpt /video_vit/pretrain_models/apple/aimv2-large-patch14-native/

# DINOv3
python align_dinov3_packing.py --ckpt /video_vit/pretrain_models/dinov3-vitl16-pretrain-lvd1689m/

# SigLIP2
python align_siglip2_packing.py --ckpt /video_vit/pretrain_models/siglip2-so400m-patch16-naflex/
```

### Test Options

- `--ckpt`: Model checkpoint path or HuggingFace Hub ID
- `--device`: Device to use (`cuda` or `cpu`)
- `--batch_size`: Number of images per batch (default: 2)
- `--image_size`: Image size for testing (must be divisible by patch_size)
- `--threshold`: Cosine similarity threshold for passing (default: 0.99)
- `--use_real_images`: Test with real images instead of random tensors

### Expected Results

```
Test Results:
  224x224:
    ✅ Min cosine:  0.99987654 (PASS)
    ✅ Mean cosine: 0.99998765 (PASS)
  
✅ ALL TESTS PASSED: Models are aligned
```

## Benefits of Packing Format

1. **Memory Efficiency**: No padding for sequences of the same length
2. **Variable-Size Support**: Efficiently batch images of different sizes
3. **FlashAttention Compatible**: Optimized for varlen attention mechanisms
4. **Same Weights**: Uses identical model weights as standard format

## Implementation Notes

### Common Utilities

The `alignment_utils.py` module provides shared functions:
- `convert_to_patches()`: Convert images to packing format
- `compute_similarity_metrics()`: Compare model outputs
- `generate_test_image()`, `load_image_as_tensor()`: Test utilities

### Architecture Patterns

**Models with Conv2d Patch Embedding** (AIMv2, DINOv3):
```
Packed input → Reconstruct images → Conv2d → Process → Extract patches → Packed output
```

**Models with Linear Patch Embedding** (SigLIP2 Naflex):
```
Packed input → Reshape with padding → Attention masks → Process → Remove padding → Packed output
```

## Troubleshooting

### Image Size Must Be Divisible by Patch Size

```python
# Resize to nearest valid size
patch_size = 14
new_h = (height // patch_size) * patch_size
new_w = (width // patch_size) * patch_size
image = F.interpolate(image, size=(new_h, new_w), mode='bilinear')
```

### Low Alignment Scores

- Check that both models use the same checkpoint
- Verify correct patch size for the model
- Try lowering threshold: `--threshold 0.98`
- Note: Some numerical precision differences are expected

### Local Checkpoint Loading

All three packing implementations support both HuggingFace Hub IDs and local filesystem paths. The implementation automatically detects the path type and handles the loading appropriately.

**AIMv2:**
```python
from vit_aim_v2_packing_hf import AIMv2Packing

# HuggingFace Hub
model = AIMv2Packing(ckpt="apple/aimv2-large-patch14-native")

# Local path
model = AIMv2Packing(ckpt="/video_vit/pretrain_models/apple/aimv2-large-patch14-native/")
```

**DINOv3:**
```python
from vit_dinov3_packing_hf import DINOv3ViTPacking

# HuggingFace Hub
model = DINOv3ViTPacking(ckpt="facebook/dinov3-large")

# Local path
model = DINOv3ViTPacking(ckpt="/video_vit/pretrain_models/dinov3-vitl16-pretrain-lvd1689m/")
```

**SigLIP2:**
```python
from vit_siglip2_packing_hf import Siglip2NaflexPacking

# HuggingFace Hub
model = Siglip2NaflexPacking(ckpt="google/siglip2-so400m-patch16-naflex")

# Local path
model = Siglip2NaflexPacking(ckpt="/video_vit/pretrain_models/siglip2-so400m-patch16-naflex/")
```

**Note:** For AIMv2, the implementation automatically detects local paths and excludes the `revision` parameter to avoid `OSError`. DINOv3 and SigLIP2 work seamlessly with both Hub and local checkpoints.

## Example: Multi-Image Processing

### AIMv2 Example
```python
from vit_aim_v2_packing_hf import AIMv2Packing

# Load model from local checkpoint
model = AIMv2Packing(ckpt="/video_vit/pretrain_models/apple/aimv2-large-patch14-native/")

# Process multiple images with different sizes (all divisible by 14)
image1 = torch.randn(3, 224, 224)  # 16×16 = 256 patches
image2 = torch.randn(3, 336, 336)  # 24×24 = 576 patches
image3 = torch.randn(3, 448, 448)  # 32×32 = 1024 patches

# Convert each to patches
patches1, grid1 = convert_to_packing_format(image1.unsqueeze(0), 14)
patches2, grid2 = convert_to_packing_format(image2.unsqueeze(0), 14)
patches3, grid3 = convert_to_packing_format(image3.unsqueeze(0), 14)

# Concatenate into single batch
hidden_states = torch.cat([patches1, patches2, patches3], dim=0)  # [1856, 588]
grid_thw = torch.cat([grid1, grid2, grid3], dim=0)  # [3, 3]

# Single forward pass
output = model(hidden_states, grid_thw)  # [1856, 1024]

# Split outputs by image
start = 0
for i, grid in enumerate([grid1, grid2, grid3]):
    num_patches = grid[0, 1] * grid[0, 2]  # h × w
    image_output = output[start:start + num_patches]
    print(f"Image {i+1} output: {image_output.shape}")
    start += num_patches
```

### DINOv3 Example
```python
from vit_dinov3_packing_hf import DINOv3ViTPacking

# Load model from local checkpoint
model = DINOv3ViTPacking(ckpt="/video_vit/pretrain_models/dinov3-vitl16-pretrain-lvd1689m/")

# Process images with patch_size=14
image1 = torch.randn(3, 224, 224)
image2 = torch.randn(3, 336, 336)

patches1, grid1 = convert_to_packing_format(image1.unsqueeze(0), 14)
patches2, grid2 = convert_to_packing_format(image2.unsqueeze(0), 14)

hidden_states = torch.cat([patches1, patches2], dim=0)  # [832, 588]
grid_thw = torch.cat([grid1, grid2], dim=0)  # [2, 3]

output = model(hidden_states, grid_thw)  # [832, 1024]
```

### SigLIP2 Example
```python
from vit_siglip2_packing_hf import Siglip2NaflexPacking

# Load model from local checkpoint
model = Siglip2NaflexPacking(ckpt="/video_vit/pretrain_models/siglip2-so400m-patch16-naflex/")

# Process images with patch_size=16
image1 = torch.randn(3, 224, 224)  # 14×14 = 196 patches
image2 = torch.randn(3, 384, 384)  # 24×24 = 576 patches

patches1, grid1 = convert_to_packing_format(image1.unsqueeze(0), 16)
patches2, grid2 = convert_to_packing_format(image2.unsqueeze(0), 16)

hidden_states = torch.cat([patches1, patches2], dim=0)  # [772, 768]
grid_thw = torch.cat([grid1, grid2], dim=0)  # [2, 3]

output = model(hidden_states, grid_thw)  # [772, 1152]
```

## License

This implementation follows the same license as the LLaVA-ViT project.

## References

- [AIMv2 Paper](https://arxiv.org/abs/2411.14402)
- [DINOv3 Paper](https://arxiv.org/abs/2304.07193)
- [SigLIP Paper](https://arxiv.org/abs/2303.15343)
- [LLaVA Project](https://llava-vl.github.io/)
