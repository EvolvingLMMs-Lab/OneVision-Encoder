# HEVC ViT Packing Tower - Usage Guide

## Overview

This document describes the `HEVCViTPackingVisionTower` implementation, which provides a packing mode for the HEVC ViT vision encoder. This tower converts batch images to the efficient packing format used by `vit_preview_v0_packing_hf.py`.

## What is Packing Mode?

The packing mode uses a more efficient input representation:
- **Standard format**: `[B, C, H, W]` (batch, channels, height, width)
- **Packing format**: `[num_patches, patch_dim]` where:
  - `num_patches = B × (H/patch_size) × (W/patch_size)`
  - `patch_dim = patch_size × patch_size × channels`

This format allows for efficient processing of multiple images with FlashAttention, even if they have different sizes.

## File Location

```
llava_next/llava/model/multimodal_encoder/hevc_vit_packing_tower.py
```

## Key Features

### 1. Input Conversion (Batch Images → Packing Format)

The tower automatically converts standard batch images to packing format:

```python
# Input: [B, C, H, W] tensor
images = torch.randn(4, 3, 224, 224)

# Internally converted to:
# hidden_states: [num_patches, patch_dim] 
#   where num_patches = 4 × 14 × 14 = 784
#   and patch_dim = 16 × 16 × 3 = 768
# grid_thw: [[1, 14, 14], [1, 14, 14], [1, 14, 14], [1, 14, 14]]
```

### 2. Output Conversion (Packing Format → Feature Tensors)

The tower converts the packing model's output back to the expected format:

```python
# Packing model output: [total_seq_len, hidden_size]
# Converted back to: [B, num_patches, hidden_size]
```

### 3. List of Images Support

The tower also supports processing a list of images (potentially with different sizes):

```python
# Input: List of [C, H, W] tensors
images = [
    torch.randn(3, 224, 224),  # Image 1: 14×14 = 196 patches
    torch.randn(3, 224, 224),  # Image 2: 14×14 = 196 patches
    torch.randn(3, 448, 448),  # Image 3: 28×28 = 784 patches
]

# Output: List of feature tensors with shapes:
# [196, hidden_size], [196, hidden_size], [784, hidden_size]
```

## Code Highlights

### Input Conversion Markers

The code uses prominent comments to mark input conversion:

```python
# ============================================================
# 【INPUT CONVERSION】: Convert batch images to packing format
# Standard format: [B, C, H, W]
# Packing format: [total_num_patches, patch_dim]
# ============================================================
```

### Output Conversion Markers

The code uses prominent comments to mark output conversion:

```python
# ============================================================
# 【OUTPUT CONVERSION】: Convert packing output back to feature format
# Packing output: [total_seq_len, hidden_size]
# Target format: [B, num_patches, hidden_size]
# ============================================================
```

## Usage

### Registering the Tower

The packing tower is automatically registered in `builder.py`:

```python
# In your config, use a model name containing "packing"
mm_vision_tower = "/path/to/hevc_vit_packing_model"

# The builder will automatically use HEVCViTPackingVisionTower
```

### Model Selection Logic

The builder uses the following logic:

1. If `"hevc_vit_packing"` or `"packing"` is in the model name → `HEVCViTPackingVisionTower`
2. If `"hevc_vit"` is in the model name → `HEVCViTVisionTower` (standard mode)

## Implementation Details

### Helper Methods

#### `_image_to_packing_input(image_tensor)`
Converts a single image `[C, H, W]` to packing format:
- Returns: `hidden_states [seq_len, patch_dim]`, `grid_thw [1, 3]`

#### `_batch_images_to_packing_input(images)`
Converts a batch of images `[B, C, H, W]` to packing format:
- Returns: `hidden_states [total_seq_len, patch_dim]`, `grid_thw [B, 3]`

### Patch Calculation

For an image of size `H × W` with `patch_size = 16`:
- `h_patches = H // 16`
- `w_patches = W // 16`
- `num_patches = h_patches × w_patches`
- `patch_dim = 16 × 16 × 3 = 768`

Example:
- 224×224 image: 14×14 = 196 patches
- 448×448 image: 28×28 = 784 patches

## Differences from Standard Tower

| Aspect | Standard Tower | Packing Tower |
|--------|---------------|---------------|
| Model | `vit_preview_v0_hf.py` | `vit_preview_v0_packing_hf.py` |
| Input Format | `[B, C, H, W]` | `[num_patches, patch_dim]` |
| Output Format | `[B, num_patches, hidden_size]` | `[num_patches, hidden_size]` → converted back |
| Batch Processing | Standard batch attention | FlashAttention with packing |
| Variable Sizes | Requires same size in batch | Supports different sizes natively |

## Requirements

1. The packing model requires FlashAttention 2:
   ```bash
   pip install flash-attn --no-build-isolation
   ```

2. A converted packing model checkpoint (use `convert_llava_vit_packing_to_hf.py`)

## Testing

The implementation has been verified for:
- ✓ Syntax correctness
- ✓ Input conversion logic (batch and list modes)
- ✓ Output conversion logic
- ✓ Integration with builder.py

## Example Flow

```
1. User provides: images [4, 3, 224, 224]

2. Input Conversion:
   → hidden_states [784, 768]  (4×196 patches, 768 = 16×16×3)
   → grid_thw [[1,14,14], [1,14,14], [1,14,14], [1,14,14]]

3. Packing Model Forward:
   → raw_output [784, hidden_size]

4. Output Conversion:
   → features [4, 196, hidden_size]

5. Return features to downstream modules
```

## Notes

- The packing tower maintains the same interface as the standard tower
- All conversion is handled internally and transparently
- The output format matches what downstream modules expect
- Supports both batch tensors and lists of images
