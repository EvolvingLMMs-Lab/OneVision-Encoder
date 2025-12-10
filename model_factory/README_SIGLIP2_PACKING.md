# Siglip2 Naflex Packing Implementation

This document describes the packing format implementation for the Siglip2 Naflex model.

## Overview

The packing format allows efficient batch processing of variable-length image sequences by concatenating all patches into a single sequence. This is similar to the approach used in Qwen2VL and other efficient vision models.

## Files Modified/Created

1. **`vit_siglip2_packing_hf.py`** - Added `Siglip2NaflexPacking` class
2. **`align_siglip2_packing.py`** - Alignment verification script
3. **This README** - Documentation

## Architecture

### Standard Format (`Siglip2Naflex` in `vit_siglip2.py`)

**Input:** 
- `pixel_values`: `[batch_size, channels, height, width]`

**Processing:**
1. Converts images to patches internally
2. Adds positional embeddings
3. Processes through transformer layers

**Output:**
- `last_hidden_state`: `[batch_size, num_patches, hidden_size]`

### Packing Format (`Siglip2NaflexPacking` in `vit_siglip2_packing_hf.py`)

**Input:**
- `hidden_states`: `[total_num_patches, patch_dim]` - Pre-patchified input
  - `total_num_patches = sum(t_i × h_i × w_i)` for all images in batch
  - `patch_dim = patch_size × patch_size × num_channels`
- `grid_thw`: `[num_images, 3]` - Dimensions for each image `[t, h, w]`
  - `t`: temporal dimension (usually 1 for single images)
  - `h`: height in patches
  - `w`: width in patches

**Processing:**
1. Reshapes packed input to batch format with padding
2. Creates attention masks for valid patches
3. Processes through the same transformer layers
4. Converts output back to packed format

**Output:**
- `last_hidden_state`: `[total_num_patches, hidden_size]` - Packed output

## Key Implementation Details

### Input Conversion (Packing Format → Batch Format)

The packing model internally converts the packed input to batch format for compatibility with the existing Siglip2 architecture:

```python
# From: [total_num_patches, patch_dim]
# To:   [batch_size, max_num_patches, patch_dim]

# Calculate patches per image from grid_thw
patches_per_image = [t * h * w for (t, h, w) in grid_thw]
max_num_patches = max(patches_per_image)

# Create padded batch tensor
pixel_values = torch.zeros(num_images, max_num_patches, patch_dim)
attention_mask = torch.zeros(num_images, max_num_patches)

# Fill with actual patches
start_idx = 0
for i in range(num_images):
    num_patches = patches_per_image[i]
    pixel_values[i, :num_patches] = hidden_states[start_idx:start_idx + num_patches]
    attention_mask[i, :num_patches] = 1
    start_idx += num_patches
```

### Output Conversion (Batch Format → Packing Format)

After processing, the output is converted back to packing format:

```python
# From: [batch_size, max_num_patches, hidden_size]
# To:   [total_num_patches, hidden_size]

output_list = []
for i in range(num_images):
    num_patches = patches_per_image[i]
    output_list.append(last_hidden_state[i, :num_patches])

packed_output = torch.cat(output_list, dim=0)
```

## Usage

### Example 1: Using Siglip2NaflexPacking Directly

```python
import torch
from model_factory.vit_siglip2_packing_hf import Siglip2NaflexPacking

# Initialize model
model = Siglip2NaflexPacking(
    ckpt="google/siglip2-so400m-patch16-naflex",
    device="cuda"
)

# Prepare packing format input
# For a batch of 2 images of 224x224 with patch_size=16:
# - Image 1: 224x224 → 14x14 patches = 196 patches
# - Image 2: 224x224 → 14x14 patches = 196 patches
# Total: 392 patches

batch_size = 2
height = width = 224
patch_size = 16
channels = 3

h_patches = height // patch_size  # 14
w_patches = width // patch_size   # 14
num_patches_per_image = h_patches * w_patches  # 196
total_patches = batch_size * num_patches_per_image  # 392

# Create pre-patchified input
patch_dim = patch_size * patch_size * channels  # 768
hidden_states = torch.randn(total_patches, patch_dim).cuda()

# Create grid_thw
grid_thw = torch.tensor([
    [1, h_patches, w_patches],  # Image 1: t=1, h=14, w=14
    [1, h_patches, w_patches],  # Image 2: t=1, h=14, w=14
], dtype=torch.long).cuda()

# Forward pass
output = model(hidden_states, grid_thw)
print(f"Output shape: {output.shape}")  # [392, hidden_size]
```

### Example 2: Converting Standard Images to Packing Format

```python
from model_factory.vit_siglip2 import Siglip2Naflex
from model_factory.vit_siglip2_packing_hf import Siglip2NaflexPacking

def convert_to_packing_format(images, patch_size):
    """
    Convert standard images to packing format.
    
    Args:
        images: [batch_size, channels, height, width]
        patch_size: Size of each patch
        
    Returns:
        hidden_states: [total_num_patches, patch_dim]
        grid_thw: [batch_size, 3]
    """
    batch_size, channels, height, width = images.shape
    h_patches = height // patch_size
    w_patches = width // patch_size
    
    # Reshape to patches
    patches = images.reshape(
        batch_size, channels,
        h_patches, patch_size,
        w_patches, patch_size
    )
    patches = patches.permute(0, 2, 4, 3, 5, 1)
    patches = patches.reshape(
        batch_size,
        h_patches * w_patches,
        patch_size * patch_size * channels
    )
    
    # Concatenate all batches
    hidden_states = patches.reshape(-1, patch_size * patch_size * channels)
    
    # Create grid_thw
    grid_thw = torch.tensor(
        [[1, h_patches, w_patches]] * batch_size,
        dtype=torch.long,
        device=images.device
    )
    
    return hidden_states, grid_thw

# Standard input
images = torch.randn(2, 3, 224, 224).cuda()

# Convert to packing format
hidden_states, grid_thw = convert_to_packing_format(images, patch_size=16)

# Use packing model
packing_model = Siglip2NaflexPacking(ckpt="google/siglip2-so400m-patch16-naflex")
output = packing_model(hidden_states, grid_thw)
```

### Example 3: Alignment Verification

Use the provided alignment script to verify consistency between standard and packing models:

**Testing with Random Tensors (default):**
```bash
python align_siglip2_packing.py \
    --ckpt google/siglip2-so400m-patch16-naflex \
    --device cuda \
    --batch_size 2 \
    --image_size 224 \
    --threshold 0.99
```

**Testing with Real Images:**
```bash
# Test with real images from model_factory/images/ directory
python align_siglip2_packing.py \
    --ckpt google/siglip2-so400m-patch16-naflex \
    --device cuda \
    --use_real_images \
    --threshold 0.99
```

The script will:
1. Load both models
2. Create random test images OR load real images (1.jpg, 2.jpg)
3. Process with standard model
4. Convert to packing format and process with packing model
5. Compare outputs and report similarity metrics

When using `--use_real_images`:
- The script loads images from `model_factory/images/1.jpg` and `model_factory/images/2.jpg`
- Each image is tested individually with its native dimensions (resized to be divisible by patch size)
- Both images are also tested together in a batch (resized to a common size)
- If images don't exist, the script will generate random test images

Expected output:
```
Siglip2 Naflex Packing Alignment Script
================================================================================
...
Testing with Real Images
================================================================================

Testing image: 1.jpg
...
✅ PASS: 1.jpg (min cosine similarity > 0.99)

Testing image: 2.jpg
...
✅ PASS: 2.jpg (min cosine similarity > 0.99)

Testing with batched real images
...
✅ PASS: Batched images (min cosine similarity > 0.99)

Summary
================================================================================
✅ ALL TESTS PASSED: Models are aligned
```

## Advantages of Packing Format

1. **Memory Efficiency**: No padding required when all images have the same size
2. **Flexibility**: Can handle variable-size images in the same batch
3. **Compatibility**: Same weights as standard model, just different I/O format
4. **Performance**: Reduces memory overhead for batched inference

## Limitations

1. **Requires Pre-patchification**: Input must be converted to patches before passing to model
2. **Temporary Padding**: Internally pads to max_num_patches for batch processing
3. **Same Architecture**: Uses the same transformer layers, so computational complexity is similar

## Testing

To verify the implementation works correctly:

1. **Syntax Check**: 
   ```bash
   python -m py_compile vit_siglip2_packing_hf.py
   python -m py_compile align_siglip2_packing.py
   ```

2. **Import Test**:
   ```bash
   python test_imports.py
   ```

3. **Alignment Test** (requires model checkpoint):
   ```bash
   python align_siglip2_packing.py --ckpt <path_to_checkpoint>
   ```

## Integration with LLaVA-ViT

The packing format can be integrated with the rest of the LLaVA-ViT codebase by:

1. Using `Siglip2NaflexPacking` in place of `Siglip2Naflex` when packing format is desired
2. Converting inputs to packing format before passing to the model
3. Converting outputs back to standard format if needed

Example integration:
```python
# In vision tower or encoder
if use_packing:
    model = Siglip2NaflexPacking(ckpt=model_path)
    # Convert inputs
    hidden_states, grid_thw = convert_to_packing_format(images, patch_size)
    output = model(hidden_states, grid_thw)
    # output is in packing format [total_num_patches, hidden_size]
else:
    model = Siglip2Naflex(ckpt=model_path)
    output = model(images)
    # output is in batch format [batch_size, num_patches, hidden_size]
```

## Future Improvements

1. **Direct Packing Support**: Modify the underlying architecture to work directly with packed sequences without internal padding
2. **FlashAttention Integration**: Use FlashAttention's variable-length sequence support for better efficiency
3. **Benchmark Performance**: Compare memory usage and throughput between standard and packing formats

## References

- Siglip2 Model: https://huggingface.co/google/siglip2-so400m-patch16-naflex
- Qwen2VL Packing Format: Similar approach for efficient vision transformers
- LLaVA-ViT Preview V0 Packing: `vit_preview_v0_packing_hf.py` for reference
