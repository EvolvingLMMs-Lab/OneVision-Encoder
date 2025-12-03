# Weight Conversion Tool: vit_preview_v0_hf → vit_preview_v0_packing_hf

## Overview

This tool converts model weights from the HuggingFace ViT format (`vit_preview_v0_hf.py`) to the packing model format (`vit_preview_v0_packing_hf.py`). The packing model uses a Qwen2VL-style input format with `grid_thw` for efficient variable-length sequence processing.

## Files

- **`convert_vit_preview_v0_hf_to_packing.py`**: Main conversion script
- **`test_convert_vit_preview_v0_hf_to_packing.py`**: Unit tests for the conversion tool

## Features

### Weight Conversion
- Converts embeddings: `embeddings.patch_embedding` → `patch_embed.proj`
- Combines attention projections: Q/K/V → QKV
- Remaps RoPE parameters: `video_rope` → `rotary_emb`
- Preserves layer normalization and MLP weights

### Comprehensive Testing
The tool includes five types of verification tests:

1. **Single Image Test**: Verifies consistency with a single 448×448 image
2. **Video Test**: Tests with 8-frame video sequences (224×224 per frame)
3. **Mixed Video+Image Test**: Validates combined processing of video (224×224) and image (448×448)
4. **Multi-Sample Test**: Tests batch processing with:
   - 3 images at different resolutions (224, 336, 1008)
   - 2 videos at different resolutions (378, 518) with 8 frames each
5. **Reload Test**: Verifies that saved models can be loaded and produce consistent results

### Additional Features
- **CLIP Image Processor**: Automatically saves CLIP image processor configuration
- **bfloat16 Support**: Handles bfloat16 precision for memory efficiency
- **CUDA Acceleration**: Automatically detects and uses CUDA if available
- **Real Image Testing**: Downloads a real COCO image for verification (fallback to synthetic data if unavailable)

## Usage

### Basic Usage

```bash
python convert_vit_preview_v0_hf_to_packing.py \
    <model_name> \
    <weight_path> \
    [--target_model_name <name>] \
    [--output_dir <dir>]
```

### Arguments

- **`model_name`** (required): Source HF model name (e.g., `hf_llava_vit_huge_ln`)
- **`weight_path`** (required): Path to the `.pth` checkpoint file
- **`--target_model_name`** (optional): Target packing model name (auto-generated if not provided)
- **`--output_dir`** (optional): Output directory for saved model (auto-generated if not provided)

### Examples

#### Convert a Huge Model
```bash
python convert_vit_preview_v0_hf_to_packing.py \
    hf_llava_vit_huge_ln \
    /path/to/huge_model.pth \
    --output_dir ./huge_packing
```

#### Convert with Custom Target Name
```bash
python convert_vit_preview_v0_hf_to_packing.py \
    hf_llava_vit_base_ln \
    /path/to/base_model.pth \
    --target_model_name hf_llava_vit_packing_base_ln \
    --output_dir ./base_packing
```

## Weight Remapping Details

### Embeddings
```
HF:      embeddings.patch_embedding.{weight,bias}
Packing: patch_embed.proj.{weight,bias}
```

### Attention Projections
```
HF:      encoder.layers.N.self_attn.{q_proj,k_proj,v_proj}.{weight,bias}
Packing: encoder.layers.N.self_attn.qkv.{weight,bias}
         (Q, K, V concatenated along dimension 0)
```

### Attention Output
```
HF:      encoder.layers.N.self_attn.out_proj.{weight,bias}
Packing: encoder.layers.N.self_attn.proj.{weight,bias}
```

### RoPE Parameters
```
HF:      video_rope.{inv_freq_t,inv_freq_h,inv_freq_w}
Packing: rotary_emb.{inv_freq_t,inv_freq_h,inv_freq_w}
```

### Other Layers
- Layer normalization: Names unchanged
- MLP layers: Names unchanged
- Head (if present): Names unchanged

## Input Format Differences

### HF Model (vit_preview_v0_hf.py)
- Input: `pixel_values` of shape `(B, C, H, W)` or `(B, C, T, H, W)`
- Optional: `visible_indices` for masking

### Packing Model (vit_preview_v0_packing_hf.py)
- Input: `hidden_states` of shape `(seq_len, patch_dim)`
  - `seq_len = sum(t*h*w for all images/videos in batch)`
  - `patch_dim = patch_size * patch_size * in_channels`
- Required: `grid_thw` of shape `(num_samples, 3)` containing `[t, h, w]` for each sample
- Optional: `patch_positions` of shape `(seq_len, 3)` for explicit RoPE positions

## Verification Process

The conversion tool automatically runs all five verification tests:

1. **Single Image Test**
   - Processes a 448×448 image
   - Compares HF and packing model outputs
   - Success criterion: Cosine similarity > 0.99

2. **Video Test**
   - Creates an 8-frame video (224×224 per frame)
   - Uses interpolated frame indices for 64-frame context
   - Compares outputs with temporal RoPE positions

3. **Mixed Video+Image Test**
   - Processes video (8 frames, 224×224) and image (448×448) together
   - Tests packing model's ability to handle mixed inputs
   - Compares with separate HF model forward passes

4. **Multi-Sample Test**
   - Processes 5 samples simultaneously:
     - Images: 224×224, 336×336, 1008×1008
     - Videos: 378×378 (8 frames), 518×518 (8 frames)
   - Verifies batch packing functionality

5. **Reload Test**
   - Saves the converted model
   - Reloads it using `from_pretrained`
   - Verifies consistency with original conversion

## Output Structure

After successful conversion, the output directory contains:

```
output_dir/
├── config.json                 # Model configuration
├── model.safetensors          # Model weights (bfloat16)
└── preprocessor_config.json   # CLIP image processor config
```

## Testing

Run the unit tests to verify the conversion tool:

```bash
python test_convert_vit_preview_v0_hf_to_packing.py
```

Expected output:
```
================================================================================
TESTING WEIGHT CONVERSION TOOL
================================================================================

... (16 tests run)

----------------------------------------------------------------------
Ran 16 tests in 0.003s

OK
```

## Requirements

### Python Packages
- `torch` (with CUDA support recommended)
- `transformers`
- `timm`
- `PIL` / `pillow`
- `requests`
- `torchvision`

### Hardware
- **Recommended**: GPU with CUDA support (for FlashAttention 2)
- **Minimum**: CPU (slower, FlashAttention tests may fail)

## Troubleshooting

### "Flash Attention not available"
The packing model requires FlashAttention 2. Install it with:
```bash
pip install flash-attn --no-build-isolation
```

### "Shape mismatch" warnings
Minor shape mismatches are logged but don't necessarily indicate errors. The tool compares outputs up to the minimum length and still validates consistency.

### "Failed to download image"
If the COCO image download fails, the tool automatically falls back to synthetic noise. This doesn't affect the validity of the conversion tests.

### CUDA out of memory
If you encounter OOM errors during testing:
1. The conversion itself doesn't require GPU
2. Tests can run on CPU (slower but functional)
3. Consider using smaller test resolutions

## Reference

This conversion tool is based on the structure and testing methodology of:
- `convert_llava_vit_packing_to_hf.py` (reverse direction)
- `convert_llava_vit_to_hf.py` (non-packing conversion)

## Model Architecture Differences

### HF Model (vit_preview_v0_hf.py)
- Uses standard attention with separate Q, K, V projections
- Processes images/videos with optional visible indices
- RoPE applied per-patch based on spatial-temporal grid

### Packing Model (vit_preview_v0_packing_hf.py)
- Uses combined QKV projection for efficiency
- Designed for variable-length packing (Qwen2VL style)
- Requires `grid_thw` to specify sample boundaries
- Supports explicit `patch_positions` for custom RoPE calculation
- Uses `flash_attn_varlen_func` for efficient attention

## License

Copyright 2025 The HuggingFace Inc. team.

Licensed under the Apache License, Version 2.0.
