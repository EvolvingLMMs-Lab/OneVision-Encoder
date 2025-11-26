# SpatialMergeProjector Usage Guide

## Overview

The `SpatialMergeProjector` performs 2x2 spatial token merging similar to Qwen2VL's merger layer. It reduces visual tokens by 4x while preserving spatial information.

## Basic Usage

### Method 1: Auto-inference (current default)

When H and W are not provided, the projector automatically infers them from the total number of patches:

```python
from llava.model.multimodal_projector.builder import SpatialMergeProjector

# Initialize projector
projector = SpatialMergeProjector(
    llm_dim=3584,      # LLM hidden dimension (e.g., Qwen2.5-7B)
    vit_dim=1024,      # Vision encoder hidden dimension
    spatial_merge_size=2
)

# Forward pass - auto-infers H and W
image_features = vision_tower(images)  # (B, N, C) where N = H * W
merged_features = projector(image_features)  # (B, N//4, llm_dim)
```

### Method 2: Explicit H and W (recommended for dynamic resolution)

When you know the exact height and width of the feature map, pass them explicitly:

```python
# Get features from vision tower
image_features = vision_tower(images)  # (B, N, C)

# Calculate H and W from your input
h = height // patch_size  # e.g., 448 // 14 = 32
w = width // patch_size   # e.g., 448 // 14 = 32

# Forward pass with explicit dimensions
merged_features = projector(image_features, height=h, width=w)
```

## Integration with LlavaViTModel

If you want to integrate the spatial merger directly into the vision model, here's how:

### Option A: Add as a module to LlavaViTModel

```python
# In vit_preview_v0_hf.py, add to __init__:
class LlavaViTModel(LlavaViTPreTrainedModel):
    def __init__(self, config: LlavaViTConfig):
        super().__init__(config)
        self.config = config
        
        self.embeddings = LlavaViTEmbeddings(config)
        self.layernorm_pre = get_norm_layer(config)
        self.encoder = LlavaViTEncoder(config)
        self.video_rope = VideoRotaryEmbeddingSplit466(config)
        
        # Add spatial merger if needed
        if getattr(config, 'use_spatial_merge', False):
            from llava.model.multimodal_projector.builder import SpatialMergeProjector
            self.spatial_merger = SpatialMergeProjector(
                llm_dim=config.hidden_size,
                vit_dim=config.hidden_size,
                spatial_merge_size=2
            )
        else:
            self.spatial_merger = None
        
        # ... rest of init

    def forward(self, pixel_values, ...):
        # ... existing code ...
        
        # After encoder
        sequence_output = encoder_outputs[0]
        
        # Apply spatial merger if enabled
        if self.spatial_merger is not None:
            h = height // self.config.patch_size
            w = width // self.config.patch_size
            sequence_output = self.spatial_merger(sequence_output, height=h, width=w)
        
        # ... rest of forward
```

### Option B: Use it in HEVCViTVisionTower wrapper

```python
# In hevc_vit_tower.py, modify forward method:
def forward(self, images):
    # Get features from vision tower
    image_forward_outs = self.vision_tower(
        images.to(device=self.device, dtype=self.dtype),
        output_hidden_states=True
    )
    image_features = self.feature_select(image_forward_outs).to(images.dtype)
    
    # Apply spatial merge if configured
    if hasattr(self, 'spatial_merger') and self.spatial_merger is not None:
        # Calculate h and w from image dimensions
        if images.dim() == 5:  # Video
            height, width = images.shape[3], images.shape[4]
        else:  # Image
            height, width = images.shape[2], images.shape[3]
        
        h = height // self.config.patch_size
        w = width // self.config.patch_size
        
        image_features = self.spatial_merger(image_features, height=h, width=w)
    
    return image_features
```

## Training Configuration

To use spatial merge in training, set the projector type:

```bash
--mm_projector_type spatial_merge
```

Or in the training script:
```bash
deepspeed llava/train/train_mem.py \
    --mm_projector_type spatial_merge \
    ...
```

### How h and w are passed during training

The training pipeline automatically handles passing h and w to the merger:

1. **In HEVCViTVisionTower**: When `return_spatial_dims=True`, the vision tower extracts h and w from the input images:
   ```python
   # Extract from image dimensions
   height, width = images.shape[-2:]  # Get H, W from input
   h = height // self.config.patch_size  # Convert to patch coordinates
   w = width // self.config.patch_size
   return image_features, h, w
   ```

2. **In encode_images (llava_arch.py)**: When `mm_projector_type == "spatial_merge"`, the encoder automatically requests and passes h and w:
   ```python
   if projector_type == "spatial_merge":
       image_features, h, w = vision_tower(images, return_spatial_dims=True)
       image_features = mm_projector(image_features, height=h, width=w)
   ```

3. **No manual intervention needed**: During training, you just set `--mm_projector_type spatial_merge` and the framework handles everything automatically.

## Examples with Different Resolutions

### Square resolutions
```python
# 448x448 image
h, w = 32, 32  # 448 // 14
features = torch.randn(1, 1024, 1024)  # (B, 32*32, C)
output = projector(features, height=h, width=w)
# Output: (1, 256, llm_dim)  # 16*16
```

### Non-square resolutions
```python
# 252x448 image
h, w = 18, 32  # 252 // 14, 448 // 14
features = torch.randn(1, 576, 1024)  # (B, 18*32, C)
output = projector(features, height=h, width=w)
# Output: (1, 144, llm_dim)  # 9*16

# 196x392 image
h, w = 14, 28  # 196 // 14, 392 // 14
features = torch.randn(1, 392, 1024)  # (B, 14*28, C)
output = projector(features, height=h, width=w)
# Output: (1, 98, llm_dim)  # 7*14
```

## Important Notes

1. **Height and Width must be divisible by 2**: Both h and w must be even numbers for 2x2 merging to work.

2. **Auto-inference prefers square-like factors**: When h and w are not provided, the algorithm chooses factors closest to square.

3. **Explicit dimensions are recommended for dynamic resolution**: If you have varying aspect ratios, always pass explicit h and w.

4. **Token count reduction**: The merger always reduces tokens by 4x (spatial_merge_size²).

## Troubleshooting

### Error: "Grid size (H, W) not divisible by merge_size 2"
- Ensure both H and W are even numbers
- Check that your image size divided by patch size results in even dimensions

### Error: "Cannot find valid H, W factors"
- This happens when auto-inference can't find valid factors
- Solution: Always pass explicit height and width parameters

### Wrong output shape
- Verify that input shape is (B, H*W, C)
- Check that h and w values match the actual feature map dimensions
