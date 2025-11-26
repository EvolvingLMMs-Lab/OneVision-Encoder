# SpatialMergeProjector Training Flow

This document explains how h and w are automatically passed to the spatial merger during training.

## Training Pipeline Flow

```
Input Images (B, C, H, W)
         |
         v
HEVCViTVisionTower.forward(images, return_spatial_dims=True)
         |
         ├─> Extract spatial dimensions:
         |   h = H // patch_size
         |   w = W // patch_size
         |
         ├─> Process through ViT encoder
         |   
         └─> Return: (image_features, h, w)
                      (B, h*w, vit_dim)
         |
         v
encode_images() [in llava_arch.py]
         |
         ├─> Check: mm_projector_type == "spatial_merge"?
         |   YES:
         |   ├─> Request spatial dims: features, h, w = vision_tower(images, return_spatial_dims=True)
         |   └─> Pass to projector: mm_projector(features, height=h, width=w)
         |   
         |   NO:
         |   └─> Standard flow: mm_projector(features)
         |
         v
SpatialMergeProjector.forward(x, height=h, width=w)
         |
         ├─> Reshape: (B, h*w, C) -> (B, h, w, C)
         |
         ├─> 2x2 Merge: (B, h, w, C) -> (B, h//2, w//2, 4*C)
         |
         └─> MLP Project: (B, h//2 * w//2, llm_dim)
         |
         v
Output: Merged features (B, N//4, llm_dim)
         |
         v
Continue to LLM...
```

## Key Code Sections

### 1. Vision Tower (hevc_vit_tower.py)

```python
def forward(self, images, return_spatial_dims=False):
    # Extract height and width from input images
    if images.ndim == 5:  # Video: (B, C, T, H, W)
        height, width = images.shape[-2:]
    else:  # Image: (B, C, H, W)
        height, width = images.shape[-2:]
    
    # Process through ViT
    image_features = self.vision_tower(images, ...)
    
    # Calculate patch coordinates
    h = height // self.config.patch_size
    w = width // self.config.patch_size
    
    if return_spatial_dims:
        return image_features, h, w
    return image_features
```

### 2. Encoder (llava_arch.py)

```python
def encode_images(self, images):
    projector_type = getattr(self.config, "mm_projector_type", "linear")
    vision_tower = self.get_model().get_vision_tower()
    
    if projector_type == "spatial_merge":
        # Request spatial dimensions
        image_features, h, w = vision_tower(images, return_spatial_dims=True)
        # Pass h and w to projector
        image_features = self.get_model().mm_projector(
            image_features, height=h, width=w
        )
    else:
        # Standard flow
        image_features = vision_tower(images)
        image_features = self.get_model().mm_projector(image_features)
    
    return image_features
```

### 3. Spatial Merge Projector (builder.py)

```python
def forward(self, x, *args, **kwargs):
    B, N, C = x.size()
    
    # Get h and w from kwargs (passed from encode_images)
    height = kwargs.get('height', None)
    width = kwargs.get('width', None)
    
    if height is None or width is None:
        # Fallback: auto-infer (not used during training)
        H, W = self._infer_hw(N)
    else:
        # Use provided dimensions (standard training path)
        H, W = height, width
    
    # Reshape and merge
    x = self.ln_q(x)
    x = x.view(B, H, W, C)
    
    # 2x2 spatial merge
    new_H = H // 2
    new_W = W // 2
    x = x.view(B, new_H, 2, new_W, 2, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(B, new_H * new_W, 4 * C)
    
    # Project to LLM dimension
    x = self.mlp(x)
    return x
```

## Configuration

In your training script (`dist_hevc_l_14_448_pretrain_2x2merge.sh`):

```bash
deepspeed --hostfile host_80 \
    llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path ${LLM_VERSION} \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type spatial_merge \  # <-- This is all you need!
    --mm_tunable_parts mm_mlp_adapter \
    ...
```

## Dynamic Resolution Support

The system automatically handles different image sizes:

| Input Size | h × w (patches) | After Merge |
|------------|-----------------|-------------|
| 448×448    | 32×32 = 1024   | 16×16 = 256 |
| 252×448    | 18×32 = 576    | 9×16 = 144  |
| 196×392    | 14×28 = 392    | 7×14 = 98   |

Each batch can have different resolutions because h and w are calculated per-batch from the actual image dimensions.

## Summary

**You don't need to manually pass h and w!** 

The training framework automatically:
1. ✅ Extracts h and w from input image dimensions
2. ✅ Passes them through the vision tower
3. ✅ Forwards them to the spatial merger
4. ✅ Handles dynamic resolutions per batch

Just set `--mm_projector_type spatial_merge` in your training script and everything works automatically.
