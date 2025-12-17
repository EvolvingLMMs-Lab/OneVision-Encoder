# AIMv2 Packing Implementation Changes

## Overview

This document describes the changes made to `vit_aim_v2_packing_hf.py` to meet all specified requirements.

## Requirements Met

### 1. ✅ Must use flash_attn_varlen_func

**Implementation:**
- Imported: `from flash_attn import flash_attn_varlen_func`
- Used in `Aimv2PackingAttention.forward()` method (lines 246-255)
- FlashAttention varlen is used for efficient variable-length sequence processing

**Code Reference:**
```python
# In Aimv2PackingAttention.forward()
attn_output = flash_attn_varlen_func(
    queries,
    keys,
    values,
    cu_seqlens_q=cu_seqlens,
    cu_seqlens_k=cu_seqlens,
    max_seqlen_q=max_seqlen,
    max_seqlen_k=max_seqlen,
    dropout_p=self.dropout if self.training else 0.0,
    softmax_scale=self.scale,
    causal=False,
)
```

### 2. ✅ Must use transformers absolute addresses

**Implementation:**
- Changed from relative imports to absolute transformers imports
- All imports use full module paths

**Code Reference:**
```python
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.aimv2.configuration_aimv2 import Aimv2VisionConfig
from transformers.models.aimv2.modeling_aimv2 import Aimv2VisionModel
```

**Before (incorrect):**
```python
from ... import initialization as init
from ...activations import ACT2FN
from .configuration_aimv2 import Aimv2Config
```

### 3. ✅ Input must be hidden_states: torch.Tensor, grid_thw: torch.Tensor

**Implementation:**
- Main forward method signature exactly matches requirement
- Input format documented in docstring

**Code Reference:**
```python
def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor):
    """
    Forward pass with pre-patchified input using FlashAttention varlen approach.
    
    Requirement #3: Input must be hidden_states: torch.Tensor, grid_thw: torch.Tensor

    Args:
        hidden_states (torch.Tensor): Pre-patchified input of shape
            [total_num_patches, patch_dim] where
            patch_dim = patch_size * patch_size * num_channels
        grid_thw (torch.Tensor): Grid dimensions of shape [num_images, 3]
            containing [t, h, w] for each image
    """
```

### 4. ✅ Position encoding can use for loops, but encoder forward cannot use for loops, must use cu_seqlens

**Implementation:**

**Position Encoding (allows for loops):**
- Located in `Aimv2VisionEmbeddings.forward()` (lines 183-200)
- Uses for loop to compute position embeddings for each image in batch
- Each image gets its own position embedding based on spatial dimensions

**Code Reference:**
```python
# In Aimv2VisionEmbeddings.forward()
# Position embeddings need to be computed for each image in the batch
for i in range(batch_size):
    height, width = spatial_shapes[i]
    num_patches = height * width
    
    if self.config.is_native:
        # Build sincos position embedding
        pos_embed = self.build_2d_sincos_position_embedding(...)
    else:
        # Use learned position embeddings
        pos_embed = self.position_embedding(pos_ids)
    
    # Add position embeddings to this image's patches
    hidden_states[i, :num_patches] = hidden_states[i, :num_patches] + pos_embed
```

**Encoder Forward (NO for loops over samples, uses cu_seqlens):**
- Located in `Aimv2PackingEncoder.forward()` (lines 307-325)
- Only loops over layers, NOT over samples/images
- Uses `cu_seqlens` for efficient batched processing with FlashAttention
- All images processed together in packed format

**Code Reference:**
```python
# In Aimv2PackingEncoder.forward()
def forward(
    self,
    inputs_embeds,
    cu_seqlens: torch.Tensor,
) -> BaseModelOutput:
    """
    Forward pass using cu_seqlens for efficient variable-length processing.
    No for loops over images - all processing is done in packed format.
    """
    hidden_states = inputs_embeds
    
    # Process all layers without for loops over samples
    for encoder_layer in self.layers:  # ← Only loops over layers!
        hidden_states = encoder_layer(
            hidden_states,
            cu_seqlens,  # ← Uses cu_seqlens, not loops
        )

    return BaseModelOutput(last_hidden_state=hidden_states)
```

### 5. ✅ Loading model weights must be same as original non-packing model

**Implementation:**
- Loads pretrained `Aimv2VisionModel` using `from_pretrained()`
- Copies all weights using `load_state_dict()`
- Maintains weight compatibility with original model

**Code Reference:**
```python
# In AIMv2Packing.__init__()
# Requirement #5: Load the vision model from pretrained checkpoint to get config and weights
vision_model = Aimv2VisionModel.from_pretrained(ckpt, trust_remote_code=True)
self.config = vision_model.config

# Copy embeddings weights (patch_embed and position_embedding)
self.embeddings.patch_embed.load_state_dict(vision_model.embeddings.patch_embed.state_dict())
self.embeddings.rms_norm.load_state_dict(vision_model.embeddings.rms_norm.state_dict())
if not self.config.is_native:
    self.embeddings.position_embedding.load_state_dict(vision_model.embeddings.position_embedding.state_dict())

# Copy encoder weights (need to map standard attention to packing attention)
for packing_layer, standard_layer in zip(self.encoder.layers, vision_model.encoder.layers):
    # Copy RMS norms
    packing_layer.rms_norm1.load_state_dict(standard_layer.rms_norm1.state_dict())
    packing_layer.rms_norm2.load_state_dict(standard_layer.rms_norm2.state_dict())

    # Copy attention projections
    packing_layer.attention.q_proj.load_state_dict(standard_layer.attention.q_proj.state_dict())
    packing_layer.attention.k_proj.load_state_dict(standard_layer.attention.k_proj.state_dict())
    packing_layer.attention.v_proj.load_state_dict(standard_layer.attention.v_proj.state_dict())
    packing_layer.attention.out_proj.load_state_dict(standard_layer.attention.out_proj.state_dict())

    # Copy MLP (FFN)
    packing_layer.ffn.load_state_dict(standard_layer.ffn.state_dict())

# Copy post RMS norm
self.rms_norm.load_state_dict(vision_model.rms_norm.state_dict())
```

## Architecture Overview

### Classes Implemented

1. **Aimv2RMSNorm** - RMS normalization layer
2. **Aimv2MLP** - MLP/FFN layer with gated activation
3. **Aimv2VisionEmbeddings** - Patch embedding with position encoding (supports for loops)
4. **Aimv2PackingAttention** - Multi-head attention using flash_attn_varlen_func
5. **Aimv2PackingEncoderLayer** - Single transformer layer with pre-norm
6. **Aimv2PackingEncoder** - Full encoder stack (uses cu_seqlens, no for loops)
7. **AIMv2Packing** - Main model class

### Data Flow

```
Input: [total_num_patches, patch_dim] + grid_thw: [num_images, 3]
  ↓
Reshape to batched format for embeddings
  ↓
Aimv2VisionEmbeddings (with position encoding via for loops)
  ↓
Convert back to packed format
  ↓
Compute cu_seqlens from grid_thw
  ↓
Aimv2PackingEncoder (uses cu_seqlens, FlashAttention varlen)
  ↓
RMS Norm
  ↓
Output: [total_num_patches, hidden_size]
```

## Key Differences from Original

1. **Attention Mechanism:**
   - Original: Uses standard attention with explicit attention masks
   - New: Uses `flash_attn_varlen_func` with `cu_seqlens` (no masks needed)

2. **Imports:**
   - Original: Relative imports from transformers internals
   - New: Absolute imports from public transformers API

3. **Input Format:**
   - Original: Batch of images [B, C, H, W]
   - New: Packed patches [total_patches, patch_dim] + grid_thw

4. **Processing:**
   - Original: Processes each image separately in batch
   - New: Processes all images together in packed format using cu_seqlens

5. **Weight Compatibility:**
   - Both load from same pretrained checkpoint
   - New model copies weights to maintain exact equivalence

## Testing

To verify the implementation produces identical outputs to the original model:

```bash
python model_factory/align_aim_v2_packing.py \
    --ckpt apple/aimv2-large-patch14-224 \
    --device cuda
```

This will:
1. Load both standard AIMv2 and AIMv2Packing models
2. Process test images through both
3. Compare outputs using cosine similarity
4. Report pass/fail (threshold: 0.99)

## Dependencies

- `torch`
- `transformers` (with aimv2 support)
- `flash-attn` (required)

Install FlashAttention:
```bash
pip install flash-attn --no-build-isolation
```

## References

- Reference implementation: `vit_siglip2_packing_hf.py`
- Reference implementation: `vit_preview_v0_packing_hf.py`
- Alignment script: `align_aim_v2_packing.py`
- Documentation: `PACKING_IMPLEMENTATION_DINOV3_AIMV2.md`
