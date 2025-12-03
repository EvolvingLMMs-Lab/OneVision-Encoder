# SigLIP2 so400m-patch16-naflex Model

This document describes the usage of the newly added SigLIP2 so400m-patch16-naflex model in the LLaVA-ViT repository.

## Overview

The `siglip2_so400m_patch16_naflex` is a SigLIP2 vision transformer model with:
- **400M parameters** (so400m = smaller optimized 400M)
- **16x16 patch size**
- **Native resolution flexibility** (naflex) - supports variable input resolutions

This model is based on Google's SigLIP2 architecture and is registered with the `timm` library for easy integration.

## Model Source

- **HuggingFace Model Hub**: `google/siglip2-so400m-patch16-naflex`
- **Official Repository**: https://huggingface.co/google/siglip2-so400m-patch16-naflex
- **Default Local Path**: `/video_vit/pretrain_models/siglip2-so400m-patch16-naflex`

Note: The model defaults to a local checkpoint path. If you want to use the HuggingFace model directly, pass the `ckpt` parameter when creating the model.

## Usage

### Basic Usage with timm

```python
import timm
import torch

# Create the model using timm framework
model = timm.create_model("siglip2_so400m_patch16_naflex", pretrained=False)

# Create test input: [batch_size, 3, height, width]
batch_size = 4
test_input = torch.randn(batch_size, 3, 224, 224).cuda()

# Get the last hidden state
last_hidden_state = model(test_input)

print(f"Input shape: {test_input.shape}")
print(f"Last hidden state shape: {last_hidden_state.shape}")
```

### Custom Checkpoint Path

You can specify a custom checkpoint path (e.g., to use HuggingFace directly):

```python
import timm

# Use HuggingFace checkpoint directly
model = timm.create_model(
    "siglip2_so400m_patch16_naflex",
    pretrained=False,
    ckpt="google/siglip2-so400m-patch16-naflex",
    device="cuda"
)

# Or use a different local checkpoint path
model = timm.create_model(
    "siglip2_so400m_patch16_naflex",
    pretrained=False,
    ckpt="/path/to/your/local/checkpoint",
    device="cuda"
)
```

### Using CPU

For testing without GPU:

```python
import timm

model = timm.create_model(
    "siglip2_so400m_patch16_naflex",
    pretrained=False,
    device="cpu"
)
```

## Model Architecture

The model uses the `Siglip2Base` class which:
1. Loads the pre-trained model from HuggingFace
2. Uses only the vision model component (`.vision_model`)
3. Extracts the last hidden state from the model
4. Operates with `torch.no_grad()` for inference

## Key Features

- **Variable Resolution**: The naflex variant supports native resolution flexibility, allowing it to handle different input sizes efficiently
- **Efficient Design**: The so400m variant is optimized for efficiency while maintaining strong performance
- **Pre-trained Weights**: Loads from local checkpoint by default: `/video_vit/pretrain_models/siglip2-so400m-patch16-naflex`
- **Flexible Checkpoint**: Can be configured to load from HuggingFace or other custom paths

## Integration with LLaVA-ViT

This model is registered in the `model_factory/vit_siglip2.py` file and can be used throughout the LLaVA-ViT framework for:
- Vision encoding in multimodal models
- Feature extraction for downstream tasks
- Training and evaluation pipelines

## Related Files

- **Model Registration**: `model_factory/vit_siglip2.py`
- **SigLIP2 Naflex Implementation**: `llava_next/llava/model/multimodal_encoder/siglip2_naflex.py`

## Example Training Script

```python
import timm
from model_factory import vit_siglip2

# The model is automatically registered when importing the module
model = timm.create_model('siglip2_so400m_patch16_naflex', pretrained=False)

# Use in your training pipeline
for batch in dataloader:
    features = model(batch['images'])
    # Your training logic here
```

## Notes

- The model defaults to using the local checkpoint `/video_vit/pretrain_models/siglip2-so400m-patch16-naflex`
- To use HuggingFace checkpoint, pass `ckpt="google/siglip2-so400m-patch16-naflex"` when creating the model
- Set `device="cpu"` for CPU-only environments
- The model runs in evaluation mode with no gradient computation by default
- For custom local checkpoints, pass `ckpt="/path/to/checkpoint"` when creating the model
