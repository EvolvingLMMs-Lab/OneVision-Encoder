#!/usr/bin/env python3
# coding=utf-8
"""
Script to upload LlavaViT model to HuggingFace Hub with AutoModel support.

This script:
1. Loads a pre-trained LlavaViT model
2. Configures it for AutoModel compatibility
3. Uploads to HuggingFace Hub with proper auto_map configuration
4. Creates model cards and documentation

Usage:
    python upload_llava_vit_to_hf.py \
        --model_name hf_llava_vit_large_ln \
        --weight_path /path/to/checkpoint.pth \
        --repo_id your-username/llava-vit-large \
        --token YOUR_HF_TOKEN

After upload, you can load the model with:
    from transformers import AutoModel
    model = AutoModel.from_pretrained("your-username/llava-vit-large", trust_remote_code=True)
"""

import argparse
import os
import sys
from pathlib import Path
import torch
import timm

# Import the model definitions
try:
    from model_factory.vit_preview_v0_hf import (
        LlavaViTConfig,
        LlavaViTModel,
        LlavaViTPreTrainedModel
    )
    from model_factory.conversion_utils import (
        get_real_coco_image,
        CLIP_MEAN,
        CLIP_STD,
    )
except ImportError:
    # If running directly, add parent to path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from model_factory.vit_preview_v0_hf import (
        LlavaViTConfig,
        LlavaViTModel,
        LlavaViTPreTrainedModel
    )
    try:
        from model_factory.conversion_utils import (
            get_real_coco_image,
            CLIP_MEAN,
            CLIP_STD,
        )
    except ImportError:
        print("[Warning] conversion_utils not found, using defaults")
        CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
        CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def update_config_for_automodel(config: LlavaViTConfig, repo_id: str) -> LlavaViTConfig:
    """
    Update the config to include auto_map for AutoModel support.
    
    Args:
        config: The model configuration
        repo_id: HuggingFace repository ID (e.g., "username/model-name")
        
    Returns:
        Updated configuration with auto_map
    """
    # Set the auto_map to enable AutoModel.from_pretrained()
    config.auto_map = {
        "AutoConfig": "configuration_llava_vit.LlavaViTConfig",
        "AutoModel": "modeling_llava_vit.LlavaViTModel",
        "AutoModelForImageClassification": "modeling_llava_vit.LlavaViTModel",
    }
    
    # Ensure model_type is set
    if not hasattr(config, 'model_type') or config.model_type is None:
        config.model_type = "llava_vit"
    
    return config


def create_model_card(repo_id: str, model_name: str, config: LlavaViTConfig) -> str:
    """
    Create a comprehensive model card for the HuggingFace Hub.
    
    Args:
        repo_id: HuggingFace repository ID
        model_name: Name of the model variant
        config: Model configuration
        
    Returns:
        Model card content as string
    """
    card = f"""---
language: en
license: apache-2.0
tags:
- vision
- image-classification
- video-understanding
- vision-transformer
- pytorch
library_name: transformers
---

# {repo_id}

## Model Description

This is a LlavaViT (Llava Vision Transformer) model trained for visual understanding tasks. The model can process both images and videos and produces rich visual embeddings.

### Model Architecture

- **Model Type**: Vision Transformer (ViT)
- **Hidden Size**: {config.hidden_size}
- **Number of Layers**: {config.num_hidden_layers}
- **Number of Attention Heads**: {config.num_attention_heads}
- **Patch Size**: {config.patch_size}
- **Image Size**: {config.image_size}
- **Intermediate Size**: {config.intermediate_size}
- **Layer Norm Type**: {config.layer_norm_type}

## Usage

### Basic Usage with AutoModel

```python
from transformers import AutoModel, CLIPImageProcessor
import torch
from PIL import Image
import requests

# Load model and processor
model = AutoModel.from_pretrained("{repo_id}", trust_remote_code=True)
processor = CLIPImageProcessor.from_pretrained("{repo_id}")

# Load and process image
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
inputs = processor(images=image, return_tensors="pt")

# Get embeddings
with torch.no_grad():
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state  # [batch_size, num_patches, hidden_size]
    pooled_output = outputs.pooler_output    # [batch_size, hidden_size]

print(f"Embeddings shape: {{embeddings.shape}}")
print(f"Pooled output shape: {{pooled_output.shape}}")
```

### Video Input

The model also supports video inputs (5D tensors):

```python
import torch

# Video input: [batch_size, channels, num_frames, height, width]
video_input = torch.randn(1, 3, 8, {config.image_size}, {config.image_size})

with torch.no_grad():
    outputs = model(pixel_values=video_input)
    video_embeddings = outputs.last_hidden_state
    
print(f"Video embeddings shape: {{video_embeddings.shape}}")
```

### Advanced Usage with Visible Indices

For efficient processing with masking:

```python
import torch

# Process with visible patch indices (for MAE-style masking)
pixel_values = torch.randn(1, 3, {config.image_size}, {config.image_size})
num_patches = ({config.image_size} // {config.patch_size}) ** 2

# Use only first 75% of patches
visible_indices = torch.arange(int(num_patches * 0.75)).unsqueeze(0)

with torch.no_grad():
    outputs = model(pixel_values=pixel_values, visible_indices=visible_indices)
    embeddings = outputs.last_hidden_state
```

## Model Details

### Input Specifications

- **Image Input**: RGB images, recommended size {config.image_size}x{config.image_size}
- **Video Input**: RGB videos, shape (B, C, T, H, W) where T is number of frames
- **Normalization**: CLIP normalization (mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])

### Output Specifications

The model returns `BaseModelOutputWithPooling` containing:
- `last_hidden_state`: Token embeddings of shape (batch_size, sequence_length, hidden_size)
- `pooler_output`: Pooled representation of shape (batch_size, hidden_size) using multi-head attention pooling
- `hidden_states`: (optional) Hidden states from all layers
- `attentions`: (optional) Attention weights from all layers

### Features

- ✅ **3D Rotary Position Embeddings (RoPE)**: Supports spatial-temporal understanding with 4:6:6 split for T, H, W dimensions
- ✅ **Flash Attention 2**: Optimized attention implementation for better performance
- ✅ **Multi-head Attention Pooling**: PMA-style pooling for rich representation
- ✅ **Flexible Input**: Supports both 2D (images) and 3D (videos) inputs
- ✅ **Masking Support**: Compatible with MAE-style visible indices for efficient training/inference

## Training

This model was trained on large-scale vision datasets with contrastive learning and masked autoencoding objectives.

## Limitations

- Requires `flash_attn` package for optimal performance
- Best performance with bfloat16 or float16 precision
- Designed for vision understanding tasks

## Citation

If you use this model, please cite:

```bibtex
@misc{{llava-vit,
  title={{LlavaViT: Vision Transformer for Multimodal Understanding}},
  author={{Your Name}},
  year={{2025}},
  url={{https://huggingface.co/{repo_id}}}
}}
```

## License

Apache 2.0
"""
    return card


def create_configuration_file(output_dir: str):
    """
    Create a standalone configuration file for the model.
    
    Args:
        output_dir: Directory to save the configuration file
    """
    config_code = '''# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Llava ViT model configuration"""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class LlavaViTConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`LlavaViTModel`]. It is used to instantiate a
    Llava ViT model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 16):
            The size (resolution) of each patch.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler.
        layer_norm_eps (`float`, *optional*, defaults to 1e-6):
            The epsilon used by the layer normalization layers.
        layer_norm_type (`str`, *optional*, defaults to `"layer_norm"`):
            The type of layer normalization to use. Supported values: `"layer_norm"`, `"rms_norm"`.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        use_head (`bool`, *optional*, defaults to `True`):
            Whether to use the pooling head.

    Example:

    ```python
    >>> from transformers import AutoModel, AutoConfig
    >>> from configuration_llava_vit import LlavaViTConfig

    >>> # Initializing a LlavaViT configuration
    >>> configuration = LlavaViTConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = AutoModel.from_pretrained("your-repo/llava-vit", trust_remote_code=True)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "llava_vit"

    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        image_size=448,
        patch_size=16,
        hidden_act="gelu",
        layer_norm_eps=1e-6,
        layer_norm_type="layer_norm",
        attention_dropout=0.0,
        initializer_range=0.02,
        rope_theta=10000.0,
        use_head=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.layer_norm_type = layer_norm_type
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.rope_theta = rope_theta
        self.use_head = use_head
'''
    
    config_path = os.path.join(output_dir, "configuration_llava_vit.py")
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(config_code)
    print(f"    ✅ Created configuration file: {config_path}")


def create_modeling_file(output_dir: str):
    """
    Copy the modeling file to the output directory.
    
    Args:
        output_dir: Directory to save the modeling file
    """
    # Get the path to vit_preview_v0_hf.py
    script_dir = os.path.dirname(os.path.abspath(__file__))
    source_file = os.path.join(script_dir, "vit_preview_v0_hf.py")
    target_file = os.path.join(output_dir, "modeling_llava_vit.py")
    
    if not os.path.exists(source_file):
        print(f"    ⚠️  Warning: Source modeling file not found at {source_file}")
        print(f"    Please manually copy vit_preview_v0_hf.py to {target_file}")
        return False
    
    # Read and modify the file
    with open(source_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Update imports to use the new configuration file
    content = content.replace(
        "from transformers.configuration_utils import PretrainedConfig",
        "from .configuration_llava_vit import LlavaViTConfig"
    )
    
    # Remove the LlavaViTConfig class definition since it's in configuration_llava_vit.py
    # Find the start and end of the config class
    config_start = content.find("class LlavaViTConfig(PretrainedConfig):")
    if config_start != -1:
        # Find the next class definition
        next_class = content.find("\nclass ", config_start + 1)
        if next_class != -1:
            # Remove the config class and its docstring
            # Keep everything before and after
            content = content[:config_start] + content[next_class + 1:]
    
    with open(target_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"    ✅ Created modeling file: {target_file}")
    return True


def upload_to_hub(
    model: LlavaViTModel,
    repo_id: str,
    token: str,
    model_name: str,
    private: bool = False,
    commit_message: str = "Upload LlavaViT model"
):
    """
    Upload model to HuggingFace Hub with AutoModel support.
    
    Args:
        model: The model to upload
        repo_id: HuggingFace repository ID (e.g., "username/model-name")
        token: HuggingFace API token
        model_name: Name of the model variant
        private: Whether to make the repository private
        commit_message: Commit message for the upload
    """
    try:
        from huggingface_hub import HfApi, create_repo
        from transformers import CLIPImageProcessor
    except ImportError:
        print("❌ Error: Please install required packages:")
        print("   pip install huggingface_hub transformers")
        sys.exit(1)
    
    # Create temporary directory for files
    output_dir = f"/tmp/llava_vit_upload_{model_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n=== Preparing Upload to {repo_id} ===")
    
    # Update config for AutoModel
    config = model.config
    config = update_config_for_automodel(config, repo_id)
    
    # Save model and config
    print("  → Saving model and configuration...")
    model.save_pretrained(output_dir)
    
    # Create and save image processor
    print("  → Creating image processor...")
    image_processor = CLIPImageProcessor(
        size={"height": config.image_size, "width": config.image_size},
        image_mean=CLIP_MEAN,
        image_std=CLIP_STD,
        do_resize=True,
        do_center_crop=True,
        do_normalize=True,
    )
    image_processor.save_pretrained(output_dir)
    
    # Create configuration and modeling files
    print("  → Creating standalone configuration and modeling files...")
    create_configuration_file(output_dir)
    create_modeling_file(output_dir)
    
    # Create model card
    print("  → Creating model card...")
    model_card = create_model_card(repo_id, model_name, config)
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(model_card)
    
    # Create example usage file
    print("  → Creating example usage file...")
    example_code = f'''"""
Example usage of {repo_id}
"""

from transformers import AutoModel, CLIPImageProcessor
import torch
from PIL import Image
import requests

def main():
    # Load model and processor
    print("Loading model...")
    model = AutoModel.from_pretrained("{repo_id}", trust_remote_code=True)
    processor = CLIPImageProcessor.from_pretrained("{repo_id}")
    
    # Load sample image
    print("Loading image...")
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    
    # Process image
    inputs = processor(images=image, return_tensors="pt")
    
    # Get embeddings
    print("Running inference...")
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state
        pooled_output = outputs.pooler_output
    
    print(f"✅ Success!")
    print(f"   Embeddings shape: {{embeddings.shape}}")
    print(f"   Pooled output shape: {{pooled_output.shape}}")

if __name__ == "__main__":
    main()
'''
    
    example_path = os.path.join(output_dir, "example_usage.py")
    with open(example_path, 'w', encoding='utf-8') as f:
        f.write(example_code)
    
    # Create or get repository
    print(f"\n  → Creating repository: {repo_id}")
    api = HfApi(token=token)
    
    try:
        create_repo(repo_id, private=private, token=token, exist_ok=True)
        print(f"    ✅ Repository created/verified")
    except Exception as e:
        print(f"    ⚠️  Repository creation issue: {e}")
    
    # Upload all files
    print(f"\n  → Uploading files to {repo_id}...")
    try:
        api.upload_folder(
            folder_path=output_dir,
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_message,
            token=token,
        )
        print(f"    ✅ Upload complete!")
    except Exception as e:
        print(f"    ❌ Upload failed: {e}")
        print(f"\n    Files are saved locally at: {output_dir}")
        print(f"    You can manually upload them to https://huggingface.co/{repo_id}")
        return False
    
    print(f"\n✅ Model successfully uploaded to: https://huggingface.co/{repo_id}")
    print(f"\n📝 You can now load it with:")
    print(f'    from transformers import AutoModel')
    print(f'    model = AutoModel.from_pretrained("{repo_id}", trust_remote_code=True)')
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Upload LlavaViT model to HuggingFace Hub with AutoModel support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload a pre-trained model
  python upload_llava_vit_to_hf.py \\
      --model_name hf_llava_vit_large_ln \\
      --weight_path /path/to/checkpoint.pth \\
      --repo_id username/llava-vit-large \\
      --token YOUR_HF_TOKEN

  # Upload a model without weights (random initialization)
  python upload_llava_vit_to_hf.py \\
      --model_name hf_llava_vit_base_ln \\
      --repo_id username/llava-vit-base \\
      --token YOUR_HF_TOKEN

  # Create a private repository
  python upload_llava_vit_to_hf.py \\
      --model_name hf_llava_vit_huge_ln \\
      --weight_path /path/to/checkpoint.pth \\
      --repo_id username/llava-vit-huge \\
      --token YOUR_HF_TOKEN \\
      --private
        """
    )
    
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        choices=[
            "hf_llava_vit_small_ln",
            "hf_llava_vit_base_ln", 
            "hf_llava_vit_large_ln",
            "hf_llava_vit_huge_ln",
            "hf_llava_vit_giant_ln"
        ],
        help="Model architecture to use (must be registered in vit_preview_v0_hf.py)"
    )
    
    parser.add_argument(
        "--weight_path",
        type=str,
        default=None,
        help="Path to model checkpoint (.pth file). If not provided, uploads model with random weights."
    )
    
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help='HuggingFace repository ID (format: "username/model-name")'
    )
    
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace API token (or set HF_TOKEN environment variable)"
    )
    
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the repository private"
    )
    
    parser.add_argument(
        "--commit_message",
        type=str,
        default="Upload LlavaViT model with AutoModel support",
        help="Commit message for the upload"
    )
    
    args = parser.parse_args()
    
    # Get token from args or environment
    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        print("❌ Error: HuggingFace token required!")
        print("   Provide via --token argument or HF_TOKEN environment variable")
        print("   Get your token at: https://huggingface.co/settings/tokens")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"  LlavaViT HuggingFace Upload Tool")
    print(f"{'='*60}")
    print(f"  Model:      {args.model_name}")
    print(f"  Weights:    {args.weight_path or 'Random initialization'}")
    print(f"  Repository: {args.repo_id}")
    print(f"  Private:    {args.private}")
    print(f"{'='*60}\n")
    
    # Create model
    print(f"📦 Creating model: {args.model_name}")
    try:
        model = timm.create_model(args.model_name, pretrained=False)
        print(f"   ✅ Model created successfully")
    except Exception as e:
        print(f"   ❌ Failed to create model: {e}")
        print(f"\n   Make sure the model is registered in vit_preview_v0_hf.py")
        sys.exit(1)
    
    # Load weights if provided
    if args.weight_path:
        if not os.path.exists(args.weight_path):
            print(f"❌ Error: Weight file not found: {args.weight_path}")
            sys.exit(1)
        
        print(f"\n📥 Loading weights from: {args.weight_path}")
        try:
            checkpoint = torch.load(args.weight_path, map_location='cpu')
            state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
            
            # Try to load state dict
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            
            # Filter out non-critical missing keys
            real_missing = [k for k in missing if "attn.bias" not in k]
            
            if len(real_missing) > 0:
                print(f"   ⚠️  Warning: {len(real_missing)} missing keys")
                for k in real_missing[:5]:
                    print(f"      - {k}")
                if len(real_missing) > 5:
                    print(f"      ... and {len(real_missing) - 5} more")
            
            if len(unexpected) > 0:
                print(f"   ⚠️  Warning: {len(unexpected)} unexpected keys")
            
            print(f"   ✅ Weights loaded")
            
        except Exception as e:
            print(f"   ❌ Failed to load weights: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        print(f"\n⚠️  No weights provided - using random initialization")
        print(f"   This is fine for testing, but you'll want to upload trained weights later")
    
    # Upload to hub
    success = upload_to_hub(
        model=model,
        repo_id=args.repo_id,
        token=token,
        model_name=args.model_name,
        private=args.private,
        commit_message=args.commit_message
    )
    
    if success:
        print(f"\n{'='*60}")
        print(f"  🎉 Upload Complete!")
        print(f"{'='*60}")
        print(f"\n  View your model at:")
        print(f"  https://huggingface.co/{args.repo_id}")
        print(f"\n  Load it with:")
        print(f"  ```python")
        print(f"  from transformers import AutoModel")
        print(f'  model = AutoModel.from_pretrained("{args.repo_id}", trust_remote_code=True)')
        print(f"  ```")
        print()
    else:
        print(f"\n❌ Upload failed - see errors above")
        sys.exit(1)


if __name__ == "__main__":
    main()
