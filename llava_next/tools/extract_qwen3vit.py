"""
Extract ViT weights from Qwen3-VL-4B-Instruct model

Usage:
    conda activate llava_xy
    python extract_qwen3vit.py \
        --source_model_path Qwen3-VL-4B-Instruct \
        --output_path Qwen3-VL-4B-ViT

Output:
    - config.json: ViT configuration file
    - model.safetensors: ViT weight file
"""

import os
import json
import argparse
from collections import OrderedDict
from typing import Dict, Any

import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm


def load_qwen3_vl_config(model_path: str) -> Dict[str, Any]:
    """Load Qwen3-VL configuration file"""
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


def create_preprocessor_config(full_config: Dict[str, Any]) -> Dict[str, Any]:
    """Create image preprocessor config from full config"""
    vision_config = full_config.get("vision_config", {})

    preprocessor_config = {
        "size": {"longest_edge": 16777216, "shortest_edge": 65536},
        "patch_size": vision_config.get("patch_size", 16),
        "temporal_patch_size": vision_config.get("temporal_patch_size", 2),
        "merge_size": vision_config.get("spatial_merge_size", 2),
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
        "processor_class": "Qwen3VLProcessor",
        "image_processor_type": "Qwen2VLImageProcessorFast",
    }

    return preprocessor_config


def create_vit_config(full_config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract ViT config from full config"""
    vision_config = full_config.get("vision_config", {})

    vit_config = {
        "model_type": "qwen3_vl_vision",
        "architectures": ["Qwen3VLVisionEncoder"],
        # Vision Encoder config
        "hidden_size": vision_config.get("hidden_size", 1024),
        "intermediate_size": vision_config.get("intermediate_size", 4096),
        "num_heads": vision_config.get("num_heads", 16),
        "depth": vision_config.get("depth", 24),
        "patch_size": vision_config.get("patch_size", 16),
        "temporal_patch_size": vision_config.get("temporal_patch_size", 2),
        "spatial_merge_size": vision_config.get("spatial_merge_size", 2),
        "in_channels": vision_config.get("in_channels", 3),
        "num_position_embeddings": vision_config.get("num_position_embeddings", 2304),
        "hidden_act": vision_config.get("hidden_act", "gelu_pytorch_tanh"),
        "out_hidden_size": vision_config.get("out_hidden_size", 2560),
        "deepstack_visual_indexes": vision_config.get("deepstack_visual_indexes", [5, 11, 17]),
        # Other metadata
        "torch_dtype": "bfloat16",
        "transformers_version": full_config.get("transformers_version", "4.57.0"),
    }

    return vit_config


def load_model_weights(model_path: str) -> Dict[str, torch.Tensor]:
    """Load model weights"""
    # Read index file
    index_path = os.path.join(model_path, "model.safetensors.index.json")

    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            index = json.load(f)
        weight_map = index.get("weight_map", {})

        # Get all unique weight files
        weight_files = set(weight_map.values())

        # Load all weights
        all_weights = {}
        for weight_file in tqdm(weight_files, desc="Loading weight files"):
            file_path = os.path.join(model_path, weight_file)
            if os.path.exists(file_path):
                weights = load_file(file_path)
                all_weights.update(weights)

        return all_weights
    else:
        # Single file mode
        single_file = os.path.join(model_path, "model.safetensors")
        if os.path.exists(single_file):
            return load_file(single_file)
        else:
            raise FileNotFoundError(f"No weight files found in {model_path}")


def extract_vit_weights(
    all_weights: Dict[str, torch.Tensor], prefix: str = "model.visual."
) -> Dict[str, torch.Tensor]:
    """Extract ViT weights from full model weights"""
    vit_weights = OrderedDict()

    for key, value in tqdm(all_weights.items(), desc="Extracting ViT weights"):
        if key.startswith(prefix):
            # Remove prefix
            new_key = key[len(prefix) :]
            vit_weights[new_key] = value

    return vit_weights


def save_vit_model(
    output_path: str,
    vit_config: Dict[str, Any],
    vit_weights: Dict[str, torch.Tensor],
    preprocessor_config: Dict[str, Any] = None,
) -> None:
    """Save ViT model"""
    os.makedirs(output_path, exist_ok=True)

    # Save config
    config_path = os.path.join(output_path, "config.json")
    with open(config_path, "w") as f:
        json.dump(vit_config, f, indent=2)
    print(f"Config saved to: {config_path}")

    # Save preprocessor config
    if preprocessor_config is not None:
        preprocessor_path = os.path.join(output_path, "preprocessor_config.json")
        with open(preprocessor_path, "w") as f:
            json.dump(preprocessor_config, f, indent=4)
        print(f"Preprocessor config saved to: {preprocessor_path}")

    # Save weights
    weights_path = os.path.join(output_path, "model.safetensors")
    save_file(vit_weights, weights_path)
    print(f"Weights saved to: {weights_path}")

    # Calculate and print statistics
    total_params = sum(v.numel() for v in vit_weights.values())
    total_size_mb = sum(v.numel() * v.element_size() for v in vit_weights.values()) / (1024 * 1024)

    print(f"\n=== ViT Model Statistics ===")
    print(f"Total parameters: {total_params:,} ({total_params / 1e6:.2f}M)")
    print(f"Total size: {total_size_mb:.2f} MB")
    print(f"Number of weight tensors: {len(vit_weights)}")

    # Print weight key list
    print(f"\n=== Weight Keys ===")
    for key in sorted(vit_weights.keys()):
        shape = tuple(vit_weights[key].shape)
        dtype = vit_weights[key].dtype
        print(f"  {key}: {shape} ({dtype})")


def verify_vit_model(output_path: str) -> None:
    """Verify the extracted ViT model"""
    print("\n=== Verifying ViT Model ===")

    try:
        # Import model definition
        import sys

        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from qwen3_vl_vision_model import Qwen3VLVisionConfig, Qwen3VLVisionEncoder

        # Load config
        with open(os.path.join(output_path, "config.json"), "r") as f:
            config_dict = json.load(f)

        config = Qwen3VLVisionConfig(**config_dict)

        # Create model
        model = Qwen3VLVisionEncoder(config)

        # Load weights
        weights = load_file(os.path.join(output_path, "model.safetensors"))

        # Check weight matching
        model_keys = set(model.state_dict().keys())
        weight_keys = set(weights.keys())

        missing_keys = model_keys - weight_keys
        unexpected_keys = weight_keys - model_keys

        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")

        if not missing_keys and not unexpected_keys:
            print("All weight keys match!")

        # Load weights into model
        model.load_state_dict(weights)
        print("Weights loaded successfully!")

        # Test forward pass
        model.eval()
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(torch.bfloat16)

        # Create test input
        # Simulate a 224x224 image, temporal=1
        patch_size = config.patch_size
        temporal_patch_size = config.temporal_patch_size
        in_channels = config.in_channels

        # Calculate patch count
        image_size = 224
        num_patches_per_side = image_size // patch_size
        num_patches = num_patches_per_side * num_patches_per_side

        # Number of pixels per patch
        patch_pixels = in_channels * temporal_patch_size * patch_size * patch_size

        # Create input
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype

        hidden_states = torch.randn(num_patches, patch_pixels, device=device, dtype=dtype)
        grid_thw = torch.tensor([[1, num_patches_per_side, num_patches_per_side]], device=device)

        # Forward pass
        with torch.no_grad():
            output, deepstack_features = model(hidden_states, grid_thw)

        print(f"Forward pass successful!")
        print(f"  Output shape: {output.last_hidden_state.shape}")
        print(f"  Number of deepstack features: {len(deepstack_features)}")
        for i, feat in enumerate(deepstack_features):
            print(f"    Deepstack feature {i}: {feat.shape}")

    except Exception as e:
        print(f"Verification failed: {e}")
        import traceback

        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Extract ViT weights from Qwen3-VL model")
    parser.add_argument(
        "--source_model_path", type=str, default="Qwen3-VL-4B-Instruct", help="Path to the source Qwen3-VL model"
    )
    parser.add_argument(
        "--output_path", type=str, default="Qwen3-VL-4B-ViT", help="Path to save the extracted ViT model"
    )
    parser.add_argument("--verify", action="store_true", help="Verify the extracted model after saving")

    args = parser.parse_args()

    print(f"=== Extracting ViT from Qwen3-VL ===")
    print(f"Source model: {args.source_model_path}")
    print(f"Output path: {args.output_path}")

    # 1. Load config
    print("\n[1/4] Loading configuration...")
    full_config = load_qwen3_vl_config(args.source_model_path)
    vit_config = create_vit_config(full_config)
    preprocessor_config = create_preprocessor_config(full_config)
    print(f"Vision config: {json.dumps(vit_config, indent=2)}")
    print(f"Preprocessor config: {json.dumps(preprocessor_config, indent=2)}")

    # 2. Load weights
    print("\n[2/4] Loading model weights...")
    all_weights = load_model_weights(args.source_model_path)
    print(f"Total weights loaded: {len(all_weights)}")

    # 3. Extract ViT weights
    print("\n[3/4] Extracting ViT weights...")
    vit_weights = extract_vit_weights(all_weights)
    print(f"ViT weights extracted: {len(vit_weights)}")

    # 4. Save model
    print("\n[4/4] Saving ViT model...")
    save_vit_model(args.output_path, vit_config, vit_weights, preprocessor_config)

    # 5. Verify (optional)
    if args.verify:
        verify_vit_model(args.output_path)

    print("\n=== Done! ===")
    print(f"ViT model saved to: {args.output_path}")


if __name__ == "__main__":
    main()
