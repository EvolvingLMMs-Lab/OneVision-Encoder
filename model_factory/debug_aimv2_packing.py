#!/usr/bin/env python3
"""
Quick debug script to test AIMv2Packing initialization.
This helps identify the exact error when loading the model.
"""

import sys
import traceback

def test_aimv2_packing_init():
    """Test AIMv2Packing initialization with detailed error reporting."""
    print("=" * 80)
    print("Testing AIMv2Packing Initialization")
    print("=" * 80)
    
    # Test imports
    print("\n1. Testing imports...")
    try:
        import torch
        print("  ✓ torch imported")
    except ImportError as e:
        print(f"  ✗ Failed to import torch: {e}")
        return 1
    
    try:
        from transformers.models.aimv2.modeling_aimv2 import Aimv2VisionModel
        print("  ✓ Aimv2VisionModel imported")
    except ImportError as e:
        print(f"  ✗ Failed to import Aimv2VisionModel: {e}")
        return 1
    
    try:
        from vit_aim_v2_packing_hf import AIMv2Packing
        print("  ✓ AIMv2Packing imported")
    except Exception as e:
        print(f"  ✗ Failed to import AIMv2Packing: {e}")
        traceback.print_exc()
        return 1
    
    # Test model initialization
    print("\n2. Testing model initialization...")
    ckpt = "/video_vit/pretrain_models/apple/aimv2-large-patch14-native/"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Checkpoint: {ckpt}")
    print(f"  Device: {device}")
    
    try:
        print("\n  Creating AIMv2Packing model...")
        model = AIMv2Packing(ckpt=ckpt, device=device)
        print("  ✓ Model created successfully!")
        print(f"  Patch size: {model.patch_size}")
        print(f"  Hidden size: {model.config.hidden_size}")
        print(f"  Number of layers: {model.config.num_hidden_layers}")
        return 0
    except Exception as e:
        print(f"\n  ✗ Failed to create model!")
        print(f"\nError type: {type(e).__name__}")
        print(f"Error message: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        
        # Try to load base model to inspect structure
        print("\n" + "=" * 80)
        print("Attempting to load base Aimv2VisionModel for inspection...")
        print("=" * 80)
        try:
            base_model = Aimv2VisionModel.from_pretrained(ckpt, trust_remote_code=True)
            print("\n✓ Base model loaded successfully")
            print(f"\nBase model attributes: {[attr for attr in dir(base_model) if not attr.startswith('_')][:20]}")
            
            if hasattr(base_model, 'embeddings'):
                print(f"\nembeddings attributes: {[attr for attr in dir(base_model.embeddings) if not attr.startswith('_')][:20]}")
            
            if hasattr(base_model, 'encoder'):
                print(f"\nencoder attributes: {[attr for attr in dir(base_model.encoder) if not attr.startswith('_')][:10]}")
                if hasattr(base_model.encoder, 'layers') and len(base_model.encoder.layers) > 0:
                    print(f"\nencoder.layers[0] attributes: {[attr for attr in dir(base_model.encoder.layers[0]) if not attr.startswith('_')]}")
        except Exception as e2:
            print(f"✗ Failed to load base model: {e2}")
            traceback.print_exc()
        
        return 1

if __name__ == "__main__":
    sys.exit(test_aimv2_packing_init())
