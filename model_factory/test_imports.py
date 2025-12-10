#!/usr/bin/env python3
"""
Quick test to verify that both models can be loaded without errors.
"""

import sys
import os

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import torch

print("Testing model imports...")

# Test importing the classes
try:
    from vit_siglip2 import Siglip2Naflex
    print("✓ Successfully imported Siglip2Naflex")
except Exception as e:
    print(f"✗ Failed to import Siglip2Naflex: {e}")
    sys.exit(1)

try:
    from vit_siglip2_packing_hf import Siglip2NaflexPacking
    print("✓ Successfully imported Siglip2NaflexPacking")
except Exception as e:
    print(f"✗ Failed to import Siglip2NaflexPacking: {e}")
    sys.exit(1)

print("\nAll imports successful!")
print("\nNote: To test with actual model loading, you need the model checkpoint.")
print("Example usage:")
print("  python align_siglip2_packing.py --ckpt <path_to_checkpoint>")
