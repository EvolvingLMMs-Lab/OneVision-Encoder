#!/usr/bin/env python3
"""
Quick test to verify that both DINOv3 models can be imported without errors.
"""

import sys
import os

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

print("Testing DINOv3 model imports...")

# Test importing the classes
try:
    from vit_dinov3 import Dinov3
    print("✓ Successfully imported Dinov3")
except Exception as e:
    print(f"✗ Failed to import Dinov3: {e}")
    sys.exit(1)

try:
    from vit_dinov3_packing_hf import DINOv3ViTPacking
    print("✓ Successfully imported DINOv3ViTPacking")
except Exception as e:
    print(f"✗ Failed to import DINOv3ViTPacking: {e}")
    sys.exit(1)

print("\nAll imports successful!")
print("\nNote: To test with actual model loading, you need the model checkpoint.")
print("Example usage:")
print("  python align_dinov3_packing.py --ckpt facebook/dinov3-base")
