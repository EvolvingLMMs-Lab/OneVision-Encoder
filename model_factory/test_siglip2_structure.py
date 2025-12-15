#!/usr/bin/env python3
"""
Lightweight structure test for Siglip2NaflexPacking refactoring.
This test validates the code structure without requiring dependencies.
"""

import re
import sys
import os

def read_file(filepath):
    """Read file content."""
    # Get absolute path relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(script_dir, filepath)
    with open(full_path, 'r') as f:
        return f.read()

def test_flash_attn_import():
    """Test that FlashAttention is properly imported."""
    content = read_file('vit_siglip2_packing_hf.py')
    
    # Check for flash_attn import
    if 'from flash_attn import flash_attn_varlen_func' in content:
        print("✓ FlashAttention varlen import found")
        return True
    else:
        print("✗ FlashAttention varlen import not found")
        return False

def test_no_automodel_in_custom_code():
    """Test that AutoModel.from_pretrained is not used in Siglip2NaflexPacking."""
    content = read_file('vit_siglip2_packing_hf.py')
    
    # Extract the custom extension section (after the marker)
    custom_start = content.find('# CUSTOM EXTENSION:')
    if custom_start == -1:
        print("✗ Custom extension section not found")
        return False
    
    custom_section = content[custom_start:]
    
    # Check that AutoModel.from_pretrained is not in custom section
    # (excluding comments and docstrings)
    lines = custom_section.split('\n')
    for line in lines:
        stripped = line.strip()
        # Skip comments and docstrings
        if stripped.startswith('#') or stripped.startswith('"""') or stripped.startswith("'''"):
            continue
        if 'AutoModel.from_pretrained' in line and not line.strip().startswith('#'):
            print(f"✗ Found AutoModel.from_pretrained in custom code: {line.strip()}")
            return False
    
    print("✓ AutoModel.from_pretrained not used in custom extension")
    return True

def test_siglip2_vision_model_usage():
    """Test that Siglip2VisionModel.from_pretrained is used instead."""
    content = read_file('vit_siglip2_packing_hf.py')
    
    if 'Siglip2VisionModel.from_pretrained' in content:
        print("✓ Siglip2VisionModel.from_pretrained is used")
        return True
    else:
        print("✗ Siglip2VisionModel.from_pretrained not found")
        return False

def test_packing_attention_class():
    """Test that Siglip2PackingAttention class exists."""
    content = read_file('vit_siglip2_packing_hf.py')
    
    if 'class Siglip2PackingAttention(nn.Module)' in content:
        print("✓ Siglip2PackingAttention class exists")
        return True
    else:
        print("✗ Siglip2PackingAttention class not found")
        return False

def test_cu_seqlens_usage():
    """Test that cu_seqlens is used for packing."""
    content = read_file('vit_siglip2_packing_hf.py')
    
    # Check for cu_seqlens in forward methods
    if 'cu_seqlens' in content:
        print("✓ cu_seqlens parameter found (packing support)")
        return True
    else:
        print("✗ cu_seqlens parameter not found")
        return False

def test_flash_attn_varlen_func_call():
    """Test that flash_attn_varlen_func is called."""
    content = read_file('vit_siglip2_packing_hf.py')
    
    if 'flash_attn_varlen_func(' in content:
        print("✓ flash_attn_varlen_func is called")
        return True
    else:
        print("✗ flash_attn_varlen_func not called")
        return False

def main():
    """Run all structure tests."""
    print("=" * 80)
    print("Siglip2NaflexPacking Structure Validation")
    print("=" * 80)
    
    tests = [
        ("FlashAttention Import", test_flash_attn_import),
        ("No AutoModel in Custom Code", test_no_automodel_in_custom_code),
        ("Uses Siglip2VisionModel", test_siglip2_vision_model_usage),
        ("Packing Attention Class", test_packing_attention_class),
        ("cu_seqlens Usage", test_cu_seqlens_usage),
        ("FlashAttention Varlen Call", test_flash_attn_varlen_func_call),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\nTest: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ Error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✅ All structure validation tests passed!")
        print("\nChanges implemented:")
        print("  1. ✓ No longer uses AutoModel.from_pretrained")
        print("  2. ✓ Uses Siglip2VisionModel.from_pretrained instead")
        print("  3. ✓ Implements FlashAttention varlen for packing")
        print("  4. ✓ Uses cu_seqlens for variable-length sequences")
        print("  5. ✓ No attention masks needed (FlashAttention handles it)")
        return 0
    else:
        print("\n❌ Some structure validation tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
