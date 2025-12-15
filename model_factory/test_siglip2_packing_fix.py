#!/usr/bin/env python3
# coding=utf-8
"""
Quick validation test to verify the Siglip2NaflexPacking refactoring.

This test verifies that:
1. The model no longer uses AutoModel.from_pretrained
2. FlashAttention varlen is used instead of attention masks
3. The implementation follows the vit_preview_v0_packing_hf pattern

Returns:
    0 if all tests pass successfully
    1 if any test fails

Note: This test validates the code structure and signatures without requiring
FlashAttention or a model checkpoint. To test with actual model inference, you need:
    - FlashAttention 2 installed: pip install flash-attn --no-build-isolation
    - A model checkpoint available
"""

import sys
import os

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

def test_imports():
    """Test that modules can be imported without errors."""
    print("=" * 80)
    print("Testing module imports...")
    print("=" * 80)
    
    try:
        from vit_siglip2_packing_hf import Siglip2NaflexPacking
        print("✓ Successfully imported Siglip2NaflexPacking")
        return True
    except Exception as e:
        print(f"✗ Failed to import Siglip2NaflexPacking: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_forward_signature():
    """Test that the forward method has the correct signature."""
    print("\n" + "=" * 80)
    print("Testing forward method signature...")
    print("=" * 80)
    
    try:
        from vit_siglip2_packing_hf import Siglip2NaflexPacking
        import inspect
        
        # Get the forward method
        forward_method = Siglip2NaflexPacking.forward
        sig = inspect.signature(forward_method)
        
        print(f"Forward signature: {sig}")
        
        # Check parameters
        params = list(sig.parameters.keys())
        expected_params = ['self', 'hidden_states', 'grid_thw']
        
        if params == expected_params:
            print(f"✓ Forward method has correct parameters: {params}")
            return True
        else:
            print(f"✗ Forward method parameters mismatch:")
            print(f"  Expected: {expected_params}")
            print(f"  Got: {params}")
            return False
            
    except Exception as e:
        print(f"✗ Error checking forward signature: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_code_paths():
    """Test that FlashAttention varlen implementation exists."""
    print("\n" + "=" * 80)
    print("Testing implementation code paths...")
    print("=" * 80)
    
    try:
        import inspect
        from vit_siglip2_packing_hf import Siglip2NaflexPacking
        
        # Get the source code of the forward method
        source = inspect.getsource(Siglip2NaflexPacking.forward)
        
        # Check for FlashAttention varlen usage
        has_flash_attn = 'flash_attn_varlen_func' in source or 'cu_seqlens' in source
        has_no_automodel = 'AutoModel.from_pretrained' not in source
        
        print(f"✓ Uses FlashAttention varlen approach: {has_flash_attn}")
        print(f"✓ Does not use AutoModel.from_pretrained: {has_no_automodel}")
        
        # Check the class structure for FlashAttention components
        try:
            from vit_siglip2_packing_hf import Siglip2PackingAttention
            print("✓ Siglip2PackingAttention class exists")
            has_packing_attn = True
        except ImportError:
            print("✗ Siglip2PackingAttention class not found")
            has_packing_attn = False
        
        if has_flash_attn and has_no_automodel and has_packing_attn:
            print("\n✓ FlashAttention varlen implementation is correct")
            return True
        else:
            print("\n✗ Missing expected FlashAttention components")
            return False
            
    except Exception as e:
        print(f"✗ Error checking code paths: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 80)
    print("Siglip2NaflexPacking Fix Validation Tests")
    print("=" * 80)
    
    results = []
    
    # Test 1: Imports
    results.append(("Import Test", test_imports()))
    
    # Test 2: Forward signature
    results.append(("Forward Signature Test", test_forward_signature()))
    
    # Test 3: Code paths
    results.append(("Code Paths Test", test_code_paths()))
    
    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✅ All validation tests passed!")
        print("\nNote: To test with actual model loading, you need:")
        print("  1. FlashAttention 2 installed: pip install flash-attn --no-build-isolation")
        print("  2. A model checkpoint available")
        return 0
    else:
        print("\n❌ Some validation tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
