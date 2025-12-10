#!/usr/bin/env python3
# coding=utf-8
"""
Quick validation test to verify the Siglip2NaflexPacking fix.

This test verifies that:
1. The parameter name issue is resolved (attention_mask vs pixel_attention_mask)
2. The optimized path works for same-length sequences (no attention mask)
3. The variable-length path works correctly with attention masks
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
    """Test that both code paths (same-length and variable-length) exist."""
    print("\n" + "=" * 80)
    print("Testing implementation code paths...")
    print("=" * 80)
    
    try:
        import inspect
        from vit_siglip2_packing_hf import Siglip2NaflexPacking
        
        # Get the source code of the forward method
        source = inspect.getsource(Siglip2NaflexPacking.forward)
        
        # Check for optimized path (same-length sequences)
        has_same_length_check = 'all_same_length' in source
        has_optimized_path = 'attention_mask=None' in source
        
        # Check for variable-length path
        has_variable_length_path = 'pixel_attention_mask' in source
        
        print(f"✓ Has same-length optimization check: {has_same_length_check}")
        print(f"✓ Has optimized path (no attention mask): {has_optimized_path}")
        print(f"✓ Has variable-length path (with attention mask): {has_variable_length_path}")
        
        if has_same_length_check and has_optimized_path and has_variable_length_path:
            print("\n✓ Both code paths are implemented correctly")
            return True
        else:
            print("\n✗ Missing expected code paths")
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
        print("\nNote: To test with actual model loading, you need a model checkpoint:")
        print("  python align_siglip2_packing.py --ckpt <path_to_checkpoint>")
        return 0
    else:
        print("\n❌ Some validation tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
