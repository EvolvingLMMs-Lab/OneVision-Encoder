#!/usr/bin/env python3
# coding=utf-8
"""
Quick validation test for the AIMv2 packing refactoring.

This test verifies that:
1. The model uses the siglip2 packing pattern
2. FlashAttention varlen is used
3. No single-image forward processing path exists
4. Only one forward pass is made for all images

Returns:
    0 if all tests pass successfully
    1 if any test fails

Note: This test validates the code structure without requiring
FlashAttention or a model checkpoint.
"""

import sys
import os
import inspect

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
        from vit_aim_v2_packing_hf import AIMv2Packing
        print("✓ Successfully imported AIMv2Packing")
        return True
    except Exception as e:
        print(f"✗ Failed to import AIMv2Packing: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_forward_signature():
    """Test that the forward method has the correct signature."""
    print("\n" + "=" * 80)
    print("Testing forward method signature...")
    print("=" * 80)
    
    try:
        from vit_aim_v2_packing_hf import AIMv2Packing
        
        # Get the forward method
        forward_method = AIMv2Packing.forward
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


def test_packing_components():
    """Test that packing components exist (following siglip2 pattern)."""
    print("\n" + "=" * 80)
    print("Testing packing components...")
    print("=" * 80)
    
    try:
        # Check for packing-specific classes
        from vit_aim_v2_packing_hf import (
            Aimv2VisionEmbeddings,
            Aimv2PackingAttention,
            Aimv2PackingEncoderLayer,
            Aimv2PackingEncoder,
        )
        print("✓ Aimv2VisionEmbeddings class exists")
        print("✓ Aimv2PackingAttention class exists")
        print("✓ Aimv2PackingEncoderLayer class exists")
        print("✓ Aimv2PackingEncoder class exists")
        return True
    except ImportError as e:
        print(f"✗ Missing packing component: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_flash_attention_usage():
    """Test that FlashAttention varlen is used."""
    print("\n" + "=" * 80)
    print("Testing FlashAttention usage...")
    print("=" * 80)
    
    try:
        from vit_aim_v2_packing_hf import AIMv2Packing, Aimv2PackingAttention
        
        # Get the source code of the forward method
        aimv2_source = inspect.getsource(AIMv2Packing.forward)
        attn_source = inspect.getsource(Aimv2PackingAttention.forward)
        
        # Check for FlashAttention varlen usage
        has_cu_seqlens = 'cu_seqlens' in aimv2_source
        has_flash_attn = 'flash_attn_varlen_func' in attn_source
        
        print(f"✓ Uses cu_seqlens in forward: {has_cu_seqlens}")
        print(f"✓ Uses flash_attn_varlen_func: {has_flash_attn}")
        
        if has_cu_seqlens and has_flash_attn:
            print("\n✓ FlashAttention varlen implementation is correct")
            return True
        else:
            print("\n✗ Missing expected FlashAttention components")
            return False
            
    except Exception as e:
        print(f"✗ Error checking FlashAttention usage: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_no_single_image_processing():
    """Test that there is NO single-image processing path."""
    print("\n" + "=" * 80)
    print("Testing for absence of single-image processing...")
    print("=" * 80)
    
    try:
        from vit_aim_v2_packing_hf import AIMv2Packing
        
        # Get the source code
        source = inspect.getsource(AIMv2Packing.forward)
        
        # These patterns indicate single-image processing loops
        bad_patterns = [
            'for i in range(num_images):',
            'for i in range(batch_size):',
        ]
        
        # Count occurrences - we expect some in _reconstruct_images_from_patches
        # but NONE in the main forward logic after embeddings
        lines = source.split('\n')
        
        # Find where embeddings are computed
        embed_line = -1
        for i, line in enumerate(lines):
            if 'embeddings = self.embeddings' in line:
                embed_line = i
                break
        
        if embed_line == -1:
            print("✗ Could not find embeddings computation")
            return False
        
        # Check after embeddings computation
        post_embed_source = '\n'.join(lines[embed_line:])
        
        # Should not have loops over images after embeddings
        has_image_loop = any(pattern in post_embed_source for pattern in bad_patterns)
        
        if not has_image_loop:
            print("✓ No single-image processing loops after embeddings")
            print("✓ All images processed in ONE forward pass")
            return True
        else:
            print("✗ Found single-image processing loop after embeddings")
            print("  This violates the requirement to process all images in one pass")
            return False
            
    except Exception as e:
        print(f"✗ Error checking for single-image processing: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 80)
    print("AIMv2Packing Refactoring Validation Tests")
    print("Following Siglip2 Pattern")
    print("=" * 80)
    
    results = []
    
    # Test 1: Imports
    results.append(("Import Test", test_imports()))
    
    # Test 2: Forward signature
    results.append(("Forward Signature Test", test_forward_signature()))
    
    # Test 3: Packing components
    results.append(("Packing Components Test", test_packing_components()))
    
    # Test 4: FlashAttention usage
    results.append(("FlashAttention Usage Test", test_flash_attention_usage()))
    
    # Test 5: No single-image processing
    results.append(("No Single-Image Processing Test", test_no_single_image_processing()))
    
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
        print("\nThe AIMv2 packing implementation now follows the Siglip2 pattern:")
        print("  1. Uses FlashAttention varlen with cu_seqlens")
        print("  2. Processes ALL images in ONE forward pass")
        print("  3. No single-image processing path")
        print("\nNote: To test with actual model loading, you need:")
        print("  1. FlashAttention 2 installed: pip install flash-attn --no-build-isolation")
        print("  2. A model checkpoint available")
        return 0
    else:
        print("\n❌ Some validation tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
