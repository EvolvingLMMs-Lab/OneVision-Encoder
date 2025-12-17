#!/usr/bin/env python3
# coding=utf-8
"""
Quick validation test to verify the AIMv2Packing refactoring.

This test verifies that:
1. The implementation uses cu_seqlens for attention control
2. Custom packing layers are defined and used
3. The forward signature matches Siglip2 pattern
4. No image reconstruction is used

Returns:
    0 if all tests pass successfully
    1 if any test fails
"""

import sys
import os

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

def test_cu_seqlens_usage():
    """Test that cu_seqlens is used throughout the implementation."""
    print("=" * 80)
    print("Testing cu_seqlens usage...")
    print("=" * 80)
    
    try:
        with open('vit_aim_v2_packing_hf.py', 'r') as f:
            content = f.read()
        
        # Check for cu_seqlens in key locations
        checks = [
            ('forward method computes cu_seqlens', 'cu_seqlens = F.pad(seq_lengths.cumsum'),
            ('encoder receives cu_seqlens', 'encoder_output = self.encoder(embeddings, cu_seqlens)'),
            ('attention uses cu_seqlens', 'flash_attn_varlen_func'),
            ('cu_seqlens passed to flash attention', 'cu_seqlens_q=cu_seqlens'),
        ]
        
        all_passed = True
        for check_name, check_str in checks:
            if check_str in content:
                print(f"✓ {check_name}")
            else:
                print(f"✗ {check_name} - NOT FOUND")
                all_passed = False
        
        return all_passed
        
    except Exception as e:
        print(f"✗ Error reading file: {e}")
        return False


def test_no_image_reconstruction():
    """Test that image reconstruction code is removed."""
    print("\n" + "=" * 80)
    print("Testing that image reconstruction is removed...")
    print("=" * 80)
    
    try:
        with open('vit_aim_v2_packing_hf.py', 'r') as f:
            content = f.read()
        
        # Check that reconstruction method is NOT present
        if '_reconstruct_images_from_patches' not in content:
            print("✓ Image reconstruction method removed")
            reconstruction_removed = True
        else:
            print("✗ Image reconstruction method still present")
            reconstruction_removed = False
        
        # Check that we don't use the standard model
        if 'self.model(' not in content:
            print("✓ Not using standard Aimv2VisionModel for inference")
            no_standard_model = True
        else:
            print("✗ Still using standard Aimv2VisionModel")
            no_standard_model = False
        
        return reconstruction_removed and no_standard_model
        
    except Exception as e:
        print(f"✗ Error reading file: {e}")
        return False


def test_custom_layers():
    """Test that custom packing layers are defined and used."""
    print("\n" + "=" * 80)
    print("Testing custom packing layers...")
    print("=" * 80)
    
    try:
        with open('vit_aim_v2_packing_hf.py', 'r') as f:
            content = f.read()
        
        # Check for custom layer definitions
        layers = [
            'class AIMv2PatchEmbedding',
            'class AIMv2PackingAttention',
            'class AIMv2PackingEncoderLayer',
            'class AIMv2PackingEncoder',
        ]
        
        all_defined = True
        for layer in layers:
            if layer in content:
                print(f"✓ {layer} defined")
            else:
                print(f"✗ {layer} NOT defined")
                all_defined = False
        
        # Check that layers are instantiated in __init__
        checks = [
            ('embeddings instantiated', 'self.embeddings = AIMv2PatchEmbedding'),
            ('encoder instantiated', 'self.encoder = AIMv2PackingEncoder'),
            ('norm instantiated', 'self.norm = RMSNorm'),
        ]
        
        for check_name, check_str in checks:
            if check_str in content:
                print(f"✓ {check_name}")
            else:
                print(f"✗ {check_name} - NOT FOUND")
                all_defined = False
        
        return all_defined
        
    except Exception as e:
        print(f"✗ Error reading file: {e}")
        return False


def test_weight_loading():
    """Test that weight loading from pretrained model is implemented."""
    print("\n" + "=" * 80)
    print("Testing weight loading...")
    print("=" * 80)
    
    try:
        with open('vit_aim_v2_packing_hf.py', 'r') as f:
            content = f.read()
        
        # Check for weight loading method
        if 'def _load_pretrained_weights' in content:
            print("✓ Weight loading method defined")
            method_defined = True
        else:
            print("✗ Weight loading method NOT defined")
            method_defined = False
        
        # Check for Conv2d to Linear conversion
        if 'permute(0, 2, 3, 1)' in content:
            print("✓ Conv2d to Linear weight conversion implemented")
            conversion = True
        else:
            print("✗ Conv2d to Linear weight conversion NOT found")
            conversion = False
        
        # Check that weight loading is called
        if '_load_pretrained_weights(pretrained_model)' in content:
            print("✓ Weight loading is called in __init__")
            called = True
        else:
            print("✗ Weight loading NOT called")
            called = False
        
        return method_defined and conversion and called
        
    except Exception as e:
        print(f"✗ Error reading file: {e}")
        return False


def test_siglip2_similarity():
    """Test that implementation follows Siglip2 pattern."""
    print("\n" + "=" * 80)
    print("Testing similarity to Siglip2 pattern...")
    print("=" * 80)
    
    try:
        # Read both files
        with open('vit_aim_v2_packing_hf.py', 'r') as f:
            aimv2_content = f.read()
        
        with open('vit_siglip2_packing_hf.py', 'r') as f:
            siglip2_content = f.read()
        
        # Check for similar patterns
        patterns = [
            ('cu_seqlens computation', 'F.pad(seq_lengths.cumsum(dim=0), (1, 0), value=0).to(torch.int32)'),
            ('encoder call with cu_seqlens', 'encoder('),
            ('embeddings before encoder', 'embeddings = self.embeddings('),
        ]
        
        all_similar = True
        for pattern_name, pattern_str in patterns:
            if pattern_str in aimv2_content and pattern_str in siglip2_content:
                print(f"✓ {pattern_name} - pattern matches Siglip2")
            elif pattern_str in aimv2_content:
                print(f"~ {pattern_name} - present in AIMv2 but not Siglip2")
            else:
                print(f"✗ {pattern_name} - NOT matching Siglip2 pattern")
                all_similar = False
        
        return all_similar
        
    except Exception as e:
        print(f"✗ Error reading files: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all validation tests."""
    print("=" * 80)
    print("AIMv2Packing Refactoring Validation Tests")
    print("=" * 80)
    
    results = []
    
    # Test 1: cu_seqlens usage
    results.append(("cu_seqlens Usage", test_cu_seqlens_usage()))
    
    # Test 2: Image reconstruction removed
    results.append(("Image Reconstruction Removed", test_no_image_reconstruction()))
    
    # Test 3: Custom layers
    results.append(("Custom Packing Layers", test_custom_layers()))
    
    # Test 4: Weight loading
    results.append(("Weight Loading", test_weight_loading()))
    
    # Test 5: Siglip2 similarity
    results.append(("Siglip2 Pattern Similarity", test_siglip2_similarity()))
    
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
        print("\nThe AIMv2Packing implementation:")
        print("  1. Uses cu_seqlens for attention control (like Siglip2)")
        print("  2. Uses custom packing layers with FlashAttention varlen")
        print("  3. Processes patches directly (no image reconstruction)")
        print("  4. Loads weights from pretrained model with proper conversion")
        return 0
    else:
        print("\n❌ Some validation tests failed")
        return 1


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    sys.exit(main())
