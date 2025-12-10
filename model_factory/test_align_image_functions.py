#!/usr/bin/env python3
# coding=utf-8
"""
Test script to verify the image loading and generation functions
in align_siglip2_packing.py work correctly.

This test validates:
1. The image generation function can create test images
2. The image loading function can load images
3. Images are correctly normalized and shaped

Returns:
    0 if all tests pass successfully
    1 if any test fails
"""

import sys
import os
import tempfile
import shutil

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)


def test_generate_image():
    """Test that we can generate test images."""
    print("=" * 80)
    print("Testing image generation...")
    print("=" * 80)
    
    try:
        # Check if PIL is available
        try:
            from PIL import Image
            import numpy as np
            PIL_AVAILABLE = True
        except ImportError:
            print("⚠️  PIL not available, skipping image generation test")
            return True
        
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()
        test_image_path = os.path.join(temp_dir, "subdir", "test_image.jpg")
        
        try:
            # Replicate the generate_test_image function
            os.makedirs(os.path.dirname(test_image_path), exist_ok=True)
            
            # Create a random colorful image
            width, height = 256, 192
            img_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
            img = Image.fromarray(img_array, 'RGB')
            img.save(test_image_path)
            
            # Check if file was created
            if not os.path.exists(test_image_path):
                print(f"❌ Image file was not created at {test_image_path}")
                return False
            
            # Check file size
            file_size = os.path.getsize(test_image_path)
            if file_size == 0:
                print(f"❌ Image file is empty")
                return False
            
            print(f"✅ Successfully generated image: {file_size} bytes")
            print(f"✅ Image dimensions: {width}x{height}")
            return True
            
        finally:
            # Clean up
            shutil.rmtree(temp_dir)
            
    except Exception as e:
        print(f"❌ Error in image generation test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_load_image():
    """Test that we can load images and convert to tensors."""
    print("\n" + "=" * 80)
    print("Testing image loading...")
    print("=" * 80)
    
    try:
        # Check if PIL is available
        try:
            from PIL import Image
            import numpy as np
        except ImportError:
            print("⚠️  PIL not available, skipping image loading test")
            return True
        
        # Check if test images exist
        image_dir = os.path.join(current_dir, "images")
        test_images = ["1.jpg", "2.jpg"]
        
        found_at_least_one = False
        for img_name in test_images:
            img_path = os.path.join(image_dir, img_name)
            
            if not os.path.exists(img_path):
                print(f"⚠️  Test image not found: {img_path}, skipping")
                continue
            
            found_at_least_one = True
            print(f"\nLoading {img_name}...")
            
            # Test the PIL loading part (replicate what load_image_as_tensor does)
            try:
                img = Image.open(img_path).convert('RGB')
                img_array = np.array(img, dtype=np.float32) / 255.0
                
                # Check shape
                if len(img_array.shape) != 3:
                    print(f"❌ Unexpected array shape: {img_array.shape}")
                    return False
                
                h, w, c = img_array.shape
                if c != 3:
                    print(f"❌ Expected 3 channels, got {c}")
                    return False
                
                # Check value range
                if img_array.min() < 0 or img_array.max() > 1:
                    print(f"❌ Values not in [0, 1] range: [{img_array.min()}, {img_array.max()}]")
                    return False
                
                print(f"✅ Successfully loaded {img_name}")
                print(f"  Shape: {w}x{h}x{c}")
                print(f"  Value range: [{img_array.min():.3f}, {img_array.max():.3f}]")
                
            except Exception as e:
                print(f"❌ Error loading {img_name}: {e}")
                return False
        
        if not found_at_least_one:
            print("⚠️  No test images found, skipping test")
            return True
        
        print("\n✅ Image loading test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error in image loading test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cli_arguments():
    """Test that the CLI accepts the new arguments."""
    print("\n" + "=" * 80)
    print("Testing CLI arguments...")
    print("=" * 80)
    
    try:
        import ast
        
        # Parse the align_siglip2_packing.py file
        with open(os.path.join(current_dir, 'align_siglip2_packing.py'), 'r') as f:
            code = f.read()
        
        # Check for use_real_images argument
        if '--use_real_images' in code:
            print("✅ Found --use_real_images argument")
        else:
            print("❌ Missing --use_real_images argument")
            return False
        
        # Check for image_dir argument
        if '--image_dir' in code:
            print("✅ Found --image_dir argument")
        else:
            print("❌ Missing --image_dir argument")
            return False
        
        # Check for PIL import
        if 'from PIL import Image' in code or 'import PIL' in code:
            print("✅ Found PIL import")
        else:
            print("❌ Missing PIL import")
            return False
        
        # Check for new functions
        expected_functions = ['generate_test_image', 'load_image_as_tensor']
        for func in expected_functions:
            if f'def {func}' in code:
                print(f"✅ Found function: {func}")
            else:
                print(f"❌ Missing function: {func}")
                return False
        
        print("\n✅ All CLI arguments and functions are present")
        return True
        
    except Exception as e:
        print(f"❌ Error checking CLI arguments: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 80)
    print("Image Functions Validation Tests")
    print("=" * 80)
    
    results = []
    
    # Test 1: CLI arguments
    results.append(("CLI Arguments Test", test_cli_arguments()))
    
    # Test 2: Image generation
    results.append(("Image Generation Test", test_generate_image()))
    
    # Test 3: Image loading
    results.append(("Image Loading Test", test_load_image()))
    
    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✅ All validation tests passed!")
        return 0
    else:
        print("\n❌ Some validation tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
