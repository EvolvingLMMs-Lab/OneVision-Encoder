#!/usr/bin/env python3
# coding=utf-8
"""
简单的测试脚本，展示如何从 HuggingFace Hub 加载 LlavaViT 模型
Simple test script showing how to load LlavaViT model from HuggingFace Hub

使用方法 / Usage:
    python test_automodel_loading.py your-username/llava-vit-model
"""

import sys
import torch


def test_automodel_loading(repo_id: str):
    """
    测试从 HuggingFace Hub 加载模型
    Test loading model from HuggingFace Hub
    
    Args:
        repo_id: HuggingFace repository ID (e.g., "username/model-name")
    """
    print(f"\n{'='*60}")
    print(f"  Testing AutoModel Loading from HuggingFace Hub")
    print(f"{'='*60}")
    print(f"  Repository: {repo_id}")
    print(f"{'='*60}\n")
    
    try:
        from transformers import AutoModel, AutoConfig, CLIPImageProcessor
        print("✅ Successfully imported transformers")
    except ImportError:
        print("❌ Failed to import transformers")
        print("   Please install: pip install transformers")
        return False
    
    # Test 1: Load configuration
    print("\n[Test 1] Loading model configuration...")
    try:
        config = AutoConfig.from_pretrained(repo_id, trust_remote_code=True)
        print(f"   ✅ Config loaded successfully")
        print(f"   Model type: {config.model_type}")
        print(f"   Hidden size: {config.hidden_size}")
        print(f"   Num layers: {config.num_hidden_layers}")
        print(f"   Image size: {config.image_size}")
    except Exception as e:
        print(f"   ❌ Failed to load config: {e}")
        return False
    
    # Test 2: Load image processor
    print("\n[Test 2] Loading image processor...")
    try:
        processor = CLIPImageProcessor.from_pretrained(repo_id)
        print(f"   ✅ Processor loaded successfully")
        print(f"   Image size: {processor.size}")
    except Exception as e:
        print(f"   ❌ Failed to load processor: {e}")
        return False
    
    # Test 3: Load model
    print("\n[Test 3] Loading model with AutoModel...")
    try:
        model = AutoModel.from_pretrained(
            repo_id, 
            trust_remote_code=True,
            torch_dtype=torch.float32  # Use float32 for CPU testing
        )
        print(f"   ✅ Model loaded successfully")
        print(f"   Model class: {model.__class__.__name__}")
        print(f"   Device: {next(model.parameters()).device}")
    except Exception as e:
        print(f"   ❌ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 4: Test forward pass with dummy input
    print("\n[Test 4] Testing forward pass with dummy input...")
    try:
        model.eval()
        
        # Create dummy input
        batch_size = 2
        image_size = config.image_size
        dummy_input = torch.randn(batch_size, 3, image_size, image_size)
        
        print(f"   Input shape: {dummy_input.shape}")
        
        with torch.no_grad():
            outputs = model(pixel_values=dummy_input)
        
        print(f"   ✅ Forward pass successful")
        print(f"   Output last_hidden_state shape: {outputs.last_hidden_state.shape}")
        if outputs.pooler_output is not None:
            print(f"   Output pooler_output shape: {outputs.pooler_output.shape}")
        
    except Exception as e:
        print(f"   ❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 5: Test with video input (5D tensor)
    print("\n[Test 5] Testing with video input...")
    try:
        num_frames = 4
        video_input = torch.randn(1, 3, num_frames, image_size, image_size)
        
        print(f"   Video input shape: {video_input.shape}")
        
        with torch.no_grad():
            outputs = model(pixel_values=video_input)
        
        print(f"   ✅ Video forward pass successful")
        print(f"   Output shape: {outputs.last_hidden_state.shape}")
        
    except Exception as e:
        print(f"   ❌ Video forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 6: Test with visible indices (masking)
    print("\n[Test 6] Testing with visible indices (masking)...")
    try:
        dummy_input = torch.randn(1, 3, image_size, image_size)
        patch_size = config.patch_size
        num_patches = (image_size // patch_size) ** 2
        
        # Use 75% of patches
        num_visible = int(num_patches * 0.75)
        visible_indices = torch.arange(num_visible).unsqueeze(0)
        
        print(f"   Total patches: {num_patches}")
        print(f"   Visible patches: {num_visible}")
        
        with torch.no_grad():
            outputs = model(
                pixel_values=dummy_input,
                visible_indices=visible_indices
            )
        
        print(f"   ✅ Masking forward pass successful")
        print(f"   Output shape: {outputs.last_hidden_state.shape}")
        
    except Exception as e:
        print(f"   ❌ Masking forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # All tests passed
    print(f"\n{'='*60}")
    print(f"  🎉 All Tests Passed!")
    print(f"{'='*60}\n")
    
    return True


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_automodel_loading.py <repo_id>")
        print("\nExample:")
        print("  python test_automodel_loading.py username/llava-vit-large")
        sys.exit(1)
    
    repo_id = sys.argv[1]
    success = test_automodel_loading(repo_id)
    
    if not success:
        print("\n❌ Some tests failed")
        sys.exit(1)
    
    print("\n📝 Next Steps:")
    print("  1. Try with real images using the example in README_UPLOAD_TO_HF.md")
    print("  2. Fine-tune the model for your specific task")
    print("  3. Share your results with the community!")
    print()


if __name__ == "__main__":
    main()
