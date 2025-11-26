"""
Example script demonstrating how to use SpatialMergeProjector

This shows the recommended way to use the spatial merger with explicit
height and width parameters for dynamic resolution support.
"""

import torch
import torch.nn as nn
from builder import SpatialMergeProjector


def example_basic_usage():
    """Basic usage with auto-inference"""
    print("=" * 60)
    print("Example 1: Basic usage with auto-inference")
    print("=" * 60)
    
    # Initialize projector
    vit_dim = 1024  # Vision encoder hidden dimension
    llm_dim = 3584  # LLM hidden dimension (e.g., Qwen2.5-7B)
    projector = SpatialMergeProjector(llm_dim=llm_dim, vit_dim=vit_dim)
    
    # Simulate features from vision tower (448px image, 14px patches -> 32x32 = 1024 patches)
    batch_size = 2
    num_patches = 1024  # 32 * 32
    features = torch.randn(batch_size, num_patches, vit_dim)
    
    print(f"Input: {features.shape}")
    
    # Forward pass - auto-infers H=32, W=32
    output = projector(features)
    
    print(f"Output: {output.shape}")
    print(f"Token reduction: {num_patches} -> {output.shape[1]} (4x reduction)")
    print()


def example_explicit_hw():
    """Recommended usage with explicit height and width"""
    print("=" * 60)
    print("Example 2: Explicit H and W (recommended)")
    print("=" * 60)
    
    # Initialize projector
    vit_dim = 1024
    llm_dim = 3584
    projector = SpatialMergeProjector(llm_dim=llm_dim, vit_dim=vit_dim)
    
    # Simulate dynamic resolution: 252px x 448px image
    # With 14px patches: h=18, w=32
    batch_size = 1
    h, w = 18, 32
    num_patches = h * w  # 576
    features = torch.randn(batch_size, num_patches, vit_dim)
    
    print(f"Input: {features.shape} (h={h}, w={w})")
    
    # Forward pass with explicit dimensions
    output = projector(features, height=h, width=w)
    
    expected_h = h // 2  # 9
    expected_w = w // 2  # 16
    print(f"Output: {output.shape} (h={expected_h}, w={expected_w})")
    print(f"Token reduction: {num_patches} -> {output.shape[1]} (4x reduction)")
    print()


def example_integration_with_vision_model():
    """Example showing integration with vision model"""
    print("=" * 60)
    print("Example 3: Integration with vision model")
    print("=" * 60)
    
    # Simulate a vision model forward pass
    class MockVisionTower(nn.Module):
        def __init__(self, hidden_size, patch_size):
            super().__init__()
            self.hidden_size = hidden_size
            self.patch_size = patch_size
            self.encoder = nn.Linear(3 * patch_size * patch_size, hidden_size)
        
        def forward(self, images):
            # images: (B, C, H, W)
            B, C, H, W = images.shape
            h = H // self.patch_size
            w = W // self.patch_size
            
            # Simulate patch embedding
            features = torch.randn(B, h * w, self.hidden_size)
            return features, h, w
    
    # Initialize components
    patch_size = 14
    vit_dim = 1024
    llm_dim = 3584
    
    vision_tower = MockVisionTower(vit_dim, patch_size)
    spatial_merger = SpatialMergeProjector(llm_dim=llm_dim, vit_dim=vit_dim)
    
    # Forward pass with dynamic resolution
    images = torch.randn(1, 3, 252, 448)  # Non-square image
    
    print(f"Input image: {images.shape}")
    
    # Get features from vision tower
    features, h, w = vision_tower(images)
    print(f"Vision features: {features.shape} (h={h}, w={w})")
    
    # Apply spatial merger with known h and w
    merged_features = spatial_merger(features, height=h, width=w)
    print(f"Merged features: {merged_features.shape} (h={h//2}, w={w//2})")
    print()


def example_batch_with_different_resolutions():
    """Example handling batch with different resolutions"""
    print("=" * 60)
    print("Example 4: Batch with different resolutions")
    print("=" * 60)
    
    # Initialize projector
    vit_dim = 1024
    llm_dim = 3584
    projector = SpatialMergeProjector(llm_dim=llm_dim, vit_dim=vit_dim)
    
    # Different resolutions in the same training run
    test_cases = [
        (32, 32, "448px square"),
        (18, 32, "252px × 448px"),
        (14, 28, "196px × 392px"),
        (24, 24, "336px square"),
    ]
    
    for h, w, desc in test_cases:
        num_patches = h * w
        features = torch.randn(1, num_patches, vit_dim)
        output = projector(features, height=h, width=w)
        
        print(f"{desc}: {h}×{w} = {num_patches} patches -> "
              f"{h//2}×{w//2} = {output.shape[1]} tokens")
    print()


def example_full_pipeline():
    """Complete example from image to LLM input"""
    print("=" * 60)
    print("Example 5: Full pipeline simulation")
    print("=" * 60)
    
    # Configuration
    patch_size = 14
    vit_dim = 1024
    llm_dim = 3584
    
    # Input image
    batch_size = 2
    image_height, image_width = 252, 448
    images = torch.randn(batch_size, 3, image_height, image_width)
    
    print(f"1. Input images: {images.shape}")
    
    # Calculate patch grid dimensions
    h = image_height // patch_size  # 18
    w = image_width // patch_size   # 32
    num_patches = h * w
    
    print(f"2. Patch grid: {h}×{w} = {num_patches} patches")
    
    # Simulate vision encoder output
    vision_features = torch.randn(batch_size, num_patches, vit_dim)
    print(f"3. Vision features: {vision_features.shape}")
    
    # Apply spatial merger
    projector = SpatialMergeProjector(llm_dim=llm_dim, vit_dim=vit_dim)
    merged_features = projector(vision_features, height=h, width=w)
    
    merged_h = h // 2  # 9
    merged_w = w // 2  # 16
    print(f"4. After spatial merge: {merged_features.shape} ({merged_h}×{merged_w} tokens)")
    print(f"5. Token reduction: {num_patches} -> {merged_features.shape[1]} (4x)")
    print(f"6. Ready for LLM input!")
    print()


if __name__ == "__main__":
    # Run all examples
    example_basic_usage()
    example_explicit_hw()
    example_integration_with_vision_model()
    example_batch_with_different_resolutions()
    example_full_pipeline()
    
    print("=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
