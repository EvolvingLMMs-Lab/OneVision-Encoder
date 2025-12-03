"""
Example usage of HEVCViTPackingVisionTower

This example demonstrates how to use the packing mode vision tower.
"""

# Example 1: Using with builder (recommended)
# ============================================
# In your config file or code, specify a model path containing "packing":

config_example = {
    "mm_vision_tower": "/path/to/hevc_vit_packing_model",  # Contains "packing" in name
    "mm_projector_type": "patch_merger",
    "mm_vision_select_layer": -1,
    "mm_vision_select_feature": "patch",
}

# The builder will automatically use HEVCViTPackingVisionTower
# from llava_next.llava.model.multimodal_encoder.builder import build_vision_tower
# tower = build_vision_tower(config)


# Example 2: Direct usage (for testing)
# =====================================
def example_batch_processing():
    """Example of processing a batch of images"""
    import torch
    from llava_next.llava.model.multimodal_encoder.hevc_vit_packing_tower import HEVCViTPackingVisionTower
    
    # Mock args (in real usage, these come from your config)
    class MockArgs:
        mm_projector_type = "patch_merger"
        mm_vision_select_layer = -1
        mm_vision_select_feature = "patch"
    
    # Initialize tower (requires a pretrained packing model)
    tower = HEVCViTPackingVisionTower(
        vision_tower="/path/to/hevc_vit_packing_model",
        args=MockArgs(),
        delay_load=False  # Set to True if loading later
    )
    
    # Prepare batch of images [B, C, H, W]
    batch_images = torch.randn(4, 3, 224, 224).cuda()
    
    # Forward pass
    # Input: [4, 3, 224, 224]
    # Internally converted to packing format: [784, 768]
    #   where 784 = 4 images × 196 patches/image
    #   and 768 = 16×16×3 (patch_dim)
    # Output: [4, 196, hidden_size]
    features = tower(batch_images)
    
    print(f"Input shape: {batch_images.shape}")
    print(f"Output shape: {features.shape}")
    return features


def example_list_processing():
    """Example of processing a list of images with different sizes"""
    import torch
    from llava_next.llava.model.multimodal_encoder.hevc_vit_packing_tower import HEVCViTPackingVisionTower
    
    class MockArgs:
        mm_projector_type = "patch_merger"
        mm_vision_select_layer = -1
        mm_vision_select_feature = "patch"
    
    tower = HEVCViTPackingVisionTower(
        vision_tower="/path/to/hevc_vit_packing_model",
        args=MockArgs(),
        delay_load=False
    )
    
    # Prepare list of images with different sizes
    images = [
        torch.randn(3, 224, 224).cuda(),  # 196 patches
        torch.randn(3, 224, 224).cuda(),  # 196 patches
        torch.randn(3, 448, 448).cuda(),  # 784 patches
    ]
    
    # Forward pass
    # Internally converted to packing format: [1176, 768]
    #   where 1176 = 196 + 196 + 784 patches
    # Output: List of [196, hidden_size], [196, hidden_size], [784, hidden_size]
    features_list = tower(images)
    
    print(f"Input: {len(images)} images")
    for i, feat in enumerate(features_list):
        print(f"  Image {i}: {feat.shape}")
    return features_list


def example_with_spatial_dims():
    """Example of getting spatial dimensions (for spatial_merge projector)"""
    import torch
    from llava_next.llava.model.multimodal_encoder.hevc_vit_packing_tower import HEVCViTPackingVisionTower
    
    class MockArgs:
        mm_projector_type = "spatial_merge"  # Using spatial merge
        mm_vision_select_layer = -1
        mm_vision_select_feature = "patch"
    
    tower = HEVCViTPackingVisionTower(
        vision_tower="/path/to/hevc_vit_packing_model",
        args=MockArgs(),
        delay_load=False
    )
    
    batch_images = torch.randn(2, 3, 224, 224).cuda()
    
    # Forward pass with spatial dimensions
    features, h, w = tower(batch_images, return_spatial_dims=True)
    
    print(f"Input shape: {batch_images.shape}")
    print(f"Output shape: {features.shape}")
    print(f"Spatial dims: h={h}, w={w}")
    return features, h, w


# Example 3: Understanding the conversion
# =======================================
def show_conversion_details():
    """
    Show how the conversion works step by step
    """
    print("=" * 60)
    print("CONVERSION DETAILS")
    print("=" * 60)
    
    # Configuration
    batch_size = 4
    channels = 3
    height = 224
    width = 224
    patch_size = 16
    
    # Calculate dimensions
    h_patches = height // patch_size  # 14
    w_patches = width // patch_size   # 14
    num_patches_per_image = h_patches * w_patches  # 196
    total_patches = batch_size * num_patches_per_image  # 784
    patch_dim = patch_size * patch_size * channels  # 768
    
    print(f"\nInput Format:")
    print(f"  Shape: [{batch_size}, {channels}, {height}, {width}]")
    print(f"  Batch size: {batch_size}")
    print(f"  Image size: {height}x{width}")
    
    print(f"\nPacking Format (Internal):")
    print(f"  hidden_states shape: [{total_patches}, {patch_dim}]")
    print(f"  - total_patches = {batch_size} × {h_patches} × {w_patches} = {total_patches}")
    print(f"  - patch_dim = {patch_size} × {patch_size} × {channels} = {patch_dim}")
    print(f"  grid_thw: {[[1, h_patches, w_patches]] * batch_size}")
    
    print(f"\nOutput Format:")
    print(f"  Shape: [{batch_size}, {num_patches_per_image}, hidden_size]")
    print(f"  - Reshaped from packing output [{total_patches}, hidden_size]")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    print("HEVCViTPackingVisionTower Usage Examples")
    print("=" * 60)
    print("\nNOTE: These examples require:")
    print("1. A converted packing model checkpoint")
    print("2. FlashAttention 2 installed")
    print("3. CUDA-compatible GPU")
    print("\n" + "=" * 60)
    
    # Show conversion details (doesn't require actual model)
    show_conversion_details()
    
    # Uncomment to run actual examples (requires model and dependencies)
    # print("\nExample 1: Batch Processing")
    # example_batch_processing()
    # 
    # print("\nExample 2: List Processing")
    # example_list_processing()
    # 
    # print("\nExample 3: With Spatial Dimensions")
    # example_with_spatial_dims()
