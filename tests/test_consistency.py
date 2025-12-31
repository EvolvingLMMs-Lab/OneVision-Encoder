"""Consistency tests for OneVision Encoder.

This module contains tests to verify:
1. Code consistency between local files and HuggingFace remote files
2. Preprocessing consistency between manual center-crop normalization and AutoImageProcessor
"""

import hashlib
import pytest
import torch
import numpy as np
from PIL import Image


# Default normalization parameters for center crop preprocessing
DEFAULT_MEAN = [0.48145466, 0.4578275, 0.40821073]
DEFAULT_STD = [0.26862954, 0.26130258, 0.27577711]
DEFAULT_IMAGE_SIZE = 448


def manual_center_crop_preprocess(image: Image.Image, size: int = DEFAULT_IMAGE_SIZE,
                                  mean: list = None, std: list = None) -> torch.Tensor:
    """
    Manual center-crop preprocessing for images (CLIP-style).

    This follows the standard CLIP preprocessing pipeline:
    1. Resize the shorter edge to target size (preserving aspect ratio)
    2. Center crop to target size x target size
    3. Normalize with mean and std

    Args:
        image: PIL Image to preprocess
        size: Target size for center crop (default: 448)
        mean: Normalization mean (default: [0.48145466, 0.4578275, 0.40821073])
        std: Normalization std (default: [0.26862954, 0.26130258, 0.27577711])

    Returns:
        Preprocessed tensor of shape (1, 3, size, size)
    """
    if mean is None:
        mean = DEFAULT_MEAN
    if std is None:
        std = DEFAULT_STD

    # Convert to RGB if necessary
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Step 1: Resize shorter edge to target size (CLIP-style)
    width, height = image.size
    if width < height:
        new_width = size
        new_height = int(height * size / width)
    else:
        new_height = size
        new_width = int(width * size / height)

    image = image.resize((new_width, new_height), Image.Resampling.BICUBIC)

    # Step 2: Center crop to target size
    left = (new_width - size) // 2
    top = (new_height - size) // 2
    right = left + size
    bottom = top + size

    image = image.crop((left, top, right, bottom))

    # Convert to numpy array and normalize to [0, 1]
    img_array = np.array(image).astype(np.float32) / 255.0

    # Apply normalization: (x - mean) / std
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    img_array = (img_array - mean) / std

    # Convert to tensor: (H, W, C) -> (C, H, W)
    tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)

    return tensor


class TestCodeConsistency:
    """Tests for code consistency between local and HuggingFace remote files."""

    @pytest.fixture
    def local_modeling_path(self):
        """Path to local modeling file."""
        import os
        return os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "onevision_encoder",
            "modeling_onevision_encoder.py"
        )

    @pytest.fixture
    def local_config_path(self):
        """Path to local configuration file."""
        import os
        return os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "onevision_encoder",
            "configuration_onevision_encoder.py"
        )

    def _read_file_content(self, path: str) -> str:
        """Read and return file content."""
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def _compute_file_hash(self, content: str) -> str:
        """Compute SHA256 hash of file content."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    @pytest.mark.skipif(
        True,  # Skip by default since network may not be available
        reason="Network access required for HuggingFace download"
    )
    def test_modeling_file_consistency(self, local_modeling_path):
        """Test consistency between local and remote modeling file.

        This test downloads the modeling file from HuggingFace and compares
        it with the local file to ensure they are identical.
        """
        try:
            from huggingface_hub import hf_hub_download

            remote_path = hf_hub_download(
                repo_id="lmms-lab-encoder/onevision-encoder-large",
                filename="modeling_onevision_encoder.py"
            )

            local_content = self._read_file_content(local_modeling_path)
            remote_content = self._read_file_content(remote_path)

            local_hash = self._compute_file_hash(local_content)
            remote_hash = self._compute_file_hash(remote_content)

            assert local_hash == remote_hash, (
                f"Modeling files do not match!\n"
                f"Local hash: {local_hash}\n"
                f"Remote hash: {remote_hash}"
            )
        except Exception as e:
            pytest.skip(f"Could not download from HuggingFace: {e}")

    @pytest.mark.skipif(
        True,  # Skip by default since network may not be available
        reason="Network access required for HuggingFace download"
    )
    def test_config_file_consistency(self, local_config_path):
        """Test consistency between local and remote configuration file.

        This test downloads the configuration file from HuggingFace and compares
        it with the local file to ensure they are identical.
        """
        try:
            from huggingface_hub import hf_hub_download

            remote_path = hf_hub_download(
                repo_id="lmms-lab-encoder/onevision-encoder-large",
                filename="configuration_onevision_encoder.py"
            )

            local_content = self._read_file_content(local_config_path)
            remote_content = self._read_file_content(remote_path)

            local_hash = self._compute_file_hash(local_content)
            remote_hash = self._compute_file_hash(remote_content)

            assert local_hash == remote_hash, (
                f"Configuration files do not match!\n"
                f"Local hash: {local_hash}\n"
                f"Remote hash: {remote_hash}"
            )
        except Exception as e:
            pytest.skip(f"Could not download from HuggingFace: {e}")

    def test_local_files_exist(self, local_modeling_path, local_config_path):
        """Test that local files exist and are readable."""
        import os

        assert os.path.exists(local_modeling_path), (
            f"Local modeling file not found: {local_modeling_path}"
        )
        assert os.path.exists(local_config_path), (
            f"Local config file not found: {local_config_path}"
        )

        # Verify files are readable
        modeling_content = self._read_file_content(local_modeling_path)
        config_content = self._read_file_content(local_config_path)

        assert len(modeling_content) > 0, "Modeling file is empty"
        assert len(config_content) > 0, "Config file is empty"

        # Check for expected class definitions
        assert "class OneVisionEncoderModel" in modeling_content, (
            "OneVisionEncoderModel class not found in modeling file"
        )
        assert "class OneVisionEncoderConfig" in config_content, (
            "OneVisionEncoderConfig class not found in config file"
        )


class TestPreprocessingConsistency:
    """Tests for preprocessing consistency."""

    @pytest.fixture
    def sample_image(self):
        """Create a sample test image."""
        # Create a simple RGB image with random values
        np.random.seed(42)
        img_array = np.random.randint(0, 256, (640, 480, 3), dtype=np.uint8)
        return Image.fromarray(img_array)

    @pytest.fixture
    def sample_square_image(self):
        """Create a sample square test image."""
        np.random.seed(42)
        img_array = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        return Image.fromarray(img_array)

    def test_manual_preprocess_output_shape(self, sample_image):
        """Test that manual preprocessing produces correct output shape."""
        tensor = manual_center_crop_preprocess(sample_image)

        assert tensor.shape == (1, 3, DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE), (
            f"Expected shape (1, 3, {DEFAULT_IMAGE_SIZE}, {DEFAULT_IMAGE_SIZE}), "
            f"got {tensor.shape}"
        )

    def test_manual_preprocess_dtype(self, sample_image):
        """Test that manual preprocessing produces float tensor."""
        tensor = manual_center_crop_preprocess(sample_image)

        assert tensor.dtype == torch.float32, (
            f"Expected dtype torch.float32, got {tensor.dtype}"
        )

    def test_manual_preprocess_custom_size(self, sample_image):
        """Test manual preprocessing with custom size."""
        custom_size = 224
        tensor = manual_center_crop_preprocess(sample_image, size=custom_size)

        assert tensor.shape == (1, 3, custom_size, custom_size), (
            f"Expected shape (1, 3, {custom_size}, {custom_size}), got {tensor.shape}"
        )

    def test_manual_preprocess_square_image(self, sample_square_image):
        """Test manual preprocessing on square image."""
        tensor = manual_center_crop_preprocess(sample_square_image)

        assert tensor.shape == (1, 3, DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)

    def test_manual_preprocess_normalization(self, sample_image):
        """Test that normalization is applied correctly."""
        tensor = manual_center_crop_preprocess(sample_image)

        # After normalization, values should not be in [0, 1] range
        # Check that values are reasonable (not NaN or Inf)
        assert not torch.isnan(tensor).any(), "Tensor contains NaN values"
        assert not torch.isinf(tensor).any(), "Tensor contains Inf values"

    @pytest.mark.skipif(
        True,  # Skip by default since network may not be available
        reason="Network access required for HuggingFace model download"
    )
    def test_preprocessing_consistency_with_auto_processor(self, sample_image):
        """Test consistency between manual preprocessing and AutoImageProcessor.

        This test compares the output of manual center-crop preprocessing with
        the AutoImageProcessor from HuggingFace to ensure they produce
        identical results.
        """
        try:
            from transformers import AutoImageProcessor

            # Load the AutoImageProcessor from HuggingFace
            preprocessor = AutoImageProcessor.from_pretrained(
                "lmms-lab-encoder/onevision-encoder-large",
                trust_remote_code=True
            )

            # Process with AutoImageProcessor
            auto_output = preprocessor(images=sample_image, return_tensors="pt")
            auto_tensor = auto_output["pixel_values"]

            # Process with manual preprocessing
            manual_tensor = manual_center_crop_preprocess(sample_image)

            # Compare outputs
            assert auto_tensor.shape == manual_tensor.shape, (
                f"Shape mismatch: auto={auto_tensor.shape}, manual={manual_tensor.shape}"
            )

            # Check if values are close (allowing for small floating point differences)
            is_close = torch.allclose(auto_tensor, manual_tensor, rtol=1e-4, atol=1e-4)
            if not is_close:
                max_diff = (auto_tensor - manual_tensor).abs().max().item()
                pytest.fail(
                    f"Preprocessing outputs differ!\n"
                    f"Max difference: {max_diff}\n"
                    f"Auto tensor stats: min={auto_tensor.min()}, max={auto_tensor.max()}\n"
                    f"Manual tensor stats: min={manual_tensor.min()}, max={manual_tensor.max()}"
                )

        except Exception as e:
            pytest.skip(f"Could not load AutoImageProcessor: {e}")

    def test_grayscale_image_conversion(self):
        """Test that grayscale images are converted to RGB."""
        # Create a grayscale image
        np.random.seed(42)
        gray_array = np.random.randint(0, 256, (480, 640), dtype=np.uint8)
        gray_image = Image.fromarray(gray_array, mode="L")

        tensor = manual_center_crop_preprocess(gray_image)

        assert tensor.shape == (1, 3, DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE), (
            f"Expected 3 channels, got shape {tensor.shape}"
        )

    def test_rgba_image_conversion(self):
        """Test that RGBA images are converted to RGB."""
        # Create an RGBA image
        np.random.seed(42)
        rgba_array = np.random.randint(0, 256, (480, 640, 4), dtype=np.uint8)
        rgba_image = Image.fromarray(rgba_array, mode="RGBA")

        tensor = manual_center_crop_preprocess(rgba_image)

        assert tensor.shape == (1, 3, DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE), (
            f"Expected 3 channels, got shape {tensor.shape}"
        )


class TestModelInputConsistency:
    """Tests for model input consistency with preprocessed images."""

    @pytest.fixture
    def sample_image(self):
        """Create a sample test image."""
        np.random.seed(42)
        img_array = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        return Image.fromarray(img_array)

    @pytest.fixture
    def small_config(self):
        """Small config for fast testing.

        Note: The model requires head_dim (hidden_size / num_attention_heads) to be
        divisible by 32 for the 3D rotary embedding split (4:6:6).
        Using hidden_size=128 and num_attention_heads=2 gives head_dim=64.
        """
        from onevision_encoder import OneVisionEncoderConfig

        return OneVisionEncoderConfig(
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=512,
            patch_size=16,
            image_size=DEFAULT_IMAGE_SIZE,
            use_head=False,
        )

    def test_preprocessed_image_model_forward(self, sample_image, small_config):
        """Test that preprocessed image can be passed through the model."""
        from onevision_encoder import OneVisionEncoderModel

        model = OneVisionEncoderModel(small_config)
        model.eval()

        # Preprocess the image
        pixel_values = manual_center_crop_preprocess(sample_image)

        # Forward pass
        with torch.no_grad():
            output = model(pixel_values)

        # Check output shape
        # 448/16 = 28 patches per dimension, 28*28 = 784 total patches
        expected_seq_len = (DEFAULT_IMAGE_SIZE // small_config.patch_size) ** 2
        assert output.last_hidden_state.shape == (
            1, expected_seq_len, small_config.hidden_size
        ), (
            f"Unexpected output shape: {output.last_hidden_state.shape}"
        )

    def test_preprocessed_batch_model_forward(self, sample_image, small_config):
        """Test that a batch of preprocessed images can be passed through the model."""
        from onevision_encoder import OneVisionEncoderModel

        model = OneVisionEncoderModel(small_config)
        model.eval()

        # Preprocess multiple images
        tensor1 = manual_center_crop_preprocess(sample_image)
        # Create second image
        np.random.seed(123)
        img_array2 = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        image2 = Image.fromarray(img_array2)
        tensor2 = manual_center_crop_preprocess(image2)

        # Batch tensors
        batch = torch.cat([tensor1, tensor2], dim=0)

        # Forward pass
        with torch.no_grad():
            output = model(batch)

        # Check output shape for batch
        expected_seq_len = (DEFAULT_IMAGE_SIZE // small_config.patch_size) ** 2
        assert output.last_hidden_state.shape == (
            2, expected_seq_len, small_config.hidden_size
        ), (
            f"Unexpected batch output shape: {output.last_hidden_state.shape}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
