"""Output consistency tests between AutoModel and OneVisionEncoderModel.

This module tests that:
1. AutoModel.from_pretrained() with flash_attention_2
2. OneVisionEncoderModel.from_pretrained()

produce identical outputs for the same input.

The test is parametrized to run across multiple transformers versions:
- 5.0.0rc1
- 4.53.1
"""

import numpy as np
import pytest
import torch
from PIL import Image

from onevision_encoder import OneVisionEncoderModel


# Transformers versions to test
TRANSFORMERS_VERSIONS = ["5.0.0rc1", "4.53.1"]


def get_current_transformers_version():
    """Get the currently installed transformers version."""
    import transformers

    return transformers.__version__


def create_test_image(size=(512, 512), seed=42):
    """Create a deterministic test image."""
    np.random.seed(seed)
    img_array = np.random.randint(0, 256, (size[0], size[1], 3), dtype=np.uint8)
    return Image.fromarray(img_array)


class TestAutoModelOutputConsistency:
    """Tests for output consistency between AutoModel and OneVisionEncoderModel."""

    @pytest.fixture
    def test_image(self):
        """Create a sample test image."""
        return create_test_image()

    @pytest.fixture
    def model_name(self):
        """Model name for loading from HuggingFace."""
        return "lmms-lab-encoder/onevision-encoder-large"

    @pytest.mark.parametrize("expected_version", TRANSFORMERS_VERSIONS)
    def test_transformers_version_compatibility(self, expected_version):
        """
        Test marker to document which transformers versions this test suite
        is designed to be compatible with.

        Note: This test simply documents the expected versions. To actually
        test across versions, you need to run the tests in different environments
        with different transformers versions installed.

        You can run this with:
        - transformers==5.0.0rc1
        - transformers==4.53.1
        """
        current_version = get_current_transformers_version()
        # This is informational - the test passes regardless of version
        # but logs the current version being tested
        print(f"\nTesting with transformers {current_version}")
        print(f"This test is designed for version: {expected_version}")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_automodel_vs_onevision_encoder_model_output_consistency(self, test_image, model_name):
        """
        Test that AutoModel.from_pretrained and OneVisionEncoderModel.from_pretrained
        produce identical outputs.

        This test:
        1. Loads model via AutoModel with flash_attention_2
        2. Loads model via OneVisionEncoderModel
        3. Preprocesses the same image
        4. Compares outputs from both models
        """
        from transformers import AutoImageProcessor, AutoModel

        # Log transformers version for debugging
        current_version = get_current_transformers_version()
        print(f"\nRunning test with transformers version: {current_version}")

        # Load model via AutoModel with flash_attention_2
        auto_model = (
            AutoModel.from_pretrained(model_name, trust_remote_code=True, attn_implementation="flash_attention_2")
            .to("cuda")
            .eval()
        )

        # Load model via OneVisionEncoderModel
        onevision_model = (
            OneVisionEncoderModel.from_pretrained(
                model_name, trust_remote_code=True, attn_implementation="flash_attention_2"
            )
            .to("cuda")
            .eval()
        )

        # Load preprocessor
        preprocessor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)

        # Preprocess image
        inputs = preprocessor(images=test_image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to("cuda")

        # Run inference with both models
        with torch.no_grad():
            with torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda"):
                auto_output = auto_model(pixel_values)
                onevision_output = onevision_model(pixel_values)

        # Compare last_hidden_state
        assert auto_output.last_hidden_state.shape == onevision_output.last_hidden_state.shape, (
            f"Shape mismatch: AutoModel={auto_output.last_hidden_state.shape}, "
            f"OneVisionEncoderModel={onevision_output.last_hidden_state.shape}"
        )

        # Check if outputs are identical
        is_close = torch.allclose(
            auto_output.last_hidden_state, onevision_output.last_hidden_state, rtol=1e-4, atol=1e-4
        )

        if not is_close:
            max_diff = (auto_output.last_hidden_state - onevision_output.last_hidden_state).abs().max().item()
            pytest.fail(
                f"Output mismatch between AutoModel and OneVisionEncoderModel!\n"
                f"Max difference: {max_diff}\n"
                f"AutoModel output stats: min={auto_output.last_hidden_state.min()}, "
                f"max={auto_output.last_hidden_state.max()}\n"
                f"OneVisionEncoderModel output stats: "
                f"min={onevision_output.last_hidden_state.min()}, "
                f"max={onevision_output.last_hidden_state.max()}"
            )

        # Compare pooler_output if present
        if auto_output.pooler_output is not None and onevision_output.pooler_output is not None:
            assert auto_output.pooler_output.shape == onevision_output.pooler_output.shape, (
                f"Pooler output shape mismatch: AutoModel={auto_output.pooler_output.shape}, "
                f"OneVisionEncoderModel={onevision_output.pooler_output.shape}"
            )

            is_close_pooler = torch.allclose(
                auto_output.pooler_output, onevision_output.pooler_output, rtol=1e-4, atol=1e-4
            )

            if not is_close_pooler:
                max_diff_pooler = (auto_output.pooler_output - onevision_output.pooler_output).abs().max().item()
                pytest.fail(f"Pooler output mismatch!\nMax difference: {max_diff_pooler}")

        # Clean up GPU memory
        del auto_model, onevision_model
        torch.cuda.empty_cache()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_automodel_vs_onevision_encoder_model_eager_attention(self, test_image, model_name):
        """
        Test output consistency with eager attention implementation.

        This tests that both loading methods produce identical results
        when using the eager (non-flash) attention implementation.
        """
        from transformers import AutoImageProcessor, AutoModel

        # Log transformers version
        current_version = get_current_transformers_version()
        print(f"\nRunning test with transformers version: {current_version}")

        # Load model via AutoModel with eager attention
        auto_model = (
            AutoModel.from_pretrained(model_name, trust_remote_code=True, attn_implementation="eager")
            .to("cuda")
            .eval()
        )

        # Load model via OneVisionEncoderModel with eager attention
        onevision_model = (
            OneVisionEncoderModel.from_pretrained(model_name, trust_remote_code=True, attn_implementation="eager")
            .to("cuda")
            .eval()
        )

        # Load preprocessor
        preprocessor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)

        # Preprocess image
        inputs = preprocessor(images=test_image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to("cuda")

        # Run inference
        with torch.no_grad():
            auto_output = auto_model(pixel_values)
            onevision_output = onevision_model(pixel_values)

        # Compare outputs
        is_close = torch.allclose(
            auto_output.last_hidden_state, onevision_output.last_hidden_state, rtol=1e-4, atol=1e-4
        )

        if not is_close:
            max_diff = (auto_output.last_hidden_state - onevision_output.last_hidden_state).abs().max().item()
            pytest.fail(f"Output mismatch with eager attention!\nMax difference: {max_diff}")

        # Clean up
        del auto_model, onevision_model
        torch.cuda.empty_cache()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_automodel_vs_onevision_encoder_model_dtype_consistency(self, test_image, model_name):
        """
        Test output consistency with different dtypes.

        This ensures both loading methods handle bfloat16 identically.
        """
        from transformers import AutoImageProcessor, AutoModel

        # Log transformers version
        current_version = get_current_transformers_version()
        print(f"\nRunning test with transformers version: {current_version}")

        # Load models with bfloat16
        auto_model = (
            AutoModel.from_pretrained(
                model_name, trust_remote_code=True, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16
            )
            .to("cuda")
            .eval()
        )

        onevision_model = (
            OneVisionEncoderModel.from_pretrained(
                model_name, trust_remote_code=True, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16
            )
            .to("cuda")
            .eval()
        )

        # Load preprocessor
        preprocessor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)

        # Preprocess image
        inputs = preprocessor(images=test_image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to("cuda", dtype=torch.bfloat16)

        # Run inference
        with torch.no_grad():
            auto_output = auto_model(pixel_values)
            onevision_output = onevision_model(pixel_values)

        # Verify dtype
        assert auto_output.last_hidden_state.dtype == torch.bfloat16, (
            f"AutoModel output dtype: {auto_output.last_hidden_state.dtype}"
        )
        assert onevision_output.last_hidden_state.dtype == torch.bfloat16, (
            f"OneVisionEncoderModel output dtype: {onevision_output.last_hidden_state.dtype}"
        )

        # Compare outputs (use larger tolerance for bf16)
        is_close = torch.allclose(
            auto_output.last_hidden_state.float(), onevision_output.last_hidden_state.float(), rtol=1e-3, atol=1e-3
        )

        if not is_close:
            max_diff = (
                (auto_output.last_hidden_state.float() - onevision_output.last_hidden_state.float()).abs().max().item()
            )
            pytest.fail(f"bfloat16 output mismatch!\nMax difference: {max_diff}")

        # Clean up
        del auto_model, onevision_model
        torch.cuda.empty_cache()


class TestDirectClassInstantiation:
    """
    Tests for output consistency using direct class instantiation.

    These tests do NOT use AutoModel - they directly instantiate OneVisionEncoderModel
    class to verify output consistency across different transformers versions.
    """

    @pytest.fixture
    def test_image(self):
        """Create a sample test image."""
        return create_test_image()

    @pytest.fixture
    def model_name(self):
        """Model name for loading from HuggingFace."""
        return "lmms-lab-encoder/onevision-encoder-large"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_direct_instantiation_flash_attention(self, test_image, model_name):
        """
        Test direct class instantiation with flash_attention_2.

        This test directly uses OneVisionEncoderModel.from_pretrained()
        without AutoModel to verify output consistency.
        """
        from onevision_encoder import OneVisionEncoderModel
        from transformers import AutoImageProcessor

        current_version = get_current_transformers_version()
        print(f"\nRunning direct instantiation test with transformers version: {current_version}")

        # Load model directly using OneVisionEncoderModel class
        model = (
            OneVisionEncoderModel.from_pretrained(
                model_name, trust_remote_code=True, attn_implementation="flash_attention_2"
            )
            .to("cuda")
            .eval()
        )

        # Load preprocessor
        preprocessor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)

        # Preprocess image
        inputs = preprocessor(images=test_image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to("cuda")

        # Run inference
        with torch.no_grad():
            with torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda"):
                output = model(pixel_values)

        # Verify output shape and values
        assert output.last_hidden_state is not None, "last_hidden_state should not be None"
        assert output.last_hidden_state.shape[0] == 1, "Batch size should be 1"
        assert not torch.isnan(output.last_hidden_state).any(), "Output contains NaN values"
        assert not torch.isinf(output.last_hidden_state).any(), "Output contains Inf values"

        print(f"Output shape: {output.last_hidden_state.shape}")
        print(
            f"Output stats: min={output.last_hidden_state.min().item():.4f}, "
            f"max={output.last_hidden_state.max().item():.4f}, "
            f"mean={output.last_hidden_state.mean().item():.4f}"
        )

        # Clean up
        del model
        torch.cuda.empty_cache()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_direct_instantiation_eager_attention(self, test_image, model_name):
        """
        Test direct class instantiation with eager attention.

        This test directly uses OneVisionEncoderModel.from_pretrained()
        with eager attention implementation.
        """
        from onevision_encoder import OneVisionEncoderModel
        from transformers import AutoImageProcessor

        current_version = get_current_transformers_version()
        print(f"\nRunning direct instantiation (eager) test with transformers version: {current_version}")

        # Load model directly using OneVisionEncoderModel class with eager attention
        model = (
            OneVisionEncoderModel.from_pretrained(model_name, trust_remote_code=True, attn_implementation="eager")
            .to("cuda")
            .eval()
        )

        # Load preprocessor
        preprocessor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)

        # Preprocess image
        inputs = preprocessor(images=test_image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to("cuda")

        # Run inference
        with torch.no_grad():
            output = model(pixel_values)

        # Verify output shape and values
        assert output.last_hidden_state is not None, "last_hidden_state should not be None"
        assert output.last_hidden_state.shape[0] == 1, "Batch size should be 1"
        assert not torch.isnan(output.last_hidden_state).any(), "Output contains NaN values"
        assert not torch.isinf(output.last_hidden_state).any(), "Output contains Inf values"

        print(f"Output shape: {output.last_hidden_state.shape}")
        print(
            f"Output stats: min={output.last_hidden_state.min().item():.4f}, "
            f"max={output.last_hidden_state.max().item():.4f}, "
            f"mean={output.last_hidden_state.mean().item():.4f}"
        )

        # Clean up
        del model
        torch.cuda.empty_cache()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_direct_instantiation_flash_vs_eager_consistency(self, test_image, model_name):
        """
        Test output consistency between flash_attention_2 and eager attention.

        This test directly instantiates two models with different attention
        implementations and compares their outputs to verify consistency.
        """
        from onevision_encoder import OneVisionEncoderModel
        from transformers import AutoImageProcessor

        current_version = get_current_transformers_version()
        print(f"\nRunning flash vs eager consistency test with transformers version: {current_version}")

        # Load model with flash_attention_2
        model_flash = (
            OneVisionEncoderModel.from_pretrained(
                model_name, trust_remote_code=True, attn_implementation="flash_attention_2"
            )
            .to("cuda")
            .eval()
        )

        # Load model with eager attention
        model_eager = (
            OneVisionEncoderModel.from_pretrained(model_name, trust_remote_code=True, attn_implementation="eager")
            .to("cuda")
            .eval()
        )

        # Load preprocessor
        preprocessor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)

        # Preprocess image
        inputs = preprocessor(images=test_image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to("cuda")

        # Run inference with both models
        with torch.no_grad():
            with torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda"):
                output_flash = model_flash(pixel_values)
                output_eager = model_eager(pixel_values)

        # Compare shapes
        assert output_flash.last_hidden_state.shape == output_eager.last_hidden_state.shape, (
            f"Shape mismatch: flash={output_flash.last_hidden_state.shape}, "
            f"eager={output_eager.last_hidden_state.shape}"
        )

        # Compare outputs (allow some tolerance due to different implementations)
        max_diff = (output_flash.last_hidden_state - output_eager.last_hidden_state).abs().max().item()
        mean_diff = (output_flash.last_hidden_state - output_eager.last_hidden_state).abs().mean().item()

        print(f"Flash vs Eager comparison:")
        print(f"  Max difference: {max_diff:.6f}")
        print(f"  Mean difference: {mean_diff:.6f}")

        # Flash and Eager attention produce similar but not identical results due to:
        # 1. Different numerical algorithms (FlashAttention uses online softmax)
        # 2. bfloat16 precision limitations with autocast
        # 3. Different memory access patterns affecting floating point accumulation
        # Tolerance of 1e-2 is appropriate for this comparison
        FLASH_EAGER_RTOL = 1e-2
        FLASH_EAGER_ATOL = 1e-2
        is_close = torch.allclose(
            output_flash.last_hidden_state,
            output_eager.last_hidden_state,
            rtol=FLASH_EAGER_RTOL,
            atol=FLASH_EAGER_ATOL,
        )

        if not is_close:
            pytest.fail(
                f"Output mismatch between flash_attention_2 and eager!\n"
                f"Max difference: {max_diff}\n"
                f"Mean difference: {mean_diff}\n"
                f"Flash stats: min={output_flash.last_hidden_state.min()}, max={output_flash.last_hidden_state.max()}\n"
                f"Eager stats: min={output_eager.last_hidden_state.min()}, max={output_eager.last_hidden_state.max()}"
            )

        # Clean up
        del model_flash, model_eager
        torch.cuda.empty_cache()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_direct_instantiation_deterministic_output(self, test_image, model_name):
        """
        Test that direct class instantiation produces deterministic outputs.

        Running the same model twice with the same input should produce
        identical outputs.
        """
        from onevision_encoder import OneVisionEncoderModel
        from transformers import AutoImageProcessor

        current_version = get_current_transformers_version()
        print(f"\nRunning deterministic output test with transformers version: {current_version}")

        # Load model
        model = (
            OneVisionEncoderModel.from_pretrained(
                model_name, trust_remote_code=True, attn_implementation="flash_attention_2"
            )
            .to("cuda")
            .eval()
        )

        # Load preprocessor
        preprocessor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)

        # Preprocess image
        inputs = preprocessor(images=test_image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to("cuda")

        # Run inference twice
        # Note: Using autocast with bfloat16 during eval mode should be deterministic
        # since dropout is disabled and no stochastic operations are performed
        with torch.no_grad():
            with torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda"):
                output1 = model(pixel_values)
                output2 = model(pixel_values)

        # Outputs should be identical
        is_identical = torch.equal(output1.last_hidden_state, output2.last_hidden_state)

        if not is_identical:
            max_diff = (output1.last_hidden_state - output2.last_hidden_state).abs().max().item()
            pytest.fail(f"Non-deterministic output detected!\nMax difference between two runs: {max_diff}")

        print("Deterministic output verified: two identical runs produce identical outputs")

        # Clean up
        del model
        torch.cuda.empty_cache()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_direct_instantiation_bfloat16(self, test_image, model_name):
        """
        Test direct class instantiation with bfloat16 dtype.
        """
        from onevision_encoder import OneVisionEncoderModel
        from transformers import AutoImageProcessor

        current_version = get_current_transformers_version()
        print(f"\nRunning bfloat16 test with transformers version: {current_version}")

        # Load model with bfloat16
        model = (
            OneVisionEncoderModel.from_pretrained(
                model_name, trust_remote_code=True, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16
            )
            .to("cuda")
            .eval()
        )

        # Load preprocessor
        preprocessor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)

        # Preprocess image
        inputs = preprocessor(images=test_image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to("cuda", dtype=torch.bfloat16)

        # Run inference
        with torch.no_grad():
            output = model(pixel_values)

        # Verify dtype
        assert output.last_hidden_state.dtype == torch.bfloat16, (
            f"Expected bfloat16 output, got: {output.last_hidden_state.dtype}"
        )

        # Verify no NaN/Inf
        assert not torch.isnan(output.last_hidden_state).any(), "Output contains NaN values"
        assert not torch.isinf(output.last_hidden_state).any(), "Output contains Inf values"

        print(f"bfloat16 output verified: dtype={output.last_hidden_state.dtype}")
        print(f"Output shape: {output.last_hidden_state.shape}")

        # Clean up
        del model
        torch.cuda.empty_cache()


class TestTransformersVersionInfo:
    """Test class to document and verify transformers version information."""

    def test_log_transformers_version(self):
        """Log the current transformers version for CI/CD tracking."""
        current_version = get_current_transformers_version()
        print(f"\n{'=' * 60}")
        print(f"Current transformers version: {current_version}")
        print(f"Tested versions: {TRANSFORMERS_VERSIONS}")
        print(f"Version in tested list: {current_version in TRANSFORMERS_VERSIONS}")
        print(f"{'=' * 60}")

        # This test always passes - it's for information only
        assert True

    def test_verify_version_list(self):
        """Verify the test version list is valid."""
        assert len(TRANSFORMERS_VERSIONS) >= 2, "Should test at least 2 transformers versions"
        assert "5.0.0rc1" in TRANSFORMERS_VERSIONS
        assert "4.53.1" in TRANSFORMERS_VERSIONS


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
