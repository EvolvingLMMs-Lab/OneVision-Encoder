"""Unit tests for OneVision Encoder model."""

import pytest
import torch

from onevision_encoder import OneVisionEncoderConfig, OneVisionEncoderModel


class TestOneVisionEncoderConfig:
    """Tests for OneVisionEncoderConfig."""

    def test_default_config(self):
        config = OneVisionEncoderConfig()
        assert config.hidden_size == 768
        assert config.num_hidden_layers == 12
        assert config.num_attention_heads == 12
        assert config.patch_size == 16
        assert config.image_size == 448

    def test_custom_config(self):
        config = OneVisionEncoderConfig(
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            patch_size=14,
            image_size=224,
        )
        assert config.hidden_size == 768
        assert config.num_hidden_layers == 12
        assert config.num_attention_heads == 12
        assert config.patch_size == 14
        assert config.image_size == 224


class TestOneVisionEncoderModel:
    """Tests for OneVisionEncoderModel."""

    @pytest.fixture
    def small_config(self):
        """Small config for fast testing."""
        return OneVisionEncoderConfig(
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=256,
            patch_size=16,
            image_size=64,
            num_frames=4,
            use_head=False,
        )

    @pytest.fixture
    def small_model(self, small_config):
        """Small model for fast testing."""
        return OneVisionEncoderModel(small_config)

    def test_model_creation(self, small_model, small_config):
        assert small_model.config.hidden_size == small_config.hidden_size
        assert small_model.config.num_hidden_layers == small_config.num_hidden_layers

    def test_forward_4d_input(self, small_model):
        """Test forward pass with 4D input (single image)."""
        batch_size = 2
        x = torch.randn(batch_size, 3, 64, 64)

        output = small_model(x)

        # 64/16 = 4 patches per dim, 4*4 = 16 patches total
        expected_seq_len = 16
        assert output.last_hidden_state.shape == (batch_size, expected_seq_len, 64)

    def test_forward_5d_input(self, small_model):
        """Test forward pass with 5D input (video frames)."""
        batch_size = 2
        num_frames = 4
        x = torch.randn(batch_size, 3, num_frames, 64, 64)

        output = small_model(x)

        # 4 frames * 16 patches per frame = 64 total patches
        expected_seq_len = num_frames * 16
        assert output.last_hidden_state.shape == (batch_size, expected_seq_len, 64)

    def test_forward_with_visible_indices(self, small_model):
        """Test forward pass with sparse visible indices."""
        batch_size = 2
        num_frames = 4
        x = torch.randn(batch_size, 3, num_frames, 64, 64)

        # select only 32 of 64 patches
        visible_indices = torch.arange(32).unsqueeze(0).expand(batch_size, -1)

        output = small_model(x, visible_indices=visible_indices)

        assert output.last_hidden_state.shape == (batch_size, 32, 64)

    def test_output_hidden_states(self, small_model):
        """Test output_hidden_states flag."""
        x = torch.randn(1, 3, 64, 64)

        output = small_model(x, output_hidden_states=True)

        assert output.hidden_states is not None
        # num_layers + 1 (initial embedding)
        assert len(output.hidden_states) == 3

    def test_model_with_head(self):
        """Test model with pooling head."""
        config = OneVisionEncoderConfig(
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=256,
            patch_size=16,
            image_size=64,
            use_head=True,
        )
        model = OneVisionEncoderModel(config)

        x = torch.randn(2, 3, 64, 64)
        output = model(x)

        assert output.pooler_output is not None
        assert output.pooler_output.shape == (2, 64)


class TestVideoRotaryEmbedding:
    """Tests for 3D Rotary Position Embedding."""

    def test_rope_shape(self):
        config = OneVisionEncoderConfig(
            hidden_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            patch_size=16,
            image_size=64,
        )
        model = OneVisionEncoderModel(config)

        # 4 patches height, 4 patches width, 2 frames
        freqs = model.video_rope(t=2, h=4, w=4)

        # 2 * 4 * 4 = 32 positions, head_dim//2 = 8
        assert freqs.shape == (32, 8)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
