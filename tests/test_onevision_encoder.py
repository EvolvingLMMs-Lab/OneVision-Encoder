"""Unit tests for OneVision Encoder model."""

import pytest
import torch

from onevision_encoder import OneVisionEncoderConfig, OneVisionEncoderModel


class TestOneVisionEncoderConfig:
    """Tests for OneVisionEncoderConfig."""

    def test_default_config(self):
        config = OneVisionEncoderConfig()
        assert config.hidden_size == 1024
        assert config.num_hidden_layers == 24
        assert config.num_attention_heads == 16
        assert config.patch_size == 14
        assert config.image_size == 448


class TestVideoRotaryEmbedding:
    """Tests for 3D Rotary Position Embedding."""

    def test_rope_shape(self):
        # The model requires head_dim (hidden_size / num_attention_heads) to be
        # divisible by 32 for the 3D rotary embedding split (4:6:6).
        # Using hidden_size=128 and num_attention_heads=2 gives head_dim=64.
        config = OneVisionEncoderConfig(
            hidden_size=128,
            num_hidden_layers=1,
            num_attention_heads=2,
            patch_size=16,
            image_size=64,
        )
        model = OneVisionEncoderModel(config)

        # 4 patches height, 4 patches width, 2 frames
        freqs = model.video_rope(t=2, h=4, w=4)

        # 2 * 4 * 4 = 32 positions, head_dim//2 = 32
        assert freqs.shape == (32, 32)

    def test_forward_from_positions_consistency(self):
        """Test consistency between forward and forward_from_positions.

        When forward_from_positions is given patch positions that form a dense
        t×h×w grid (same ordering as forward), both methods should produce
        identical outputs.
        """
        # The model requires head_dim (hidden_size / num_attention_heads) to be
        # divisible by 32 for the 3D rotary embedding split (4:6:6).
        # Using hidden_size=128 and num_attention_heads=2 gives head_dim=64.
        config = OneVisionEncoderConfig(
            hidden_size=128,
            num_hidden_layers=1,
            num_attention_heads=2,
            patch_size=16,
            image_size=64,
        )
        model = OneVisionEncoderModel(config)

        t, h, w = 2, 4, 4

        # Get frequencies using forward method
        freqs_forward = model.video_rope(t=t, h=h, w=w)

        # Build patch positions for dense grid matching forward's ordering
        # forward uses: t_ids = arange(t).repeat_interleave(h * w)
        #               h_ids = arange(h).repeat_interleave(w).repeat(t)
        #               w_ids = arange(w).repeat(h).repeat(t)
        device = freqs_forward.device
        t_ids = torch.arange(t, device=device).repeat_interleave(h * w)
        h_ids = torch.arange(h, device=device).repeat_interleave(w).repeat(t)
        w_ids = torch.arange(w, device=device).repeat(h).repeat(t)
        patch_positions = torch.stack([t_ids, h_ids, w_ids], dim=-1)

        # Get frequencies using forward_from_positions method
        freqs_from_positions = model.video_rope.forward_from_positions(patch_positions)

        # Both should have the same shape
        assert freqs_forward.shape == freqs_from_positions.shape, (
            f"Shape mismatch: forward={freqs_forward.shape}, forward_from_positions={freqs_from_positions.shape}"
        )

        # Both should produce identical values
        assert torch.allclose(freqs_forward, freqs_from_positions, rtol=1e-5, atol=1e-5), (
            f"Value mismatch between forward and forward_from_positions. "
            f"Max difference: {(freqs_forward - freqs_from_positions).abs().max().item()}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
