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

    def test_forward_from_positions_batched(self):
        """Test batched forward_from_positions with 3D input [batch_size, seq_len, 3].

        This test verifies that forward_from_positions correctly handles batched inputs
        and produces the same result as calling it on each batch element separately.
        """
        config = OneVisionEncoderConfig(
            hidden_size=128,
            num_hidden_layers=1,
            num_attention_heads=2,
            patch_size=16,
            image_size=64,
        )
        model = OneVisionEncoderModel(config)

        batch_size = 2
        t, h, w = 2, 4, 4
        seq_len = t * h * w

        device = model.video_rope.inv_freq_t.device

        # Create patch positions for dense grid (same for both batches in this test)
        t_ids = torch.arange(t, device=device).repeat_interleave(h * w)
        h_ids = torch.arange(h, device=device).repeat_interleave(w).repeat(t)
        w_ids = torch.arange(w, device=device).repeat(h).repeat(t)
        patch_positions_2d = torch.stack([t_ids, h_ids, w_ids], dim=-1)  # [seq_len, 3]

        # Create batched input [batch_size, seq_len, 3]
        patch_positions_3d = patch_positions_2d.unsqueeze(0).expand(batch_size, -1, -1)

        # Get frequencies using 2D input
        freqs_2d = model.video_rope.forward_from_positions(patch_positions_2d)  # [seq_len, half]

        # Get frequencies using 3D batched input
        freqs_3d = model.video_rope.forward_from_positions(patch_positions_3d)  # [batch_size, seq_len, half]

        # Check shapes
        assert freqs_2d.shape == (seq_len, model.video_rope.half), (
            f"2D shape mismatch: expected ({seq_len}, {model.video_rope.half}), got {freqs_2d.shape}"
        )
        assert freqs_3d.shape == (batch_size, seq_len, model.video_rope.half), (
            f"3D shape mismatch: expected ({batch_size}, {seq_len}, {model.video_rope.half}), got {freqs_3d.shape}"
        )

        # Check that each batch element matches the 2D result
        for b in range(batch_size):
            assert torch.allclose(freqs_2d, freqs_3d[b], rtol=1e-5, atol=1e-5), (
                f"Batch {b} value mismatch. Max diff: {(freqs_2d - freqs_3d[b]).abs().max().item()}"
            )

    def test_forward_from_positions_temporal_scaling(self):
        """Test that temporal positions in [0, 64) range produce valid RoPE frequencies.

        This test simulates the chunk_wise_sampling use case where interpolated frame
        indices are scaled to the range [0, target_frames) where target_frames=64.
        """
        config = OneVisionEncoderConfig(
            hidden_size=128,
            num_hidden_layers=1,
            num_attention_heads=2,
            patch_size=16,
            image_size=64,
        )
        model = OneVisionEncoderModel(config)

        device = model.video_rope.inv_freq_t.device
        batch_size = 2
        num_frames = 8  # sampled frames
        patches_per_frame = 16  # 4x4 spatial patches
        target_frames = 64

        # Simulate interpolated indices in [0, target_frames-1] range
        # For num_frames sampled frames from a video, spread across target_frames
        interpolated_t = torch.linspace(0, target_frames - 1, num_frames, dtype=torch.long, device=device)

        # Spatial positions for each frame (4x4 grid)
        h_ids = torch.arange(4, device=device).repeat_interleave(4)  # [0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3]
        w_ids = torch.arange(4, device=device).repeat(4)  # [0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3]

        # Build patch_positions [batch_size, seq_len, 3] for chunk_wise_sampling
        seq_len = num_frames * patches_per_frame
        t_positions = interpolated_t.unsqueeze(-1).expand(-1, patches_per_frame).reshape(-1)  # [seq_len]
        h_positions = h_ids.unsqueeze(0).expand(num_frames, -1).reshape(-1)  # [seq_len]
        w_positions = w_ids.unsqueeze(0).expand(num_frames, -1).reshape(-1)  # [seq_len]

        patch_positions_2d = torch.stack([t_positions, h_positions, w_positions], dim=-1)  # [seq_len, 3]
        patch_positions_3d = patch_positions_2d.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, seq_len, 3]

        # Get frequencies
        freqs = model.video_rope.forward_from_positions(patch_positions_3d)

        # Verify shape
        assert freqs.shape == (batch_size, seq_len, model.video_rope.half), (
            f"Shape mismatch: expected ({batch_size}, {seq_len}, {model.video_rope.half}), got {freqs.shape}"
        )

        # Verify that temporal positions scaled to [0, 64) don't cause any issues
        # (no NaN/Inf values)
        assert torch.isfinite(freqs).all(), "RoPE frequencies contain NaN or Inf values"

        # Verify temporal dimension contribution
        # For the same spatial position but different temporal positions,
        # the temporal part of freqs should differ
        frame_0_patch_0 = freqs[0, 0, :]  # t=0
        frame_7_patch_0 = freqs[0, 7 * patches_per_frame, :]  # t=63
        assert not torch.allclose(frame_0_patch_0, frame_7_patch_0), (
            "RoPE frequencies should differ for different temporal positions"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
