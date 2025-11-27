#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate an animated GIF to visualize how LLaVA-ViT processes video frames 
with residual encoding.

The first frame is kept in full, while subsequent frames are encoded as 
residuals (differences from the first frame).

Usage:
    python generate_vit_residual_gif.py --video /path/to/video.mp4 --output output.gif
    python generate_vit_residual_gif.py --demo --output demo.gif  # Generate demo with synthetic data
"""

import argparse
import os
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Cross-platform font paths - try common locations on different OSes
FONT_PATHS = [
    # Linux
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    # macOS
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial.ttf",
    # Windows
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
]


def get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Get a font with cross-platform support, falling back to default if needed."""
    for font_path in FONT_PATHS:
        if os.path.exists(font_path):
            # Prefer bold fonts for bold requests
            if bold and "Bold" in font_path or "bd" in font_path.lower():
                try:
                    return ImageFont.truetype(font_path, size)
                except OSError:
                    continue
            elif not bold and "Bold" not in font_path and "bd" not in font_path.lower():
                try:
                    return ImageFont.truetype(font_path, size)
                except OSError:
                    continue
    # Fallback: try any available font
    for font_path in FONT_PATHS:
        if os.path.exists(font_path):
            try:
                return ImageFont.truetype(font_path, size)
            except OSError:
                continue
    # Ultimate fallback
    return ImageFont.load_default()


def load_video_frames(video_path: str, num_frames: int = 8, resize: Tuple[int, int] = (224, 224)) -> Optional[np.ndarray]:
    """Load video frames using OpenCV. Returns None if video cannot be loaded."""
    if not os.path.exists(video_path):
        print(f"Warning: Video file not found: {video_path}")
        return None
    
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Warning: Cannot open video: {video_path}")
            return None
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            print(f"Warning: Video has no frames: {video_path}")
            cap.release()
            return None
            
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, resize)
                frames.append(frame)
        cap.release()
        
        if len(frames) == 0:
            print(f"Warning: Could not read any frames from: {video_path}")
            return None
            
        return np.array(frames)
    except ImportError:
        print("Warning: OpenCV not available.")
        return None
    except Exception as e:
        print(f"Warning: Error loading video: {e}")
        return None


def generate_synthetic_frames(num_frames: int = 8, size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """Generate synthetic frames for demo purposes using vectorized operations."""
    frames = []
    base_color = np.array([100, 150, 200], dtype=np.float32)
    
    # Create coordinate grids for vectorized operations
    y_coords, x_coords = np.mgrid[0:size[1], 0:size[0]]
    
    for i in range(num_frames):
        # Create gradient background using vectorized operations
        sin_component = 30 * np.sin(2 * np.pi * x_coords / size[0])
        cos_component = 20 * np.cos(2 * np.pi * y_coords / size[1])
        
        frame = np.zeros((size[1], size[0], 3), dtype=np.float32)
        frame[:, :, 0] = base_color[0] + sin_component
        frame[:, :, 1] = base_color[1] + cos_component
        frame[:, :, 2] = base_color[2]
        
        # Add a moving circle using vectorized distance calculation
        center_x = size[0] * 0.3 + size[0] * 0.4 * i / max(num_frames - 1, 1)
        center_y = size[1] * 0.5 + size[1] * 0.2 * np.sin(2 * np.pi * i / num_frames)
        
        dist = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
        circle_mask = dist < 30
        
        # Apply orange color to circle
        frame[circle_mask, 0] = 255
        frame[circle_mask, 1] = 100
        frame[circle_mask, 2] = 50
        
        frames.append(np.clip(frame, 0, 255).astype(np.uint8))
    
    return np.array(frames)


def compute_residual(current_frame: np.ndarray, reference_frame: np.ndarray) -> np.ndarray:
    """Compute the residual (difference) between current frame and reference frame."""
    residual = current_frame.astype(np.float32) - reference_frame.astype(np.float32)
    # Normalize to 0-255 range for visualization
    residual_normalized = (residual + 255) / 2  # Map [-255, 255] to [0, 255]
    return residual_normalized.clip(0, 255).astype(np.uint8)


def compute_masked_residual(
    current_frame: np.ndarray, 
    reference_frame: np.ndarray, 
    patch_size: int = 16,
    threshold: float = 10.0
) -> np.ndarray:
    """
    Compute the residual with masking - only show patches that have significant changes.
    
    Patches with low residual values (below threshold) are set to white (blank),
    indicating they are not being transmitted/input to the model.
    
    Args:
        current_frame: The current frame (P frame)
        reference_frame: The reference frame (I frame, first frame)
        patch_size: Size of each patch
        threshold: Mean absolute difference threshold to consider a patch as significant.
                   Default of 10.0 was chosen empirically to filter out noise while
                   preserving meaningful motion/change detection in typical video content.
                   Higher values = fewer patches shown, lower values = more patches shown.
                   
    Returns:
        Masked residual where non-significant patches are blank (white)
    """
    # Compute raw residual
    residual = current_frame.astype(np.float32) - reference_frame.astype(np.float32)
    
    # Create output frame - start with white (blank)
    h, w = current_frame.shape[:2]
    masked_residual = np.ones((h, w, 3), dtype=np.uint8) * 255  # White background
    
    # Normalize residual for visualization
    residual_normalized = ((residual + 255) / 2).clip(0, 255).astype(np.uint8)
    
    # Process each patch
    num_patches_h = h // patch_size
    num_patches_w = w // patch_size
    
    for i in range(num_patches_h):
        for j in range(num_patches_w):
            # Extract patch region
            y_start = i * patch_size
            y_end = (i + 1) * patch_size
            x_start = j * patch_size
            x_end = (j + 1) * patch_size
            
            # Get the raw residual for this patch
            patch_residual = residual[y_start:y_end, x_start:x_end]
            
            # Compute mean absolute difference for this patch
            patch_mad = np.mean(np.abs(patch_residual))
            
            # If the patch has significant change, show it
            if patch_mad > threshold:
                masked_residual[y_start:y_end, x_start:x_end] = \
                    residual_normalized[y_start:y_end, x_start:x_end]
    
    return masked_residual


def create_patch_grid(frame: np.ndarray, patch_size: int = 16) -> np.ndarray:
    """Add patch grid overlay to show how the image is divided into patches."""
    frame_with_grid = frame.copy()
    h, w = frame.shape[:2]
    
    # Draw grid lines
    for i in range(0, h, patch_size):
        frame_with_grid[i:i+1, :] = [200, 200, 200]
    for j in range(0, w, patch_size):
        frame_with_grid[:, j:j+1] = [200, 200, 200]
    
    return frame_with_grid


def create_visualization_frame(
    frames: np.ndarray,
    frame_idx: int,
    patch_size: int = 16,
    canvas_size: Tuple[int, int] = (1200, 600)
) -> Image.Image:
    """Create a single visualization frame showing what is input to ViT with position information."""
    canvas = Image.new('RGB', canvas_size, color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    
    # Load fonts using the cross-platform helper function
    font_title = get_font(24, bold=True)
    font_label = get_font(16, bold=False)
    font_small = get_font(12, bold=False)
    
    # Title
    title = f"ViT Input - Frame {frame_idx + 1}/{len(frames)}"
    if frame_idx == 0:
        title += " (I-Frame: Full Input)"
    else:
        title += " (P-Frame: Only Changed Patches)"
    draw.text((canvas_size[0] // 2 - 250, 20), title, fill=(0, 0, 0), font=font_title)
    
    frame_height, frame_width = frames[0].shape[:2]
    num_patches_h = frame_height // patch_size
    num_patches_w = frame_width // patch_size
    
    # Large display size for the main visualization
    display_size = 400
    patch_display_size = display_size // num_patches_h
    
    # Calculate center position for the main ViT input visualization
    main_x = (canvas_size[0] - display_size) // 2
    main_y = 80
    
    # Draw the ViT input with position grid
    if frame_idx == 0:
        # I-frame: show full frame with patch grid and position labels
        frame_with_grid = create_patch_grid(frames[0], patch_size)
        frame_img = Image.fromarray(frame_with_grid)
        frame_resized = frame_img.resize((display_size, display_size), Image.Resampling.NEAREST)
        canvas.paste(frame_resized, (main_x, main_y))
    else:
        # P-frame: show masked residual - only changed patches visible, others blank
        masked_residual = compute_masked_residual(frames[frame_idx], frames[0], patch_size)
        residual_grid = create_patch_grid(masked_residual, patch_size)
        residual_img = Image.fromarray(residual_grid)
        residual_resized = residual_img.resize((display_size, display_size), Image.Resampling.NEAREST)
        canvas.paste(residual_resized, (main_x, main_y))
    
    # Draw position labels on the grid
    for i in range(num_patches_h):
        for j in range(num_patches_w):
            x = main_x + j * patch_display_size
            y = main_y + i * patch_display_size
            # Draw patch position label (row, col)
            pos_label = f"{i},{j}"
            # Draw a small label at the corner of each patch
            draw.text((x + 2, y + 2), pos_label, fill=(100, 100, 100), font=font_small)
    
    # Draw grid lines on top
    for i in range(num_patches_h + 1):
        y = main_y + i * patch_display_size
        draw.line([(main_x, y), (main_x + display_size, y)], fill=(150, 150, 150), width=1)
    for j in range(num_patches_w + 1):
        x = main_x + j * patch_display_size
        draw.line([(x, main_y), (x, main_y + display_size)], fill=(150, 150, 150), width=1)
    
    # Draw border
    draw.rectangle([main_x - 2, main_y - 2, main_x + display_size + 2, main_y + display_size + 2], 
                   outline=(0, 0, 0), width=2)
    
    # Label below the main image
    if frame_idx == 0:
        label = "All patches are input (I-Frame)"
        label_color = (0, 100, 0)
    else:
        label = "White patches = Not Input | Colored patches = Input to ViT"
        label_color = (100, 0, 0)
    draw.text((main_x, main_y + display_size + 10), label, fill=label_color, font=font_label)
    
    # === Legend at bottom ===
    legend_y = main_y + display_size + 50
    draw.text((50, legend_y), "Position Format: (row, col)", fill=(0, 0, 0), font=font_label)
    
    return canvas


def create_architecture_frame(canvas_size: Tuple[int, int] = (1200, 700)) -> Image.Image:
    """Create a frame showing the overall ViT architecture."""
    canvas = Image.new('RGB', canvas_size, color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    
    # Load fonts using the cross-platform helper function
    font_title = get_font(24, bold=True)
    font_label = get_font(14, bold=False)
    font_small = get_font(12, bold=False)
    
    # Title
    draw.text((canvas_size[0] // 2 - 250, 20), "LLaVA-ViT Architecture: Video Residual Encoding", 
              fill=(0, 0, 0), font=font_title)
    
    # Draw the architecture diagram
    # Video Input
    draw.rectangle([50, 100, 200, 200], fill=(200, 220, 240), outline=(0, 0, 100), width=2)
    draw.text((80, 140), "Video Input\n(T×H×W×C)", fill=(0, 0, 0), font=font_label)
    
    # Arrow
    draw.line([(200, 150), (250, 150)], fill=(100, 100, 100), width=2)
    draw.polygon([(245, 145), (250, 150), (245, 155)], fill=(100, 100, 100))
    
    # Frame Decomposition
    draw.rectangle([250, 80, 400, 220], fill=(220, 240, 220), outline=(0, 100, 0), width=2)
    draw.text((260, 90), "Frame\nDecomposition", fill=(0, 0, 0), font=font_label)
    
    # Frame 1 (reference)
    draw.rectangle([260, 130, 310, 160], fill=(100, 200, 100), outline=(0, 100, 0), width=1)
    draw.text((265, 135), "F₁", fill=(255, 255, 255), font=font_small)
    
    # Residual frames
    for i in range(3):
        y = 170 + i * 15
        draw.rectangle([260, y, 310, y + 12], fill=(200, 100, 100), outline=(100, 0, 0), width=1)
        draw.text((265, y + 1), f"ΔF{i+2}", fill=(255, 255, 255), font=font_small)
    
    draw.text((320, 130), "Reference\nFrame", fill=(0, 100, 0), font=font_small)
    draw.text((320, 170), "Residual\nFrames", fill=(100, 0, 0), font=font_small)
    
    # Arrow
    draw.line([(400, 150), (450, 150)], fill=(100, 100, 100), width=2)
    draw.polygon([(445, 145), (450, 150), (445, 155)], fill=(100, 100, 100))
    
    # Patch Embedding
    draw.rectangle([450, 80, 580, 220], fill=(240, 230, 200), outline=(150, 100, 0), width=2)
    draw.text((460, 90), "Patch\nEmbedding", fill=(0, 0, 0), font=font_label)
    
    # Draw patches
    patch_start_x = 460
    patch_start_y = 130
    for i in range(4):
        for j in range(4):
            x = patch_start_x + j * 20
            y = patch_start_y + i * 20
            color = (150 + i * 20, 200 - j * 20, 180)
            draw.rectangle([x, y, x + 18, y + 18], fill=color, outline=(100, 100, 100), width=1)
    
    # Arrow
    draw.line([(580, 150), (630, 150)], fill=(100, 100, 100), width=2)
    draw.polygon([(625, 145), (630, 150), (625, 155)], fill=(100, 100, 100))
    
    # ViT Encoder
    draw.rectangle([630, 60, 800, 240], fill=(230, 220, 250), outline=(100, 0, 150), width=2)
    draw.text((670, 70), "ViT Encoder", fill=(0, 0, 0), font=font_label)
    
    # Encoder layers
    for i in range(6):
        y = 100 + i * 22
        draw.rectangle([645, y, 785, y + 18], fill=(200, 190, 220), outline=(100, 0, 150), width=1)
        draw.text((650, y + 2), f"Transformer Layer {i+1}", fill=(0, 0, 0), font=font_small)
    
    # Arrow
    draw.line([(800, 150), (850, 150)], fill=(100, 100, 100), width=2)
    draw.polygon([(845, 145), (850, 150), (845, 155)], fill=(100, 100, 100))
    
    # Output Features
    draw.rectangle([850, 100, 1000, 200], fill=(200, 240, 240), outline=(0, 100, 100), width=2)
    draw.text((880, 140), "Video\nFeatures", fill=(0, 0, 0), font=font_label)
    
    # Add RoPE notation
    # 3D RoPE explanation: The 4:6:6 split allocates embedding dimensions for T:H:W
    # - 4 parts for temporal (T) dimension
    # - 6 parts for height (H) dimension  
    # - 6 parts for width (W) dimension
    # This enables the model to learn separate positional encodings for time and space
    draw.text((670, 250), "3D RoPE (4:6:6 split)", fill=(100, 0, 150), font=font_small)
    draw.text((670, 270), "T:H:W dimension allocation", fill=(100, 0, 150), font=font_small)
    
    # Legend at bottom
    legend_y = 350
    draw.text((50, legend_y), "Key Components:", fill=(0, 0, 0), font=font_label)
    
    legend_items = [
        ((100, 200, 100), "F₁: First frame (reference, kept in full)"),
        ((200, 100, 100), "ΔFₜ: Residual frames (difference from F₁)"),
        ((200, 190, 220), "Transformer Layers: Self-attention + FFN"),
        ((200, 240, 240), "Output: Encoded video features for LLM"),
    ]
    
    for i, (color, text) in enumerate(legend_items):
        x = 50 + (i % 2) * 500
        y = legend_y + 30 + (i // 2) * 25
        draw.rectangle([x, y, x + 15, y + 15], fill=color, outline=(100, 100, 100), width=1)
        draw.text((x + 25, y), text, fill=(0, 0, 0), font=font_small)
    
    # Mathematical explanation
    math_y = 450
    draw.text((50, math_y), "Mathematical Formulation:", fill=(0, 0, 0), font=font_label)
    draw.text((50, math_y + 25), "1. x₁ = PatchEmbed(Frame₁)  — Full first frame embedding", fill=(0, 0, 0), font=font_small)
    draw.text((50, math_y + 45), "2. xₜ = PatchEmbed(Frameₜ - Frame₁)  — Residual frame embedding (t > 1)", fill=(0, 0, 0), font=font_small)
    draw.text((50, math_y + 65), "3. X = [x₁; x₂; ...; xₜ]  — Concatenate all patch embeddings", fill=(0, 0, 0), font=font_small)
    draw.text((50, math_y + 85), "4. Z = ViTEncoder(X + RoPE)  — Apply transformer with 3D positional encoding", fill=(0, 0, 0), font=font_small)
    
    # Benefits section
    benefits_y = 560
    draw.text((50, benefits_y), "Benefits of Residual Encoding:", fill=(0, 0, 0), font=font_label)
    benefits = [
        "• Reduces temporal redundancy in video frames",
        "• First frame provides full context",
        "• Subsequent frames encode only changes (motion, new objects)",
        "• More efficient representation for video understanding",
    ]
    for i, benefit in enumerate(benefits):
        draw.text((50, benefits_y + 25 + i * 20), benefit, fill=(0, 0, 100), font=font_small)
    
    return canvas


def generate_gif(
    frames: np.ndarray,
    output_path: str,
    patch_size: int = 16,
    duration: int = 500,
    canvas_size: Tuple[int, int] = (1200, 600),
    include_architecture: bool = True
) -> None:
    """Generate the animated GIF showing the ViT processing pipeline."""
    gif_frames: List[Image.Image] = []
    
    # Add architecture overview frame first
    if include_architecture:
        arch_frame = create_architecture_frame((canvas_size[0], 700))
        # Show architecture frame longer
        for _ in range(3):
            gif_frames.append(arch_frame)
    
    # Create visualization frames for each video frame
    for frame_idx in range(len(frames)):
        viz_frame = create_visualization_frame(frames, frame_idx, patch_size, canvas_size)
        gif_frames.append(viz_frame)
    
    # Save as GIF
    if gif_frames:
        gif_frames[0].save(
            output_path,
            save_all=True,
            append_images=gif_frames[1:],
            duration=duration,
            loop=0
        )
        print(f"GIF saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate an animated GIF visualizing LLaVA-ViT video processing with residual encoding"
    )
    parser.add_argument("--video", type=str, help="Path to input video file")
    parser.add_argument("--demo", action="store_true", help="Generate demo with synthetic frames")
    parser.add_argument("--output", type=str, default="vit_residual_encoding.gif", help="Output GIF path")
    parser.add_argument("--num-frames", type=int, default=8, help="Number of frames to sample from video")
    parser.add_argument("--patch-size", type=int, default=16, help="ViT patch size")
    parser.add_argument("--duration", type=int, default=800, help="Duration of each frame in ms")
    parser.add_argument("--width", type=int, default=1200, help="Canvas width")
    parser.add_argument("--height", type=int, default=600, help="Canvas height")
    parser.add_argument("--no-architecture", action="store_true", help="Skip architecture overview frame")
    
    args = parser.parse_args()
    
    frames = None
    
    if args.demo:
        print("Generating demo with synthetic frames...")
        frames = generate_synthetic_frames(args.num_frames)
    elif args.video is not None:
        print(f"Loading video: {args.video}")
        frames = load_video_frames(args.video, args.num_frames)
        if frames is None:
            print("Falling back to demo mode with synthetic frames...")
            frames = generate_synthetic_frames(args.num_frames)
    else:
        print("No video specified, generating demo with synthetic frames...")
        frames = generate_synthetic_frames(args.num_frames)
    
    print(f"Loaded {len(frames)} frames with shape: {frames[0].shape}")
    
    generate_gif(
        frames=frames,
        output_path=args.output,
        patch_size=args.patch_size,
        duration=args.duration,
        canvas_size=(args.width, args.height),
        include_architecture=not args.no_architecture
    )


if __name__ == "__main__":
    main()
