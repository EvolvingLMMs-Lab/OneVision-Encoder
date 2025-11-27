#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate an animated video to visualize how LLaVA-ViT processes video frames 
with residual encoding.

The first frame is kept in full, while subsequent frames are encoded as 
residuals (differences from the first frame). For P-frames, only the most 
important tokens (based on residual magnitude) are kept.

Features:
- Token limiting: 64 frames max, with top 196*7=1372 most important tokens for P-frames
- Video output (MP4) for better quality
- Side-by-side visualization of original frame and residual

Usage:
    python generate_vit_residual_gif.py --video /path/to/video.mp4 --output output.mp4
    python generate_vit_residual_gif.py --demo --output demo.mp4  # Generate demo with synthetic data
"""

import argparse
import os
from pathlib import Path
from typing import Optional, Tuple, List

import imageio
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


def draw_rounded_rectangle(
    draw: ImageDraw.Draw,
    xy: List[int],
    radius: int = 10,
    fill: Optional[Tuple[int, int, int]] = None,
    outline: Optional[Tuple[int, int, int]] = None,
    width: int = 1
) -> None:
    """Draw a rounded rectangle with fallback for older Pillow versions."""
    try:
        # Try using the built-in rounded_rectangle (Pillow >= 8.2.0)
        draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)
    except AttributeError:
        # Fallback: draw regular rectangle for older versions
        draw.rectangle(xy, fill=fill, outline=outline, width=width)


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
    threshold: float = 10.0,
    max_tokens: Optional[int] = None
) -> Tuple[np.ndarray, int]:
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
        max_tokens: Maximum number of tokens to keep. If None, use threshold-based filtering.
                   
    Returns:
        Tuple of (masked residual where non-significant patches are blank, number of tokens kept)
    """
    # Compute raw residual
    residual = current_frame.astype(np.float32) - reference_frame.astype(np.float32)
    
    # Create output frame - start with white (blank)
    h, w = current_frame.shape[:2]
    masked_residual = np.ones((h, w, 3), dtype=np.uint8) * 255  # White background
    
    # Normalize residual for visualization
    residual_normalized = ((residual + 255) / 2).clip(0, 255).astype(np.uint8)
    
    # Process each patch and collect importance scores
    num_patches_h = h // patch_size
    num_patches_w = w // patch_size
    
    patch_scores = []
    for i in range(num_patches_h):
        for j in range(num_patches_w):
            y_start = i * patch_size
            y_end = (i + 1) * patch_size
            x_start = j * patch_size
            x_end = (j + 1) * patch_size
            
            patch_residual = residual[y_start:y_end, x_start:x_end]
            patch_mad = np.mean(np.abs(patch_residual))
            patch_scores.append((patch_mad, i, j, y_start, y_end, x_start, x_end))
    
    # Sort by importance (descending) and select top tokens
    patch_scores.sort(key=lambda x: x[0], reverse=True)
    
    if max_tokens is not None:
        # Keep only the top max_tokens patches
        selected_patches = patch_scores[:max_tokens]
    else:
        # Use threshold-based filtering
        selected_patches = [p for p in patch_scores if p[0] > threshold]
    
    tokens_kept = len(selected_patches)
    
    for score, i, j, y_start, y_end, x_start, x_end in selected_patches:
        masked_residual[y_start:y_end, x_start:x_end] = \
            residual_normalized[y_start:y_end, x_start:x_end]
    
    return masked_residual, tokens_kept


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
    canvas_size: Tuple[int, int] = (1600, 720),
    max_tokens_per_frame: Optional[int] = None,
    total_frames: int = 64
) -> Image.Image:
    """Create a single visualization frame showing original frame, I-frame reference, and residual.
    
    For P-frames (frame_idx > 0), displays three panels:
    - Left: Original current frame
    - Center: First frame (I-frame reference)
    - Right: Residual visualization with top tokens
    
    Args:
        frames: All video frames
        frame_idx: Current frame index
        patch_size: Size of each ViT patch
        canvas_size: Size of the output canvas
        max_tokens_per_frame: Maximum tokens to keep per P-frame (default: 196*7/63 for 64 frames)
        total_frames: Total number of frames (default 64, used to calculate tokens per frame)
    """
    # Calculate max tokens per frame if not specified
    # For 64 frames: first frame gets 196 tokens, remaining 63 frames share 196*7 = 1372 tokens
    if max_tokens_per_frame is None:
        max_tokens_per_frame = (196 * 7) // (total_frames - 1) if total_frames > 1 else 196
    
    # Use a dark gradient background for modern look
    canvas = Image.new('RGB', canvas_size, color=(25, 28, 35))
    draw = ImageDraw.Draw(canvas)
    
    # Load fonts
    font_title = get_font(28, bold=True)
    font_label = get_font(18, bold=False)
    font_small = get_font(14, bold=False)
    font_stats = get_font(16, bold=True)
    
    frame_height, frame_width = frames[0].shape[:2]
    num_patches_h = frame_height // patch_size
    num_patches_w = frame_width // patch_size
    total_patches = num_patches_h * num_patches_w
    
    # Header area with gradient-like background
    header_height = 70
    for y in range(header_height):
        alpha = 1 - (y / header_height) * 0.3
        color = (int(40 * alpha), int(45 * alpha), int(55 * alpha))
        draw.line([(0, y), (canvas_size[0], y)], fill=color)
    
    # Title with modern styling
    title = f"LLaVA-ViT Residual Encoding"
    draw.text((canvas_size[0] // 2 - 180, 15), title, fill=(255, 255, 255), font=font_title)
    
    # Frame counter with accent color
    frame_info = f"Frame {frame_idx + 1}/{len(frames)}"
    frame_type = "I-Frame (Full)" if frame_idx == 0 else f"P-Frame (Top {max_tokens_per_frame} tokens)"
    draw.text((canvas_size[0] // 2 - 120, 45), f"{frame_info} • {frame_type}", 
              fill=(100, 200, 255), font=font_label)
    
    main_y = 90
    
    if frame_idx == 0:
        # I-frame: show original frame larger and centered
        display_size = 450
        gap = 100
        total_width = display_size * 2 + gap
        left_x = (canvas_size[0] - total_width) // 2
        right_x = left_x + display_size + gap
        
        # === Left: Original Frame ===
        _draw_frame_panel(
            canvas, draw, frames[0], left_x, main_y, display_size,
            "Original Frame", (100, 200, 255), patch_size, font_label, show_grid=False
        )
        
        # === Right: ViT Input (same as original for I-frame) ===
        _draw_frame_panel(
            canvas, draw, frames[0], right_x, main_y, display_size,
            "ViT Input (All 196 Tokens)", (100, 255, 150), patch_size, font_label, show_grid=True
        )
        
        # Stats bar
        stats_y = main_y + display_size + 60
        _draw_stats_bar(
            draw, canvas_size[0] // 2, stats_y,
            f"Total Tokens: {total_patches}", 
            f"Compression: None (I-Frame)",
            font_stats
        )
        
    else:
        # P-frame: show three panels
        display_size = 320
        gap = 60
        total_width = display_size * 3 + gap * 2
        left_x = (canvas_size[0] - total_width) // 2
        center_x = left_x + display_size + gap
        right_x = center_x + display_size + gap
        
        # === Left: Original Current Frame ===
        _draw_frame_panel(
            canvas, draw, frames[frame_idx], left_x, main_y, display_size,
            f"Original Frame {frame_idx + 1}", (100, 200, 255), patch_size, font_label, show_grid=False
        )
        
        # === Center: Reference I-Frame ===
        _draw_frame_panel(
            canvas, draw, frames[0], center_x, main_y, display_size,
            "Reference (Frame 1)", (100, 255, 150), patch_size, font_label, show_grid=False
        )
        
        # === Right: Residual with Top Tokens ===
        masked_residual, tokens_kept = compute_masked_residual(
            frames[frame_idx], frames[0], patch_size, max_tokens=max_tokens_per_frame
        )
        _draw_frame_panel(
            canvas, draw, masked_residual, right_x, main_y, display_size,
            f"ViT Input ({tokens_kept} Tokens)", (255, 150, 100), patch_size, font_label, 
            show_grid=True, is_residual=True
        )
        
        # Draw flow arrows
        arrow_y = main_y + display_size // 2
        _draw_arrow(draw, left_x + display_size + 10, arrow_y, center_x - 10, arrow_y, (150, 150, 150))
        _draw_arrow(draw, center_x + display_size + 10, arrow_y, right_x - 10, arrow_y, (150, 150, 150))
        
        # Stats bar
        stats_y = main_y + display_size + 60
        compression = (1 - tokens_kept / total_patches) * 100
        _draw_stats_bar(
            draw, canvas_size[0] // 2, stats_y,
            f"Tokens: {tokens_kept}/{total_patches}", 
            f"Compression: {compression:.1f}%",
            font_stats
        )
    
    # Footer legend
    footer_y = canvas_size[1] - 50
    _draw_legend(draw, canvas_size[0] // 2, footer_y, frame_idx, font_small)
    
    return canvas


def _draw_frame_panel(
    canvas: Image.Image, 
    draw: ImageDraw.Draw,
    frame: np.ndarray, 
    x: int, 
    y: int, 
    size: int,
    label: str,
    border_color: Tuple[int, int, int],
    patch_size: int,
    font: ImageFont.FreeTypeFont,
    show_grid: bool = False,
    is_residual: bool = False
) -> None:
    """Draw a single frame panel with border, label, and optional grid."""
    # Draw shadow for depth effect
    shadow_offset = 4
    draw.rectangle(
        [x + shadow_offset, y + shadow_offset, x + size + shadow_offset, y + size + shadow_offset],
        fill=(15, 18, 22)
    )
    
    # Draw frame
    frame_img = Image.fromarray(frame)
    frame_resized = frame_img.resize((size, size), Image.Resampling.LANCZOS)
    canvas.paste(frame_resized, (x, y))
    
    # Draw border with glow effect
    for i in range(3):
        alpha = 1 - i * 0.3
        color = tuple(int(c * alpha) for c in border_color)
        draw.rectangle(
            [x - i - 1, y - i - 1, x + size + i + 1, y + size + i + 1],
            outline=color,
            width=1
        )
    
    # Draw grid if requested
    if show_grid:
        num_patches = frame.shape[0] // patch_size
        patch_display_size = size // num_patches if num_patches > 0 else size
        if num_patches > 0:
            for i in range(num_patches + 1):
                line_y = y + i * patch_display_size
                line_x = x + i * patch_display_size
                grid_color = (80, 80, 80) if is_residual else (60, 60, 60)
                if i < num_patches:
                    draw.line([(x, line_y), (x + size, line_y)], fill=grid_color, width=1)
                if i < num_patches:
                    draw.line([(line_x, y), (line_x, y + size)], fill=grid_color, width=1)
    
    # Draw label below
    label_x = x + (size - len(label) * 8) // 2
    draw.text((label_x, y + size + 10), label, fill=border_color, font=font)


def _draw_arrow(
    draw: ImageDraw.Draw, 
    x1: int, y1: int, 
    x2: int, y2: int, 
    color: Tuple[int, int, int]
) -> None:
    """Draw an arrow from (x1, y1) to (x2, y2)."""
    draw.line([(x1, y1), (x2, y2)], fill=color, width=2)
    # Arrowhead
    arrow_size = 8
    draw.polygon([
        (x2 - arrow_size, y1 - arrow_size // 2),
        (x2, y1),
        (x2 - arrow_size, y1 + arrow_size // 2)
    ], fill=color)


def _draw_stats_bar(
    draw: ImageDraw.Draw,
    center_x: int,
    y: int,
    stat1: str,
    stat2: str,
    font: ImageFont.FreeTypeFont
) -> None:
    """Draw a stats bar with two statistics."""
    # Background pill
    bar_width = 400
    bar_height = 30
    bar_x = center_x - bar_width // 2
    
    draw_rounded_rectangle(
        draw,
        [bar_x, y, bar_x + bar_width, y + bar_height],
        radius=15,
        fill=(45, 50, 60)
    )
    
    # Stats text
    draw.text((bar_x + 30, y + 5), stat1, fill=(200, 220, 255), font=font)
    draw.text((bar_x + bar_width // 2 + 20, y + 5), stat2, fill=(255, 200, 150), font=font)


def _draw_legend(
    draw: ImageDraw.Draw,
    center_x: int,
    y: int,
    frame_idx: int,
    font: ImageFont.FreeTypeFont
) -> None:
    """Draw the legend at the bottom of the frame."""
    legend_items = [
        ("●", (100, 200, 255), "Original"),
        ("●", (100, 255, 150), "Reference"),
    ]
    if frame_idx > 0:
        legend_items.append(("●", (255, 150, 100), "Residual (Top Tokens)"))
    
    total_width = sum(len(item[2]) * 10 + 40 for item in legend_items)
    start_x = center_x - total_width // 2
    
    for dot, color, text in legend_items:
        draw.text((start_x, y), dot, fill=color, font=font)
        draw.text((start_x + 15, y), text, fill=(180, 180, 180), font=font)
        start_x += len(text) * 10 + 50


def create_architecture_frame(canvas_size: Tuple[int, int] = (1600, 720)) -> Image.Image:
    """Create a frame showing the overall ViT architecture with modern dark theme."""
    canvas = Image.new('RGB', canvas_size, color=(25, 28, 35))
    draw = ImageDraw.Draw(canvas)
    
    # Load fonts using the cross-platform helper function
    font_title = get_font(32, bold=True)
    font_subtitle = get_font(20, bold=True)
    font_label = get_font(16, bold=False)
    font_small = get_font(14, bold=False)
    
    # Header area with gradient-like background
    header_height = 80
    for y in range(header_height):
        alpha = 1 - (y / header_height) * 0.3
        color = (int(40 * alpha), int(45 * alpha), int(55 * alpha))
        draw.line([(0, y), (canvas_size[0], y)], fill=color)
    
    # Title
    title = "LLaVA-ViT: Video Residual Encoding Architecture"
    draw.text((canvas_size[0] // 2 - 300, 20), title, fill=(255, 255, 255), font=font_title)
    
    # Subtitle
    subtitle = "Efficient Video Understanding with Token Compression"
    draw.text((canvas_size[0] // 2 - 220, 55), subtitle, fill=(100, 200, 255), font=font_label)
    
    # Calculate layout - scale for wider canvas
    base_x = 80
    box_width = 160
    box_height = 140
    gap = 30
    y_center = 250
    
    # === Video Input Box ===
    x = base_x
    draw_rounded_rectangle(
        draw, [x, y_center - box_height//2, x + box_width, y_center + box_height//2],
        radius=10, fill=(45, 60, 80), outline=(100, 150, 200), width=2
    )
    draw.text((x + 30, y_center - 30), "Video Input", fill=(200, 220, 255), font=font_label)
    draw.text((x + 35, y_center), "(T×H×W×C)", fill=(150, 170, 200), font=font_small)
    draw.text((x + 25, y_center + 25), "64 frames", fill=(100, 200, 255), font=font_small)
    
    # Arrow
    _draw_arrow(draw, x + box_width + 5, y_center, x + box_width + gap - 5, y_center, (100, 150, 200))
    
    # === Frame Decomposition Box ===
    x = base_x + box_width + gap
    draw_rounded_rectangle(
        draw, [x, y_center - box_height//2, x + box_width, y_center + box_height//2],
        radius=10, fill=(45, 70, 55), outline=(100, 200, 150), width=2
    )
    draw.text((x + 20, y_center - 60), "Frame Split", fill=(150, 255, 180), font=font_label)
    
    # I-Frame
    draw_rounded_rectangle(draw, [x + 15, y_center - 35, x + 75, y_center - 5], radius=5, 
                           fill=(80, 180, 120), outline=(100, 255, 150), width=1)
    draw.text((x + 25, y_center - 28), "F₁", fill=(255, 255, 255), font=font_small)
    
    # P-Frames
    for i in range(3):
        py = y_center + 5 + i * 22
        draw_rounded_rectangle(draw, [x + 15, py, x + 75, py + 18], radius=3,
                              fill=(180, 100, 80), outline=(255, 150, 100), width=1)
        draw.text((x + 22, py + 2), f"ΔF{i+2}", fill=(255, 255, 255), font=font_small)
    
    draw.text((x + 85, y_center - 25), "I-Frame", fill=(100, 255, 150), font=font_small)
    draw.text((x + 85, y_center + 15), "P-Frames", fill=(255, 150, 100), font=font_small)
    
    # Arrow
    _draw_arrow(draw, x + box_width + 5, y_center, x + box_width + gap - 5, y_center, (100, 200, 150))
    
    # === Token Selection Box ===
    x = base_x + (box_width + gap) * 2
    draw_rounded_rectangle(
        draw, [x, y_center - box_height//2, x + box_width, y_center + box_height//2],
        radius=10, fill=(70, 55, 45), outline=(255, 180, 100), width=2
    )
    draw.text((x + 15, y_center - 60), "Token Selection", fill=(255, 200, 150), font=font_label)
    
    # Token grid
    for i in range(4):
        for j in range(4):
            tx = x + 25 + j * 25
            ty = y_center - 30 + i * 25
            # Some tokens are highlighted (selected)
            if (i + j) % 3 == 0:
                color = (255, 180, 100)
            else:
                color = (60, 50, 45)
            draw.rectangle([tx, ty, tx + 20, ty + 20], fill=color, outline=(100, 80, 60), width=1)
    
    draw.text((x + 20, y_center + 45), "Top 196×7", fill=(255, 180, 100), font=font_small)
    
    # Arrow
    _draw_arrow(draw, x + box_width + 5, y_center, x + box_width + gap - 5, y_center, (255, 180, 100))
    
    # === Patch Embedding Box ===
    x = base_x + (box_width + gap) * 3
    draw_rounded_rectangle(
        draw, [x, y_center - box_height//2, x + box_width, y_center + box_height//2],
        radius=10, fill=(60, 55, 70), outline=(180, 150, 220), width=2
    )
    draw.text((x + 15, y_center - 60), "Patch Embed", fill=(200, 180, 255), font=font_label)
    
    # Patch grid with colors
    for i in range(4):
        for j in range(4):
            px = x + 25 + j * 25
            py = y_center - 30 + i * 25
            color = (100 + i * 30, 120 + j * 20, 180 - i * 15)
            draw.rectangle([px, py, px + 20, py + 20], fill=color, outline=(120, 100, 150), width=1)
    
    draw.text((x + 20, y_center + 45), "+ Position", fill=(180, 150, 220), font=font_small)
    
    # Arrow
    _draw_arrow(draw, x + box_width + 5, y_center, x + box_width + gap - 5, y_center, (180, 150, 220))
    
    # === ViT Encoder Box ===
    x = base_x + (box_width + gap) * 4
    encoder_height = box_height + 40
    draw_rounded_rectangle(
        draw, [x, y_center - encoder_height//2, x + box_width + 40, y_center + encoder_height//2],
        radius=10, fill=(55, 50, 70), outline=(150, 100, 200), width=2
    )
    draw.text((x + 30, y_center - 75), "ViT Encoder", fill=(200, 170, 255), font=font_subtitle)
    
    # Transformer layers
    layer_colors = [(120, 100, 160), (130, 110, 170), (140, 120, 180), (150, 130, 190)]
    for i in range(4):
        ly = y_center - 45 + i * 28
        draw_rounded_rectangle(draw, [x + 15, ly, x + box_width + 25, ly + 22], radius=5,
                              fill=layer_colors[i], outline=(180, 150, 220), width=1)
        draw.text((x + 25, ly + 3), f"Transformer Layer {i+1}", fill=(255, 255, 255), font=font_small)
    
    draw.text((x + 35, y_center + 55), "3D RoPE", fill=(150, 100, 200), font=font_small)
    
    # Arrow
    _draw_arrow(draw, x + box_width + 45, y_center, x + box_width + gap + 35, y_center, (150, 100, 200))
    
    # === Output Features Box ===
    x = base_x + (box_width + gap) * 5 + 40
    draw_rounded_rectangle(
        draw, [x, y_center - box_height//2, x + box_width, y_center + box_height//2],
        radius=10, fill=(45, 65, 70), outline=(100, 200, 200), width=2
    )
    draw.text((x + 30, y_center - 30), "Video", fill=(150, 230, 230), font=font_label)
    draw.text((x + 25, y_center), "Features", fill=(150, 230, 230), font=font_label)
    draw.text((x + 20, y_center + 30), "→ LLM", fill=(100, 200, 255), font=font_small)
    
    # === Info Section at Bottom ===
    info_y = 430
    
    # Key points box
    draw_rounded_rectangle(draw, [50, info_y, canvas_size[0] // 2 - 30, info_y + 220], 
                           radius=10, fill=(35, 38, 48), outline=(80, 85, 100), width=1)
    draw.text((70, info_y + 15), "Key Features", fill=(255, 255, 255), font=font_subtitle)
    
    features = [
        ("●", (100, 255, 150), " I-Frame: Full 196 tokens (reference)"),
        ("●", (255, 150, 100), " P-Frames: Top important tokens only"),
        ("●", (255, 180, 100), " Token limit: 196×7 = 1372 for P-frames"),
        ("●", (180, 150, 220), " 3D RoPE: T:H:W position encoding"),
        ("●", (100, 200, 255), " Compression: ~85% token reduction"),
    ]
    
    for i, (dot, color, text) in enumerate(features):
        fy = info_y + 50 + i * 32
        draw.text((70, fy), dot, fill=color, font=font_label)
        draw.text((85, fy), text, fill=(200, 200, 210), font=font_label)
    
    # Formulas box
    draw_rounded_rectangle(draw, [canvas_size[0] // 2 + 30, info_y, canvas_size[0] - 50, info_y + 220], 
                           radius=10, fill=(35, 38, 48), outline=(80, 85, 100), width=1)
    draw.text((canvas_size[0] // 2 + 50, info_y + 15), "Processing Pipeline", fill=(255, 255, 255), font=font_subtitle)
    
    formulas = [
        "1. x₁ = PatchEmbed(Frame₁)",
        "2. xₜ = PatchEmbed(Frameₜ - Frame₁)  for t > 1",
        "3. Select top-k tokens by importance",
        "4. X = [x₁; TopK(x₂); ...; TopK(xₜ)]",
        "5. Z = ViTEncoder(X + 3D-RoPE)",
    ]
    
    for i, formula in enumerate(formulas):
        fy = info_y + 50 + i * 32
        draw.text((canvas_size[0] // 2 + 50, fy), formula, fill=(180, 200, 255), font=font_label)
    
    return canvas



def generate_video(
    frames: np.ndarray,
    output_path: str,
    patch_size: int = 16,
    fps: int = 2,
    canvas_size: Tuple[int, int] = (1600, 720),
    include_architecture: bool = True,
    max_tokens_per_frame: Optional[int] = None,
    total_frames: int = 64
) -> None:
    """Generate an MP4 video showing the ViT processing pipeline.
    
    Args:
        frames: Video frames to process
        output_path: Output video path (should end with .mp4)
        patch_size: ViT patch size
        fps: Frames per second for the output video
        canvas_size: Size of each frame in the video
        include_architecture: Whether to include architecture overview
        max_tokens_per_frame: Maximum tokens to keep per P-frame
        total_frames: Total number of frames (used to calculate tokens per frame)
    """
    video_frames: List[np.ndarray] = []
    
    # Add architecture overview frame first (shown for 3 seconds)
    if include_architecture:
        arch_frame = create_architecture_frame((canvas_size[0], canvas_size[1]))
        arch_array = np.array(arch_frame)
        # Repeat for 3 seconds
        for _ in range(fps * 3):
            video_frames.append(arch_array)
    
    # Create visualization frames for each video frame
    for frame_idx in range(len(frames)):
        viz_frame = create_visualization_frame(
            frames, frame_idx, patch_size, canvas_size,
            max_tokens_per_frame=max_tokens_per_frame,
            total_frames=total_frames
        )
        video_frames.append(np.array(viz_frame))
    
    # Ensure output path ends with .mp4
    if not output_path.lower().endswith('.mp4'):
        output_path = output_path.rsplit('.', 1)[0] + '.mp4'
    
    # Save as MP4 video using imageio
    if video_frames:
        imageio.mimwrite(
            output_path,
            video_frames,
            fps=fps,
            codec='libx264',
            quality=9,
            pixelformat='yuv420p'
        )
        print(f"Video saved to: {output_path}")
        print(f"  - Total frames: {len(video_frames)}")
        print(f"  - Duration: {len(video_frames) / fps:.1f} seconds")
        print(f"  - Resolution: {canvas_size[0]}x{canvas_size[1]}")


def generate_gif(
    frames: np.ndarray,
    output_path: str,
    patch_size: int = 16,
    duration: int = 500,
    canvas_size: Tuple[int, int] = (1600, 720),
    include_architecture: bool = True,
    max_tokens_per_frame: Optional[int] = None,
    total_frames: int = 64
) -> None:
    """Generate the animated GIF showing the ViT processing pipeline (legacy support)."""
    gif_frames: List[Image.Image] = []
    
    # Add architecture overview frame first
    if include_architecture:
        arch_frame = create_architecture_frame((canvas_size[0], canvas_size[1]))
        # Show architecture frame longer
        for _ in range(3):
            gif_frames.append(arch_frame)
    
    # Create visualization frames for each video frame
    for frame_idx in range(len(frames)):
        viz_frame = create_visualization_frame(
            frames, frame_idx, patch_size, canvas_size,
            max_tokens_per_frame=max_tokens_per_frame,
            total_frames=total_frames
        )
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
        description="Generate an animated video visualizing LLaVA-ViT video processing with residual encoding"
    )
    parser.add_argument("--video", type=str, help="Path to input video file")
    parser.add_argument("--demo", action="store_true", help="Generate demo with synthetic frames")
    parser.add_argument("--output", type=str, default="vit_residual_encoding.mp4", help="Output video path")
    parser.add_argument("--num-frames", type=int, default=64, help="Number of frames to sample from video (default: 64)")
    parser.add_argument("--patch-size", type=int, default=16, help="ViT patch size")
    parser.add_argument("--fps", type=int, default=4, help="Frames per second for video output")
    parser.add_argument("--width", type=int, default=1600, help="Canvas width")
    parser.add_argument("--height", type=int, default=720, help="Canvas height (should be divisible by 16)")
    parser.add_argument("--no-architecture", action="store_true", help="Skip architecture overview frame")
    parser.add_argument("--max-tokens", type=int, default=None, 
                        help="Max tokens per P-frame (default: 196*7/(num_frames-1))")
    parser.add_argument("--gif", action="store_true", help="Output as GIF instead of video")
    parser.add_argument("--duration", type=int, default=800, help="Duration of each frame in ms (for GIF)")
    
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
    
    # Calculate max tokens per frame if not specified
    max_tokens = args.max_tokens
    if max_tokens is None and args.num_frames > 1:
        # 196*7 tokens distributed across (num_frames-1) P-frames
        max_tokens = (196 * 7) // (args.num_frames - 1)
        print(f"Using {max_tokens} tokens per P-frame ({196 * 7} tokens / {args.num_frames - 1} P-frames)")
    
    if args.gif:
        generate_gif(
            frames=frames,
            output_path=args.output,
            patch_size=args.patch_size,
            duration=args.duration,
            canvas_size=(args.width, args.height),
            include_architecture=not args.no_architecture,
            max_tokens_per_frame=max_tokens,
            total_frames=args.num_frames
        )
    else:
        generate_video(
            frames=frames,
            output_path=args.output,
            patch_size=args.patch_size,
            fps=args.fps,
            canvas_size=(args.width, args.height),
            include_architecture=not args.no_architecture,
            max_tokens_per_frame=max_tokens,
            total_frames=args.num_frames
        )


if __name__ == "__main__":
    main()
