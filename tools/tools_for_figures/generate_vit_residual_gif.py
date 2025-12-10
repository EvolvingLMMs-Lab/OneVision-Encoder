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
- Animated cube building: Progressive spatiotemporal visualization showing I-frame and P-frames

Usage:
    # Generate standard residual encoding video
    python generate_vit_residual_gif.py --video /path/to/video.mp4 --output output.mp4
    
    # Generate demo with synthetic data
    python generate_vit_residual_gif.py --demo --output demo.mp4
    
    # Generate animated cube building GIF showing residual encoding
    python generate_vit_residual_gif.py --demo --animated-cube residual_cube.gif
    
    # Customize animated cube building
    python generate_vit_residual_gif.py --video /path/to/video.mp4 \
        --animated-cube cube.gif --cube-scale 0.6 --cube-duration 400 \
        --cube-max-frames 16 --no-cube-transparency
    
    # Generate both standard visualization and animated cube
    python generate_vit_residual_gif.py --demo --output demo.mp4 \
        --animated-cube residual_cube.gif
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


def compute_global_token_selection(
    frames: np.ndarray,
    patch_size: int = 16,
    total_tokens: int = 196 * 7
) -> dict:
    """
    Compute global token selection across all P-frames.
    
    Instead of distributing tokens equally per P-frame, this function computes
    importance scores for ALL tokens across ALL P-frames and selects the top
    `total_tokens` tokens globally based on importance.
    
    Args:
        frames: All video frames (T, H, W, C)
        patch_size: Size of each ViT patch
        total_tokens: Total number of tokens to select across all P-frames (default: 196*7=1372)
        
    Returns:
        Dictionary mapping frame_idx -> set of (patch_i, patch_j) that are selected for that frame
    """
    reference_frame = frames[0].astype(np.float32)
    h, w = frames[0].shape[:2]
    num_patches_h = h // patch_size
    num_patches_w = w // patch_size
    
    # Collect all patch scores across all P-frames
    all_patch_scores = []
    
    for frame_idx in range(1, len(frames)):
        current_frame = frames[frame_idx].astype(np.float32)
        residual = current_frame - reference_frame
        
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                y_start = i * patch_size
                y_end = (i + 1) * patch_size
                x_start = j * patch_size
                x_end = (j + 1) * patch_size
                
                patch_residual = residual[y_start:y_end, x_start:x_end]
                patch_mad = np.mean(np.abs(patch_residual))
                all_patch_scores.append((patch_mad, frame_idx, i, j))
    
    # Sort by importance (descending) and select top tokens globally
    all_patch_scores.sort(key=lambda x: x[0], reverse=True)
    selected_global = all_patch_scores[:total_tokens]
    
    # Build a dictionary mapping frame_idx -> set of selected (i, j) patches
    selected_by_frame = {}
    for _, frame_idx, i, j in selected_global:
        if frame_idx not in selected_by_frame:
            selected_by_frame[frame_idx] = set()
        selected_by_frame[frame_idx].add((i, j))
    
    return selected_by_frame


def compute_masked_residual_with_selection(
    current_frame: np.ndarray, 
    reference_frame: np.ndarray, 
    selected_patches_set: set,
    patch_size: int = 16
) -> Tuple[np.ndarray, int]:
    """
    Compute the residual with masking based on pre-computed global selection.
    
    Args:
        current_frame: The current frame (P frame)
        reference_frame: The reference frame (I frame, first frame)
        selected_patches_set: Set of (i, j) tuples indicating which patches are selected
        patch_size: Size of each patch
                   
    Returns:
        Tuple of (masked residual where non-selected patches are blank, number of tokens kept)
    """
    # Compute raw residual
    residual = current_frame.astype(np.float32) - reference_frame.astype(np.float32)
    
    # Create output frame - start with white (blank)
    h, w = current_frame.shape[:2]
    masked_residual = np.ones((h, w, 3), dtype=np.uint8) * 255  # White background
    
    # Normalize residual for visualization
    residual_normalized = ((residual + 255) / 2).clip(0, 255).astype(np.uint8)
    
    # Apply only the selected patches
    for i, j in selected_patches_set:
        y_start = i * patch_size
        y_end = (i + 1) * patch_size
        x_start = j * patch_size
        x_end = (j + 1) * patch_size
        masked_residual[y_start:y_end, x_start:x_end] = \
            residual_normalized[y_start:y_end, x_start:x_end]
    
    return masked_residual, len(selected_patches_set)


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
    total_frames: int = 64,
    global_selection: Optional[dict] = None
) -> Image.Image:
    """Create a single visualization frame showing original frame, I-frame reference, and residual.
    
    For P-frames (frame_idx > 0), displays three panels:
    - Left: Reference I-Frame
    - Center: Original current frame
    - Right: Residual visualization with selected tokens
    
    Args:
        frames: All video frames
        frame_idx: Current frame index
        patch_size: Size of each ViT patch
        canvas_size: Size of the output canvas
        max_tokens_per_frame: Maximum tokens to keep per P-frame (used when global_selection is None)
        total_frames: Total number of frames (default 64, used to calculate tokens per frame)
        global_selection: Pre-computed global token selection (frame_idx -> set of (i, j) patches)
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
    if frame_idx == 0:
        frame_type = "I-Frame (Full)"
    elif global_selection is not None:
        tokens_this_frame = len(global_selection.get(frame_idx, set()))
        frame_type = f"P-Frame ({tokens_this_frame} tokens)"
    else:
        frame_type = f"P-Frame (Top {max_tokens_per_frame} tokens)"
    draw.text((canvas_size[0] // 2 - 120, 45), f"{frame_info} • {frame_type}", 
              fill=(100, 200, 255), font=font_label)
    
    main_y = 90
    
    if frame_idx == 0:
        # I-frame: show original frame larger and centered
        # Use display size that is a multiple of num_patches for perfect grid alignment
        display_size = num_patches_h * 32  # 14 * 32 = 448
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
        # Use display size that is a multiple of num_patches for perfect grid alignment
        display_size = num_patches_h * 23  # 14 * 23 = 322
        gap = 60
        total_width = display_size * 3 + gap * 2
        left_x = (canvas_size[0] - total_width) // 2
        center_x = left_x + display_size + gap
        right_x = center_x + display_size + gap
        
        # === Left: Reference I-Frame ===
        _draw_frame_panel(
            canvas, draw, frames[0], left_x, main_y, display_size,
            "Reference (Frame 1)", (100, 255, 150), patch_size, font_label, show_grid=False
        )
        
        # === Center: Original Current Frame ===
        _draw_frame_panel(
            canvas, draw, frames[frame_idx], center_x, main_y, display_size,
            f"Original Frame {frame_idx + 1}", (100, 200, 255), patch_size, font_label, show_grid=False
        )
        
        # === Right: Residual with Selected Tokens ===
        if global_selection is not None and frame_idx in global_selection:
            # Use pre-computed global selection
            selected_patches_set = global_selection[frame_idx]
            masked_residual, tokens_kept = compute_masked_residual_with_selection(
                frames[frame_idx], frames[0], selected_patches_set, patch_size
            )
        elif global_selection is not None:
            # This frame has no selected tokens in global selection
            h, w = frames[frame_idx].shape[:2]
            masked_residual = np.ones((h, w, 3), dtype=np.uint8) * 255  # All white
            tokens_kept = 0
        else:
            # Fall back to per-frame selection (legacy behavior)
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
        if num_patches > 0:
            # Draw grid lines that perfectly align with the resized patches
            # The last grid line should be exactly at the edge (x + size, y + size)
            for i in range(num_patches + 1):
                # Use linear interpolation to ensure perfect alignment
                # First line at 0, last line at size
                line_y = y + (i * size) // num_patches
                line_x = x + (i * size) // num_patches
                grid_color = (80, 80, 80) if is_residual else (60, 60, 60)
                # Draw horizontal lines
                draw.line([(x, line_y), (x + size, line_y)], fill=grid_color, width=1)
                # Draw vertical lines
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



def create_animated_cube_building_residual(
    frames: np.ndarray,
    output_path: str,
    patch_size: int = 16,
    offset_x: int = 15,
    offset_y: int = 15,
    max_frames: Optional[int] = None,
    frame_scale: float = 0.5,
    add_labels: bool = True,
    duration: int = 300,
    transparency: bool = True,
    final_hold_frames: int = 3,
    total_p_frame_tokens: int = 196 * 7
) -> None:
    """
    Create an animated GIF showing the residual encoding cube being built frame by frame.
    
    This creates an animation that starts from the I-frame (full frame) and progressively adds
    P-frames (residual frames) to build the complete spatiotemporal cube visualization with 
    optional transparency effects for depth perception.
    
    The first frame is shown as the full reference frame (I-frame), and subsequent frames
    show residuals with only the most important tokens highlighted.
    
    Args:
        frames: np.ndarray of shape (num_frames, height, width, 3)
        output_path: Path to save the animated GIF
        patch_size: ViT patch size for residual computation
        offset_x: Horizontal offset between consecutive frames (default: 15)
        offset_y: Vertical offset between consecutive frames (default: 15)
        max_frames: Maximum number of frames to include (None = all frames)
        frame_scale: Scale factor for frames (default: 0.5)
        add_labels: Whether to add frame numbers
        duration: Duration of each animation frame in milliseconds (default: 300)
        transparency: Whether to apply transparency effects for depth (default: True)
        final_hold_frames: Number of times to repeat final frame for emphasis (default: 3)
        total_p_frame_tokens: Total tokens to select across all P-frames (default: 196*7=1372)
    """
    # Limit number of frames if needed
    if max_frames is not None and len(frames) > max_frames:
        # Sample frames evenly
        indices = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
        frames = frames[indices]
        print(f"Using {max_frames} frames sampled from total frames")
    
    num_frames = len(frames)
    frame_height, frame_width = frames[0].shape[:2]
    
    # Compute global token selection for residuals
    global_selection = compute_global_token_selection(frames, patch_size, total_p_frame_tokens)
    
    # Prepare frames for visualization
    display_frames = []
    for i in range(num_frames):
        if i == 0:
            # First frame: show original (I-frame)
            display_frames.append(frames[i])
        else:
            # P-frames: show residual with selected tokens
            if i in global_selection:
                selected_patches_set = global_selection[i]
                masked_residual, _ = compute_masked_residual_with_selection(
                    frames[i], frames[0], selected_patches_set, patch_size
                )
            else:
                # No tokens selected for this frame
                h, w = frames[i].shape[:2]
                masked_residual = np.ones((h, w, 3), dtype=np.uint8) * 255
            display_frames.append(masked_residual)
    
    # Scale frames
    scaled_width = int(frame_width * frame_scale)
    scaled_height = int(frame_height * frame_scale)
    
    # Calculate canvas size
    canvas_width = scaled_width + (num_frames - 1) * offset_x + 100
    canvas_height = scaled_height + (num_frames - 1) * offset_y + 100
    
    # Font for labels
    font = get_font(16, bold=True) if add_labels else None
    
    print(f"Creating animated cube building with residual encoding...")
    print(f"  Frames: {num_frames}")
    print(f"  Frame size: {scaled_width}x{scaled_height}")
    print(f"  Canvas size: {canvas_width}x{canvas_height}")
    print(f"  Offset: ({offset_x}, {offset_y})")
    print(f"  Transparency: {transparency}")
    
    gif_frames = []
    
    # Create animation: add frames one by one
    for current_frame_count in range(1, num_frames + 1):
        # Create canvas with white background (RGBA for transparency support)
        canvas = Image.new('RGBA', (canvas_width, canvas_height), (255, 255, 255, 255))
        
        # Draw frames in order (0 to current_frame_count-1), so later frames overlay earlier frames
        for i in range(current_frame_count):
            frame = display_frames[i]
            
            # Resize frame
            frame_img = Image.fromarray(frame)
            frame_img = frame_img.resize((scaled_width, scaled_height), Image.Resampling.LANCZOS)
            
            # Convert to RGBA for transparency effects
            if frame_img.mode != 'RGBA':
                frame_img = frame_img.convert('RGBA')
            
            # Apply transparency for depth effect (earlier frames more transparent, later frames more opaque)
            if transparency:
                # Calculate alpha: later frames (higher i, drawn last) are more opaque
                # Range from 60% (earlier frames) to 100% (later frames)
                # For single frame (current_frame_count=1), use 100% opacity
                if current_frame_count == 1:
                    alpha_factor = 1.0
                else:
                    alpha_factor = 0.6 + (0.4 * i / (current_frame_count - 1))
                # Create alpha mask
                alpha = frame_img.split()[3]
                alpha = alpha.point(lambda p: int(p * alpha_factor))
                frame_img.putalpha(alpha)
            
            # Calculate position (earlier frames at top-left, later frames at bottom-right)
            x = 50 + i * offset_x
            y = 50 + i * offset_y
            
            # Add subtle shadow for depth
            shadow = Image.new('RGBA', (scaled_width + 4, scaled_height + 4), (0, 0, 0, 50))
            canvas.paste(shadow, (x + 2, y + 2), shadow)
            
            # Paste frame with transparency
            canvas.paste(frame_img, (x, y), frame_img)
            
            # Add frame border for clarity
            draw = ImageDraw.Draw(canvas)
            # Use different colors for I-frame vs P-frames
            if i == 0:
                border_color = (100, 255, 150, 255)  # Green for I-frame
            else:
                border_color = (255, 150, 100, 255)  # Orange for P-frames
            draw.rectangle(
                [x, y, x + scaled_width, y + scaled_height],
                outline=border_color,
                width=2
            )
            
            # Add label if requested (only for the most recent frames)
            if add_labels and i >= current_frame_count - min(5, current_frame_count):
                if i == 0:
                    label = f"I-Frame"
                else:
                    label = f"P{i}"
                # Get text size
                bbox = draw.textbbox((0, 0), label, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                
                # Position label at bottom-right of frame
                label_x = x + scaled_width - text_width - 5
                label_y = y + scaled_height - text_height - 5
                
                # Draw background for label
                draw.rectangle(
                    [label_x - 3, label_y - 2, label_x + text_width + 3, label_y + text_height + 2],
                    fill=(255, 255, 255, 230)
                )
                
                # Draw text
                draw.text((label_x, label_y), label, fill=(0, 0, 0, 255), font=font)
        
        # Convert RGBA to RGB for GIF (with white background)
        rgb_canvas = Image.new('RGB', canvas.size, (255, 255, 255))
        rgb_canvas.paste(canvas, (0, 0), canvas)
        gif_frames.append(rgb_canvas)
    
    # Hold the final frame for longer
    for _ in range(final_hold_frames):
        gif_frames.append(gif_frames[-1])
    
    # Save as animated GIF
    gif_frames[0].save(
        output_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=duration,
        loop=0
    )
    
    print(f"\nAnimated cube building GIF saved to: {output_path}")
    print(f"  Total animation frames: {len(gif_frames)}")
    print(f"  Duration per frame: {duration}ms")
    print(f"  Total duration: {len(gif_frames) * duration / 1000:.1f}s")


def generate_video(
    frames: np.ndarray,
    output_path: str,
    patch_size: int = 16,
    fps: int = 2,
    canvas_size: Tuple[int, int] = (1600, 720),
    include_architecture: bool = True,
    max_tokens_per_frame: Optional[int] = None,
    total_frames: int = 64,
    total_p_frame_tokens: int = 196 * 7
) -> None:
    """Generate an MP4 video showing the ViT processing pipeline.
    
    Args:
        frames: Video frames to process
        output_path: Output video path (should end with .mp4)
        patch_size: ViT patch size
        fps: Frames per second for the output video
        canvas_size: Size of each frame in the video
        include_architecture: Whether to include architecture overview
        max_tokens_per_frame: Maximum tokens to keep per P-frame (legacy, not used with global selection)
        total_frames: Total number of frames (used to calculate tokens per frame)
        total_p_frame_tokens: Total tokens to select across all P-frames (default: 196*7=1372)
    """
    video_frames: List[np.ndarray] = []
    
    # Compute global token selection across all P-frames
    global_selection = compute_global_token_selection(frames, patch_size, total_p_frame_tokens)
    total_selected = sum(len(v) for v in global_selection.values())
    print(f"Global token selection: {total_selected} tokens across {len(global_selection)} P-frames")
    
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
            total_frames=total_frames,
            global_selection=global_selection
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
    total_frames: int = 64,
    total_p_frame_tokens: int = 196 * 7
) -> None:
    """Generate the animated GIF showing the ViT processing pipeline (legacy support).
    
    Args:
        frames: Video frames to process
        output_path: Output GIF path
        patch_size: ViT patch size
        duration: Duration of each frame in milliseconds
        canvas_size: Size of each frame in the GIF
        include_architecture: Whether to include architecture overview
        max_tokens_per_frame: Maximum tokens to keep per P-frame (legacy, not used with global selection)
        total_frames: Total number of frames
        total_p_frame_tokens: Total tokens to select across all P-frames (default: 196*7=1372)
    """
    gif_frames: List[Image.Image] = []
    
    # Compute global token selection across all P-frames
    global_selection = compute_global_token_selection(frames, patch_size, total_p_frame_tokens)
    total_selected = sum(len(v) for v in global_selection.values())
    print(f"Global token selection: {total_selected} tokens across {len(global_selection)} P-frames")
    
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
            total_frames=total_frames,
            global_selection=global_selection
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
    parser.add_argument("--total-tokens", type=int, default=196 * 7, 
                        help="Total tokens to select across all P-frames (default: 196*7=1372)")
    parser.add_argument("--gif", action="store_true", help="Output as GIF instead of video")
    parser.add_argument("--duration", type=int, default=800, help="Duration of each frame in ms (for GIF)")
    
    # Animated cube building options
    parser.add_argument("--animated-cube", type=str,
                        help="Create animated GIF showing residual cube being built frame by frame (e.g., 'cube_animation.gif')")
    parser.add_argument("--cube-offset-x", type=int, default=15,
                        help="Horizontal offset between frames in animated cube (default: 15)")
    parser.add_argument("--cube-offset-y", type=int, default=15,
                        help="Vertical offset between frames in animated cube (default: 15)")
    parser.add_argument("--cube-scale", type=float, default=0.5,
                        help="Scale factor for frames in animated cube (default: 0.5)")
    parser.add_argument("--cube-max-frames", type=int,
                        help="Maximum number of frames in animated cube (default: all frames)")
    parser.add_argument("--cube-duration", type=int, default=300,
                        help="Duration of each frame in animated cube GIF in milliseconds (default: 300)")
    parser.add_argument("--no-cube-transparency", action="store_true",
                        help="Disable transparency effects in animated cube (frames won't fade with depth)")
    parser.add_argument("--no-cube-labels", action="store_true",
                        help="Don't add frame labels to animated cube")
    
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
    print(f"Selecting top {args.total_tokens} tokens globally across {len(frames) - 1} P-frames")
    
    # Generate animated cube building if requested
    if args.animated_cube:
        print(f"\nGenerating animated cube building GIF...")
        create_animated_cube_building_residual(
            frames=frames,
            output_path=args.animated_cube,
            patch_size=args.patch_size,
            offset_x=args.cube_offset_x,
            offset_y=args.cube_offset_y,
            max_frames=args.cube_max_frames,
            frame_scale=args.cube_scale,
            add_labels=not args.no_cube_labels,
            duration=args.cube_duration,
            transparency=not args.no_cube_transparency,
            total_p_frame_tokens=args.total_tokens
        )
    
    if args.gif:
        generate_gif(
            frames=frames,
            output_path=args.output,
            patch_size=args.patch_size,
            duration=args.duration,
            canvas_size=(args.width, args.height),
            include_architecture=not args.no_architecture,
            total_frames=len(frames),
            total_p_frame_tokens=args.total_tokens
        )
    else:
        generate_video(
            frames=frames,
            output_path=args.output,
            patch_size=args.patch_size,
            fps=args.fps,
            canvas_size=(args.width, args.height),
            include_architecture=not args.no_architecture,
            total_frames=len(frames),
            total_p_frame_tokens=args.total_tokens
        )


if __name__ == "__main__":
    main()
