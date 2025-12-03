#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate an animated GIF to visualize the comparison between CLIP's contrastive learning
and the global contrastive learning approach.

Key differences:
1. CLIP: Image-Text pairs within a batch (limited negative samples)
2. Global: No text encoder, 1M concept centers from offline clustering as negatives

Usage:
    python generate_global_contrastive_comparison.py --output comparison.gif
    python generate_global_contrastive_comparison.py --output comparison.mp4 --video
"""

import argparse
import os
from pathlib import Path
from typing import Optional, Tuple, List

import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Cross-platform font paths
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
    """Get a font with cross-platform support."""
    for font_path in FONT_PATHS:
        if os.path.exists(font_path):
            if bold and ("Bold" in font_path or "bd" in font_path.lower()):
                try:
                    return ImageFont.truetype(font_path, size)
                except OSError:
                    continue
            elif not bold and ("Bold" not in font_path and "bd" not in font_path.lower()):
                try:
                    return ImageFont.truetype(font_path, size)
                except OSError:
                    continue
    # Fallback
    for font_path in FONT_PATHS:
        if os.path.exists(font_path):
            try:
                return ImageFont.truetype(font_path, size)
            except OSError:
                continue
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
        draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)
    except AttributeError:
        draw.rectangle(xy, fill=fill, outline=outline, width=width)


def create_title_frame(canvas_size: Tuple[int, int] = (1920, 1080)) -> Image.Image:
    """Create an introduction title frame."""
    canvas = Image.new('RGB', canvas_size, color=(15, 20, 30))
    draw = ImageDraw.Draw(canvas)
    
    font_title = get_font(56, bold=True)
    font_subtitle = get_font(32, bold=False)
    font_text = get_font(24, bold=False)
    
    # Gradient background
    for y in range(canvas_size[1]):
        alpha = 1 - (y / canvas_size[1]) * 0.5
        color = (int(15 * alpha), int(20 * alpha), int(30 + 20 * alpha))
        draw.line([(0, y), (canvas_size[0], y)], fill=color)
    
    # Title
    title = "Contrastive Learning Comparison"
    title_w = len(title) * 35
    draw.text((canvas_size[0] // 2 - title_w // 2, 200), title, 
              fill=(255, 255, 255), font=font_title)
    
    # Subtitle
    subtitle = "CLIP vs. Global Contrastive Learning"
    subtitle_w = len(subtitle) * 20
    draw.text((canvas_size[0] // 2 - subtitle_w // 2, 280), subtitle,
              fill=(100, 200, 255), font=font_subtitle)
    
    # Divider line
    draw.line([(canvas_size[0] // 2 - 400, 350), (canvas_size[0] // 2 + 400, 350)],
              fill=(80, 100, 120), width=2)
    
    # Content boxes
    y_start = 420
    box_width = 700
    box_height = 500
    gap = 120
    
    # CLIP box
    clip_x = canvas_size[0] // 2 - box_width - gap // 2
    draw_rounded_rectangle(draw, [clip_x, y_start, clip_x + box_width, y_start + box_height],
                          radius=15, fill=(30, 40, 55), outline=(100, 150, 200), width=3)
    
    draw.text((clip_x + 250, y_start + 30), "CLIP", fill=(100, 200, 255), font=font_subtitle)
    
    clip_features = [
        "• Image-Text pairs",
        "• Batch-level contrastive",
        "• Limited negative samples",
        "  (batch size: 32-1024)",
        "• Dual encoders:",
        "  - Image Encoder",
        "  - Text Encoder",
        "• Cross-modal matching"
    ]
    
    for i, feature in enumerate(clip_features):
        draw.text((clip_x + 50, y_start + 120 + i * 48), feature,
                 fill=(200, 220, 240), font=font_text)
    
    # Global box
    global_x = canvas_size[0] // 2 + gap // 2
    draw_rounded_rectangle(draw, [global_x, y_start, global_x + box_width, y_start + box_height],
                          radius=15, fill=(40, 35, 30), outline=(255, 180, 100), width=3)
    
    draw.text((global_x + 150, y_start + 30), "Global Contrastive",
              fill=(255, 180, 100), font=font_subtitle)
    
    global_features = [
        "• Image only (no text)",
        "• Global negative sampling",
        "• 1M concept centers",
        "  (from offline clustering)",
        "• Single encoder:",
        "  - Image Encoder only",
        "• Sample negatives from",
        "  concept bank each batch"
    ]
    
    for i, feature in enumerate(global_features):
        draw.text((global_x + 50, y_start + 120 + i * 48), feature,
                 fill=(255, 220, 180), font=font_text)
    
    return canvas


def create_clip_frame(
    canvas_size: Tuple[int, int] = (1920, 1080),
    animation_step: int = 0
) -> Image.Image:
    """Create a frame showing CLIP's contrastive learning."""
    canvas = Image.new('RGB', canvas_size, color=(15, 20, 30))
    draw = ImageDraw.Draw(canvas)
    
    font_title = get_font(40, bold=True)
    font_label = get_font(22, bold=False)
    font_small = get_font(18, bold=False)
    
    # Title
    draw.text((50, 40), "CLIP: Batch-Level Image-Text Contrastive Learning",
              fill=(100, 200, 255), font=font_title)
    
    # Batch size indicator
    batch_size = 8
    draw.text((50, 100), f"Batch Size: {batch_size} pairs",
              fill=(150, 180, 200), font=font_label)
    
    # Layout
    img_encoder_x = 200
    text_encoder_x = 1500
    y_start = 200
    item_height = 100
    gap = 10
    
    # Draw images on the left
    draw.text((img_encoder_x - 100, 160), "Images", fill=(100, 200, 255), font=font_label)
    
    image_colors = [
        (255, 100, 100), (100, 255, 100), (100, 100, 255), (255, 255, 100),
        (255, 100, 255), (100, 255, 255), (200, 150, 100), (150, 100, 200)
    ]
    
    for i in range(batch_size):
        y = y_start + i * (item_height + gap)
        # Image box
        draw_rounded_rectangle(draw, [img_encoder_x - 80, y, img_encoder_x + 20, y + item_height],
                              radius=8, fill=image_colors[i], outline=(200, 200, 200), width=2)
        draw.text((img_encoder_x - 65, y + 35), f"I{i+1}", fill=(255, 255, 255), font=font_label)
    
    # Image Encoder
    encoder_x = img_encoder_x + 150
    encoder_width = 180
    encoder_height = batch_size * (item_height + gap) - gap
    draw_rounded_rectangle(draw, [encoder_x, y_start, encoder_x + encoder_width, y_start + encoder_height],
                          radius=10, fill=(40, 60, 80), outline=(100, 150, 200), width=3)
    draw.text((encoder_x + 20, y_start + encoder_height // 2 - 20), "Image\nEncoder",
              fill=(200, 220, 255), font=font_label)
    
    # Image embeddings
    emb_x = encoder_x + encoder_width + 100
    for i in range(batch_size):
        y = y_start + i * (item_height + gap)
        # Embedding representation
        draw.ellipse([emb_x, y + 30, emb_x + 40, y + 70],
                    fill=image_colors[i], outline=(200, 200, 200), width=2)
    
    # Draw texts on the right
    draw.text((text_encoder_x + 100, 160), "Texts", fill=(255, 180, 100), font=font_label)
    
    text_colors = image_colors  # Same colors for matching pairs
    
    for i in range(batch_size):
        y = y_start + i * (item_height + gap)
        # Text box
        draw_rounded_rectangle(draw, [text_encoder_x + 80, y, text_encoder_x + 180, y + item_height],
                              radius=8, fill=text_colors[i], outline=(200, 200, 200), width=2)
        draw.text((text_encoder_x + 105, y + 35), f"T{i+1}", fill=(255, 255, 255), font=font_label)
    
    # Text Encoder
    text_enc_x = text_encoder_x - 150
    draw_rounded_rectangle(draw, [text_enc_x, y_start, text_enc_x + encoder_width, y_start + encoder_height],
                          radius=10, fill=(60, 50, 40), outline=(255, 180, 100), width=3)
    draw.text((text_enc_x + 30, y_start + encoder_height // 2 - 20), "Text\nEncoder",
              fill=(255, 220, 180), font=font_label)
    
    # Text embeddings
    text_emb_x = text_enc_x - 80
    for i in range(batch_size):
        y = y_start + i * (item_height + gap)
        draw.ellipse([text_emb_x, y + 30, text_emb_x + 40, y + 70],
                    fill=text_colors[i], outline=(200, 200, 200), width=2)
    
    # Contrastive matrix in the center
    matrix_size = 400
    matrix_x = canvas_size[0] // 2 - matrix_size // 2
    matrix_y = y_start + 50
    
    draw.text((matrix_x + 100, matrix_y - 40), "Similarity Matrix",
              fill=(200, 200, 200), font=font_label)
    
    cell_size = matrix_size // batch_size
    
    # Animation: highlight matching pairs
    highlight_pair = (animation_step // 3) % batch_size
    
    for i in range(batch_size):
        for j in range(batch_size):
            x = matrix_x + j * cell_size
            y = matrix_y + i * cell_size
            
            # Diagonal are positive pairs, off-diagonal are negatives
            if i == j:
                # Positive pair
                if i == highlight_pair:
                    color = (0, 255, 0)
                    alpha = 220
                else:
                    color = (0, 200, 0)
                    alpha = 150
            else:
                # Negative pair
                if i == highlight_pair or j == highlight_pair:
                    color = (255, 0, 0)
                    alpha = 100
                else:
                    color = (150, 0, 0)
                    alpha = 60
            
            fill_color = color + (alpha,)
            # PIL doesn't support alpha directly, so we approximate
            dark_bg = (20, 25, 35)
            final_color = tuple(int(dark_bg[k] * (1 - alpha/255) + color[k] * (alpha/255)) for k in range(3))
            
            draw.rectangle([x, y, x + cell_size - 2, y + cell_size - 2],
                         fill=final_color, outline=(100, 100, 100), width=1)
    
    # Info box
    info_y = y_start + encoder_height + 80
    draw_rounded_rectangle(draw, [200, info_y, canvas_size[0] - 200, info_y + 150],
                          radius=10, fill=(30, 35, 45), outline=(100, 120, 150), width=2)
    
    info_lines = [
        f"• Positive pairs: {batch_size} (diagonal elements)",
        f"• Negative pairs: {batch_size * (batch_size - 1)} (off-diagonal elements)",
        f"• Total comparisons: {batch_size * batch_size}",
        "• Challenge: Limited negative samples scale with batch size"
    ]
    
    for i, line in enumerate(info_lines):
        draw.text((230, info_y + 20 + i * 32), line, fill=(200, 220, 240), font=font_small)
    
    return canvas


def create_global_frame(
    canvas_size: Tuple[int, int] = (1920, 1080),
    animation_step: int = 0
) -> Image.Image:
    """Create a frame showing Global Contrastive Learning."""
    canvas = Image.new('RGB', canvas_size, color=(15, 20, 30))
    draw = ImageDraw.Draw(canvas)
    
    font_title = get_font(40, bold=True)
    font_label = get_font(22, bold=False)
    font_small = get_font(18, bold=False)
    font_tiny = get_font(14, bold=False)
    
    # Title
    draw.text((50, 40), "Global Contrastive Learning: 1M Concept Centers",
              fill=(255, 180, 100), font=font_title)
    
    # Layout
    batch_size = 8
    sampled_negatives = 1024
    total_concepts = 1000000
    
    draw.text((50, 100), f"Batch: {batch_size} images | Sampled Negatives: {sampled_negatives:,} | Total Concepts: {total_concepts:,}",
              fill=(150, 180, 200), font=font_label)
    
    # Left side: Images and encoder
    img_x = 150
    y_start = 200
    item_height = 80
    gap = 15
    
    draw.text((img_x - 50, 160), "Images", fill=(255, 180, 100), font=font_label)
    
    image_colors = [
        (255, 100, 100), (100, 255, 100), (100, 100, 255), (255, 255, 100),
        (255, 100, 255), (100, 255, 255), (200, 150, 100), (150, 100, 200)
    ]
    
    for i in range(batch_size):
        y = y_start + i * (item_height + gap)
        draw_rounded_rectangle(draw, [img_x - 60, y, img_x + 40, y + item_height],
                              radius=8, fill=image_colors[i], outline=(200, 200, 200), width=2)
        draw.text((img_x - 45, y + 25), f"I{i+1}", fill=(255, 255, 255), font=font_label)
    
    # Image Encoder (no text encoder!)
    encoder_x = img_x + 180
    encoder_width = 180
    encoder_height = batch_size * (item_height + gap) - gap
    draw_rounded_rectangle(draw, [encoder_x, y_start, encoder_x + encoder_width, y_start + encoder_height],
                          radius=10, fill=(60, 50, 40), outline=(255, 180, 100), width=3)
    draw.text((encoder_x + 30, y_start + encoder_height // 2 - 20), "Image\nEncoder",
              fill=(255, 220, 180), font=font_label)
    
    # Image embeddings
    emb_x = encoder_x + encoder_width + 80
    for i in range(batch_size):
        y = y_start + i * (item_height + gap)
        draw.ellipse([emb_x, y + 20, emb_x + 40, y + 60],
                    fill=image_colors[i], outline=(200, 200, 200), width=2)
    
    # Concept Bank visualization (right side)
    bank_x = 1100
    bank_y = 180
    bank_width = 700
    bank_height = 650
    
    draw_rounded_rectangle(draw, [bank_x, bank_y, bank_x + bank_width, bank_y + bank_height],
                          radius=12, fill=(25, 30, 40), outline=(150, 120, 80), width=3)
    
    draw.text((bank_x + 200, bank_y + 20), "Concept Centers Bank",
              fill=(255, 200, 120), font=font_label)
    draw.text((bank_x + 220, bank_y + 50), f"({total_concepts:,} centers from offline clustering)",
              fill=(180, 160, 120), font=font_small)
    
    # Draw concept centers as a cloud of dots
    rng = np.random.default_rng(42)
    num_visible_concepts = 200
    
    # Animation: highlight sampled concepts
    highlight_step = animation_step % 30
    sampled_indices = set(rng.choice(num_visible_concepts, size=min(20, num_visible_concepts), replace=False))
    
    for i in range(num_visible_concepts):
        cx = bank_x + 50 + rng.integers(0, bank_width - 100)
        cy = bank_y + 100 + rng.integers(0, bank_height - 150)
        
        # Different colors for different concept clusters
        cluster_id = i % 10
        base_colors = [
            (255, 150, 150), (150, 255, 150), (150, 150, 255),
            (255, 255, 150), (255, 150, 255), (150, 255, 255),
            (200, 180, 150), (180, 150, 200), (150, 200, 180),
            (220, 180, 200)
        ]
        
        if i in sampled_indices and highlight_step > 15:
            # Highlighted sampled negative
            color = (255, 100, 0)
            size = 8
        else:
            color = base_colors[cluster_id]
            size = 4
        
        draw.ellipse([cx - size, cy - size, cx + size, cy + size],
                    fill=color, outline=(100, 100, 100), width=1)
    
    # Sampling annotation
    sample_box_y = bank_y + bank_height - 100
    draw_rounded_rectangle(draw, [bank_x + 50, sample_box_y, bank_x + bank_width - 50, sample_box_y + 80],
                          radius=8, fill=(40, 35, 30), outline=(255, 150, 50), width=2)
    
    draw.text((bank_x + 100, sample_box_y + 15),
              f"✓ Each batch samples {sampled_negatives:,} negatives",
              fill=(255, 200, 100), font=font_small)
    draw.text((bank_x + 100, sample_box_y + 45),
              "✓ Concept centers updated via offline clustering",
              fill=(255, 200, 100), font=font_small)
    
    # Connection lines (showing similarity computation)
    if highlight_step > 10:
        for i in range(min(3, batch_size)):
            start_x = emb_x + 40
            start_y = y_start + i * (item_height + gap) + 40
            
            # Draw a few lines to sampled concepts
            for j, concept_idx in enumerate(list(sampled_indices)[:5]):
                end_x = bank_x + 50
                end_y = bank_y + 150 + concept_idx * 2
                
                # Draw dotted line
                line_color = (100, 150, 200) if j == 0 else (80, 100, 120)
                draw.line([(start_x, start_y), (end_x, end_y)],
                         fill=line_color, width=1)
    
    # Info box at bottom
    info_y = y_start + encoder_height + 100
    draw_rounded_rectangle(draw, [100, info_y, canvas_size[0] - 100, info_y + 170],
                          radius=10, fill=(30, 35, 45), outline=(150, 120, 80), width=2)
    
    info_lines = [
        "• No text encoder - pure visual representation learning",
        f"• Positive: Current image embedding vs. itself",
        f"• Negatives: {sampled_negatives:,} sampled from {total_concepts:,} concept centers per batch",
        "• Concept centers from offline clustering (e.g., K-means on large dataset)",
        "• Advantages: Much larger negative pool, better separability, no text dependency"
    ]
    
    for i, line in enumerate(info_lines):
        draw.text((130, info_y + 15 + i * 30), line, fill=(200, 220, 240), font=font_small)
    
    return canvas


def create_comparison_frame(
    canvas_size: Tuple[int, int] = (1920, 1080)
) -> Image.Image:
    """Create a side-by-side comparison summary frame."""
    canvas = Image.new('RGB', canvas_size, color=(15, 20, 30))
    draw = ImageDraw.Draw(canvas)
    
    font_title = get_font(48, bold=True)
    font_subtitle = get_font(28, bold=True)
    font_text = get_font(22, bold=False)
    
    # Title
    draw.text((canvas_size[0] // 2 - 250, 50), "Key Differences",
              fill=(255, 255, 255), font=font_title)
    
    # Divider
    draw.line([(canvas_size[0] // 2, 150), (canvas_size[0] // 2, canvas_size[1] - 100)],
              fill=(80, 90, 110), width=4)
    
    # CLIP side
    clip_x = 100
    y_start = 180
    
    draw.text((clip_x + 200, y_start), "CLIP Approach",
              fill=(100, 200, 255), font=font_subtitle)
    
    clip_points = [
        ("Architecture", "Dual encoders (Image + Text)"),
        ("Input", "Image-Text pairs"),
        ("Negatives", "Within batch (32-1024 pairs)"),
        ("Limitation", "Limited by batch size"),
        ("Benefit", "Cross-modal alignment"),
        ("Scale", "~400M pairs training"),
    ]
    
    for i, (label, value) in enumerate(clip_points):
        y = y_start + 80 + i * 120
        draw.text((clip_x, y), label + ":", fill=(150, 180, 220), font=font_text)
        
        # Value box
        draw_rounded_rectangle(draw, [clip_x, y + 35, clip_x + 750, y + 90],
                              radius=8, fill=(35, 50, 70), outline=(100, 150, 200), width=2)
        draw.text((clip_x + 20, y + 47), value, fill=(200, 220, 255), font=font_text)
    
    # Global side
    global_x = canvas_size[0] // 2 + 100
    
    draw.text((global_x + 100, y_start), "Global Contrastive",
              fill=(255, 180, 100), font=font_subtitle)
    
    global_points = [
        ("Architecture", "Single encoder (Image only)"),
        ("Input", "Images only"),
        ("Negatives", "Sampled from 1M concept centers"),
        ("Limitation", "No text alignment"),
        ("Benefit", "Massive negative pool"),
        ("Scale", "~1M concept centers"),
    ]
    
    for i, (label, value) in enumerate(global_points):
        y = y_start + 80 + i * 120
        draw.text((global_x, y), label + ":", fill=(200, 160, 120), font=font_text)
        
        # Value box
        draw_rounded_rectangle(draw, [global_x, y + 35, global_x + 750, y + 90],
                              radius=8, fill=(45, 40, 35), outline=(255, 180, 100), width=2)
        draw.text((global_x + 20, y + 47), value, fill=(255, 220, 180), font=font_text)
    
    return canvas


def generate_animation(
    output_path: str,
    fps: int = 2,
    canvas_size: Tuple[int, int] = (1920, 1080),
    as_video: bool = False
) -> None:
    """Generate the comparison animation."""
    frames: List[np.ndarray] = []
    
    print("Generating frames...")
    
    # 1. Title frame (3 seconds)
    print("  - Title frame")
    title_frame = create_title_frame(canvas_size)
    for _ in range(fps * 3):
        frames.append(np.array(title_frame))
    
    # 2. CLIP frames with animation (8 seconds)
    print("  - CLIP animation frames")
    clip_frames = fps * 8
    for i in range(clip_frames):
        frame = create_clip_frame(canvas_size, i)
        frames.append(np.array(frame))
    
    # 3. Global frames with animation (8 seconds)
    print("  - Global contrastive animation frames")
    global_frames = fps * 8
    for i in range(global_frames):
        frame = create_global_frame(canvas_size, i)
        frames.append(np.array(frame))
    
    # 4. Comparison frame (4 seconds)
    print("  - Comparison summary frame")
    comparison_frame = create_comparison_frame(canvas_size)
    for _ in range(fps * 4):
        frames.append(np.array(comparison_frame))
    
    # Save output
    if as_video:
        if not output_path.lower().endswith('.mp4'):
            output_path = output_path.rsplit('.', 1)[0] + '.mp4'
        
        print(f"\nSaving video to: {output_path}")
        imageio.mimwrite(
            output_path,
            frames,
            fps=fps,
            codec='libx264',
            pixelformat='yuv420p'
        )
    else:
        if not output_path.lower().endswith('.gif'):
            output_path = output_path.rsplit('.', 1)[0] + '.gif'
        
        print(f"\nSaving GIF to: {output_path}")
        pil_frames = [Image.fromarray(f) for f in frames]
        pil_frames[0].save(
            output_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=int(1000 / fps),
            loop=0
        )
    
    print(f"✓ Animation saved successfully!")
    print(f"  - Total frames: {len(frames)}")
    print(f"  - Duration: {len(frames) / fps:.1f} seconds")
    print(f"  - Resolution: {canvas_size[0]}x{canvas_size[1]}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate animated comparison of CLIP vs Global Contrastive Learning"
    )
    parser.add_argument("--output", type=str, default="global_contrastive_comparison.gif",
                       help="Output file path (default: global_contrastive_comparison.gif)")
    parser.add_argument("--video", action="store_true",
                       help="Output as MP4 video instead of GIF")
    parser.add_argument("--fps", type=int, default=2,
                       help="Frames per second (default: 2)")
    parser.add_argument("--width", type=int, default=1920,
                       help="Canvas width (default: 1920)")
    parser.add_argument("--height", type=int, default=1080,
                       help="Canvas height (default: 1080)")
    
    args = parser.parse_args()
    
    generate_animation(
        output_path=args.output,
        fps=args.fps,
        canvas_size=(args.width, args.height),
        as_video=args.video
    )


if __name__ == "__main__":
    main()
