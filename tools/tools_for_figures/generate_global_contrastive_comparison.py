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
    """Create an introduction title frame with publication-quality design."""
    # Create smooth gradient background (soft blue to white)
    canvas = Image.new('RGB', canvas_size, color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    
    font_title = get_font(64, bold=True)
    font_subtitle = get_font(36, bold=False)
    font_text = get_font(26, bold=False)
    
    # Sophisticated gradient background with smooth transition
    for y in range(canvas_size[1]):
        alpha = y / canvas_size[1]
        # Smooth gradient from light blue to pure white
        r = int(240 + (255 - 240) * alpha ** 0.7)
        g = int(247 + (255 - 247) * alpha ** 0.7)
        b = int(255)
        draw.line([(0, y), (canvas_size[0], y)], fill=(r, g, b))
    
    # Main title with shadow effect for depth
    title = "Visual Contrastive Learning"
    bbox = draw.textbbox((0, 0), title, font=font_title)
    title_w = bbox[2] - bbox[0]
    title_x = canvas_size[0] // 2 - title_w // 2
    title_y = 180
    
    # Shadow
    draw.text((title_x + 3, title_y + 3), title, fill=(180, 190, 200), font=font_title)
    # Main text
    draw.text((title_x, title_y), title, fill=(30, 64, 175), font=font_title)
    
    # Elegant subtitle
    subtitle = "Comparing CLIP and Global Contrastive Approaches"
    bbox = draw.textbbox((0, 0), subtitle, font=font_subtitle)
    subtitle_w = bbox[2] - bbox[0]
    draw.text((canvas_size[0] // 2 - subtitle_w // 2, 270), subtitle,
              fill=(71, 85, 105), font=font_subtitle)
    
    # Elegant divider with gradient effect
    div_y = 350
    for i in range(3):
        opacity = 255 - i * 60
        draw.line([(canvas_size[0] // 2 - 500, div_y + i), (canvas_size[0] // 2 + 500, div_y + i)],
                  fill=(200 - i*20, 210 - i*20, 230 - i*10), width=1)
    
    # Enhanced content boxes with better spacing and visual hierarchy
    y_start = 440
    box_width = 750
    box_height = 520
    gap = 140
    
    # CLIP box with sophisticated styling
    clip_x = canvas_size[0] // 2 - box_width - gap // 2
    
    # Add subtle shadow for depth
    for offset in range(8, 0, -2):
        shadow_alpha = 20 - offset * 2
        shadow_color = (200, 205, 210)
        draw_rounded_rectangle(draw, 
            [clip_x + offset, y_start + offset, 
             clip_x + box_width + offset, y_start + box_height + offset],
            radius=20, fill=None, outline=shadow_color, width=1)
    
    draw_rounded_rectangle(draw, [clip_x, y_start, clip_x + box_width, y_start + box_height],
                          radius=20, fill=(249, 250, 251), outline=(147, 197, 253), width=4)
    
    # Title with icon-like accent
    draw.text((clip_x + 280, y_start + 40), "CLIP", fill=(37, 99, 235), font=font_subtitle)
    draw.text((clip_x + 200, y_start + 42), "▶", fill=(96, 165, 250), font=font_subtitle)
    
    clip_features = [
        "• Image-Text paired learning",
        "• Batch-level contrastive loss",
        "• Limited negative samples",
        "  (batch size: 32-1024)",
        "  (max ~32K negatives)",
        "• Dual encoder architecture:",
        "  › Image Encoder (ViT)",
        "  › Text Encoder (Transformer)",
        "• Cross-modal semantic alignment"
    ]
    
    for i, feature in enumerate(clip_features):
        draw.text((clip_x + 70, y_start + 140 + i * 48), feature,
                 fill=(51, 65, 85), font=font_text)
    
    # Global box with sophisticated styling
    global_x = canvas_size[0] // 2 + gap // 2
    
    # Add subtle shadow for depth
    for offset in range(8, 0, -2):
        shadow_color = (200, 205, 210)
        draw_rounded_rectangle(draw, 
            [global_x + offset, y_start + offset, 
             global_x + box_width + offset, y_start + box_height + offset],
            radius=20, fill=None, outline=shadow_color, width=1)
    
    draw_rounded_rectangle(draw, [global_x, y_start, global_x + box_width, y_start + box_height],
                          radius=20, fill=(240, 249, 255), outline=(59, 130, 246), width=4)
    
    # Title with icon-like accent
    draw.text((global_x + 180, y_start + 40), "Global Contrastive",
              fill=(30, 64, 175), font=font_subtitle)
    draw.text((global_x + 100, y_start + 42), "◆", fill=(59, 130, 246), font=font_subtitle)
    
    global_features = [
        "• Image-only representation",
        "• Global negative sampling",
        "• 1M concept centers pool",
        "  (offline clustering)",
        "• Single encoder architecture:",
        "  › Image Encoder only",
        "• Massive negative sampling:",
        "  › 1024+ negatives per batch"
    ]
    
    for i, feature in enumerate(global_features):
        draw.text((global_x + 70, y_start + 140 + i * 48), feature,
                 fill=(23, 37, 84), font=font_text)
    
    return canvas


def create_clip_frame(
    canvas_size: Tuple[int, int] = (1920, 1080),
    animation_step: int = 0
) -> Image.Image:
    """Create a frame showing CLIP's contrastive learning with publication-quality design."""
    # Elegant gradient background
    canvas = Image.new('RGB', canvas_size, color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    
    # Smooth gradient from top to bottom
    for y in range(canvas_size[1]):
        alpha = (y / canvas_size[1]) ** 0.6
        r = int(252 + (255 - 252) * alpha)
        g = int(252 + (255 - 252) * alpha)
        b = int(253 + (255 - 253) * alpha)
        draw.line([(0, y), (canvas_size[0], y)], fill=(r, g, b))
    
    font_title = get_font(44, bold=True)
    font_label = get_font(24, bold=False)
    font_small = get_font(20, bold=False)
    
    # Title with shadow for depth
    title_text = "CLIP: Batch-Level Image-Text Contrastive Learning"
    draw.text((53, 43), title_text, fill=(180, 190, 200), font=font_title)
    draw.text((50, 40), title_text, fill=(37, 99, 235), font=font_title)
    
    # Enhanced subtitle
    batch_size = 8
    draw.text((50, 105), f"Batch Size: {batch_size} pairs | Max ~32K negatives in large batches",
              fill=(71, 85, 105), font=font_label)
    
    # Layout with improved spacing
    img_encoder_x = 200
    text_encoder_x = 1500
    y_start = 210
    item_height = 95
    gap = 12
    
    # Draw images on the left with label
    draw.text((img_encoder_x - 90, 165), "Images", fill=(37, 99, 235), font=font_label)
    
    # Modern color palette
    image_colors = [
        (239, 68, 68), (34, 197, 94), (59, 130, 246), (251, 191, 36),
        (168, 85, 247), (20, 184, 166), (249, 115, 22), (236, 72, 153)
    ]
    
    for i in range(batch_size):
        y = y_start + i * (item_height + gap)
        
        # Add subtle shadow for depth
        for offset in range(4, 0, -1):
            shadow_color = (220 - offset*10, 220 - offset*10, 220 - offset*10)
            draw_rounded_rectangle(draw, 
                [img_encoder_x - 80 + offset, y + offset, 
                 img_encoder_x + 20 + offset, y + item_height + offset],
                radius=10, fill=None, outline=shadow_color, width=1)
        
        # Image box with gradient effect
        draw_rounded_rectangle(draw, [img_encoder_x - 80, y, img_encoder_x + 20, y + item_height],
                              radius=10, fill=image_colors[i], outline=(255, 255, 255), width=3)
        draw.text((img_encoder_x - 63, y + 32), f"I{i+1}", fill=(255, 255, 255), font=font_label)
    
    # Image Encoder with enhanced styling
    encoder_x = img_encoder_x + 160
    encoder_width = 200
    encoder_height = batch_size * (item_height + gap) - gap
    
    # Shadow for depth
    for offset in range(6, 0, -2):
        shadow_color = (200 - offset*8, 205 - offset*8, 215 - offset*8)
        draw_rounded_rectangle(draw, 
            [encoder_x + offset, y_start + offset, 
             encoder_x + encoder_width + offset, y_start + encoder_height + offset],
            radius=15, fill=None, outline=shadow_color, width=1)
    
    draw_rounded_rectangle(draw, [encoder_x, y_start, encoder_x + encoder_width, y_start + encoder_height],
                          radius=15, fill=(30, 58, 138), outline=(96, 165, 250), width=4)
    
    # Encoder label with better positioning
    draw.text((encoder_x + 48, y_start + encoder_height // 2 - 35), "Image",
              fill=(191, 219, 254), font=font_label)
    draw.text((encoder_x + 35, y_start + encoder_height // 2 - 5), "Encoder",
              fill=(191, 219, 254), font=font_label)
    draw.text((encoder_x + 40, y_start + encoder_height // 2 + 30), "(ViT-L)",
              fill=(147, 197, 253), font=font_small)
    
    # Image embeddings with enhanced appearance
    emb_x = encoder_x + encoder_width + 110
    for i in range(batch_size):
        y = y_start + i * (item_height + gap)
        # Subtle glow effect
        for offset in range(3, 0, -1):
            glow_alpha = 80 - offset * 20
            draw.ellipse([emb_x + 20 - offset*3, y + 28 - offset*3, 
                         emb_x + 20 + offset*3, y + 68 + offset*3],
                        fill=None, outline=image_colors[i], width=1)
        # Main embedding
        draw.ellipse([emb_x, y + 28, emb_x + 40, y + 68],
                    fill=image_colors[i], outline=(255, 255, 255), width=3)
    
    # Draw texts on the right with label
    draw.text((text_encoder_x + 110, 165), "Texts", fill=(249, 115, 22), font=font_label)
    
    text_colors = image_colors  # Same colors for matching pairs
    
    for i in range(batch_size):
        y = y_start + i * (item_height + gap)
        
        # Add shadow for depth
        for offset in range(4, 0, -1):
            shadow_color = (220 - offset*10, 220 - offset*10, 220 - offset*10)
            draw_rounded_rectangle(draw, 
                [text_encoder_x + 80 + offset, y + offset, 
                 text_encoder_x + 180 + offset, y + item_height + offset],
                radius=10, fill=None, outline=shadow_color, width=1)
        
        # Text box
        draw_rounded_rectangle(draw, [text_encoder_x + 80, y, text_encoder_x + 180, y + item_height],
                              radius=10, fill=text_colors[i], outline=(255, 255, 255), width=3)
        draw.text((text_encoder_x + 107, y + 32), f"T{i+1}", fill=(255, 255, 255), font=font_label)
    
    # Text Encoder with enhanced styling
    text_enc_x = text_encoder_x - 160
    
    # Shadow for depth
    for offset in range(6, 0, -2):
        shadow_color = (200 - offset*8, 205 - offset*8, 215 - offset*8)
        draw_rounded_rectangle(draw, 
            [text_enc_x + offset, y_start + offset, 
             text_enc_x + encoder_width + offset, y_start + encoder_height + offset],
            radius=15, fill=None, outline=shadow_color, width=1)
    
    draw_rounded_rectangle(draw, [text_enc_x, y_start, text_enc_x + encoder_width, y_start + encoder_height],
                          radius=15, fill=(124, 58, 237), outline=(196, 181, 253), width=4)
    
    # Encoder label
    draw.text((text_enc_x + 58, y_start + encoder_height // 2 - 35), "Text",
              fill=(233, 213, 255), font=font_label)
    draw.text((text_enc_x + 35, y_start + encoder_height // 2 - 5), "Encoder",
              fill=(233, 213, 255), font=font_label)
    draw.text((text_enc_x + 15, y_start + encoder_height // 2 + 30), "(Transformer)",
              fill=(196, 181, 253), font=font_small)
    
    # Text embeddings
    text_emb_x = text_enc_x - 90
    for i in range(batch_size):
        y = y_start + i * (item_height + gap)
        # Subtle glow effect
        for offset in range(3, 0, -1):
            draw.ellipse([text_emb_x + 20 - offset*3, y + 28 - offset*3, 
                         text_emb_x + 20 + offset*3, y + 68 + offset*3],
                        fill=None, outline=text_colors[i], width=1)
        # Main embedding
        draw.ellipse([text_emb_x, y + 28, text_emb_x + 40, y + 68],
                    fill=text_colors[i], outline=(255, 255, 255), width=3)
    
    # Enhanced contrastive matrix in the center
    matrix_size = 420
    matrix_x = canvas_size[0] // 2 - matrix_size // 2
    matrix_y = y_start + 80
    
    draw.text((matrix_x + 110, matrix_y - 45), "Similarity Matrix",
              fill=(71, 85, 105), font=font_label)
    
    cell_size = matrix_size // batch_size
    
    # Animation: highlight matching pairs with smooth effect
    highlight_pair = (animation_step // 3) % batch_size
    
    for i in range(batch_size):
        for j in range(batch_size):
            x = matrix_x + j * cell_size
            y = matrix_y + i * cell_size
            
            # Diagonal are positive pairs, off-diagonal are negatives
            if i == j:
                # Positive pair with vibrant green
                if i == highlight_pair:
                    color = (34, 197, 94)  # Bright green
                    alpha = 240
                else:
                    color = (74, 222, 128)  # Light green
                    alpha = 180
            else:
                # Negative pair with red tones
                if i == highlight_pair or j == highlight_pair:
                    color = (239, 68, 68)  # Bright red
                    alpha = 120
                else:
                    color = (252, 165, 165)  # Light red
                    alpha = 80
            
            # Blend with background
            bg = (248, 250, 252)
            final_color = tuple(int(bg[k] * (1 - alpha/255) + color[k] * (alpha/255)) for k in range(3))
            
            draw.rectangle([x + 1, y + 1, x + cell_size - 2, y + cell_size - 2],
                         fill=final_color, outline=(203, 213, 225), width=1)
    
    # Enhanced info box at bottom
    info_y = y_start + encoder_height + 100
    info_height = 160
    
    # Shadow for info box
    for offset in range(6, 0, -2):
        shadow_color = (200 - offset*8, 205 - offset*8, 215 - offset*8)
        draw_rounded_rectangle(draw, 
            [200 + offset, info_y + offset, 
             canvas_size[0] - 200 + offset, info_y + info_height + offset],
            radius=15, fill=None, outline=shadow_color, width=1)
    
    draw_rounded_rectangle(draw, [200, info_y, canvas_size[0] - 200, info_y + info_height],
                          radius=15, fill=(248, 250, 252), outline=(147, 197, 253), width=4)
    
    info_lines = [
        f"⦿ Positive pairs: {batch_size} (diagonal - matching image-text pairs)",
        f"⦿ Negative pairs: {batch_size * (batch_size - 1)} (off-diagonal - mismatched pairs)",
        f"⦿ Total comparisons: {batch_size * batch_size} within batch",
        "⦿ Limitation: Negative samples scale with batch size (max ~32K in largest batches)"
    ]
    
    for i, line in enumerate(info_lines):
        draw.text((240, info_y + 24 + i * 34), line, fill=(51, 65, 85), font=font_small)
    
    return canvas


def create_global_frame(
    canvas_size: Tuple[int, int] = (1920, 1080),
    animation_step: int = 0
) -> Image.Image:
    """Create a frame showing Global Contrastive Learning with publication-quality design."""
    # Elegant gradient background
    canvas = Image.new('RGB', canvas_size, color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    
    # Smooth gradient
    for y in range(canvas_size[1]):
        alpha = (y / canvas_size[1]) ** 0.6
        r = int(250 + (255 - 250) * alpha)
        g = int(251 + (255 - 251) * alpha)
        b = int(254 + (255 - 254) * alpha)
        draw.line([(0, y), (canvas_size[0], y)], fill=(r, g, b))
    
    font_title = get_font(44, bold=True)
    font_label = get_font(24, bold=False)
    font_small = get_font(20, bold=False)
    font_tiny = get_font(16, bold=False)
    
    # Title with shadow for depth
    title_text = "Global Contrastive Learning: 1M Concept Centers"
    draw.text((53, 43), title_text, fill=(180, 190, 200), font=font_title)
    draw.text((50, 40), title_text, fill=(30, 64, 175), font=font_title)
    
    # Layout
    batch_size = 8
    sampled_negatives = 1024
    total_concepts = 1000000
    num_positive_centers = 10
    
    draw.text((50, 105), f"Batch: {batch_size} images | Sampled Negatives: {sampled_negatives:,} | Total Concepts: {total_concepts:,}",
              fill=(71, 85, 105), font=font_label)
    
    # Left side: Images and encoder with improved layout
    img_x = 150
    y_start = 210
    item_height = 75
    gap = 20
    
    draw.text((img_x - 40, 165), "Images", fill=(30, 64, 175), font=font_label)
    
    # Modern color palette matching CLIP
    image_colors = [
        (239, 68, 68), (34, 197, 94), (59, 130, 246), (251, 191, 36),
        (168, 85, 247), (20, 184, 166), (249, 115, 22), (236, 72, 153)
    ]
    
    # Animation: cycle through samples
    current_sample = (animation_step // 6) % batch_size
    sample_phase = (animation_step % 6)
    
    for i in range(batch_size):
        y = y_start + i * (item_height + gap)
        
        # Highlight current sample being processed
        if i == current_sample and sample_phase >= 2:
            outline_color = (251, 191, 36)  # Vibrant yellow
            outline_width = 4
            glow = True
        else:
            outline_color = (203, 213, 225)
            outline_width = 2
            glow = False
        
        # Draw sophisticated glow effect
        if glow:
            for offset in range(6, 0, -2):
                alpha = 120 - offset * 15
                glow_alpha = alpha / 255
                glow_r = int(255 * glow_alpha + 255 * (1 - glow_alpha))
                glow_g = int(200 * glow_alpha + 255 * (1 - glow_alpha))
                glow_b = int(0 * glow_alpha + 255 * (1 - glow_alpha))
                draw_rounded_rectangle(draw, 
                    [img_x - 60 - offset*2, y - offset*2, 
                     img_x + 40 + offset*2, y + item_height + offset*2],
                    radius=12, fill=None, outline=(glow_r, glow_g, glow_b), width=2)
        
        # Add shadow for depth
        for offset in range(4, 0, -1):
            shadow_color = (220 - offset*10, 220 - offset*10, 220 - offset*10)
            draw_rounded_rectangle(draw, 
                [img_x - 60 + offset, y + offset, 
                 img_x + 40 + offset, y + item_height + offset],
                radius=12, fill=None, outline=shadow_color, width=1)
        
        draw_rounded_rectangle(draw, [img_x - 60, y, img_x + 40, y + item_height],
                              radius=12, fill=image_colors[i], outline=outline_color, width=outline_width)
        draw.text((img_x - 43, y + 22), f"I{i+1}", fill=(255, 255, 255), font=font_label)
    
    # Image Encoder with sophisticated styling
    encoder_x = img_x + 190
    encoder_width = 200
    encoder_height = batch_size * (item_height + gap) - gap
    
    # Shadow for depth
    for offset in range(6, 0, -2):
        shadow_color = (200 - offset*8, 205 - offset*8, 215 - offset*8)
        draw_rounded_rectangle(draw, 
            [encoder_x + offset, y_start + offset, 
             encoder_x + encoder_width + offset, y_start + encoder_height + offset],
            radius=15, fill=None, outline=shadow_color, width=1)
    
    draw_rounded_rectangle(draw, [encoder_x, y_start, encoder_x + encoder_width, y_start + encoder_height],
                          radius=15, fill=(30, 58, 138), outline=(59, 130, 246), width=4)
    
    # Add encoder details with better typography
    draw.text((encoder_x + 48, y_start + encoder_height // 2 - 40), "Image",
              fill=(191, 219, 254), font=font_label)
    draw.text((encoder_x + 35, y_start + encoder_height // 2 - 10), "Encoder",
              fill=(191, 219, 254), font=font_label)
    draw.text((encoder_x + 40, y_start + encoder_height // 2 + 25), "(ViT-L/14)",
              fill=(147, 197, 253), font=font_small)
    
    # Image embeddings with sophisticated animation
    emb_x = encoder_x + encoder_width + 100
    for i in range(batch_size):
        y = y_start + i * (item_height + gap)
        
        # Enhanced pulse effect for current sample
        if i == current_sample and sample_phase >= 2:
            pulse = 1.0 + 0.15 * np.sin(animation_step * 0.5)
            size = int(20 * pulse)
            # Multi-layer glow
            for offset in range(4, 0, -1):
                glow_size = size + offset * 2
                draw.ellipse([emb_x + 20 - glow_size, y + 12, 
                            emb_x + 20 + glow_size, y + 52],
                           fill=None, outline=(251, 191, 36), width=1)
            draw.ellipse([emb_x + 20 - size, y + 12, emb_x + 20 + size, y + 52],
                        fill=image_colors[i], outline=(251, 191, 36), width=4)
        else:
            # Normal state with subtle glow
            for offset in range(2, 0, -1):
                draw.ellipse([emb_x + 20 - offset*3, y + 12 - offset*3, 
                            emb_x + 20 + offset*3, y + 52 + offset*3],
                           fill=None, outline=image_colors[i], width=1)
            draw.ellipse([emb_x, y + 12, emb_x + 40, y + 52],
                        fill=image_colors[i], outline=(255, 255, 255), width=3)
    
    # Concept Bank visualization (right side) - Publication quality
    bank_x = 1000
    bank_y = 165
    bank_width = 820
    bank_height = 720
    
    # Sophisticated gradient background
    for i in range(bank_height):
        alpha = (i / bank_height) ** 0.8
        # Deep blue gradient
        r = int(240 + (250 - 240) * alpha)
        g = int(245 + (252 - 245) * alpha)
        b = int(250 + (255 - 250) * alpha)
        draw.line([(bank_x, bank_y + i), (bank_x + bank_width, bank_y + i)], fill=(r, g, b))
    
    # Shadow for depth
    for offset in range(8, 0, -2):
        shadow_color = (200 - offset*6, 205 - offset*6, 215 - offset*6)
        draw_rounded_rectangle(draw, 
            [bank_x + offset, bank_y + offset, 
             bank_x + bank_width + offset, bank_y + bank_height + offset],
            radius=20, fill=None, outline=shadow_color, width=1)
    
    draw_rounded_rectangle(draw, [bank_x, bank_y, bank_x + bank_width, bank_y + bank_height],
                          radius=20, fill=None, outline=(59, 130, 246), width=5)
    
    # Title with better styling
    title_text = "Concept Centers Bank"
    bbox = draw.textbbox((0, 0), title_text, font=font_label)
    title_w = bbox[2] - bbox[0]
    draw.text((bank_x + (bank_width - title_w) // 2, bank_y + 25), title_text,
              fill=(30, 64, 175), font=font_label)
    
    subtitle_text = f"({total_concepts:,} centers from offline clustering)"
    bbox = draw.textbbox((0, 0), subtitle_text, font=font_small)
    subtitle_w = bbox[2] - bbox[0]
    draw.text((bank_x + (bank_width - subtitle_w) // 2, bank_y + 60), subtitle_text,
              fill=(71, 85, 105), font=font_small)
    
    # Create stable random positions for concept centers
    rng = np.random.default_rng(42)
    num_visible_concepts = 320
    concept_positions = []
    for i in range(num_visible_concepts):
        cx = bank_x + 100 + rng.integers(0, bank_width - 200)
        cy = bank_y + 130 + rng.integers(0, bank_height - 240)
        concept_positions.append((cx, cy, i))
    
    # Determine which concepts to highlight based on current sample
    positive_centers = set()
    negative_centers = set()
    
    if sample_phase >= 3:
        # Select 10 positive centers for current sample - clustered together
        sample_seed = current_sample * 1000
        pos_rng = np.random.default_rng(sample_seed)
        
        # Pick a random center point for the positive cluster
        cluster_center_idx = pos_rng.integers(0, num_visible_concepts)
        cluster_cx, cluster_cy, _ = concept_positions[cluster_center_idx]
        
        # Find the 10 closest centers to the cluster center
        distances = []
        for i, (cx, cy, idx) in enumerate(concept_positions):
            dist = (cx - cluster_cx)**2 + (cy - cluster_cy)**2
            distances.append((dist, i))
        
        # Sort by distance and take the 10 closest
        distances.sort()
        positive_indices = [distances[i][1] for i in range(min(num_positive_centers, num_visible_concepts))]
        positive_centers = set(positive_indices)
        
        # Select random negative centers - 20% of all visible concepts, scattered
        neg_rng = np.random.default_rng(sample_seed + 1)
        available = [i for i in range(num_visible_concepts) if i not in positive_centers]
        num_visible_negatives = int(num_visible_concepts * 0.2)
        negative_indices = neg_rng.choice(len(available), size=min(num_visible_negatives, len(available)), replace=False)
        negative_centers = set(available[i] for i in negative_indices)
    
    # Draw concept centers with publication-quality styling
    modern_palette = [
        (239, 68, 68), (34, 197, 94), (59, 130, 246), (251, 191, 36),
        (168, 85, 247), (20, 184, 166), (249, 115, 22), (236, 72, 153),
        (220, 38, 38), (101, 163, 13), (29, 78, 216), (217, 119, 6)
    ]
    
    for cx, cy, i in concept_positions:
        # Use modern color palette
        cluster_id = i % len(modern_palette)
        base_color = modern_palette[cluster_id]
        
        if i in positive_centers:
            # Positive centers - vibrant green with sophisticated glow
            color = (34, 197, 94)
            size = 10
            # Multi-layer glow
            for offset in range(4, 0, -1):
                glow_alpha = (5 - offset) * 30
                glow_r = int(34 + (255 - 34) * (1 - glow_alpha/255))
                glow_g = int(197 + (255 - 197) * (1 - glow_alpha/255))
                glow_b = int(94 + (255 - 94) * (1 - glow_alpha/255))
                draw.ellipse([cx - size - offset*3, cy - size - offset*3, 
                            cx + size + offset*3, cy + size + offset*3],
                           fill=None, outline=(glow_r, glow_g, glow_b), width=2)
            draw.ellipse([cx - size, cy - size, cx + size, cy + size],
                        fill=color, outline=(255, 255, 255), width=2)
        elif i in negative_centers:
            # Negative centers - vibrant red/orange with glow
            color = (239, 68, 68)
            size = 9
            # Multi-layer glow
            for offset in range(3, 0, -1):
                glow_alpha = (4 - offset) * 30
                glow_r = int(239 + (255 - 239) * (1 - glow_alpha/255))
                glow_g = int(68 + (255 - 68) * (1 - glow_alpha/255))
                glow_b = int(68 + (255 - 68) * (1 - glow_alpha/255))
                draw.ellipse([cx - size - offset*2, cy - size - offset*2, 
                            cx + size + offset*2, cy + size + offset*2],
                           fill=None, outline=(glow_r, glow_g, glow_b), width=1)
            draw.ellipse([cx - size, cy - size, cx + size, cy + size],
                        fill=color, outline=(255, 255, 255), width=2)
        else:
            # Regular centers with subtle styling
            color = base_color
            size = 5
            # Subtle border
            draw.ellipse([cx - size, cy - size, cx + size, cy + size],
                        fill=color, outline=(255, 255, 255), width=1)
    
    # Draw elegant connection lines from current sample to centers
    if sample_phase >= 4 and current_sample < batch_size:
        start_x = emb_x + 40
        start_y = y_start + current_sample * (item_height + gap) + 32
        
        # Lines to positive centers (green) with smooth curves
        for cx, cy, i in concept_positions:
            if i in positive_centers:
                # Draw smooth animated line
                draw.line([(start_x, start_y), (cx, cy)], 
                         fill=(74, 222, 128), width=2)
        
        # Lines to negative centers (red/orange)
        if sample_phase >= 5:
            for cx, cy, i in concept_positions[:60]:  # Limit to avoid clutter
                if i in negative_centers:
                    # Thinner lines for negatives
                    draw.line([(start_x, start_y), (cx, cy)], 
                             fill=(252, 165, 165), width=1)
    
    # Publication-quality legend box
    legend_x = bank_x + 60
    legend_y = bank_y + bank_height - 140
    legend_width = bank_width - 120
    legend_height = 120
    
    # Shadow for legend
    for offset in range(4, 0, -1):
        shadow_color = (200 - offset*10, 205 - offset*10, 215 - offset*10)
        draw_rounded_rectangle(draw, 
            [legend_x + offset, legend_y + offset, 
             legend_x + legend_width + offset, legend_y + legend_height + offset],
            radius=15, fill=None, outline=shadow_color, width=1)
    
    draw_rounded_rectangle(draw, [legend_x, legend_y, legend_x + legend_width, legend_y + legend_height],
                          radius=15, fill=(248, 250, 252), outline=(147, 197, 253), width=4)
    
    # Legend items with visual indicators
    num_visible_negatives = int(num_visible_concepts * 0.2)
    legend_items = [
        ("Selected Sample", (251, 191, 36), 10),
        (f"{num_positive_centers} Positive Centers", (34, 197, 94), 10),
        (f"{num_visible_negatives} Sampled Negatives", (239, 68, 68), 9),
        ("Other Concepts", (148, 163, 184), 5)
    ]
    
    item_x = legend_x + 40
    item_width = legend_width // 2 - 20
    
    for idx, (label, color, size) in enumerate(legend_items):
        row = idx // 2
        col = idx % 2
        x = item_x + col * item_width
        y = legend_y + 25 + row * 45
        
        # Draw indicator circle
        draw.ellipse([x, y + 5, x + size*2, y + 5 + size*2],
                    fill=color, outline=(255, 255, 255), width=2)
        
        # Draw label
        draw.text((x + 30, y + 3), label, fill=(51, 65, 85), font=font_small)
    
    # Enhanced info box at bottom
    info_y = y_start + encoder_height + 130
    info_height = 140
    
    # Shadow for info box
    for offset in range(6, 0, -2):
        shadow_color = (200 - offset*8, 205 - offset*8, 215 - offset*8)
        draw_rounded_rectangle(draw, 
            [100 + offset, info_y + offset, 
             canvas_size[0] - 100 + offset, info_y + info_height + offset],
            radius=15, fill=None, outline=shadow_color, width=1)
    
    draw_rounded_rectangle(draw, [100, info_y, canvas_size[0] - 100, info_y + info_height],
                          radius=15, fill=(248, 250, 252), outline=(59, 130, 246), width=4)
    
    info_lines = [
        "✓ Pure visual representation learning without text encoder",
        f"✓ Each sample: {num_positive_centers} clustered positives + {sampled_negatives:,} sampled negatives",
        f"✓ Positives clustered together; negatives scattered across concept space",
        "✓ Concept bank built via offline clustering (e.g., K-means on ImageNet-21K)"
    ]
    
    for i, line in enumerate(info_lines):
        draw.text((140, info_y + 22 + i * 32), line, fill=(51, 65, 85), font=font_small)
    
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
    
    # 3. Global frames with enhanced sampling animation (12 seconds - longer to show sampling)
    print("  - Global contrastive animation frames")
    global_frames = fps * 12
    for i in range(global_frames):
        frame = create_global_frame(canvas_size, i)
        frames.append(np.array(frame))
    
    # Note: Comparison frame removed as per requirements
    
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
    print(f"  - Enhanced with publication-quality design")
    print(f"  - Removed: Key differences frame (as requested)")


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
