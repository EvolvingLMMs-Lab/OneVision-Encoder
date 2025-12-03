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
    """Create an introduction title frame with webpage-matching colors."""
    # Light background matching webpage (#eff6ff to #ffffff)
    canvas = Image.new('RGB', canvas_size, color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    
    font_title = get_font(56, bold=True)
    font_subtitle = get_font(32, bold=False)
    font_text = get_font(24, bold=False)
    
    # Gradient background (light blue to white)
    for y in range(canvas_size[1]):
        alpha = y / canvas_size[1]
        # From #eff6ff (239, 246, 255) to white (255, 255, 255)
        r = int(239 + (255 - 239) * alpha)
        g = int(246 + (255 - 246) * alpha)
        b = int(255)
        draw.line([(0, y), (canvas_size[0], y)], fill=(r, g, b))
    
    # Title with blue gradient (#2563eb to #4f46e5)
    title = "Cluster Discrimination Visualization"
    title_w = len(title) * 35
    draw.text((canvas_size[0] // 2 - title_w // 2, 200), title, 
              fill=(37, 99, 235), font=font_title)  # #2563eb
    
    # Subtitle
    subtitle = "CLIP vs. Global Contrastive Learning"
    subtitle_w = len(subtitle) * 20
    draw.text((canvas_size[0] // 2 - subtitle_w // 2, 280), subtitle,
              fill=(79, 70, 229), font=font_subtitle)  # #4f46e5
    
    # Divider line
    draw.line([(canvas_size[0] // 2 - 400, 350), (canvas_size[0] // 2 + 400, 350)],
              fill=(203, 213, 225), width=2)  # slate-300
    
    # Content boxes
    y_start = 420
    box_width = 700
    box_height = 500
    gap = 120
    
    # CLIP box with blue theme
    clip_x = canvas_size[0] // 2 - box_width - gap // 2
    draw_rounded_rectangle(draw, [clip_x, y_start, clip_x + box_width, y_start + box_height],
                          radius=15, fill=(248, 250, 252), outline=(148, 163, 184), width=3)  # slate-50, slate-400
    
    draw.text((clip_x + 250, y_start + 30), "CLIP", fill=(37, 99, 235), font=font_subtitle)  # #2563eb
    
    clip_features = [
        "• Image-Text pairs",
        "• Batch-level contrastive",
        "• Limited negative samples",
        "  (batch size: 32-1024,",
        "   max ~32K negatives)",
        "• Dual encoders:",
        "  - Image Encoder",
        "  - Text Encoder",
        "• Cross-modal matching"
    ]
    
    for i, feature in enumerate(clip_features):
        draw.text((clip_x + 50, y_start + 120 + i * 48), feature,
                 fill=(71, 85, 105), font=font_text)  # slate-600
    
    # Global box with blue theme
    global_x = canvas_size[0] // 2 + gap // 2
    draw_rounded_rectangle(draw, [global_x, y_start, global_x + box_width, y_start + box_height],
                          radius=15, fill=(239, 246, 255), outline=(96, 165, 250), width=3)  # blue-50, blue-400
    
    draw.text((global_x + 150, y_start + 30), "Global Contrastive",
              fill=(29, 78, 216), font=font_subtitle)  # blue-700
    
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
                 fill=(30, 58, 138), font=font_text)  # blue-900
    
    return canvas


def create_clip_frame(
    canvas_size: Tuple[int, int] = (1920, 1080),
    animation_step: int = 0
) -> Image.Image:
    """Create a frame showing CLIP's contrastive learning with light theme."""
    canvas = Image.new('RGB', canvas_size, color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    
    font_title = get_font(40, bold=True)
    font_label = get_font(22, bold=False)
    font_small = get_font(18, bold=False)
    
    # Title
    draw.text((50, 40), "CLIP: Batch-Level Image-Text Contrastive Learning",
              fill=(37, 99, 235), font=font_title)  # #2563eb
    
    # Batch size indicator - Updated to mention 32K negatives
    batch_size = 8
    draw.text((50, 100), f"Batch Size: {batch_size} pairs | Max ~32K negatives in large batches",
              fill=(71, 85, 105), font=font_label)  # slate-600
    
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
        "• Challenge: Limited negative samples scale with batch size (max ~32K in large batches)"
    ]
    
    for i, line in enumerate(info_lines):
        draw.text((230, info_y + 20 + i * 32), line, fill=(200, 220, 240), font=font_small)
    
    return canvas


def create_global_frame(
    canvas_size: Tuple[int, int] = (1920, 1080),
    animation_step: int = 0
) -> Image.Image:
    """Create a frame showing Global Contrastive Learning with sampling animation and light theme."""
    canvas = Image.new('RGB', canvas_size, color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    
    font_title = get_font(40, bold=True)
    font_label = get_font(22, bold=False)
    font_small = get_font(18, bold=False)
    font_tiny = get_font(14, bold=False)
    
    # Title with blue color
    draw.text((50, 40), "Global Contrastive Learning: 1M Concept Centers",
              fill=(29, 78, 216), font=font_title)  # blue-700
    
    # Layout
    batch_size = 8
    sampled_negatives = 1024
    total_concepts = 1000000
    num_positive_centers = 10
    
    draw.text((50, 100), f"Batch: {batch_size} images | Sampled Negatives: {sampled_negatives:,} | Total Concepts: {total_concepts:,}",
              fill=(71, 85, 105), font=font_label)  # slate-600
    
    # Left side: Images and encoder
    img_x = 150
    y_start = 200
    item_height = 70
    gap = 18
    
    draw.text((img_x - 50, 160), "Images", fill=(255, 200, 120), font=font_label)
    
    image_colors = [
        (255, 100, 100), (100, 255, 100), (100, 100, 255), (255, 255, 100),
        (255, 100, 255), (100, 255, 255), (200, 150, 100), (150, 100, 200)
    ]
    
    # Animation: cycle through samples
    current_sample = (animation_step // 6) % batch_size
    sample_phase = (animation_step % 6)
    
    for i in range(batch_size):
        y = y_start + i * (item_height + gap)
        
        # Highlight current sample being processed
        if i == current_sample and sample_phase >= 2:
            outline_color = (255, 255, 100)
            outline_width = 4
            glow = True
        else:
            outline_color = (180, 180, 180)
            outline_width = 2
            glow = False
        
        # Draw glow effect for selected sample
        if glow:
            for offset in range(3, 0, -1):
                alpha = 80 - offset * 20
                glow_color = tuple(int(c * alpha / 255) for c in outline_color)
                draw_rounded_rectangle(draw, 
                    [img_x - 60 - offset*2, y - offset*2, 
                     img_x + 40 + offset*2, y + item_height + offset*2],
                    radius=10, fill=None, outline=glow_color, width=1)
        
        draw_rounded_rectangle(draw, [img_x - 60, y, img_x + 40, y + item_height],
                              radius=8, fill=image_colors[i], outline=outline_color, width=outline_width)
        draw.text((img_x - 45, y + 20), f"I{i+1}", fill=(255, 255, 255), font=font_label)
    
    # Image Encoder with enhanced styling
    encoder_x = img_x + 180
    encoder_width = 180
    encoder_height = batch_size * (item_height + gap) - gap
    draw_rounded_rectangle(draw, [encoder_x, y_start, encoder_x + encoder_width, y_start + encoder_height],
                          radius=12, fill=(50, 45, 40), outline=(255, 200, 120), width=3)
    
    # Add encoder details
    draw.text((encoder_x + 35, y_start + encoder_height // 2 - 30), "Image",
              fill=(255, 230, 200), font=font_label)
    draw.text((encoder_x + 25, y_start + encoder_height // 2 - 5), "Encoder",
              fill=(255, 230, 200), font=font_label)
    draw.text((encoder_x + 15, y_start + encoder_height // 2 + 25), "(ViT-L/14)",
              fill=(180, 160, 140), font=font_small)
    
    # Image embeddings with animation
    emb_x = encoder_x + encoder_width + 80
    for i in range(batch_size):
        y = y_start + i * (item_height + gap)
        
        # Pulse effect for current sample
        if i == current_sample and sample_phase >= 2:
            pulse = 1.0 + 0.2 * np.sin(animation_step * 0.5)
            size = int(20 * pulse)
            draw.ellipse([emb_x + 20 - size, y + 15, emb_x + 20 + size, y + 55],
                        fill=image_colors[i], outline=(255, 255, 100), width=3)
        else:
            draw.ellipse([emb_x, y + 15, emb_x + 40, y + 55],
                        fill=image_colors[i], outline=(180, 180, 180), width=2)
    
    # Concept Bank visualization (right side) - Enhanced
    bank_x = 1000
    bank_y = 160
    bank_width = 800
    bank_height = 700
    
    # Bank background with gradient
    for i in range(bank_height):
        alpha = 0.3 + 0.7 * (i / bank_height)
        color = tuple(int(25 + 15 * alpha) for _ in range(3))
        draw.line([(bank_x, bank_y + i), (bank_x + bank_width, bank_y + i)], fill=color)
    
    draw_rounded_rectangle(draw, [bank_x, bank_y, bank_x + bank_width, bank_y + bank_height],
                          radius=15, fill=None, outline=(200, 160, 100), width=4)
    
    draw.text((bank_x + 220, bank_y + 20), "Concept Centers Bank",
              fill=(255, 220, 140), font=font_label)
    draw.text((bank_x + 180, bank_y + 55), f"({total_concepts:,} centers from offline clustering)",
              fill=(200, 180, 140), font=font_small)
    
    # Create stable random positions for concept centers
    rng = np.random.default_rng(42)
    num_visible_concepts = 300
    concept_positions = []
    for i in range(num_visible_concepts):
        cx = bank_x + 80 + rng.integers(0, bank_width - 160)
        cy = bank_y + 120 + rng.integers(0, bank_height - 200)
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
        num_visible_negatives = int(num_visible_concepts * 0.2)  # 20% of all visible concepts
        negative_indices = neg_rng.choice(len(available), size=min(num_visible_negatives, len(available)), replace=False)
        negative_centers = set(available[i] for i in negative_indices)
    
    # Draw concept centers with enhanced styling
    for cx, cy, i in concept_positions:
        # Different colors for different concept clusters
        cluster_id = i % 10
        base_colors = [
            (255, 180, 180), (180, 255, 180), (180, 180, 255),
            (255, 255, 180), (255, 180, 255), (180, 255, 255),
            (220, 200, 180), (200, 180, 220), (180, 220, 200),
            (240, 200, 220)
        ]
        
        if i in positive_centers:
            # Positive centers - green with glow
            color = (100, 255, 100)
            size = 9
            glow_color = (150, 255, 150)
            # Draw glow
            for offset in range(2, 0, -1):
                draw.ellipse([cx - size - offset*2, cy - size - offset*2, 
                            cx + size + offset*2, cy + size + offset*2],
                           fill=None, outline=glow_color, width=1)
            draw.ellipse([cx - size, cy - size, cx + size, cy + size],
                        fill=color, outline=(50, 200, 50), width=2)
        elif i in negative_centers:
            # Negative centers - red/orange with glow
            color = (255, 100, 50)
            size = 8
            glow_color = (255, 150, 100)
            # Draw glow
            for offset in range(2, 0, -1):
                draw.ellipse([cx - size - offset*2, cy - size - offset*2, 
                            cx + size + offset*2, cy + size + offset*2],
                           fill=None, outline=glow_color, width=1)
            draw.ellipse([cx - size, cy - size, cx + size, cy + size],
                        fill=color, outline=(200, 50, 0), width=2)
        else:
            # Regular centers
            color = base_colors[cluster_id]
            size = 4
            draw.ellipse([cx - size, cy - size, cx + size, cy + size],
                        fill=color, outline=(120, 120, 120), width=1)
    
    # Draw connection lines from current sample to positive/negative centers
    if sample_phase >= 4 and current_sample < batch_size:
        start_x = emb_x + 40
        start_y = y_start + current_sample * (item_height + gap) + 35
        
        # Lines to positive centers (green)
        for cx, cy, i in concept_positions:
            if i in positive_centers:
                # Draw animated dashed line
                dash_length = 10
                gap_length = 5
                total_length = np.sqrt((cx - start_x)**2 + (cy - start_y)**2)
                num_dashes = int(total_length / (dash_length + gap_length))
                
                for d in range(num_dashes):
                    t1 = d * (dash_length + gap_length) / total_length
                    t2 = (d * (dash_length + gap_length) + dash_length) / total_length
                    x1 = int(start_x + t1 * (cx - start_x))
                    y1 = int(start_y + t1 * (cy - start_y))
                    x2 = int(start_x + t2 * (cx - start_x))
                    y2 = int(start_y + t2 * (cy - start_y))
                    draw.line([(x1, y1), (x2, y2)], fill=(100, 255, 100), width=2)
        
        # Lines to negative centers (red/orange)
        if sample_phase >= 5:
            for cx, cy, i in concept_positions[:50]:  # Limit to avoid clutter
                if i in negative_centers:
                    draw.line([(start_x, start_y), (cx, cy)], 
                            fill=(255, 100, 50), width=1)
    
    # Legend box - Enhanced
    legend_x = bank_x + 50
    legend_y = bank_y + bank_height - 130
    legend_width = bank_width - 100
    legend_height = 110
    
    draw_rounded_rectangle(draw, [legend_x, legend_y, legend_x + legend_width, legend_y + legend_height],
                          radius=10, fill=(20, 25, 35), outline=(255, 180, 100), width=3)
    
    # Legend items with visual indicators
    num_visible_negatives = int(num_visible_concepts * 0.2)  # Calculate 20%
    legend_items = [
        ("Selected Sample", (255, 255, 100), "circle"),
        (f"{num_positive_centers} Positive Centers (Clustered)", (100, 255, 100), "circle"),
        (f"{num_visible_negatives} Sampled Negatives (20%)", (255, 100, 50), "circle"),
        ("Other Concepts", (180, 180, 200), "circle")
    ]
    
    item_x = legend_x + 30
    item_width = legend_width // 2 - 20
    
    for idx, (label, color, shape) in enumerate(legend_items):
        row = idx // 2
        col = idx % 2
        x = item_x + col * item_width
        y = legend_y + 20 + row * 40
        
        # Draw indicator
        if shape == "circle":
            size = 8
            draw.ellipse([x, y + 5, x + size*2, y + 5 + size*2],
                        fill=color, outline=(200, 200, 200), width=1)
        
        # Draw label
        draw.text((x + 25, y + 3), label, fill=(220, 220, 240), font=font_small)
    
    # Info box at bottom - Enhanced
    info_y = y_start + encoder_height + 120
    info_height = 130
    draw_rounded_rectangle(draw, [80, info_y, canvas_size[0] - 80, info_y + info_height],
                          radius=12, fill=(25, 30, 40), outline=(200, 160, 100), width=3)
    
    info_lines = [
        "✓ No text encoder - pure visual representation learning",
        f"✓ Each sample matched with {num_positive_centers} clustered positive centers + {sampled_negatives:,} sampled negatives",
        f"✓ Positive centers clustered together; negatives scattered (20% of visible concepts)",
        "✓ Concept centers from offline clustering (e.g., K-means on large-scale dataset)"
    ]
    
    for i, line in enumerate(info_lines):
        draw.text((110, info_y + 20 + i * 28), line, fill=(220, 230, 240), font=font_small)
    
    return canvas


def create_comparison_frame(
    canvas_size: Tuple[int, int] = (1920, 1080)
) -> Image.Image:
    """Create a side-by-side comparison summary frame with light theme."""
    canvas = Image.new('RGB', canvas_size, color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    
    font_title = get_font(48, bold=True)
    font_subtitle = get_font(28, bold=True)
    font_text = get_font(22, bold=False)
    
    # Title
    draw.text((canvas_size[0] // 2 - 250, 50), "Key Differences",
              fill=(37, 99, 235), font=font_title)  # #2563eb
    
    # Divider
    draw.line([(canvas_size[0] // 2, 150), (canvas_size[0] // 2, canvas_size[1] - 100)],
              fill=(203, 213, 225), width=4)  # slate-300
    
    # CLIP side
    clip_x = 100
    y_start = 180
    
    draw.text((clip_x + 200, y_start), "CLIP Approach",
              fill=(37, 99, 235), font=font_subtitle)  # #2563eb
    
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
        draw.text((clip_x, y), label + ":", fill=(71, 85, 105), font=font_text)  # slate-600
        
        # Value box
        draw_rounded_rectangle(draw, [clip_x, y + 35, clip_x + 750, y + 90],
                              radius=8, fill=(248, 250, 252), outline=(148, 163, 184), width=2)  # slate-50, slate-400
        draw.text((clip_x + 20, y + 47), value, fill=(30, 41, 59), font=font_text)  # slate-800
    
    # Global side
    global_x = canvas_size[0] // 2 + 100
    
    draw.text((global_x + 100, y_start), "Global Contrastive",
              fill=(29, 78, 216), font=font_subtitle)  # blue-700
    
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
        draw.text((global_x, y), label + ":", fill=(71, 85, 105), font=font_text)  # slate-600
        
        # Value box
        draw_rounded_rectangle(draw, [global_x, y + 35, global_x + 750, y + 90],
                              radius=8, fill=(239, 246, 255), outline=(96, 165, 250), width=2)  # blue-50, blue-400
        draw.text((global_x + 20, y + 47), value, fill=(30, 58, 138), font=font_text)  # blue-900
    
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
