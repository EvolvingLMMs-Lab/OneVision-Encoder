#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成动画GIF来可视化CLIP对比学习和全局对比学习方法的比较
Generate an animated GIF to visualize the comparison between CLIP's contrastive learning
and the global contrastive learning approach.

主要区别 / Key differences:
1. CLIP: 批次内的图像-文本对 (有限的负样本) / Image-Text pairs within a batch (limited negative samples)
2. Global: 无文本编码器，使用离线聚类得到的100万概念中心作为负样本 / No text encoder, 1M concept centers from offline clustering as negatives

使用方法 / Usage:
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

# 跨平台字体路径 (CVPR风格：优先使用Times/Serif字体)
# Cross-platform font paths (CVPR style: prioritize Times/Serif fonts)
FONT_PATHS = [
    # Linux - Times New Roman / Serif fonts for CVPR style
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    # Fallback to sans-serif if serif not available
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    # macOS - Times New Roman
    "/Library/Fonts/Times New Roman.ttf",
    "/Library/Fonts/Times New Roman Bold.ttf",
    "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial.ttf",
    # Windows - Times New Roman for CVPR style
    "C:/Windows/Fonts/times.ttf",
    "C:/Windows/Fonts/timesbd.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
]

# Animation configuration constants
# 动画配置常量
CLIP_ANIMATION_EXAMPLES = 3  # Number of examples to animate in CLIP (out of 8 total)
CONCEPT_CENTER_GRAY = (180, 180, 180)  # Gray color for non-sampled concept centers
CONCEPT_CENTER_GRAY_BORDER = (120, 120, 120)  # Border color for non-sampled centers
FAINT_LINE_COLOR = (200, 200, 200)  # Color for faint connection lines


def get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """
    获取跨平台支持的字体 (CVPR风格)
    Get a font with cross-platform support (CVPR style)
    
    Args:
        size: 字体大小 / Font size
        bold: 是否使用粗体 / Whether to use bold font
    
    Returns:
        ImageFont.FreeTypeFont: 字体对象 / Font object
    """
    # 首先尝试查找匹配bold参数的字体
    # First try to find fonts matching the bold parameter
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
    
    # 回退：尝试任何可用的字体
    # Fallback: try any available font
    for font_path in FONT_PATHS:
        if os.path.exists(font_path):
            try:
                return ImageFont.truetype(font_path, size)
            except OSError:
                continue
    
    # 最终回退：使用默认字体
    # Final fallback: use default font
    return ImageFont.load_default()


def draw_rounded_rectangle(
    draw: ImageDraw.Draw,
    xy: List[int],
    radius: int = 10,
    fill: Optional[Tuple[int, int, int]] = None,
    outline: Optional[Tuple[int, int, int]] = None,
    width: int = 1
) -> None:
    """
    绘制圆角矩形，支持旧版本Pillow的回退方案
    Draw a rounded rectangle with fallback for older Pillow versions
    
    Args:
        draw: ImageDraw对象 / ImageDraw object
        xy: 坐标列表 [x1, y1, x2, y2] / Coordinate list [x1, y1, x2, y2]
        radius: 圆角半径 / Corner radius
        fill: 填充颜色 / Fill color
        outline: 边框颜色 / Outline color
        width: 边框宽度 / Border width
    """
    try:
        draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)
    except AttributeError:
        # 如果Pillow版本不支持rounded_rectangle，使用普通矩形
        # If Pillow version doesn't support rounded_rectangle, use regular rectangle
        draw.rectangle(xy, fill=fill, outline=outline, width=width)


def create_title_frame(canvas_size: Tuple[int, int] = (1920, 1080)) -> Image.Image:
    """
    创建扁平化风格的标题帧 (CLIP/SAM论文风格)
    Create a flat-style title frame (CLIP/SAM paper style)
    
    Args:
        canvas_size: 画布大小 / Canvas size (width, height)
    
    Returns:
        Image.Image: 标题帧图像 / Title frame image
    """
    # 创建纯白色背景 (扁平化风格)
    # Create pure white background (flat style)
    canvas = Image.new('RGB', canvas_size, color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    
    # 使用较小的字体以避免超出框
    # Use smaller fonts to avoid overflow
    font_title = get_font(52, bold=True)
    font_subtitle = get_font(30, bold=False)
    font_text = get_font(22, bold=False)
    
    # 主标题 (无阴影，扁平化)
    # Main title (no shadow, flat)
    title = "Visual Contrastive Learning"
    bbox = draw.textbbox((0, 0), title, font=font_title)
    title_w = bbox[2] - bbox[0]
    title_x = canvas_size[0] // 2 - title_w // 2
    title_y = 160
    draw.text((title_x, title_y), title, fill=(0, 0, 0), font=font_title)
    
    # 副标题
    # Subtitle
    subtitle = "Comparing CLIP and Global Contrastive Approaches"
    bbox = draw.textbbox((0, 0), subtitle, font=font_subtitle)
    subtitle_w = bbox[2] - bbox[0]
    draw.text((canvas_size[0] // 2 - subtitle_w // 2, 230), subtitle,
              fill=(60, 60, 60), font=font_subtitle)
    
    # 简单分隔线 (扁平化)
    # Simple divider line (flat)
    div_y = 290
    draw.line([(canvas_size[0] // 2 - 400, div_y), (canvas_size[0] // 2 + 400, div_y)],
              fill=(180, 180, 180), width=2)
    
    # 对比框布局 (扁平化，无阴影)
    # Comparison boxes layout (flat, no shadows)
    y_start = 360
    box_width = 700
    box_height = 480
    gap = 120
    
    # CLIP框 (扁平化设计)
    # CLIP box (flat design)
    clip_x = canvas_size[0] // 2 - box_width - gap // 2
    
    # 使用简单的矩形边框，无阴影
    # Use simple rectangle border, no shadows
    draw_rounded_rectangle(draw, [clip_x, y_start, clip_x + box_width, y_start + box_height],
                          radius=15, fill=(245, 245, 250), outline=(100, 100, 150), width=3)
    
    # 标题
    # Title
    draw.text((clip_x + 290, y_start + 30), "CLIP", fill=(50, 50, 100), font=font_subtitle)
    
    # 特征列表
    # Feature list
    clip_features = [
        "• Image-Text paired learning",
        "• Batch-level contrastive loss",
        "• Limited negative samples",
        "  (batch size: 32-1024)",
        "• Dual encoder architecture:",
        "  › Image Encoder",
        "  › Text Encoder",
        "• Cross-modal alignment"
    ]
    
    for i, feature in enumerate(clip_features):
        draw.text((clip_x + 60, y_start + 100 + i * 42), feature,
                 fill=(40, 40, 40), font=font_text)
    
    # Global框 (扁平化设计)
    # Global box (flat design)
    global_x = canvas_size[0] // 2 + gap // 2
    
    # 使用简单的矩形边框，无阴影
    # Use simple rectangle border, no shadows
    draw_rounded_rectangle(draw, [global_x, y_start, global_x + box_width, y_start + box_height],
                          radius=15, fill=(240, 245, 255), outline=(80, 120, 180), width=3)
    
    # 标题
    # Title
    draw.text((global_x + 200, y_start + 30), "Global Contrastive",
              fill=(40, 80, 140), font=font_subtitle)
    
    # 特征列表
    # Feature list
    global_features = [
        "• Image-only representation",
        "• Global negative sampling",
        "• 1M concept centers pool",
        "  (offline clustering)",
        "• Single encoder:",
        "  › Image Encoder only",
        "• Massive negative sampling:",
        "  › 1024+ negatives per batch"
    ]
    
    for i, feature in enumerate(global_features):
        draw.text((global_x + 60, y_start + 100 + i * 42), feature,
                 fill=(40, 40, 40), font=font_text)
    
    return canvas


def create_clip_frame(
    canvas_size: Tuple[int, int] = (1920, 1080),
    animation_step: int = 0
) -> Image.Image:
    """
    创建展示CLIP对比学习的扁平化帧 (CLIP/SAM论文风格)
    Create a flat-style frame showing CLIP's contrastive learning (CLIP/SAM paper style)
    
    Args:
        canvas_size: 画布大小 / Canvas size (width, height)
        animation_step: 动画步骤 / Animation step for highlighting
    
    Returns:
        Image.Image: CLIP帧图像 / CLIP frame image
    """
    # 纯白色背景 (扁平化)
    # Pure white background (flat)
    canvas = Image.new('RGB', canvas_size, color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    
    font_title = get_font(40, bold=True)
    font_label = get_font(22, bold=False)
    font_small = get_font(18, bold=False)
    
    # 标题 (无阴影)
    # Title (no shadow)
    title_text = "CLIP: Batch-Level Image-Text Contrastive Learning"
    draw.text((50, 40), title_text, fill=(0, 0, 0), font=font_title)
    
    # 副标题
    # Subtitle
    batch_size = 8
    draw.text((50, 100), f"Batch Size: {batch_size} pairs | Max ~32K negatives in large batches",
              fill=(80, 80, 80), font=font_label)
    
    # 布局参数
    # Layout parameters
    img_encoder_x = 200
    text_encoder_x = 1620
    y_start = 200
    item_height = 95
    gap = 12
    
    # 左侧图像标签
    # Left side images label
    draw.text((img_encoder_x - 90, 155), "Images", fill=(0, 0, 0), font=font_label)
    
    # 扁平化配色方案
    # Flat color scheme
    image_colors = [
        (239, 68, 68), (34, 197, 94), (59, 130, 246), (251, 191, 36),
        (168, 85, 247), (20, 184, 166), (249, 115, 22), (236, 72, 153)
    ]
    
    # 绘制图像框 (无阴影，扁平化)
    # Draw image boxes (no shadows, flat)
    for i in range(batch_size):
        y = y_start + i * (item_height + gap)
        draw_rounded_rectangle(draw, [img_encoder_x - 80, y, img_encoder_x + 20, y + item_height],
                              radius=8, fill=image_colors[i], outline=(0, 0, 0), width=2)
        draw.text((img_encoder_x - 63, y + 32), f"I{i+1}", fill=(255, 255, 255), font=font_label)
    
    # 图像编码器 (扁平化，无阴影，改进设计)
    # Image Encoder (flat, no shadows, improved design)
    encoder_x = img_encoder_x + 180
    encoder_width = 180
    encoder_height = batch_size * (item_height + gap) - gap
    
    # 绘制渐变效果的编码器背景
    # Draw encoder with gradient-like effect
    draw_rounded_rectangle(draw, [encoder_x, y_start, encoder_x + encoder_width, y_start + encoder_height],
                          radius=12, fill=(200, 210, 240), outline=(60, 80, 140), width=4)
    
    # 添加内部装饰线条，模拟神经网络层
    # Add internal decorative lines to simulate neural network layers
    for i in range(3):
        offset = encoder_height // 4 * (i + 1)
        y_line = y_start + offset
        draw.line([(encoder_x + 20, y_line), (encoder_x + encoder_width - 20, y_line)],
                 fill=(150, 160, 200), width=2)
    
    # 编码器标签 (改进的文字位置和样式)
    # Encoder label (improved text position and style)
    draw.text((encoder_x + 40, y_start + encoder_height // 2 - 25), "Image",
              fill=(20, 40, 100), font=get_font(24, bold=True))
    draw.text((encoder_x + 30, y_start + encoder_height // 2 + 5), "Encoder",
              fill=(20, 40, 100), font=get_font(24, bold=True))
    
    # 图像嵌入 (扁平化圆圈，无发光)
    # Image embeddings (flat circles, no glow)
    emb_x = encoder_x + encoder_width + 90
    for i in range(batch_size):
        y = y_start + i * (item_height + gap)
        draw.ellipse([emb_x, y + 28, emb_x + 40, y + 68],
                    fill=image_colors[i], outline=(0, 0, 0), width=3)
    
    # 右侧文本标签
    # Right side texts label
    draw.text((text_encoder_x + 110, 155), "Texts", fill=(0, 0, 0), font=font_label)
    
    text_colors = image_colors  # 匹配的配对使用相同颜色 / Same colors for matching pairs
    
    # 绘制文本框 (无阴影，扁平化)
    # Draw text boxes (no shadows, flat)
    for i in range(batch_size):
        y = y_start + i * (item_height + gap)
        draw_rounded_rectangle(draw, [text_encoder_x + 80, y, text_encoder_x + 180, y + item_height],
                              radius=8, fill=text_colors[i], outline=(0, 0, 0), width=2)
        draw.text((text_encoder_x + 107, y + 32), f"T{i+1}", fill=(255, 255, 255), font=font_label)
    
    # 文本编码器 (扁平化，无阴影，改进设计)
    # Text Encoder (flat, no shadows, improved design)
    text_enc_x = text_encoder_x - 180
    
    # 绘制渐变效果的编码器背景
    # Draw encoder with gradient-like effect
    draw_rounded_rectangle(draw, [text_enc_x, y_start, text_enc_x + encoder_width, y_start + encoder_height],
                          radius=12, fill=(230, 210, 245), outline=(120, 60, 180), width=4)
    
    # 添加内部装饰线条，模拟神经网络层
    # Add internal decorative lines to simulate neural network layers
    for i in range(3):
        offset = encoder_height // 4 * (i + 1)
        y_line = y_start + offset
        draw.line([(text_enc_x + 20, y_line), (text_enc_x + encoder_width - 20, y_line)],
                 fill=(190, 160, 215), width=2)
    
    # 编码器标签 (改进的文字位置和样式)
    # Encoder label (improved text position and style)
    draw.text((text_enc_x + 50, y_start + encoder_height // 2 - 25), "Text",
              fill=(80, 30, 120), font=get_font(24, bold=True))
    draw.text((text_enc_x + 30, y_start + encoder_height // 2 + 5), "Encoder",
              fill=(80, 30, 120), font=get_font(24, bold=True))
    
    # 文本嵌入 (扁平化圆圈，无发光)
    # Text embeddings (flat circles, no glow)
    text_emb_x = text_enc_x - 90
    for i in range(batch_size):
        y = y_start + i * (item_height + gap)
        draw.ellipse([text_emb_x, y + 28, text_emb_x + 40, y + 68],
                    fill=text_colors[i], outline=(0, 0, 0), width=3)
    
    # 中间的相似度矩阵 (扁平化，放大并居中在两个编码器之间)
    # Contrastive matrix in the center (flat, enlarged and centered between encoders)
    matrix_size = 480
    matrix_x = canvas_size[0] // 2 - matrix_size // 2
    matrix_y = y_start + 30
    
    draw.text((matrix_x + 130, matrix_y - 45), "Similarity Matrix",
              fill=(0, 0, 0), font=get_font(26, bold=True))
    
    cell_size = matrix_size // batch_size
    
    # 动画：只高亮显示前N个配对 (由CLIP_ANIMATION_EXAMPLES定义)
    # Animation: only highlight first N pairs (defined by CLIP_ANIMATION_EXAMPLES)
    # Note: This intentionally limits highlighting to the first N pairs to simplify the visualization
    highlight_pair = (animation_step // 3) % CLIP_ANIMATION_EXAMPLES  # Only cycle through first N
    
    # 绘制从图像嵌入到矩阵列的连接线
    # Draw connection lines from image embeddings to matrix columns
    for i in range(batch_size):
        img_emb_y = y_start + i * (item_height + gap) + 48
        matrix_col_x = matrix_x + i * cell_size + cell_size // 2
        # 轻度的连接线
        if i == highlight_pair:
            draw.line([(emb_x + 40, img_emb_y), (matrix_col_x, matrix_y)],
                     fill=image_colors[i], width=3)
        elif i < CLIP_ANIMATION_EXAMPLES:  # Show faint lines for first N
            draw.line([(emb_x + 40, img_emb_y), (matrix_col_x, matrix_y)],
                     fill=FAINT_LINE_COLOR, width=1)
    
    # 绘制从文本嵌入到矩阵行的连接线
    # Draw connection lines from text embeddings to matrix rows
    for i in range(batch_size):
        text_emb_y = y_start + i * (item_height + gap) + 48
        matrix_row_y = matrix_y + i * cell_size + cell_size // 2
        # 轻度的连接线
        if i == highlight_pair:
            draw.line([(text_emb_x + 40, text_emb_y), (matrix_x + matrix_size, matrix_row_y)],
                     fill=text_colors[i], width=3)
        elif i < CLIP_ANIMATION_EXAMPLES:  # Show faint lines for first N
            draw.line([(text_emb_x + 40, text_emb_y), (matrix_x + matrix_size, matrix_row_y)],
                     fill=FAINT_LINE_COLOR, width=1)
    
    for i in range(batch_size):
        for j in range(batch_size):
            x = matrix_x + j * cell_size
            y = matrix_y + i * cell_size
            
            # 对角线是正样本对，非对角线是负样本
            # Diagonal are positive pairs, off-diagonal are negatives
            if i == j:
                # 正样本对 (绿色)
                # Positive pair (green)
                if i == highlight_pair and i < CLIP_ANIMATION_EXAMPLES:
                    color = (74, 222, 128)  # 亮绿色 / Bright green (only for first N)
                else:
                    color = (187, 247, 208)  # 浅绿色 / Light green
            else:
                # 负样本对 (红色)
                # Negative pair (red)
                # Only highlight negative pairs within the first NxN submatrix for clarity
                is_within_animation_range = (i < CLIP_ANIMATION_EXAMPLES and j < CLIP_ANIMATION_EXAMPLES)
                is_highlighted = (i == highlight_pair or j == highlight_pair)
                
                if is_highlighted and is_within_animation_range:
                    color = (252, 165, 165)  # 亮红色 / Bright red (only for first NxN)
                else:
                    color = (254, 226, 226)  # 浅红色 / Light red
            
            draw.rectangle([x + 1, y + 1, x + cell_size - 2, y + cell_size - 2],
                         fill=color, outline=(160, 160, 160), width=2)
    
    # 底部信息框 (扁平化)
    # Info box at bottom (flat)
    info_y = y_start + encoder_height + 100
    info_height = 140
    
    draw_rounded_rectangle(draw, [200, info_y, canvas_size[0] - 200, info_y + info_height],
                          radius=10, fill=(245, 245, 250), outline=(120, 120, 150), width=2)
    
    info_lines = [
        f"⦿ Positive pairs: {batch_size} (diagonal - matching image-text pairs)",
        f"⦿ Negative pairs: {batch_size * (batch_size - 1)} (off-diagonal - mismatched pairs)",
        f"⦿ Total comparisons: {batch_size * batch_size} within batch",
        "⦿ Limitation: Negative samples scale with batch size (max ~32K in largest batches)"
    ]
    
    for i, line in enumerate(info_lines):
        draw.text((240, info_y + 18 + i * 30), line, fill=(40, 40, 40), font=font_small)
    
    return canvas


def create_global_frame(
    canvas_size: Tuple[int, int] = (1920, 1080),
    animation_step: int = 0
) -> Image.Image:
    """
    创建展示全局对比学习的扁平化帧 (CLIP/SAM论文风格)
    Create a flat-style frame showing Global Contrastive Learning (CLIP/SAM paper style)
    
    Args:
        canvas_size: 画布大小 / Canvas size (width, height)
        animation_step: 动画步骤 / Animation step for highlighting
    
    Returns:
        Image.Image: 全局对比学习帧图像 / Global contrastive frame image
    """
    # 纯白色背景 (扁平化)
    # Pure white background (flat)
    canvas = Image.new('RGB', canvas_size, color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    
    font_title = get_font(40, bold=True)
    font_label = get_font(22, bold=False)
    font_small = get_font(18, bold=False)
    
    # 标题 (无阴影)
    # Title (no shadow)
    title_text = "Global Contrastive Learning: 1M Concept Centers"
    draw.text((50, 40), title_text, fill=(0, 0, 0), font=font_title)
    
    # 布局参数
    # Layout parameters
    batch_size = 8
    sampled_negatives = 1024
    total_concepts = 1000000
    num_positive_centers = 10
    
    draw.text((50, 100), f"Batch: {batch_size} images | Sampled Negatives: {sampled_negatives:,} | Total Concepts: {total_concepts:,}",
              fill=(80, 80, 80), font=font_label)
    
    # 左侧：图像和编码器
    # Left side: Images and encoder
    img_x = 150
    y_start = 200
    item_height = 75
    gap = 20
    
    draw.text((img_x - 40, 155), "Images", fill=(0, 0, 0), font=font_label)
    
    # 扁平化配色方案
    # Flat color scheme
    image_colors = [
        (239, 68, 68), (34, 197, 94), (59, 130, 246), (251, 191, 36),
        (168, 85, 247), (20, 184, 166), (249, 115, 22), (236, 72, 153)
    ]
    
    # 动画：循环遍历样本
    # Animation: cycle through samples
    current_sample = (animation_step // 6) % batch_size
    sample_phase = (animation_step % 6)
    
    # 绘制图像框 (无阴影，扁平化)
    # Draw image boxes (no shadows, flat)
    for i in range(batch_size):
        y = y_start + i * (item_height + gap)
        
        # 高亮当前处理的样本
        # Highlight current sample being processed
        if i == current_sample and sample_phase >= 2:
            outline_color = (251, 191, 36)  # 鲜明黄色 / Vibrant yellow
            outline_width = 4
        else:
            outline_color = (0, 0, 0)
            outline_width = 2
        
        draw_rounded_rectangle(draw, [img_x - 60, y, img_x + 40, y + item_height],
                              radius=8, fill=image_colors[i], outline=outline_color, width=outline_width)
        draw.text((img_x - 43, y + 22), f"I{i+1}", fill=(255, 255, 255), font=font_label)
    
    # 图像编码器 (扁平化，无阴影)
    # Image Encoder (flat, no shadows)
    encoder_x = img_x + 190
    encoder_width = 200
    encoder_height = batch_size * (item_height + gap) - gap
    
    draw_rounded_rectangle(draw, [encoder_x, y_start, encoder_x + encoder_width, y_start + encoder_height],
                          radius=10, fill=(220, 220, 240), outline=(80, 80, 120), width=3)
    
    # 编码器标签 (仅"Image Encoder"，不显示ViT-L/14)
    # Encoder label (only "Image Encoder", no ViT-L/14)
    draw.text((encoder_x + 48, y_start + encoder_height // 2 - 20), "Image",
              fill=(40, 40, 80), font=font_label)
    draw.text((encoder_x + 35, y_start + encoder_height // 2 + 10), "Encoder",
              fill=(40, 40, 80), font=font_label)
    
    # 图像嵌入 (扁平化圆圈，无发光)
    # Image embeddings (flat circles, no glow)
    emb_x = encoder_x + encoder_width + 100
    for i in range(batch_size):
        y = y_start + i * (item_height + gap)
        
        # 当前样本的脉冲效果 (扁平化版本)
        # Pulse effect for current sample (flat version)
        if i == current_sample and sample_phase >= 2:
            outline_color = (251, 191, 36)
            outline_width = 3
        else:
            outline_color = (0, 0, 0)
            outline_width = 2
        
        draw.ellipse([emb_x, y + 12, emb_x + 40, y + 52],
                    fill=image_colors[i], outline=outline_color, width=outline_width)
    
    # 概念中心库可视化 (右侧) - 扁平化风格
    # Concept Bank visualization (right side) - Flat style
    bank_x = 1000
    bank_y = 155
    bank_width = 820
    bank_height = 720
    
    # 简单的矩形边框 (无阴影，扁平化)
    # Simple rectangle border (no shadows, flat)
    draw_rounded_rectangle(draw, [bank_x, bank_y, bank_x + bank_width, bank_y + bank_height],
                          radius=15, fill=(245, 248, 252), outline=(100, 120, 150), width=3)
    
    # 标题
    # Title
    title_text = "Concept Centers Bank"
    bbox = draw.textbbox((0, 0), title_text, font=font_label)
    title_w = bbox[2] - bbox[0]
    draw.text((bank_x + (bank_width - title_w) // 2, bank_y + 25), title_text,
              fill=(0, 0, 0), font=font_label)
    
    subtitle_text = f"({total_concepts:,} centers from offline clustering)"
    bbox = draw.textbbox((0, 0), subtitle_text, font=font_small)
    subtitle_w = bbox[2] - bbox[0]
    draw.text((bank_x + (bank_width - subtitle_w) // 2, bank_y + 55), subtitle_text,
              fill=(80, 80, 80), font=font_small)
    
    # 创建稳定的随机位置用于概念中心
    # Create stable random positions for concept centers
    rng = np.random.default_rng(42)
    num_visible_concepts = 320
    concept_positions = []
    for i in range(num_visible_concepts):
        cx = bank_x + 100 + rng.integers(0, bank_width - 200)
        cy = bank_y + 130 + rng.integers(0, bank_height - 240)
        concept_positions.append((cx, cy, i))
    
    # 确定要高亮的概念（基于当前样本）
    # Determine which concepts to highlight based on current sample
    positive_centers = set()
    negative_centers = set()
    
    if sample_phase >= 3:
        # 为当前样本选择10个正样本中心 - 聚集在一起
        # Select 10 positive centers for current sample - clustered together
        sample_seed = current_sample * 1000
        pos_rng = np.random.default_rng(sample_seed)
        
        # 选择一个随机中心点作为正样本聚类中心
        # Pick a random center point for the positive cluster
        cluster_center_idx = pos_rng.integers(0, num_visible_concepts)
        cluster_cx, cluster_cy, _ = concept_positions[cluster_center_idx]
        
        # 找到距离聚类中心最近的10个中心
        # Find the 10 closest centers to the cluster center
        distances = []
        for i, (cx, cy, idx) in enumerate(concept_positions):
            dist = (cx - cluster_cx)**2 + (cy - cluster_cy)**2
            distances.append((dist, i))
        
        # 按距离排序并取最近的10个
        # Sort by distance and take the 10 closest
        distances.sort()
        positive_indices = [distances[i][1] for i in range(min(num_positive_centers, num_visible_concepts))]
        positive_centers = set(positive_indices)
        
        # 选择随机负样本中心 - 占所有可见概念的20%，分散分布
        # Select random negative centers - 20% of all visible concepts, scattered
        neg_rng = np.random.default_rng(sample_seed + 1)
        available = [i for i in range(num_visible_concepts) if i not in positive_centers]
        num_visible_negatives = int(num_visible_concepts * 0.2)
        negative_indices = neg_rng.choice(len(available), size=min(num_visible_negatives, len(available)), replace=False)
        negative_centers = set(available[i] for i in negative_indices)
    
    # 绘制概念中心 (扁平化风格，未采样的为灰色)
    # Draw concept centers (flat style, non-sampled in gray)
    modern_palette = [
        (239, 68, 68), (34, 197, 94), (59, 130, 246), (251, 191, 36),
        (168, 85, 247), (20, 184, 166), (249, 115, 22), (236, 72, 153),
        (220, 38, 38), (101, 163, 13), (29, 78, 216), (217, 119, 6)
    ]
    
    for cx, cy, i in concept_positions:
        if i in positive_centers:
            # 正样本中心 - 绿色 (扁平化，无发光)
            # Positive centers - green (flat, no glow)
            color = (74, 222, 128)
            size = 9
            draw.ellipse([cx - size, cy - size, cx + size, cy + size],
                        fill=color, outline=(0, 0, 0), width=2)
        elif i in negative_centers:
            # 负样本中心 - 红色 (扁平化，无发光)
            # Negative centers - red (flat, no glow)
            color = (252, 165, 165)
            size = 8
            draw.ellipse([cx - size, cy - size, cx + size, cy + size],
                        fill=color, outline=(0, 0, 0), width=2)
        else:
            # 未采样的中心 - 灰色 (扁平化)
            # Non-sampled centers - gray (flat)
            color = CONCEPT_CENTER_GRAY  # 灰色替代彩色
            size = 5
            draw.ellipse([cx - size, cy - size, cx + size, cy + size],
                        fill=color, outline=CONCEPT_CENTER_GRAY_BORDER, width=1)
    
    # 从当前样本到中心的连接线 (扁平化)
    # Connection lines from current sample to centers (flat)
    if sample_phase >= 4 and current_sample < batch_size:
        start_x = emb_x + 40
        start_y = y_start + current_sample * (item_height + gap) + 32
        
        # 到正样本中心的线 (绿色)
        # Lines to positive centers (green)
        for cx, cy, i in concept_positions:
            if i in positive_centers:
                draw.line([(start_x, start_y), (cx, cy)], 
                         fill=(74, 222, 128), width=2)
        
        # 到负样本中心的线 (红色)
        # Lines to negative centers (red)
        if sample_phase >= 5:
            for cx, cy, i in concept_positions[:60]:  # 限制数量以避免混乱 / Limit to avoid clutter
                if i in negative_centers:
                    draw.line([(start_x, start_y), (cx, cy)], 
                             fill=(252, 165, 165), width=1)
    
    # 图例框 (扁平化)
    # Legend box (flat)
    legend_x = bank_x + 60
    legend_y = bank_y + bank_height - 130
    legend_width = bank_width - 120
    legend_height = 110
    
    draw_rounded_rectangle(draw, [legend_x, legend_y, legend_x + legend_width, legend_y + legend_height],
                          radius=10, fill=(255, 255, 255), outline=(120, 120, 150), width=2)
    
    # 图例项
    # Legend items
    num_visible_negatives = int(num_visible_concepts * 0.2)
    legend_items = [
        ("Selected Sample", (251, 191, 36), 9),
        (f"{num_positive_centers} Positive Centers", (74, 222, 128), 9),
        (f"{num_visible_negatives} Sampled Negatives", (252, 165, 165), 8),
        ("Other Concepts", (148, 163, 184), 5)
    ]
    
    item_x = legend_x + 40
    item_width = legend_width // 2 - 20
    
    for idx, (label, color, size) in enumerate(legend_items):
        row = idx // 2
        col = idx % 2
        x = item_x + col * item_width
        y = legend_y + 20 + row * 40
        
        # 绘制指示圆圈
        # Draw indicator circle
        draw.ellipse([x, y + 5, x + size*2, y + 5 + size*2],
                    fill=color, outline=(0, 0, 0), width=1)
        
        # 绘制标签
        # Draw label
        draw.text((x + 25, y + 3), label, fill=(40, 40, 40), font=font_small)
    
    # 底部信息框 (扁平化)
    # Info box at bottom (flat)
    info_y = y_start + encoder_height + 120
    info_height = 120
    
    draw_rounded_rectangle(draw, [100, info_y, canvas_size[0] - 100, info_y + info_height],
                          radius=10, fill=(245, 245, 250), outline=(120, 120, 150), width=2)
    
    info_lines = [
        "✓ Pure visual representation learning without text encoder",
        f"✓ Each sample: {num_positive_centers} clustered positives + {sampled_negatives:,} sampled negatives",
        f"✓ Positives clustered together; negatives scattered across concept space",
        "✓ Concept bank built via offline clustering (e.g., K-means on ImageNet-21K)"
    ]
    
    for i, line in enumerate(info_lines):
        draw.text((140, info_y + 16 + i * 28), line, fill=(40, 40, 40), font=font_small)
    
    return canvas





def generate_animation(
    output_path: str,
    fps: int = 2,
    canvas_size: Tuple[int, int] = (1920, 1080),
    as_video: bool = False
) -> None:
    """
    生成对比动画 (扁平化CLIP/SAM论文风格)
    Generate the comparison animation (flat CLIP/SAM paper style)
    
    Args:
        output_path: 输出文件路径 / Output file path
        fps: 每秒帧数 / Frames per second
        canvas_size: 画布大小 / Canvas size (width, height)
        as_video: 是否输出为视频格式 / Whether to output as video format
    """
    frames: List[np.ndarray] = []
    
    print("生成帧... / Generating frames...")
    
    # 1. 标题帧 (3秒)
    # 1. Title frame (3 seconds)
    print("  - 标题帧 / Title frame")
    title_frame = create_title_frame(canvas_size)
    for _ in range(fps * 3):
        frames.append(np.array(title_frame))
    
    # 2. CLIP帧与动画 (8秒)
    # 2. CLIP frames with animation (8 seconds)
    print("  - CLIP动画帧 / CLIP animation frames")
    clip_frames = fps * 8
    for i in range(clip_frames):
        frame = create_clip_frame(canvas_size, i)
        frames.append(np.array(frame))
    
    # 3. 全局对比学习帧与增强采样动画 (12秒 - 更长以展示采样过程)
    # 3. Global frames with enhanced sampling animation (12 seconds - longer to show sampling)
    print("  - 全局对比学习动画帧 / Global contrastive animation frames")
    global_frames = fps * 12
    for i in range(global_frames):
        frame = create_global_frame(canvas_size, i)
        frames.append(np.array(frame))
    
    # 注意：按要求移除了对比帧
    # Note: Comparison frame removed as per requirements
    
    # 保存输出
    # Save output
    if as_video:
        # 确保输出路径有.mp4扩展名
        # Ensure output path has .mp4 extension
        if not output_path.lower().endswith('.mp4'):
            output_path = output_path.rsplit('.', 1)[0] + '.mp4'
        
        print(f"\n保存视频到 / Saving video to: {output_path}")
        imageio.mimwrite(
            output_path,
            frames,
            fps=fps,
            codec='libx264',
            pixelformat='yuv420p'
        )
    else:
        # 确保输出路径有.gif扩展名
        # Ensure output path has .gif extension
        if not output_path.lower().endswith('.gif'):
            output_path = output_path.rsplit('.', 1)[0] + '.gif'
        
        print(f"\n保存GIF到 / Saving GIF to: {output_path}")
        pil_frames = [Image.fromarray(f) for f in frames]
        pil_frames[0].save(
            output_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=int(1000 / fps),
            loop=0
        )
    
    # 输出统计信息
    # Output statistics
    print(f"✓ 动画保存成功! / Animation saved successfully!")
    print(f"  - 总帧数 / Total frames: {len(frames)}")
    print(f"  - 时长 / Duration: {len(frames) / fps:.1f} seconds")
    print(f"  - 分辨率 / Resolution: {canvas_size[0]}x{canvas_size[1]}")
    print(f"  - 扁平化CLIP/SAM论文风格 / Flat CLIP/SAM paper style")
    print(f"  - 已移除关键差异帧 / Removed: Key differences frame")


def main():
    """
    主函数：解析命令行参数并生成动画
    Main function: parse command line arguments and generate animation
    """
    parser = argparse.ArgumentParser(
        description="生成CLIP与全局对比学习的动画对比 / Generate animated comparison of CLIP vs Global Contrastive Learning"
    )
    parser.add_argument("--output", type=str, default="global_contrastive_comparison.gif",
                       help="输出文件路径 / Output file path (default: global_contrastive_comparison.gif)")
    parser.add_argument("--video", action="store_true",
                       help="输出为MP4视频而非GIF / Output as MP4 video instead of GIF")
    parser.add_argument("--fps", type=int, default=2,
                       help="每秒帧数 / Frames per second (default: 2)")
    parser.add_argument("--width", type=int, default=1920,
                       help="画布宽度 / Canvas width (default: 1920)")
    parser.add_argument("--height", type=int, default=1080,
                       help="画布高度 / Canvas height (default: 1080)")
    
    args = parser.parse_args()
    
    # 生成动画
    # Generate animation
    generate_animation(
        output_path=args.output,
        fps=args.fps,
        canvas_size=(args.width, args.height),
        as_video=args.video
    )


if __name__ == "__main__":
    main()
