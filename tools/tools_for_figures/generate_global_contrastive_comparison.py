#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成动画GIF来可视化CLIP对比学习和全局对比学习方法的比较
Generate animated GIFs to visualize the comparison between CLIP's contrastive learning
and the global contrastive learning approach.

主要区别 / Key differences:
1. CLIP: 批次内的图像-文本对 (有限的负样本) / Image-Text pairs within a batch (limited negative samples)
2. Global: 无文本编码器，使用离线聚类得到的100万概念中心作为负样本 / No text encoder, 1M concept centers from offline clustering as negatives

使用方法 / Usage:
    # 生成两个独立的GIF文件 / Generate two separate GIF files
    python generate_global_contrastive_comparison.py --output comparison --separate
    
    # 仅生成CLIP动画 / Generate CLIP animation only
    python generate_global_contrastive_comparison.py --output clip --clip-only
    
    # 仅生成全局对比学习动画 / Generate Global animation only
    python generate_global_contrastive_comparison.py --output global --global-only
    
    # 生成包含两者的组合动画（向后兼容） / Generate combined animation (backward compatible)
    python generate_global_contrastive_comparison.py --output comparison.gif
    
    # 生成视频格式 / Generate as video format
    python generate_global_contrastive_comparison.py --output comparison --separate --video

更新内容 / Updates:
- 移除了标题帧 / Removed title frame
- 支持生成独立的CLIP和Global动画 / Support separate CLIP and Global animations
- 所有超参数集中在文件顶部，便于修改 / All hyperparameters centralized at top for easy modification
- 优化了视觉效果和配色 / Optimized visual effects and color scheme
"""

import argparse
import os
from pathlib import Path
from typing import Optional, Tuple, List

import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ============================================================================
# HYPERPARAMETERS CONFIGURATION / 超参数配置
# ============================================================================
# These parameters can be easily modified to adjust the animation appearance
# 这些参数可以轻松修改以调整动画外观

# Canvas and Animation Settings / 画布和动画设置
# ----------------------------------------------------------------------------
DEFAULT_CANVAS_WIDTH = 1920  # Canvas width in pixels / 画布宽度（像素）
DEFAULT_CANVAS_HEIGHT = 1080  # Canvas height in pixels / 画布高度（像素）
DEFAULT_FPS = 2  # Frames per second for animation / 动画帧率（每秒帧数）

# Animation Timing / 动画时长
# ----------------------------------------------------------------------------
CLIP_ANIMATION_DURATION = 8  # Duration in seconds for CLIP animation / CLIP动画时长（秒）
GLOBAL_ANIMATION_DURATION = 12  # Duration in seconds for Global animation / Global动画时长（秒）

# CLIP Animation Settings / CLIP动画设置
# ----------------------------------------------------------------------------
CLIP_BATCH_SIZE = 8  # Number of image-text pairs in CLIP batch / CLIP批次中的图像-文本对数量
CLIP_ANIMATION_EXAMPLES = 3  # Number of examples to animate in CLIP / CLIP中动画展示的样本数量

# Global Contrastive Settings / 全局对比学习设置
# ----------------------------------------------------------------------------
GLOBAL_BATCH_SIZE = 8  # Number of images in Global batch / Global批次中的图像数量
GLOBAL_SAMPLED_NEGATIVES = 1024  # Number of sampled negative centers / 采样的负样本中心数量
GLOBAL_TOTAL_CONCEPTS = 1000000  # Total concept centers in bank / 概念库中总的概念中心数量
GLOBAL_NUM_POSITIVE_CENTERS = 10  # Number of positive centers per sample / 每个样本的正样本中心数量
GLOBAL_NUM_VISIBLE_CONCEPTS = 320  # Number of visible concept centers / 可见的概念中心数量
GLOBAL_VISIBLE_NEGATIVES_RATIO = 0.2  # Ratio of visible negatives to display / 可见负样本显示比例

# Layout and Sizing / 布局和尺寸
# ----------------------------------------------------------------------------
# CLIP layout
CLIP_ITEM_HEIGHT = 95  # Height of each image/text item in CLIP / CLIP中每个图像/文本项的高度
CLIP_ITEM_GAP = 12  # Gap between items in CLIP / CLIP中项之间的间隙
CLIP_ENCODER_WIDTH = 180  # Width of encoder boxes / 编码器框的宽度

# Global layout
GLOBAL_ITEM_HEIGHT = 75  # Height of each image item in Global / Global中每个图像项的高度
GLOBAL_ITEM_GAP = 20  # Gap between items in Global / Global中项之间的间隙
GLOBAL_ENCODER_WIDTH = 200  # Width of encoder box / 编码器框的宽度
GLOBAL_BANK_WIDTH = 820  # Width of concept bank visualization / 概念库可视化的宽度
GLOBAL_BANK_HEIGHT = 720  # Height of concept bank visualization / 概念库可视化的高度

# Color Palette - SAM-style / 配色方案 - SAM风格
# ----------------------------------------------------------------------------
# SAM-style color palette for images - more sophisticated and harmonious
# SAM风格配色方案 - 更精致和谐
IMAGE_COLORS_SAM = [
    (255, 107, 107),  # Coral red - softer, more professional / 珊瑚红 - 柔和专业
    (78, 205, 196),   # Turquoise - calming / 青绿色 - 平静
    (99, 110, 250),   # Indigo blue - modern / 靛蓝 - 现代
    (255, 195, 113),  # Peach - warm / 桃色 - 温暖
    (162, 155, 254),  # Light purple - elegant / 浅紫 - 优雅
    (69, 183, 209),   # Sky blue - fresh / 天蓝 - 清新
    (255, 159, 64),   # Orange - energetic / 橙色 - 活力
    (255, 99, 164),   # Pink - vibrant / 粉色 - 鲜活
]

# Matrix colors - SAM-style with better contrast / 矩阵颜色 - SAM风格，更好的对比度
POSITIVE_COLOR_BRIGHT = (52, 211, 153)  # Emerald green - professional / 祖母绿 - 专业
POSITIVE_COLOR_LIGHT = (209, 250, 229)  # Very light green / 极浅绿
NEGATIVE_COLOR_BRIGHT = (248, 113, 113)  # Modern red / 现代红
NEGATIVE_COLOR_LIGHT = (254, 242, 242)  # Very light red/pink / 极浅红/粉

# Concept center colors / 概念中心颜色
CONCEPT_CENTER_GRAY = (220, 220, 225)  # Lighter gray for non-sampled concept centers / 浅灰色（未采样）
CONCEPT_CENTER_GRAY_BORDER = (180, 180, 190)  # Lighter border color for non-sampled centers / 浅灰边框
FAINT_LINE_COLOR = (210, 210, 215)  # Lighter color for faint connection lines / 浅色连接线

# Typography / 字体设置
# ----------------------------------------------------------------------------
FONT_SIZE_TITLE = 40  # Main title font size / 主标题字体大小
FONT_SIZE_LABEL = 22  # Label font size / 标签字体大小
FONT_SIZE_SMALL = 18  # Small text font size / 小字体大小

# Visual Enhancement / 视觉增强
# ----------------------------------------------------------------------------
BORDER_RADIUS = 12  # Radius for rounded corners / 圆角半径
HIGHLIGHT_BORDER_WIDTH = 3  # Width of highlight borders / 高亮边框宽度
STANDARD_BORDER_WIDTH = 2  # Width of standard borders / 标准边框宽度
SHADOW_OFFSET = 2  # Offset for subtle shadow effects / 微妙阴影效果的偏移量

# Background colors / 背景颜色
BACKGROUND_WHITE = (255, 255, 255)  # Pure white background / 纯白色背景
ENCODER_BG_IMAGE = (235, 240, 250)  # Light blue for image encoder / 图像编码器的浅蓝色
ENCODER_BG_TEXT = (248, 240, 252)  # Light purple for text encoder / 文本编码器的浅紫色
INFO_BOX_BG = (245, 245, 250)  # Light gray for info boxes / 信息框的浅灰色
CONCEPT_BANK_BG = (245, 248, 252)  # Very light blue for concept bank / 概念库的极浅蓝色

# ============================================================================
# FONT PATHS / 字体路径
# ============================================================================
# Cross-platform font paths (CVPR style: prioritize Times/Serif fonts)
# 跨平台字体路径 (CVPR风格：优先使用Times/Serif字体)
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
    
    font_title = get_font(FONT_SIZE_TITLE, bold=True)
    font_label = get_font(FONT_SIZE_LABEL, bold=False)
    font_small = get_font(FONT_SIZE_SMALL, bold=False)
    
    # 标题 (无阴影)
    # Title (no shadow)
    title_text = "CLIP: Batch-Level Image-Text Contrastive Learning"
    draw.text((50, 40), title_text, fill=(0, 0, 0), font=font_title)
    
    # 副标题
    # Subtitle
    batch_size = CLIP_BATCH_SIZE
    draw.text((50, 100), f"Batch Size: {batch_size} pairs | Max ~32K negatives in large batches",
              fill=(80, 80, 80), font=font_label)
    
    # 布局参数
    # Layout parameters
    img_encoder_x = 200
    text_encoder_x = 1620
    y_start = 200
    item_height = CLIP_ITEM_HEIGHT
    gap = CLIP_ITEM_GAP
    
    # 左侧图像标签
    # Left side images label
    draw.text((img_encoder_x - 90, 155), "Images", fill=(0, 0, 0), font=font_label)
    
    # SAM风格配色方案 - 更专业和谐
    # SAM-style color scheme - more professional and harmonious
    image_colors = IMAGE_COLORS_SAM
    
    # 绘制图像框 (无阴影，扁平化)
    # Draw image boxes (no shadows, flat)
    for i in range(batch_size):
        y = y_start + i * (item_height + gap)
        draw_rounded_rectangle(draw, [img_encoder_x - 80, y, img_encoder_x + 20, y + item_height],
                              radius=8, fill=image_colors[i], outline=(0, 0, 0), width=2)
        draw.text((img_encoder_x - 63, y + 32), f"I{i+1}", fill=(255, 255, 255), font=font_label)
    
    # 图像编码器 (SAM风格，优雅设计)
    # Image Encoder (SAM-style, elegant design)
    encoder_x = img_encoder_x + 180
    encoder_width = CLIP_ENCODER_WIDTH
    encoder_height = batch_size * (item_height + gap) - gap
    
    # 使用更优雅的渐变效果背景
    # Use more elegant gradient-like background
    draw_rounded_rectangle(draw, [encoder_x, y_start, encoder_x + encoder_width, y_start + encoder_height],
                          radius=BORDER_RADIUS, fill=(235, 240, 250), outline=(100, 120, 200), width=HIGHLIGHT_BORDER_WIDTH)
    
    # 添加精致的内部装饰线条，模拟神经网络层
    # Add refined internal decorative lines to simulate neural network layers
    for i in range(4):
        offset = encoder_height // 5 * (i + 1)
        y_line = y_start + offset
        draw.line([(encoder_x + 30, y_line), (encoder_x + encoder_width - 30, y_line)],
                 fill=(180, 190, 220), width=2)
    
    # 编码器标签 (优化的文字位置和样式)
    # Encoder label (optimized text position and style)
    draw.text((encoder_x + 40, y_start + encoder_height // 2 - 25), "Image",
              fill=(40, 60, 140), font=get_font(24, bold=True))
    draw.text((encoder_x + 30, y_start + encoder_height // 2 + 5), "Encoder",
              fill=(40, 60, 140), font=get_font(24, bold=True))
    
    # 图像嵌入 (SAM风格圆圈，优雅)
    # Image embeddings (SAM-style circles, elegant)
    emb_x = encoder_x + encoder_width + 90
    for i in range(batch_size):
        y = y_start + i * (item_height + gap)
        # 添加微妙的外圈以增强视觉层次
        # Add subtle outer ring for visual hierarchy
        draw.ellipse([emb_x - 2, y + 26, emb_x + 42, y + 70],
                    fill=(240, 240, 245), outline=None)
        draw.ellipse([emb_x, y + 28, emb_x + 40, y + 68],
                    fill=image_colors[i], outline=(255, 255, 255), width=2)
    
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
    
    # 文本编码器 (SAM风格，优雅设计)
    # Text Encoder (SAM-style, elegant design)
    text_enc_x = text_encoder_x - 180
    
    # 使用更优雅的渐变效果背景
    # Use more elegant gradient-like background
    draw_rounded_rectangle(draw, [text_enc_x, y_start, text_enc_x + encoder_width, y_start + encoder_height],
                          radius=BORDER_RADIUS, fill=(248, 240, 252), outline=(160, 100, 200), width=HIGHLIGHT_BORDER_WIDTH)
    
    # 添加精致的内部装饰线条，模拟神经网络层
    # Add refined internal decorative lines to simulate neural network layers
    for i in range(4):
        offset = encoder_height // 5 * (i + 1)
        y_line = y_start + offset
        draw.line([(text_enc_x + 30, y_line), (text_enc_x + encoder_width - 30, y_line)],
                 fill=(210, 180, 225), width=2)
    
    # 编码器标签 (优化的文字位置和样式)
    # Encoder label (optimized text position and style)
    draw.text((text_enc_x + 50, y_start + encoder_height // 2 - 25), "Text",
              fill=(100, 50, 140), font=get_font(24, bold=True))
    draw.text((text_enc_x + 30, y_start + encoder_height // 2 + 5), "Encoder",
              fill=(100, 50, 140), font=get_font(24, bold=True))
    
    # 文本嵌入 (SAM风格圆圈，优雅)
    # Text embeddings (SAM-style circles, elegant)
    text_emb_x = text_enc_x - 90
    for i in range(batch_size):
        y = y_start + i * (item_height + gap)
        # 添加微妙的外圈以增强视觉层次
        # Add subtle outer ring for visual hierarchy
        draw.ellipse([text_emb_x - 2, y + 26, text_emb_x + 42, y + 70],
                    fill=(240, 240, 245), outline=None)
        draw.ellipse([text_emb_x, y + 28, text_emb_x + 40, y + 68],
                    fill=text_colors[i], outline=(255, 255, 255), width=2)
    
    # 中间的相似度矩阵 (扁平化，放大并居中在两个编码器之间)
    # Contrastive matrix in the center (flat, enlarged and centered between encoders)
    matrix_size = 480
    # Better centering: position matrix exactly between left and right embeddings
    left_embedding_center = emb_x + 20  # Center of left embeddings
    right_embedding_center = text_emb_x + 20  # Center of right embeddings (will be defined later)
    # For now, use canvas center with slight adjustment for visual balance
    matrix_x = canvas_size[0] // 2 - matrix_size // 2
    # Vertically center the matrix with the encoder height
    matrix_y = y_start + (encoder_height - matrix_size) // 2
    
    # Draw title with proper centering
    title_text = "Similarity Matrix"
    bbox = draw.textbbox((0, 0), title_text, font=get_font(26, bold=True))
    title_width = bbox[2] - bbox[0]
    title_x = matrix_x + (matrix_size - title_width) // 2
    draw.text((title_x, matrix_y - 45), title_text,
              fill=(0, 0, 0), font=get_font(26, bold=True))
    
    cell_size = matrix_size // batch_size
    
    # 动画：只高亮显示前N个配对 (由CLIP_ANIMATION_EXAMPLES定义)
    # Animation: only highlight first N pairs (defined by CLIP_ANIMATION_EXAMPLES)
    # Note: This intentionally limits ANIMATION highlighting to first N pairs for clarity,
    # but ALL samples are calculated and shown in the matrix
    highlight_pair = (animation_step // 3) % CLIP_ANIMATION_EXAMPLES  # Only cycle through first N
    
    # 绘制从图像嵌入到矩阵列的连接线 (所有样本都显示，突出动画样本)
    # Draw connection lines from image embeddings to matrix columns (all samples shown, animated ones highlighted)
    for i in range(batch_size):
        img_emb_y = y_start + i * (item_height + gap) + 48
        matrix_col_x = matrix_x + i * cell_size + cell_size // 2
        # 所有样本都画连接线，动画样本用彩色，其他用浅灰色
        # All samples get lines: animated ones in color, others in light gray
        if i == highlight_pair:
            # Draw with slight curve for elegance / 使用轻微的曲线增加优雅感
            draw.line([(emb_x + 40, img_emb_y), (matrix_col_x, matrix_y)],
                     fill=image_colors[i], width=3)
        else:
            # Show all connection lines to indicate all samples are calculated
            draw.line([(emb_x + 40, img_emb_y), (matrix_col_x, matrix_y)],
                     fill=FAINT_LINE_COLOR, width=1)
    
    # 绘制从文本嵌入到矩阵行的连接线 (所有样本都显示，突出动画样本)
    # Draw connection lines from text embeddings to matrix rows (all samples shown, animated ones highlighted)
    for i in range(batch_size):
        text_emb_y = y_start + i * (item_height + gap) + 48
        matrix_row_y = matrix_y + i * cell_size + cell_size // 2
        # 所有样本都画连接线，动画样本用彩色，其他用浅灰色
        # All samples get lines: animated ones in color, others in light gray
        if i == highlight_pair:
            # Draw with slight curve for elegance / 使用轻微的曲线增加优雅感
            draw.line([(text_emb_x + 40, text_emb_y), (matrix_x + matrix_size, matrix_row_y)],
                     fill=text_colors[i], width=3)
        else:
            # Show all connection lines to indicate all samples are calculated
            draw.line([(text_emb_x + 40, text_emb_y), (matrix_x + matrix_size, matrix_row_y)],
                     fill=FAINT_LINE_COLOR, width=1)
    
    for i in range(batch_size):
        for j in range(batch_size):
            x = matrix_x + j * cell_size
            y = matrix_y + i * cell_size
            
            # 对角线是正样本对，非对角线是负样本 - 所有样本都要计算
            # Diagonal are positive pairs, off-diagonal are negatives - ALL are calculated
            
            # Calculate if this cell is highlighted (used for both color and border)
            is_highlighted = (i == highlight_pair or j == highlight_pair)
            
            if i == j:
                # 正样本对 (绿色) - 只有动画样本用亮绿色，使用SAM风格颜色
                # Positive pair (green) - only animated samples use bright green, SAM-style colors
                if i == highlight_pair and i < CLIP_ANIMATION_EXAMPLES:
                    color = POSITIVE_COLOR_BRIGHT  # SAM风格亮绿色
                else:
                    color = POSITIVE_COLOR_LIGHT  # SAM风格浅绿色
            else:
                # 负样本对 (红色) - 所有负样本都显示，但只有动画相关的用亮色，使用SAM风格颜色
                # Negative pair (red) - ALL negatives shown, SAM-style colors
                if is_highlighted and highlight_pair < CLIP_ANIMATION_EXAMPLES:
                    color = NEGATIVE_COLOR_BRIGHT  # SAM风格亮红色
                else:
                    color = NEGATIVE_COLOR_LIGHT  # SAM风格浅红色
            
            # SAM风格的边框样式 - 更精致
            # SAM-style border styling - more refined
            border_color = (200, 200, 205)  # 柔和的灰色边框
            border_width = 1
            
            # 高亮单元格使用更明显的边框
            # Highlighted cells use more prominent borders
            if i == j and i == highlight_pair and i < CLIP_ANIMATION_EXAMPLES:
                border_color = (16, 185, 129)  # 绿色边框 - 更深的emerald
                border_width = 3
            elif is_highlighted and highlight_pair < CLIP_ANIMATION_EXAMPLES and i != j:
                border_color = (239, 68, 68)  # 红色边框 - 更鲜艳
                border_width = 2
            
            # 绘制带圆角的矩形以获得更优雅的外观
            # Draw with rounded corners for more elegant look
            draw.rectangle([x + 2, y + 2, x + cell_size - 3, y + cell_size - 3],
                         fill=color, outline=border_color, width=border_width)
    
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
    
    font_title = get_font(FONT_SIZE_TITLE, bold=True)
    font_label = get_font(FONT_SIZE_LABEL, bold=False)
    font_small = get_font(FONT_SIZE_SMALL, bold=False)
    
    # 标题 (无阴影)
    # Title (no shadow)
    title_text = "Global Contrastive Learning: 1M Concept Centers"
    draw.text((50, 40), title_text, fill=(0, 0, 0), font=font_title)
    
    # 布局参数
    # Layout parameters
    batch_size = GLOBAL_BATCH_SIZE
    sampled_negatives = GLOBAL_SAMPLED_NEGATIVES
    total_concepts = GLOBAL_TOTAL_CONCEPTS
    num_positive_centers = GLOBAL_NUM_POSITIVE_CENTERS
    
    draw.text((50, 100), f"Batch: {batch_size} images | Sampled Negatives: {sampled_negatives:,} | Total Concepts: {total_concepts:,}",
              fill=(80, 80, 80), font=font_label)
    
    # 左侧：图像和编码器
    # Left side: Images and encoder
    img_x = 150
    y_start = 200
    item_height = GLOBAL_ITEM_HEIGHT
    gap = GLOBAL_ITEM_GAP
    
    draw.text((img_x - 40, 155), "Images", fill=(0, 0, 0), font=font_label)
    
    # SAM风格配色方案
    # SAM-style color scheme
    image_colors = IMAGE_COLORS_SAM
    
    # 动画：循环遍历样本
    # Animation: cycle through samples
    current_sample = (animation_step // 6) % batch_size
    sample_phase = (animation_step % 6)
    
    # 绘制图像框 (无阴影，扁平化)
    # Draw image boxes (no shadows, flat)
    for i in range(batch_size):
        y = y_start + i * (item_height + gap)
        
        # 高亮当前处理的样本 - SAM风格
        # Highlight current sample being processed - SAM-style
        if i == current_sample and sample_phase >= 2:
            outline_color = (255, 195, 113)  # SAM风格温暖的橙色高亮
            outline_width = 3
        else:
            outline_color = (100, 100, 110)  # 柔和的灰色边框
            outline_width = 2
        
        draw_rounded_rectangle(draw, [img_x - 60, y, img_x + 40, y + item_height],
                              radius=8, fill=image_colors[i], outline=outline_color, width=outline_width)
        draw.text((img_x - 43, y + 22), f"I{i+1}", fill=(255, 255, 255), font=font_label)
    
    # 图像编码器 (SAM风格)
    # Image Encoder (SAM-style)
    encoder_x = img_x + 190
    encoder_width = GLOBAL_ENCODER_WIDTH
    encoder_height = batch_size * (item_height + gap) - gap
    
    draw_rounded_rectangle(draw, [encoder_x, y_start, encoder_x + encoder_width, y_start + encoder_height],
                          radius=BORDER_RADIUS, fill=(235, 240, 250), outline=(100, 120, 200), width=HIGHLIGHT_BORDER_WIDTH)
    
    # 添加精致的内部装饰线条
    # Add refined internal decorative lines
    for i in range(4):
        offset = encoder_height // 5 * (i + 1)
        y_line = y_start + offset
        draw.line([(encoder_x + 30, y_line), (encoder_x + encoder_width - 30, y_line)],
                 fill=(180, 190, 220), width=2)
    
    # 编码器标签
    # Encoder label
    draw.text((encoder_x + 48, y_start + encoder_height // 2 - 20), "Image",
              fill=(40, 60, 140), font=font_label)
    draw.text((encoder_x + 35, y_start + encoder_height // 2 + 10), "Encoder",
              fill=(40, 60, 140), font=font_label)
    
    # 图像嵌入 (SAM风格圆圈)
    # Image embeddings (SAM-style circles)
    emb_x = encoder_x + encoder_width + 100
    for i in range(batch_size):
        y = y_start + i * (item_height + gap)
        
        # 当前样本的优雅高亮效果
        # Elegant highlight effect for current sample
        if i == current_sample and sample_phase >= 2:
            outline_color = (255, 195, 113)
            outline_width = 3
            # 添加外圈
            draw.ellipse([emb_x - 3, y + 9, emb_x + 43, y + 55],
                        fill=(255, 220, 160), outline=None)
        else:
            outline_color = (255, 255, 255)
            outline_width = 2
        
        draw.ellipse([emb_x, y + 12, emb_x + 40, y + 52],
                    fill=image_colors[i], outline=outline_color, width=outline_width)
    
    # 概念中心库可视化 (右侧) - 扁平化风格
    # Concept Bank visualization (right side) - Flat style
    bank_x = 1000
    bank_y = 155
    bank_width = GLOBAL_BANK_WIDTH
    bank_height = GLOBAL_BANK_HEIGHT
    
    # 简单的矩形边框 (无阴影，扁平化)
    # Simple rectangle border (no shadows, flat)
    draw_rounded_rectangle(draw, [bank_x, bank_y, bank_x + bank_width, bank_y + bank_height],
                          radius=15, fill=(245, 248, 252), outline=(100, 120, 150), width=HIGHLIGHT_BORDER_WIDTH)
    
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
    num_visible_concepts = GLOBAL_NUM_VISIBLE_CONCEPTS
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
        num_visible_negatives = int(num_visible_concepts * GLOBAL_VISIBLE_NEGATIVES_RATIO)
        negative_indices = neg_rng.choice(len(available), size=min(num_visible_negatives, len(available)), replace=False)
        negative_centers = set(available[i] for i in negative_indices)
    
    # 绘制概念中心 (SAM风格，未采样的为灰色)
    # Draw concept centers (SAM-style, non-sampled in gray)
    
    for cx, cy, i in concept_positions:
        if i in positive_centers:
            # 正样本中心 - SAM风格绿色，增强视觉效果
            # Positive centers - SAM-style green with enhanced visual effect
            color = POSITIVE_COLOR_BRIGHT
            size = 8
            # 添加多层外圈增强视觉层次感
            # Add multiple outer rings for enhanced visual hierarchy
            draw.ellipse([cx - size - 3, cy - size - 3, cx + size + 3, cy + size + 3],
                        fill=(200, 245, 220), outline=None)
            draw.ellipse([cx - size - 1, cy - size - 1, cx + size + 1, cy + size + 1],
                        fill=(160, 240, 200), outline=None)
            draw.ellipse([cx - size, cy - size, cx + size, cy + size],
                        fill=color, outline=(255, 255, 255), width=2)
        elif i in negative_centers:
            # 负样本中心 - SAM风格红色，轻微外圈
            # Negative centers - SAM-style red with subtle outer ring
            color = NEGATIVE_COLOR_BRIGHT
            size = 7
            # 添加轻微的外圈
            # Add subtle outer ring
            draw.ellipse([cx - size - 1, cy - size - 1, cx + size + 1, cy + size + 1],
                        fill=(255, 180, 180), outline=None)
            draw.ellipse([cx - size, cy - size, cx + size, cy + size],
                        fill=color, outline=(255, 255, 255), width=1)
        else:
            # 未采样的中心 - SAM风格灰色
            # Non-sampled centers - SAM-style gray
            color = CONCEPT_CENTER_GRAY
            size = 5
            draw.ellipse([cx - size, cy - size, cx + size, cy + size],
                        fill=color, outline=CONCEPT_CENTER_GRAY_BORDER, width=1)
    
    # 从当前样本到中心的连接线 (SAM风格)
    # Connection lines from current sample to centers (SAM-style)
    if sample_phase >= 4 and current_sample < batch_size:
        start_x = emb_x + 40
        start_y = y_start + current_sample * (item_height + gap) + 32
        
        # 到正样本中心的线 (SAM风格绿色)
        # Lines to positive centers (SAM-style green)
        for cx, cy, i in concept_positions:
            if i in positive_centers:
                draw.line([(start_x, start_y), (cx, cy)], 
                         fill=POSITIVE_COLOR_BRIGHT, width=2)
        
        # 到负样本中心的线 (SAM风格红色)
        # Lines to negative centers (SAM-style red)
        if sample_phase >= 5:
            for cx, cy, i in concept_positions[:60]:  # 限制数量以避免混乱 / Limit to avoid clutter
                if i in negative_centers:
                    draw.line([(start_x, start_y), (cx, cy)], 
                             fill=NEGATIVE_COLOR_BRIGHT, width=1)
    
    # 图例框 (扁平化)
    # Legend box (flat)
    legend_x = bank_x + 60
    legend_y = bank_y + bank_height - 130
    legend_width = bank_width - 120
    legend_height = 110
    
    draw_rounded_rectangle(draw, [legend_x, legend_y, legend_x + legend_width, legend_y + legend_height],
                          radius=10, fill=(255, 255, 255), outline=(120, 120, 150), width=2)
    
    # 图例项 - SAM风格颜色
    # Legend items - SAM-style colors
    num_visible_negatives = int(num_visible_concepts * GLOBAL_VISIBLE_NEGATIVES_RATIO)
    legend_items = [
        ("Selected Sample", (255, 195, 113), 9),  # SAM风格橙色
        (f"{num_positive_centers} Positive Centers", POSITIVE_COLOR_BRIGHT, 8),  # SAM风格绿色
        (f"{num_visible_negatives} Sampled Negatives", NEGATIVE_COLOR_BRIGHT, 7),  # SAM风格红色
        ("Other Concepts", CONCEPT_CENTER_GRAY, 5)  # SAM风格灰色
    ]
    
    item_x = legend_x + 40
    item_width = legend_width // 2 - 20
    
    for idx, (label, color, size) in enumerate(legend_items):
        row = idx // 2
        col = idx % 2
        x = item_x + col * item_width
        y = legend_y + 20 + row * 40
        
        # 绘制指示圆圈 - SAM风格
        # Draw indicator circle - SAM-style
        draw.ellipse([x, y + 5, x + size*2, y + 5 + size*2],
                    fill=color, outline=(255, 255, 255), width=1)
        
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





def generate_clip_animation(
    output_path: str,
    fps: int = 2,
    canvas_size: Tuple[int, int] = (1920, 1080),
    as_video: bool = False
) -> None:
    """
    生成CLIP动画 (扁平化CLIP/SAM论文风格)
    Generate CLIP animation (flat CLIP/SAM paper style)
    
    Args:
        output_path: 输出文件路径 / Output file path
        fps: 每秒帧数 / Frames per second
        canvas_size: 画布大小 / Canvas size (width, height)
        as_video: 是否输出为视频格式 / Whether to output as video format
    """
    frames: List[np.ndarray] = []
    
    print("生成CLIP动画帧... / Generating CLIP animation frames...")
    
    # CLIP帧与动画
    # CLIP frames with animation
    clip_frames = fps * CLIP_ANIMATION_DURATION
    for i in range(clip_frames):
        frame = create_clip_frame(canvas_size, i)
        frames.append(np.array(frame))
    
    # 保存输出
    # Save output
    if as_video:
        # 确保输出路径有.mp4扩展名
        # Ensure output path has .mp4 extension
        if not output_path.lower().endswith('.mp4'):
            output_path = output_path.rsplit('.', 1)[0] + '.mp4'
        
        print(f"\n保存CLIP视频到 / Saving CLIP video to: {output_path}")
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
        
        print(f"\n保存CLIP GIF到 / Saving CLIP GIF to: {output_path}")
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
    print(f"✓ CLIP动画保存成功! / CLIP animation saved successfully!")
    print(f"  - 总帧数 / Total frames: {len(frames)}")
    print(f"  - 时长 / Duration: {len(frames) / fps:.1f} seconds")
    print(f"  - 分辨率 / Resolution: {canvas_size[0]}x{canvas_size[1]}")
    print(f"  - 扁平化CLIP/SAM论文风格 / Flat CLIP/SAM paper style")


def generate_global_animation(
    output_path: str,
    fps: int = 2,
    canvas_size: Tuple[int, int] = (1920, 1080),
    as_video: bool = False
) -> None:
    """
    生成全局对比学习动画 (扁平化CLIP/SAM论文风格)
    Generate Global Contrastive Learning animation (flat CLIP/SAM paper style)
    
    Args:
        output_path: 输出文件路径 / Output file path
        fps: 每秒帧数 / Frames per second
        canvas_size: 画布大小 / Canvas size (width, height)
        as_video: 是否输出为视频格式 / Whether to output as video format
    """
    frames: List[np.ndarray] = []
    
    print("生成全局对比学习动画帧... / Generating Global Contrastive animation frames...")
    
    # 全局对比学习帧与增强采样动画
    # Global frames with enhanced sampling animation
    global_frames = fps * GLOBAL_ANIMATION_DURATION
    for i in range(global_frames):
        frame = create_global_frame(canvas_size, i)
        frames.append(np.array(frame))
    
    # 保存输出
    # Save output
    if as_video:
        # 确保输出路径有.mp4扩展名
        # Ensure output path has .mp4 extension
        if not output_path.lower().endswith('.mp4'):
            output_path = output_path.rsplit('.', 1)[0] + '.mp4'
        
        print(f"\n保存全局对比学习视频到 / Saving Global video to: {output_path}")
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
        
        print(f"\n保存全局对比学习GIF到 / Saving Global GIF to: {output_path}")
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
    print(f"✓ 全局对比学习动画保存成功! / Global animation saved successfully!")
    print(f"  - 总帧数 / Total frames: {len(frames)}")
    print(f"  - 时长 / Duration: {len(frames) / fps:.1f} seconds")
    print(f"  - 分辨率 / Resolution: {canvas_size[0]}x{canvas_size[1]}")
    print(f"  - 扁平化CLIP/SAM论文风格 / Flat CLIP/SAM paper style")


def generate_animation(
    output_path: str,
    fps: int = 2,
    canvas_size: Tuple[int, int] = (1920, 1080),
    as_video: bool = False
) -> None:
    """
    生成对比动画 (扁平化CLIP/SAM论文风格)
    Generate the comparison animation (flat CLIP/SAM paper style)
    
    此函数已弃用，请使用 generate_clip_animation 和 generate_global_animation
    This function is deprecated, use generate_clip_animation and generate_global_animation instead
    
    Args:
        output_path: 输出文件路径 / Output file path
        fps: 每秒帧数 / Frames per second
        canvas_size: 画布大小 / Canvas size (width, height)
        as_video: 是否输出为视频格式 / Whether to output as video format
    """
    frames: List[np.ndarray] = []
    
    print("生成帧... / Generating frames...")
    
    # 注意：标题帧已移除
    # Note: Title frame removed
    
    # 1. CLIP帧与动画 (8秒)
    # 1. CLIP frames with animation (8 seconds)
    print("  - CLIP动画帧 / CLIP animation frames")
    clip_frames = fps * CLIP_ANIMATION_DURATION
    for i in range(clip_frames):
        frame = create_clip_frame(canvas_size, i)
        frames.append(np.array(frame))
    
    # 2. 全局对比学习帧与增强采样动画 (12秒 - 更长以展示采样过程)
    # 2. Global frames with enhanced sampling animation (12 seconds - longer to show sampling)
    print("  - 全局对比学习动画帧 / Global contrastive animation frames")
    global_frames = fps * GLOBAL_ANIMATION_DURATION
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
    print(f"  - 已移除标题帧 / Removed: Title frame")


def main():
    """
    主函数：解析命令行参数并生成动画
    Main function: parse command line arguments and generate animation
    """
    parser = argparse.ArgumentParser(
        description="生成CLIP与全局对比学习的动画对比 / Generate animated comparison of CLIP vs Global Contrastive Learning"
    )
    parser.add_argument("--output", type=str, default="comparison",
                       help="输出文件路径（不含扩展名）或前缀 / Output file path (without extension) or prefix (default: comparison)")
    parser.add_argument("--video", action="store_true",
                       help="输出为MP4视频而非GIF / Output as MP4 video instead of GIF")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS,
                       help=f"每秒帧数 / Frames per second (default: {DEFAULT_FPS})")
    parser.add_argument("--width", type=int, default=DEFAULT_CANVAS_WIDTH,
                       help=f"画布宽度 / Canvas width (default: {DEFAULT_CANVAS_WIDTH})")
    parser.add_argument("--height", type=int, default=DEFAULT_CANVAS_HEIGHT,
                       help=f"画布高度 / Canvas height (default: {DEFAULT_CANVAS_HEIGHT})")
    parser.add_argument("--separate", action="store_true",
                       help="生成两个独立的GIF/视频文件（CLIP和Global） / Generate two separate GIF/video files (CLIP and Global)")
    parser.add_argument("--clip-only", action="store_true",
                       help="仅生成CLIP动画 / Generate CLIP animation only")
    parser.add_argument("--global-only", action="store_true",
                       help="仅生成全局对比学习动画 / Generate Global animation only")
    
    args = parser.parse_args()
    
    canvas_size = (args.width, args.height)
    
    # 如果指定了separate或单独生成选项，创建两个独立文件
    # If separate or individual options are specified, create two separate files
    if args.separate or args.clip_only or args.global_only:
        ext = ".mp4" if args.video else ".gif"
        
        # 移除输出路径中的扩展名
        # Remove extension from output path
        base_output = args.output
        for extension in ['.gif', '.mp4', '.GIF', '.MP4']:
            if base_output.endswith(extension):
                base_output = base_output[:-len(extension)]
                break
        
        if not args.global_only:
            # 生成CLIP动画
            # Generate CLIP animation
            clip_output = f"{base_output}_clip{ext}"
            print(f"\n{'='*60}")
            print(f"生成CLIP动画 / Generating CLIP Animation")
            print(f"{'='*60}")
            generate_clip_animation(
                output_path=clip_output,
                fps=args.fps,
                canvas_size=canvas_size,
                as_video=args.video
            )
        
        if not args.clip_only:
            # 生成全局对比学习动画
            # Generate Global animation
            global_output = f"{base_output}_global{ext}"
            print(f"\n{'='*60}")
            print(f"生成全局对比学习动画 / Generating Global Contrastive Animation")
            print(f"{'='*60}")
            generate_global_animation(
                output_path=global_output,
                fps=args.fps,
                canvas_size=canvas_size,
                as_video=args.video
            )
        
        print(f"\n{'='*60}")
        print("✓ 所有动画生成完成! / All animations generated successfully!")
        print(f"{'='*60}")
    else:
        # 默认行为：生成包含CLIP和Global的组合动画
        # Default behavior: generate combined animation with CLIP and Global
        print(f"\n{'='*60}")
        print(f"生成组合动画 / Generating Combined Animation")
        print(f"{'='*60}")
        generate_animation(
            output_path=args.output,
            fps=args.fps,
            canvas_size=canvas_size,
            as_video=args.video
        )


if __name__ == "__main__":
    main()
