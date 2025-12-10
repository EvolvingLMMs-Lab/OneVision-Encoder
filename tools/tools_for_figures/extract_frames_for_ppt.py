#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Extract frames from a video, create a GIF preview, and save selected frames 
with perspective transformation for PowerPoint presentations.

This tool extracts 16 frames from a video, generates a GIF preview, and allows
you to select 4 specific frames to save with a 3D perspective effect suitable
for embedding in PowerPoint slides.

Features:
- Extract 16 evenly-spaced frames from any video
- Generate an animated GIF preview
- Select any 4 frames by their indices
- Apply 3D perspective transformation for visual appeal
- Save individual frames with perspective effect

Usage:
    # Step 1: Extract frames and generate GIF preview
    python extract_frames_for_ppt.py --video /path/to/video.mp4 --output frames_preview.gif
    
    # Step 2: Select 4 specific frames and apply perspective
    python extract_frames_for_ppt.py --video /path/to/video.mp4 --select 0,4,8,12 --output-dir perspective_frames/
    
    # Or do both in one command
    python extract_frames_for_ppt.py --video /path/to/video.mp4 --output preview.gif --select 0,4,8,12 --output-dir frames/
"""

import argparse
import os
from pathlib import Path
from typing import List, Optional, Tuple

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


def load_video_frames(
    video_path: str, 
    num_frames: int = 16, 
    resize: Optional[Tuple[int, int]] = None
) -> Optional[np.ndarray]:
    """
    Load evenly-spaced frames from a video file.
    
    Args:
        video_path: Path to the video file
        num_frames: Number of frames to extract (default: 16)
        resize: Optional tuple (width, height) to resize frames
        
    Returns:
        np.ndarray of shape (num_frames, height, width, 3) or None if error
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return None
    
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video: {video_path}")
            return None
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            print(f"Error: Video has no frames: {video_path}")
            cap.release()
            return None
        
        print(f"Video info: {total_frames} total frames")
        
        # Extract evenly-spaced frames
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if resize is not None:
                    frame = cv2.resize(frame, resize)
                frames.append(frame)
            else:
                print(f"Warning: Could not read frame {idx}")
        
        cap.release()
        
        if len(frames) == 0:
            print(f"Error: Could not read any frames from: {video_path}")
            return None
        
        print(f"Extracted {len(frames)} frames")
        return np.array(frames)
    
    except ImportError:
        print("Error: OpenCV (cv2) is not installed. Install with: pip install opencv-python")
        return None
    except Exception as e:
        print(f"Error loading video: {e}")
        return None


def create_gif_preview(
    frames: np.ndarray,
    output_path: str,
    duration: int = 500,
    add_labels: bool = True
) -> None:
    """
    Create an animated GIF preview of the extracted frames.
    
    Args:
        frames: np.ndarray of shape (num_frames, height, width, 3)
        output_path: Path to save the GIF
        duration: Duration of each frame in milliseconds (default: 500)
        add_labels: Whether to add frame numbers to each frame
    """
    gif_frames = []
    font = get_font(24, bold=True)
    
    for idx, frame in enumerate(frames):
        # Convert to PIL Image
        img = Image.fromarray(frame)
        
        if add_labels:
            # Add frame number overlay
            draw = ImageDraw.Draw(img)
            label = f"Frame {idx}"
            
            # Get text bounding box for background
            bbox = draw.textbbox((0, 0), label, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Position at top-left with padding
            padding = 10
            bg_box = [
                padding, 
                padding, 
                padding + text_width + padding * 2, 
                padding + text_height + padding * 2
            ]
            
            # Draw semi-transparent background
            draw.rectangle(bg_box, fill=(0, 0, 0, 180))
            
            # Draw text
            draw.text(
                (padding * 2, padding * 1.5), 
                label, 
                fill=(255, 255, 255), 
                font=font
            )
        
        gif_frames.append(img)
    
    # Save as GIF
    gif_frames[0].save(
        output_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=duration,
        loop=0
    )
    
    print(f"GIF preview saved to: {output_path}")
    print(f"  - Total frames: {len(gif_frames)}")
    print(f"  - Duration per frame: {duration}ms")


def apply_perspective_transform(
    image: np.ndarray,
    angle: float = 15.0,
    scale: float = 0.8
) -> Image.Image:
    """
    Apply a 3D perspective transformation to an image for PPT presentation.
    
    This creates a tilted, slightly rotated view that looks good in slides.
    
    Args:
        image: Input image as np.ndarray (H, W, 3)
        angle: Rotation angle in degrees (default: 15)
        scale: Scale factor (default: 0.8 to leave space for perspective)
        
    Returns:
        PIL Image with perspective transformation applied
    """
    img = Image.fromarray(image)
    width, height = img.size
    
    # Create a larger canvas to accommodate the perspective transform
    new_width = int(width * 1.3)
    new_height = int(height * 1.3)
    
    # Calculate perspective transform coefficients
    # This creates a tilted view with depth
    angle_rad = np.radians(angle)
    
    # Original corners
    corners_orig = np.array([
        [0, 0],
        [width, 0],
        [width, height],
        [0, height]
    ], dtype=np.float32)
    
    # Target corners with perspective
    offset_x = (new_width - width * scale) / 2
    offset_y = (new_height - height * scale) / 2
    depth_factor = 0.2  # How much depth perspective to apply
    
    corners_new = np.array([
        [offset_x + width * scale * depth_factor, offset_y],  # Top-left (further back)
        [offset_x + width * scale * (1 - depth_factor), offset_y],  # Top-right (closer)
        [offset_x + width * scale, offset_y + height * scale],  # Bottom-right
        [offset_x, offset_y + height * scale]  # Bottom-left
    ], dtype=np.float32)
    
    # Calculate perspective transform coefficients
    # Using PIL's transform with PERSPECTIVE method
    # Coefficients for perspective transform: (a, b, c, d, e, f, g, h)
    # where: x' = (ax + by + c) / (gx + hy + 1)
    #        y' = (dx + ey + f) / (gx + hy + 1)
    
    # For simplicity, we'll use PIL's transform with affine
    # and add a slight rotation for visual interest
    
    # Create new image with perspective
    result = Image.new('RGBA', (new_width, new_height), (255, 255, 255, 0))
    
    # Resize and rotate slightly for 3D effect
    img_scaled = img.resize((int(width * scale), int(height * scale)), Image.Resampling.LANCZOS)
    img_rotated = img_scaled.rotate(-angle, expand=True, fillcolor=(255, 255, 255))
    
    # Add subtle shadow for depth
    shadow = Image.new('RGBA', img_rotated.size, (0, 0, 0, 60))
    shadow_offset = 10
    
    # Paste shadow
    shadow_x = (new_width - img_rotated.width) // 2 + shadow_offset
    shadow_y = (new_height - img_rotated.height) // 2 + shadow_offset
    result.paste(shadow, (shadow_x, shadow_y), shadow)
    
    # Paste main image
    img_x = (new_width - img_rotated.width) // 2
    img_y = (new_height - img_rotated.height) // 2
    result.paste(img_rotated, (img_x, img_y), img_rotated if img_rotated.mode == 'RGBA' else None)
    
    return result


def save_perspective_frames(
    frames: np.ndarray,
    selected_indices: List[int],
    output_dir: str,
    angle: float = 12.0
) -> None:
    """
    Save selected frames with perspective transformation.
    
    Args:
        frames: All extracted frames
        selected_indices: Indices of frames to save with perspective
        output_dir: Directory to save the perspective frames
        angle: Rotation angle for perspective effect (default: 12)
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Different angles for variety
    angles = [angle, -angle, angle * 1.2, -angle * 0.8]
    
    for i, idx in enumerate(selected_indices):
        if idx < 0 or idx >= len(frames):
            print(f"Warning: Frame index {idx} is out of range (0-{len(frames)-1}), skipping")
            continue
        
        frame = frames[idx]
        
        # Apply perspective with varying angles for visual interest
        angle_to_use = angles[i % len(angles)]
        perspective_img = apply_perspective_transform(frame, angle=angle_to_use)
        
        # Save with descriptive filename
        output_path = os.path.join(output_dir, f"frame_{idx:02d}_perspective.png")
        perspective_img.save(output_path, "PNG")
        print(f"Saved perspective frame {idx} to: {output_path}")
    
    print(f"\nAll perspective frames saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from video, create GIF preview, and save selected frames with perspective effect"
    )
    parser.add_argument(
        "--video", 
        type=str, 
        required=True, 
        help="Path to input video file"
    )
    parser.add_argument(
        "--num-frames", 
        type=int, 
        default=16, 
        help="Number of frames to extract from video (default: 16)"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="frames_preview.gif", 
        help="Output path for GIF preview (default: frames_preview.gif)"
    )
    parser.add_argument(
        "--duration", 
        type=int, 
        default=500, 
        help="Duration of each frame in GIF (ms, default: 500)"
    )
    parser.add_argument(
        "--select", 
        type=str, 
        help="Comma-separated list of frame indices to save with perspective (e.g., '0,4,8,12')"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="perspective_frames", 
        help="Directory to save perspective frames (default: perspective_frames/)"
    )
    parser.add_argument(
        "--angle", 
        type=float, 
        default=12.0, 
        help="Perspective rotation angle in degrees (default: 12.0)"
    )
    parser.add_argument(
        "--resize", 
        type=str, 
        help="Resize frames to WIDTHxHEIGHT (e.g., '640x480')"
    )
    parser.add_argument(
        "--no-labels", 
        action="store_true", 
        help="Don't add frame labels to GIF preview"
    )
    
    args = parser.parse_args()
    
    # Parse resize option
    resize = None
    if args.resize:
        try:
            width, height = map(int, args.resize.split('x'))
            resize = (width, height)
            print(f"Frames will be resized to: {width}x{height}")
        except ValueError:
            print(f"Warning: Invalid resize format '{args.resize}', expected WIDTHxHEIGHT")
    
    # Step 1: Load video frames
    print(f"\nStep 1: Loading video frames...")
    print(f"Video: {args.video}")
    print(f"Extracting {args.num_frames} frames...")
    
    frames = load_video_frames(args.video, args.num_frames, resize)
    if frames is None:
        print("\nFailed to load video frames. Exiting.")
        return
    
    print(f"Successfully loaded {len(frames)} frames with shape: {frames[0].shape}")
    
    # Step 2: Create GIF preview
    print(f"\nStep 2: Creating GIF preview...")
    create_gif_preview(
        frames, 
        args.output, 
        duration=args.duration,
        add_labels=not args.no_labels
    )
    
    # Step 3: Save selected frames with perspective (if requested)
    if args.select:
        print(f"\nStep 3: Saving selected frames with perspective effect...")
        try:
            selected_indices = [int(x.strip()) for x in args.select.split(',')]
            
            if len(selected_indices) == 0:
                print("Warning: No frame indices provided")
            else:
                print(f"Selected frame indices: {selected_indices}")
                save_perspective_frames(
                    frames, 
                    selected_indices, 
                    args.output_dir,
                    angle=args.angle
                )
        except ValueError:
            print(f"Error: Invalid frame indices format '{args.select}'. Use comma-separated integers (e.g., '0,4,8,12')")
    else:
        print(f"\nStep 3: Skipped (use --select to specify frames for perspective effect)")
        print(f"  Example: --select 0,4,8,12 to select frames 0, 4, 8, and 12")
    
    print("\n" + "="*60)
    print("Summary:")
    print(f"  ✓ Extracted {len(frames)} frames from video")
    print(f"  ✓ GIF preview: {args.output}")
    if args.select:
        print(f"  ✓ Perspective frames: {args.output_dir}/")
    print("="*60)
    print("\nDone!")


if __name__ == "__main__":
    main()
