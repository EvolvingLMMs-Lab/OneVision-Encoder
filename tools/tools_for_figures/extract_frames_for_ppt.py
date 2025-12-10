#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Extract frames from a video, create a GIF preview, spatiotemporal volume visualization,
and save selected frames with perspective transformation for PowerPoint presentations.

This tool extracts frames from a video (either sampled or all frames at full frame rate),
generates various visualizations including:
- Animated GIF preview
- Spatiotemporal volume visualization (space-time cube with oblique projection)
- Individual frames with 3D perspective effect

Features:
- Extract N evenly-spaced frames OR all frames at full frame rate
- Generate an animated GIF preview
- Create spatiotemporal cube visualization (video cube / space-time cube)
- Select specific frames by their indices
- Apply 3D perspective transformation for visual appeal
- Save individual frames with perspective effect

Usage Examples:
    # Extract all frames and create spatiotemporal cube visualization
    python extract_frames_for_ppt.py --video /path/to/video.mp4 --all-frames --spacetime-cube spacetime.png
    
    # Extract all frames with custom cube parameters
    python extract_frames_for_ppt.py --video /path/to/video.mp4 --all-frames \
        --spacetime-cube spacetime.png --cube-offset-x 20 --cube-offset-y 20 --cube-scale 0.4
    
    # Traditional usage: Extract 16 frames and generate GIF preview
    python extract_frames_for_ppt.py --video /path/to/video.mp4 --output frames_preview.gif

    # Select 4 specific frames and apply perspective
    python extract_frames_for_ppt.py --video /path/to/video.mp4 --select 0,4,8,12 --output-dir perspective_frames/

    # Do everything in one command
    python extract_frames_for_ppt.py --video /path/to/video.mp4 \
        --all-frames --spacetime-cube spacetime.png \
        --output preview.gif --select 0,10,20,30 --output-dir frames/
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
    resize: Optional[Tuple[int, int]] = None,
    extract_all: bool = False
) -> Optional[np.ndarray]:
    """
    Load evenly-spaced frames from a video file using decord.

    Args:
        video_path: Path to the video file
        num_frames: Number of frames to extract (default: 16)
        resize: Optional tuple (width, height) to resize frames
        extract_all: If True, extract all frames at full frame rate (default: False)

    Returns:
        np.ndarray of shape (num_frames, height, width, 3) or None if error
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return None

    try:
        from decord import VideoReader, cpu

        # Load video with decord
        # context=cpu(0) ensures we decode on CPU, consistent with typical script usage
        vr = VideoReader(video_path, ctx=cpu(0))

        total_frames = len(vr)
        if total_frames == 0:
            print(f"Error: Video has no frames: {video_path}")
            return None

        print(f"Video info: {total_frames} total frames")

        # Extract frames: all frames or evenly-spaced subset
        if extract_all:
            indices = np.arange(total_frames)
            print(f"Extracting ALL {total_frames} frames at full frame rate")
        else:
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            print(f"Extracting {num_frames} evenly-spaced frames")

        # Get batch of frames (decord returns RGB directly)
        frames_batch = vr.get_batch(indices).asnumpy()

        frames = []
        for frame in frames_batch:
            # Resize if needed
            if resize is not None:
                # Decord returns numpy array, we can use PIL or cv2 to resize
                # Since we removed cv2 dependency for loading, let's use PIL for resizing
                img = Image.fromarray(frame)
                img = img.resize(resize, Image.Resampling.LANCZOS)
                frame = np.array(img)

            frames.append(frame)

        if len(frames) == 0:
            print(f"Error: Could not read any frames from: {video_path}")
            return None

        print(f"Extracted {len(frames)} frames")
        return np.array(frames)

    except ImportError:
        print("Error: decord is not installed. Install with: pip install decord")
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
    angle: float = 12.0,
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


def create_spatiotemporal_cube(
    frames: np.ndarray,
    output_path: str,
    offset_x: int = 15,
    offset_y: int = 15,
    max_frames: Optional[int] = None,
    frame_scale: float = 0.5,
    add_labels: bool = True
) -> None:
    """
    Create a spatiotemporal volume visualization (space-time cube) with oblique projection.
    
    This creates a cascade/stacked view where frames are layered along the time axis
    with diagonal offsets, creating a 3D video cube effect suitable for presentations.

    Args:
        frames: np.ndarray of shape (num_frames, height, width, 3)
        output_path: Path to save the visualization
        offset_x: Horizontal offset between consecutive frames (default: 15)
        offset_y: Vertical offset between consecutive frames (default: 15)
        max_frames: Maximum number of frames to include (None = all frames)
        frame_scale: Scale factor for frames (default: 0.5)
        add_labels: Whether to add frame numbers
    """
    # Limit number of frames if needed
    if max_frames is not None and len(frames) > max_frames:
        # Sample frames evenly
        indices = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
        frames = frames[indices]
        print(f"Using {max_frames} frames sampled from total frames")
    
    num_frames = len(frames)
    frame_height, frame_width = frames[0].shape[:2]
    
    # Scale frames
    scaled_width = int(frame_width * frame_scale)
    scaled_height = int(frame_height * frame_scale)
    
    # Calculate canvas size
    # The canvas needs to accommodate all frames with their offsets
    canvas_width = scaled_width + (num_frames - 1) * offset_x + 100
    canvas_height = scaled_height + (num_frames - 1) * offset_y + 100
    
    # Create canvas with white background
    canvas = Image.new('RGB', (canvas_width, canvas_height), (255, 255, 255))
    
    # Font for labels
    font = get_font(16, bold=True) if add_labels else None
    
    print(f"Creating spatiotemporal cube visualization...")
    print(f"  Frames: {num_frames}")
    print(f"  Frame size: {scaled_width}x{scaled_height}")
    print(f"  Canvas size: {canvas_width}x{canvas_height}")
    print(f"  Offset: ({offset_x}, {offset_y})")
    
    # Draw frames from back to front (reverse order for proper layering)
    for i in range(num_frames - 1, -1, -1):
        frame = frames[i]
        
        # Resize frame
        frame_img = Image.fromarray(frame)
        frame_img = frame_img.resize((scaled_width, scaled_height), Image.Resampling.LANCZOS)
        
        # Calculate position (back frames at top-left, front frames at bottom-right)
        x = 50 + i * offset_x
        y = 50 + i * offset_y
        
        # Add subtle shadow for depth (optional)
        shadow = Image.new('RGBA', (scaled_width + 4, scaled_height + 4), (0, 0, 0, 50))
        canvas.paste(shadow, (x + 2, y + 2), shadow)
        
        # Paste frame
        canvas.paste(frame_img, (x, y))
        
        # Add frame border for clarity
        draw = ImageDraw.Draw(canvas)
        draw.rectangle(
            [x, y, x + scaled_width, y + scaled_height],
            outline=(100, 100, 100),
            width=2
        )
        
        # Add label if requested (only for front-most frames to avoid clutter)
        if add_labels and i >= num_frames - min(10, num_frames):
            label = f"t={i}"
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
                fill=(255, 255, 255, 200)
            )
            
            # Draw text
            draw.text((label_x, label_y), label, fill=(0, 0, 0), font=font)
    
    # Save the result
    canvas.save(output_path, "PNG")
    print(f"\nSpatiotemporal cube visualization saved to: {output_path}")
    print(f"  Image size: {canvas_width}x{canvas_height}")


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
        description="Extract frames from video, create GIF preview, spatiotemporal cube visualization, and save selected frames with perspective effect"
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
        help="Number of frames to extract from video (default: 16, ignored if --all-frames is set)"
    )
    parser.add_argument(
        "--all-frames",
        action="store_true",
        help="Extract ALL frames at full frame rate (overrides --num-frames)"
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
    
    # Spatiotemporal cube options
    parser.add_argument(
        "--spacetime-cube",
        type=str,
        help="Create spatiotemporal volume visualization (space-time cube) and save to this path (e.g., 'spacetime.png')"
    )
    parser.add_argument(
        "--cube-offset-x",
        type=int,
        default=15,
        help="Horizontal offset between frames in space-time cube (default: 15)"
    )
    parser.add_argument(
        "--cube-offset-y",
        type=int,
        default=15,
        help="Vertical offset between frames in space-time cube (default: 15)"
    )
    parser.add_argument(
        "--cube-max-frames",
        type=int,
        help="Maximum number of frames to include in space-time cube (default: all frames)"
    )
    parser.add_argument(
        "--cube-scale",
        type=float,
        default=0.5,
        help="Scale factor for frames in space-time cube (default: 0.5)"
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
    
    if args.all_frames:
        print(f"Extracting ALL frames at full frame rate...")
        frames = load_video_frames(args.video, resize=resize, extract_all=True)
    else:
        print(f"Extracting {args.num_frames} frames...")
        frames = load_video_frames(args.video, args.num_frames, resize)
    
    if frames is None:
        print("\nFailed to load video frames. Exiting.")
        return

    print(f"Successfully loaded {len(frames)} frames with shape: {frames[0].shape}")

    # Step 2: Create spatiotemporal cube if requested
    if args.spacetime_cube:
        print(f"\nStep 2: Creating spatiotemporal cube visualization...")
        create_spatiotemporal_cube(
            frames,
            args.spacetime_cube,
            offset_x=args.cube_offset_x,
            offset_y=args.cube_offset_y,
            max_frames=args.cube_max_frames,
            frame_scale=args.cube_scale,
            add_labels=not args.no_labels
        )
        step_num = 3
    else:
        step_num = 2

    # Step 2/3: Create GIF preview
    print(f"\nStep {step_num}: Creating GIF preview...")
    create_gif_preview(
        frames,
        args.output,
        duration=args.duration,
        add_labels=not args.no_labels
    )
    step_num += 1

    # Step 3/4: Save selected frames with perspective (if requested)
    if args.select:
        print(f"\nStep {step_num}: Saving selected frames with perspective effect...")
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
        print(f"\nStep {step_num}: Skipped (use --select to specify frames for perspective effect)")
        print(f"  Example: --select 0,4,8,12 to select frames 0, 4, 8, and 12")

    print("\n" + "="*60)
    print("Summary:")
    print(f"  ✓ Extracted {len(frames)} frames from video")
    if args.spacetime_cube:
        print(f"  ✓ Spatiotemporal cube: {args.spacetime_cube}")
    print(f"  ✓ GIF preview: {args.output}")
    if args.select:
        print(f"  ✓ Perspective frames: {args.output_dir}/")
    print("="*60)
    print("\nDone!")


if __name__ == "__main__":
    main()