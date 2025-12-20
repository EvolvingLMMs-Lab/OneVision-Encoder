#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Extract frames from a video and create cube-based GIF visualizations.

This simplified tool creates two types of spatiotemporal cube visualizations:
- Animated cube building (GIF showing frames being added one by one with transparency)
- Residual-based animated cube (highlights changed regions, darkens unchanged regions)

Usage Examples:
    # Create animated cube building GIF
    python extract_frames_for_ppt.py --video /path/to/video.mp4 --animated-cube cube_animation.gif

    # Create residual cube GIF
    python extract_frames_for_ppt.py --video /path/to/video.mp4 --residual-gif residual_cube.gif

    # Create both with custom parameters
    python extract_frames_for_ppt.py --video /path/to/video.mp4 \
        --animated-cube test.gif --animation-duration 300 \
        --residual-gif residual_preview.gif --residual-threshold 20
"""

import argparse
import os
from typing import Optional, Tuple

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


def compute_residual_mask(
    current_frame: np.ndarray,
    reference_frame: np.ndarray,
    patch_size: int = 16,
    threshold: float = 10.0
) -> np.ndarray:
    """
    Compute a binary mask indicating which patches have high residual values.

    Args:
        current_frame: Current frame to compare
        reference_frame: Reference frame (typically the first frame)
        patch_size: Size of each patch for residual computation
        threshold: Threshold for considering a patch as having high residual

    Returns:
        Binary mask where 1 = high residual (keep), 0 = low residual (darken)

    Note:
        Partial patches at image boundaries (when dimensions are not evenly divisible
        by patch_size) are not processed and will have mask value of 0 (darkened).
    """
    # Compute raw residual
    residual = current_frame.astype(np.float32) - reference_frame.astype(np.float32)

    h, w = current_frame.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)

    # Process each patch
    num_patches_h = h // patch_size
    num_patches_w = w // patch_size

    for i in range(num_patches_h):
        for j in range(num_patches_w):
            y_start = i * patch_size
            y_end = (i + 1) * patch_size
            x_start = j * patch_size
            x_end = (j + 1) * patch_size

            # Calculate mean absolute difference for this patch
            patch_residual = residual[y_start:y_end, x_start:x_end]
            patch_mad = np.mean(np.abs(patch_residual))

            # If residual is high, mark this patch in the mask
            if patch_mad > threshold:
                mask[y_start:y_end, x_start:x_end] = 1.0

    return mask


def apply_residual_darkening(
    frame: np.ndarray,
    mask: np.ndarray,
    darken_factor: float = 0.3
) -> np.ndarray:
    """
    Darken regions of the frame based on the residual mask.

    Args:
        frame: Original frame
        mask: Binary mask (1 = keep bright, 0 = darken)
        darken_factor: Factor to darken low-residual regions (0-1, lower = darker)

    Returns:
        Frame with low-residual regions darkened
    """
    # Expand mask to 3 channels
    mask_3d = np.stack([mask, mask, mask], axis=2)

    # Apply darkening: high residual regions stay bright, low residual regions are darkened
    darkened_frame = frame.astype(np.float32) * (mask_3d + (1 - mask_3d) * darken_factor)

    return darkened_frame.clip(0, 255).astype(np.uint8)


def create_residual_gif_preview(
    frames: np.ndarray,
    output_path: str,
    duration: int = 500,
    add_labels: bool = True,
    patch_size: int = 16,
    threshold: float = 10.0,
    darken_factor: float = 0.3,
    offset_x: int = 15,
    offset_y: int = 15,
    max_frames: Optional[int] = None,
    frame_scale: float = 0.5,
    transparency: bool = True,
    final_hold_frames: int = 3
) -> None:
    """
    Create an animated GIF showing the residual cube being built frame by frame.

    This creates an animation that starts from a single frame (reference) and progressively adds
    more frames to build the complete spatiotemporal cube visualization with residual highlighting.
    The first frame is shown normally as the reference. Subsequent frames have their
    low-residual patches (patches with little change from the first frame) darkened,
    while high-residual patches (patches with significant change) remain bright.

    Args:
        frames: np.ndarray of shape (num_frames, height, width, 3)
        output_path: Path to save the residual GIF
        duration: Duration of each animation frame in milliseconds (default: 500)
        add_labels: Whether to add frame numbers to each frame
        patch_size: Size of patches for residual computation (default: 16)
        threshold: Threshold for high residual detection (default: 10.0)
        darken_factor: Factor to darken low-residual regions (default: 0.3)
        offset_x: Horizontal offset between consecutive frames (default: 15)
        offset_y: Vertical offset between consecutive frames (default: 15)
        max_frames: Maximum number of frames to include (None = all frames)
        frame_scale: Scale factor for frames (default: 0.5)
        transparency: Whether to apply transparency effects for depth (default: True)
        final_hold_frames: Number of times to repeat final frame for emphasis (default: 3)
    """
    # Limit number of frames if needed
    original_frame_count = len(frames)
    if max_frames is not None and original_frame_count > max_frames:
        # Sample frames evenly
        indices = np.linspace(0, original_frame_count - 1, max_frames, dtype=int)
        frames = frames[indices]
        print(f"Using {max_frames} frames sampled from {original_frame_count} total frames")

    # Process frames with residual highlighting
    reference_frame = frames[0]
    processed_frames = []

    for idx, frame in enumerate(frames):
        if idx == 0:
            # First frame is the reference - show it normally
            processed_frame = frame.copy()
        else:
            # Compute residual mask and apply darkening
            mask = compute_residual_mask(frame, reference_frame, patch_size, threshold)
            processed_frame = apply_residual_darkening(frame, mask, darken_factor)

        processed_frames.append(processed_frame)

    num_frames = len(processed_frames)
    frame_height, frame_width = processed_frames[0].shape[:2]

    # Scale frames
    scaled_width = int(frame_width * frame_scale)
    scaled_height = int(frame_height * frame_scale)

    # Calculate canvas size
    canvas_width = scaled_width + (num_frames - 1) * offset_x + 100
    canvas_height = scaled_height + (num_frames - 1) * offset_y + 100

    # Font for labels
    font = get_font(16, bold=True) if add_labels else None

    print(f"Creating animated residual cube building visualization...")
    print(f"  Frames: {num_frames}")
    print(f"  Frame size: {scaled_width}x{scaled_height}")
    print(f"  Canvas size: {canvas_width}x{canvas_height}")
    print(f"  Offset: ({offset_x}, {offset_y})")
    print(f"  Transparency: {transparency}")
    print(f"  Patch size: {patch_size}x{patch_size}")
    print(f"  Residual threshold: {threshold}")

    gif_frames = []

    # Create animation: add frames one by one
    for current_frame_count in range(1, num_frames + 1):
        # Create canvas with white background (RGBA for transparency support)
        canvas = Image.new('RGBA', (canvas_width, canvas_height), (255, 255, 255, 255))

        # Draw frames in order (0 to current_frame_count-1), so later frames overlay earlier frames
        for i in range(current_frame_count):
            frame = processed_frames[i]

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
            # Use different colors for reference frame vs residual frames
            if i == 0:
                border_color = (100, 255, 150, 255)  # Green for reference frame
            else:
                border_color = (255, 150, 100, 255)  # Orange for residual frames
            draw.rectangle(
                [x, y, x + scaled_width, y + scaled_height],
                outline=border_color,
                width=2
            )

            # Add label if requested (only for the most recent frames)
            if add_labels and i >= current_frame_count - min(5, current_frame_count):
                if i == 0:
                    label = f"Ref"
                else:
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

    print(f"\nAnimated residual cube GIF saved to: {output_path}")
    print(f"  Total animation frames: {len(gif_frames)}")
    print(f"  Duration per frame: {duration}ms")


def create_animated_cube_building(
    frames: np.ndarray,
    output_path: str,
    offset_x: int = 15,
    offset_y: int = 15,
    max_frames: Optional[int] = None,
    frame_scale: float = 0.5,
    add_labels: bool = True,
    duration: int = 300,
    transparency: bool = True,
    final_hold_frames: int = 3
) -> None:
    """
    Create an animated GIF showing the spatiotemporal cube being built frame by frame.

    This creates an animation that starts from a single frame and progressively adds
    more frames to build the complete spatiotemporal cube visualization with optional
    transparency effects for depth perception.

    Args:
        frames: np.ndarray of shape (num_frames, height, width, 3)
        output_path: Path to save the animated GIF
        offset_x: Horizontal offset between consecutive frames (default: 15)
        offset_y: Vertical offset between consecutive frames (default: 15)
        max_frames: Maximum number of frames to include (None = all frames)
        frame_scale: Scale factor for frames (default: 0.5)
        add_labels: Whether to add frame numbers
        duration: Duration of each animation frame in milliseconds (default: 300)
        transparency: Whether to apply transparency effects for depth (default: True)
        final_hold_frames: Number of times to repeat final frame for emphasis (default: 3)
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
    canvas_width = scaled_width + (num_frames - 1) * offset_x + 100
    canvas_height = scaled_height + (num_frames - 1) * offset_y + 100

    # Font for labels
    font = get_font(16, bold=True) if add_labels else None

    print(f"Creating animated cube building visualization...")
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
            frame = frames[i]

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
            draw.rectangle(
                [x, y, x + scaled_width, y + scaled_height],
                outline=(100, 100, 100, 255),
                width=2
            )

            # Add label if requested (only for the most recent frames)
            if add_labels and i >= current_frame_count - min(5, current_frame_count):
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


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from video and create cube-based GIF visualizations"
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
        "--resize",
        type=str,
        help="Resize frames to WIDTHxHEIGHT (e.g., '640x480')"
    )
    parser.add_argument(
        "--no-labels",
        action="store_true",
        help="Don't add frame labels to cube visualizations"
    )

    # Cube visualization options
    parser.add_argument(
        "--cube-offset-x",
        type=int,
        default=15,
        help="Horizontal offset between frames in cube (default: 15)"
    )
    parser.add_argument(
        "--cube-offset-y",
        type=int,
        default=15,
        help="Vertical offset between frames in cube (default: 15)"
    )
    parser.add_argument(
        "--cube-max-frames",
        type=int,
        help="Maximum number of frames to include in cube (default: all frames)"
    )
    parser.add_argument(
        "--cube-scale",
        type=float,
        default=0.5,
        help="Scale factor for frames in cube (default: 0.5)"
    )

    # Animated cube building options
    parser.add_argument(
        "--animated-cube",
        type=str,
        help="Create animated GIF showing cube being built frame by frame (e.g., 'cube_animation.gif')"
    )
    parser.add_argument(
        "--animation-duration",
        type=int,
        default=300,
        help="Duration of each frame in animated cube GIF in milliseconds (default: 300)"
    )
    parser.add_argument(
        "--no-transparency",
        action="store_true",
        help="Disable transparency effects in animated cube (frames won't fade with depth)"
    )

    # Residual GIF options
    parser.add_argument(
        "--residual-gif",
        type=str,
        help="Create animated residual cube GIF with high-residual patches highlighted (e.g., 'residual_cube.gif')"
    )
    parser.add_argument(
        "--residual-patch-size",
        type=int,
        default=16,
        help="Patch size for residual computation (default: 16)"
    )
    parser.add_argument(
        "--residual-threshold",
        type=float,
        default=10.0,
        help="Threshold for high residual detection (default: 10.0)"
    )
    parser.add_argument(
        "--residual-darken-factor",
        type=float,
        default=0.3,
        help="Factor to darken low-residual regions, 0-1 (default: 0.3, lower = darker)"
    )

    args = parser.parse_args()

    # Check that at least one output is requested
    if not args.animated_cube and not args.residual_gif:
        print("Error: At least one output must be specified (--animated-cube or --residual-gif)")
        parser.print_help()
        return

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

    # Step 2: Create animated cube building if requested
    step_num = 2
    if args.animated_cube:
        print(f"\nStep {step_num}: Creating animated cube building GIF...")
        create_animated_cube_building(
            frames,
            args.animated_cube,
            offset_x=args.cube_offset_x,
            offset_y=args.cube_offset_y,
            max_frames=args.cube_max_frames,
            frame_scale=args.cube_scale,
            add_labels=not args.no_labels,
            duration=args.animation_duration,
            transparency=not args.no_transparency
        )
        step_num += 1

    # Step 3: Create residual GIF if requested
    if args.residual_gif:
        print(f"\nStep {step_num}: Creating residual-based cube GIF...")
        create_residual_gif_preview(
            frames,
            args.residual_gif,
            duration=args.animation_duration,
            add_labels=not args.no_labels,
            patch_size=args.residual_patch_size,
            threshold=args.residual_threshold,
            darken_factor=args.residual_darken_factor,
            offset_x=args.cube_offset_x,
            offset_y=args.cube_offset_y,
            max_frames=args.cube_max_frames,
            frame_scale=args.cube_scale,
            transparency=not args.no_transparency
        )
        step_num += 1

    print("\n" + "="*60)
    print("Summary:")
    print(f"  ✓ Extracted {len(frames)} frames from video")
    if args.animated_cube:
        print(f"  ✓ Animated cube building: {args.animated_cube}")
    if args.residual_gif:
        print(f"  ✓ Residual cube GIF: {args.residual_gif}")
    print("="*60)
    print("\nDone!")



if __name__ == "__main__":
    main()