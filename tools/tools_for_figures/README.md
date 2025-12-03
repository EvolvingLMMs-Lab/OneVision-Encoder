# Visualization Tools for Figures

This directory contains tools for generating visualizations and animations for research figures and presentations.

## Scripts

### 1. generate_vit_residual_gif.py

Generates animations visualizing how LLaVA-ViT processes video frames with residual encoding.

**Features:**
- Shows I-frame (full frame) and P-frames (residual encoding)
- Visualizes token selection and compression
- Supports both video input and synthetic demo data

**Usage:**
```bash
# Generate demo with synthetic data
python generate_vit_residual_gif.py --demo --output demo.mp4

# Process real video
python generate_vit_residual_gif.py --video /path/to/video.mp4 --output output.mp4

# Generate GIF instead of video
python generate_vit_residual_gif.py --demo --output demo.gif --gif

# Customize parameters
python generate_vit_residual_gif.py --demo --output custom.mp4 \
    --num-frames 32 --fps 4 --width 1920 --height 1080
```

**Parameters:**
- `--video`: Path to input video file
- `--demo`: Generate demo with synthetic frames
- `--output`: Output file path (default: vit_residual_encoding.mp4)
- `--num-frames`: Number of frames to sample (default: 64)
- `--patch-size`: ViT patch size (default: 16)
- `--fps`: Frames per second for video output (default: 4)
- `--width`: Canvas width (default: 1600)
- `--height`: Canvas height (default: 720)
- `--total-tokens`: Total tokens across all P-frames (default: 1372)
- `--gif`: Output as GIF instead of video
- `--duration`: Duration per frame in ms for GIF (default: 800)

### 2. generate_global_contrastive_comparison.py

Generates animations comparing CLIP's batch-level contrastive learning with global contrastive learning using 1M concept centers.

**Features:**
- Side-by-side comparison of CLIP vs Global Contrastive Learning
- Animated similarity computations
- Visual representation of concept banks and sampling
- Clear explanation of key differences

**Usage:**
```bash
# Generate GIF (default)
python generate_global_contrastive_comparison.py --output comparison.gif

# Generate MP4 video (better quality)
python generate_global_contrastive_comparison.py --output comparison.mp4 --video

# Customize parameters
python generate_global_contrastive_comparison.py \
    --output custom_comparison.gif \
    --fps 3 --width 1920 --height 1080
```

**Parameters:**
- `--output`: Output file path (default: global_contrastive_comparison.gif)
- `--video`: Output as MP4 video instead of GIF
- `--fps`: Frames per second (default: 2)
- `--width`: Canvas width (default: 1920)
- `--height`: Canvas height (default: 1080)

**Key Differences Visualized:**

| Aspect | CLIP | Global Contrastive |
|--------|------|-------------------|
| Architecture | Dual encoders (Image + Text) | Single encoder (Image only) |
| Input | Image-Text pairs | Images only |
| Negatives | Within batch (32-1024) | Sampled from 1M concept centers |
| Negative Pool | Limited by batch size | Massive (1M concepts) |
| Training | Cross-modal alignment | Pure visual representation |

## Requirements

Install dependencies:
```bash
pip install imageio imageio-ffmpeg pillow numpy
```

For video processing in generate_vit_residual_gif.py, also install:
```bash
pip install opencv-python
```

## Output Examples

Generated files can be found in this directory:
- `global_contrastive_vs_clip.gif` - Animated GIF comparison
- `global_contrastive_vs_clip.mp4` - Video comparison (better quality)

## Tips

1. **For presentations:** Use MP4 format with `--video` flag for better quality
2. **For web/docs:** Use GIF format for easier embedding
3. **Animation speed:** Adjust `--fps` parameter (2-4 recommended)
4. **Resolution:** Standard HD (1920x1080) works well, adjust based on needs

## Notes

- Both scripts work cross-platform (Linux, macOS, Windows)
- Font rendering uses system fonts with automatic fallback
- Older Pillow versions may show rectangles instead of rounded rectangles
