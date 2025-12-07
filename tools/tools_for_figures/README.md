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

**Now uses Manim for high-quality mathematical animations!**

**Features:**
- Side-by-side comparison of CLIP vs Global Contrastive Learning
- Animated sampling process showing how samples are selected
- Highlights 10 positive class centers for each selected sample
- Displays randomly sampled negative centers with visual emphasis
- Multiple samples processed in sequence (similar to CLIP's approach)
- Smooth animations with professional quality using Manim
- Shows connection lines from samples to positive/negative centers
- Legend box explaining different types of centers
- CLIP section updated to mention 32K negative samples capability

**Usage:**
```bash
# Preview in low quality
manim generate_global_contrastive_comparison.py ComparisonVideo -pql

# Preview in high quality
manim generate_global_contrastive_comparison.py ComparisonVideo -pqh

# Render high quality MP4 video
manim generate_global_contrastive_comparison.py ComparisonVideo --format mp4 -qh

# Render medium quality MP4 video (faster)
manim generate_global_contrastive_comparison.py ComparisonVideo --format mp4 -qm

# Render individual scenes
manim generate_global_contrastive_comparison.py TitleScene -pql
manim generate_global_contrastive_comparison.py CLIPScene -pql
manim generate_global_contrastive_comparison.py GlobalContrastiveScene -pql
manim generate_global_contrastive_comparison.py ComparisonScene -pql
```

**Output:**
The video will be saved in:
`media/videos/generate_global_contrastive_comparison/[quality]/ComparisonVideo.mp4`

Where `[quality]` is one of:
- `480p15` for low quality
- `720p30` for medium quality  
- `1080p60` for high quality

**Animation Phases:**
1. **Title Scene (~5s):** Introduction and overview with feature comparison boxes
2. **CLIP Scene (~10s):** Shows batch-level contrastive learning with animated similarity matrix
3. **Global Contrastive Scene (~15s):** Enhanced sampling animation showing:
   - Sequential sample selection with highlight effects
   - 10 positive centers highlighted in green
   - ~25 sampled negative centers highlighted in orange
   - Animated connection lines from samples to centers
4. **Comparison Scene (~8s):** Side-by-side key differences

**Key Differences Visualized:**

| Aspect | CLIP | Global Contrastive |
|--------|------|-------------------|
| Architecture | Dual encoders (Image + Text) | Single encoder (Image only) |
| Input | Image-Text pairs | Images only |
| Negatives | Within batch (32-1024, max ~32K) | Sampled from 1M concept centers |
| Negative Pool | Limited by batch size | Massive (1M concepts) |
| Training | Cross-modal alignment | Pure visual representation |
| Positive Matching | Image-Text pairs | Image + 10 positive class centers |

## Requirements

Install dependencies:
```bash
# For generate_global_contrastive_comparison.py (uses Manim)
pip install manim

# For generate_vit_residual_gif.py (uses PIL/imageio)
pip install imageio imageio-ffmpeg pillow numpy
```

For video processing in generate_vit_residual_gif.py, also install:
```bash
pip install opencv-python
```

**System Dependencies for Manim:**
On Linux (Ubuntu/Debian):
```bash
sudo apt-get install libcairo2-dev libpango1.0-dev ffmpeg
```

On macOS:
```bash
brew install cairo pango ffmpeg
```

On Windows, follow the [Manim installation guide](https://docs.manim.community/en/stable/installation.html).

## Output Examples

Generated files:
- **generate_global_contrastive_comparison.py**: `media/videos/generate_global_contrastive_comparison/[quality]/ComparisonVideo.mp4`
- **generate_vit_residual_gif.py**: `vit_residual_encoding.mp4` or `vit_residual_encoding.gif`

## Tips

1. **For presentations:** 
   - Use Manim with `-qh` flag for best quality
   - For `generate_global_contrastive_comparison.py`, use: `manim generate_global_contrastive_comparison.py ComparisonVideo --format mp4 -qh`
   - For `generate_vit_residual_gif.py`, use MP4 format with `--video` flag
2. **For web/docs:** 
   - Use medium quality `-qm` for reasonable file sizes
   - For `generate_vit_residual_gif.py`, use GIF format for easier embedding
3. **Animation speed:** 
   - Manim animations are timed automatically in the code
   - For PIL-based animations, adjust `--fps` parameter (2-4 recommended)
4. **Resolution:** 
   - Manim: Quality flag determines resolution (480p, 720p, 1080p)
   - PIL-based: Standard HD (1920x1080) works well, adjust `--width` and `--height` as needed
5. **Preview during development:**
   - Use `-pql` (preview low quality) for fast iteration
   - Use `-pqh` (preview high quality) for final check before rendering

## Notes

- **generate_global_contrastive_comparison.py** now uses **Manim** for professional mathematical animations
- **generate_vit_residual_gif.py** still uses PIL/imageio for frame-by-frame control
- Both scripts work cross-platform (Linux, macOS, Windows)
- Font rendering uses system fonts with automatic fallback
- Manim provides smoother animations and better quality than GIF-based approaches
