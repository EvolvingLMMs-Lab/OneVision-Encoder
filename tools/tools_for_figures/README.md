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

### 2. extract_frames_for_ppt.py

Extract frames from a video, create an animated GIF preview, spatiotemporal volume visualization (space-time cube), and save selected frames with 3D perspective transformation for PowerPoint presentations.

**Features:**
- Extract N evenly-spaced frames OR all frames at full frame rate
- Generate an animated GIF preview with frame labels
- **NEW:** Create spatiotemporal cube visualization (video cube / space-time cube with oblique projection)
- Select specific frames to save separately
- Apply 3D perspective transformation with rotation for visual appeal
- Varying angles for each frame for dynamic presentation

**Usage:**
```bash
# NEW: Extract all frames and create spatiotemporal cube visualization
python extract_frames_for_ppt.py --video /path/to/video.mp4 \
    --all-frames \
    --spacetime-cube spacetime.png

# Customize spatiotemporal cube parameters
python extract_frames_for_ppt.py --video /path/to/video.mp4 \
    --all-frames \
    --spacetime-cube spacetime.png \
    --cube-offset-x 20 \
    --cube-offset-y 20 \
    --cube-scale 0.4 \
    --cube-max-frames 50 \
    --resize 320x240

# Traditional: Extract 16 frames and create GIF preview
python extract_frames_for_ppt.py --video /path/to/video.mp4 --output preview.gif

# Select frames and apply perspective transformation
python extract_frames_for_ppt.py --video /path/to/video.mp4 \
    --select 0,5,10,15 \
    --output-dir ppt_frames/

# Do everything in one command
python extract_frames_for_ppt.py --video /path/to/video.mp4 \
    --all-frames \
    --spacetime-cube spacetime.png \
    --output preview.gif \
    --select 0,10,20,30 \
    --output-dir ppt_frames/
```

**Parameters:**
- `--video`: Path to input video file (required)
- `--num-frames`: Number of frames to extract (default: 16, ignored if --all-frames is set)
- `--all-frames`: Extract ALL frames at full frame rate (overrides --num-frames)
- `--output`: Output path for GIF preview (default: frames_preview.gif)
- `--duration`: Duration per frame in GIF in milliseconds (default: 500)
- `--select`: Comma-separated frame indices for perspective effect (e.g., '0,4,8,12')
- `--output-dir`: Directory to save perspective frames (default: perspective_frames/)
- `--angle`: Perspective rotation angle in degrees (default: 12.0)
- `--resize`: Resize frames to WIDTHxHEIGHT (e.g., '640x480')
- `--no-labels`: Don't add frame labels to GIF preview

**Spatiotemporal Cube Parameters:**
- `--spacetime-cube`: Path to save space-time cube visualization (e.g., 'spacetime.png')
- `--cube-offset-x`: Horizontal offset between frames in pixels (default: 15)
- `--cube-offset-y`: Vertical offset between frames in pixels (default: 15)
- `--cube-max-frames`: Maximum number of frames to include (default: all frames)
- `--cube-scale`: Scale factor for frames (default: 0.5, smaller = more compact)

**Workflows:**

*Spatiotemporal Cube Workflow (NEW):*
1. Extract all frames from your video at full frame rate with `--all-frames`
2. Create a space-time cube visualization with `--spacetime-cube`
3. Adjust offset and scale parameters to control the 3D appearance
4. Use the generated PNG image in your PowerPoint presentation

*Traditional Workflow:*
1. Extract frames from your video and generate a GIF preview
2. View the GIF to identify which frames you want for your presentation
3. Run again with `--select` to create perspective versions of chosen frames
4. Import the perspective PNGs into your PowerPoint slides

**Output:**
- GIF preview: Animated preview showing frames with frame numbers
- **Spatiotemporal cube:** PNG image showing all frames stacked with oblique projection (cascade/stacked view)
- Perspective frames: Individual PNG files with 3D rotation effect, saved with transparency

**Spatiotemporal Cube Visualization:**

The spatiotemporal cube (also known as space-time cube or video cube) creates a 3D visualization where:
- Frames are stacked along the time axis
- Uses oblique projection to maintain frame proportions
- Creates a cascade/layered view effect
- Perfect for showing temporal progression in presentations
- Front frames show time labels (t=0, t=1, etc.)

### 3. generate_global_contrastive_comparison.py

Generates animations comparing CLIP's batch-level contrastive learning with global contrastive learning using 1M concept centers.

**Features:**
- Publication-quality visual design (CLIP/SAM paper style)
- Generates separate animations for CLIP and Global Contrastive Learning
- Option to create combined animation or individual GIFs
- All hyperparameters centralized at the top of the file for easy customization
- Animated sampling process showing how samples are selected
- Highlights 10 positive class centers for each selected sample
- Displays randomly sampled negative centers with visual emphasis
- Multiple samples processed in sequence
- Sophisticated visual aesthetics with multi-layer glow effects
- Enhanced color palette with Chinese and English documentation
- Professional flat design without title frame (direct to content)

**Usage:**
```bash
# Generate two separate GIF files (recommended)
python generate_global_contrastive_comparison.py --output comparison --separate

# Generate CLIP animation only
python generate_global_contrastive_comparison.py --output clip --clip-only

# Generate Global animation only
python generate_global_contrastive_comparison.py --output global --global-only

# Generate combined animation (backward compatible)
python generate_global_contrastive_comparison.py --output comparison.gif

# Generate as MP4 video with separate files
python generate_global_contrastive_comparison.py --output comparison --separate --video

# Customize parameters
python generate_global_contrastive_comparison.py \
    --output custom \
    --separate \
    --fps 3 --width 1920 --height 1080
```

**Parameters:**
- `--output`: Output file path (without extension) or prefix (default: comparison)
- `--video`: Output as MP4 video instead of GIF
- `--fps`: Frames per second (default: 2)
- `--width`: Canvas width (default: 1920)
- `--height`: Canvas height (default: 1080)
- `--separate`: Generate two separate GIF/video files (CLIP and Global)
- `--clip-only`: Generate CLIP animation only
- `--global-only`: Generate Global animation only

**Output Files (with --separate flag):**
- `{output}_clip.gif` or `{output}_clip.mp4` - CLIP contrastive learning animation
- `{output}_global.gif` or `{output}_global.mp4` - Global contrastive learning animation

**Animation Details:**
1. **CLIP Animation (8s):** Shows batch-level contrastive learning with similarity matrix
2. **Global Contrastive Animation (12s):** Shows sampling from 1M concept centers with:
   - Sequential sample selection with elegant highlight effects
   - 10 positive centers clustered together with multi-layer glow effects
   - ~64 sampled negative centers scattered across concept space
   - Smooth animated connection lines from samples to centers
   - Enhanced concept bank visualization with depth effects

**Customizable Hyperparameters (at top of file):**
All parameters are now centralized at the top of the script for easy modification:
- Animation timing (CLIP_ANIMATION_DURATION, GLOBAL_ANIMATION_DURATION)
- Batch sizes (CLIP_BATCH_SIZE, GLOBAL_BATCH_SIZE)
- Concept center settings (GLOBAL_TOTAL_CONCEPTS, GLOBAL_NUM_POSITIVE_CENTERS)
- Layout dimensions (item heights, gaps, encoder widths)
- Color palette (IMAGE_COLORS_SAM, positive/negative colors)
- Typography (font sizes for title, label, small text)
- Visual enhancements (border radius, border widths, glow effects)

**Design Updates:**
- **Title Frame Removed:** Direct entry into content without title slide
- **Enhanced Color Palette:** More vibrant and harmonious SAM-style colors with bilingual comments
- **Multi-layer Glow Effects:** Positive centers have 3 layers, negatives have subtle glow
- **Better Visual Hierarchy:** Improved sizing and spacing throughout
- **Flat Design:** Clean, modern aesthetic without unnecessary shadows
- **Improved Connection Lines:** Smoother curves and better visual flow

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
pip install imageio imageio-ffmpeg pillow numpy opencv-python
```

**Note:** `opencv-python` is required for video processing in both `generate_vit_residual_gif.py` and `extract_frames_for_ppt.py`.

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
