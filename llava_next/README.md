# OneVision Encoder - LLaVA Next

This repository contains the LLaVA-Next implementation for OneVision Encoder models with codec-based video understanding.

## Table of Contents

- [Quick Start](#quick-start)
  - [Docker Setup (Recommended)](#1--docker-recommended)
- [Training Data Preparation](#training-data-preparation)
  - [Data Format](#data-format)
  - [Conversion Pipeline](#training-data-conversion-pipeline)
- [Evaluation](#evaluation)
  - [Offline Codec Assets](#preparing-offline-codec-assets-for-evaluation)
  - [Running Evaluation](#running-evaluation)
  - [Troubleshooting](#troubleshooting)

---

## Quick Start

### 1. 🐳 Docker (Recommended)

We strongly recommend using the Docker environment for a seamless experience. The following instructions are tailored for the A100 80GB GPU environment.

```bash
# Clone repository
git clone https://github.com/EvolvingLMMs-Lab/OneVision-Encoder
cd OneVision-Encoder/llava_next

# Build Docker image
docker build -t ov_encoder_llava:26.01 .

# Run container
docker run -it --gpus all \
    --ipc host --net host --privileged --cap-add IPC_LOCK \
    --ulimit memlock=-1 --ulimit stack=67108864 --rm \
    -v $(pwd):/workspace/OV-Encoder-Llava \
    -w /workspace/OV-Encoder-Llava \
    --name "ov_encoder_llava_container" \
    ov_encoder_llava:26.01 bash -c "service ssh restart; bash"
```

---

## Training Data Preparation

Training data for codec mode requires precomputed visual assets (mosaic images + position indices). Each training sample contains:
- Pre-extracted frame images (e.g., 8 frames per video)
- Position indices file (`positions_thw.npy`) encoding temporal-height-width coordinates

### Data Format

#### Original (Raw) Video Format

Raw video training data uses JSON array format with direct video paths:

```json
[
  {
    "id": "YVQwAEKZpaU",
    "conversations": [
      {"from": "human", "value": "<video>\nWhat is the background setting?"},
      {"from": "gpt", "value": "A clear blue sky with spectators."}
    ],
    "video": "/path/to/videos/ytb_YVQwAEKZpaU.mp4"
  }
]
```

#### Converted Codec Format (JSONL)

Each line in the training JSONL should follow this format:

```json
{
  "id": "sample_unique_id",
  "conversations": [
    {"from": "human", "value": "<image>\n<image>\n<image>\n<image>\n<image>\n<image>\n<image>\n<image>\nYour question here?"},
    {"from": "gpt", "value": "Model response here."}
  ],
  "image": [
    "/path/to/frame_000.jpg",
    "/path/to/frame_001.jpg",
    "/path/to/frame_002.jpg",
    "/path/to/frame_003.jpg",
    "/path/to/frame_004.jpg",
    "/path/to/frame_005.jpg",
    "/path/to/frame_006.jpg",
    "/path/to/frame_007.jpg"
  ],
  "positions_thw": "/path/to/positions_thw.npy"
}
```

#### Format Comparison

| Field | Raw Format | Codec Format |
|-------|------------|--------------|
| Visual token | `<video>` (single) | `<image>` × N (one per frame) |
| Visual path | `video`: single mp4 path | `image`: list of frame paths |
| Position info | Not required | `positions_thw`: npy file path |
| File format | JSON array | JSONL (one sample per line) |

#### Key Fields

| Field | Description |
|-------|-------------|
| `id` | Unique sample identifier |
| `conversations` | Multi-turn conversation in human/gpt format |
| `image` | List of frame image paths (8 frames for codec mode) |
| `positions_thw` | Path to numpy file containing patch position indices |

> **Note:** The number of `<image>` tokens in the conversation must match the number of images in the `image` list.

#### Position Indices Format (`positions_thw.npy`)

The `positions_thw.npy` file contains patch position coordinates:

| Property | Description |
|----------|-------------|
| **Shape** | `[num_patches, 3]` where each row is `[t, h, w]` |
| **Dtype** | `int32` |
| **Coordinates** | `t`: temporal index, `h`: height position, `w`: width position |

Example: For 8 frames with 36×36 patches each → shape `[10368, 3]`

```python
import numpy as np
positions = np.load("positions_thw.npy")
# positions.shape = (10368, 3)
# positions[:5] = [[0,0,0], [0,0,1], [0,0,2], [0,0,3], [0,0,4]]
```

#### Mixed Training Data

You can mix video (codec) and image data in the same JSONL. For image-only samples:

```json
{
  "id": "image_sample_id",
  "conversations": [
    {"from": "human", "value": "<image>\nDescribe this image."},
    {"from": "gpt", "value": "Description here."}
  ],
  "image": "/path/to/single_image.jpg"
}
```

Image samples do not require the `positions_thw` field.

#### Directory Structure

```
training_data_root/
├── images/
│   └── shard00/
│       └── sample_<unique_key>/
│           ├── video_000.jpg
│           ├── video_001.jpg
│           ├── ...
│           ├── video_007.jpg
│           └── positions_thw.npy
└── train.jsonl
```

### Training Data Conversion Pipeline

Convert raw video data to codec format using the two-stage pipeline:

```
Raw Video Data (JSON with <video> token)
        ↓
Stage 1: Extract codec info (MV/Residual energy) → visidx_thw.npy, frame_ids.npy
        ↓
Stage 2: Pack frames into 8 images → positions_thw.npy, training.jsonl (with <image> tokens)
```

#### Quick Start (Demo)

```bash
cd llava_next

# Run the complete pipeline with sample videos
bash examples/training_data_demo/run_training_data_pipeline.sh
```

#### Manual Execution

**Stage 1: Extract codec information**

```bash
python Compressed_Video_Reader/tool/stage1.py \
    --dataset_path /path/to/raw_videos.json \
    --out_root /path/to/stage1_output \
    --sequence_length 64 \
    --keep_frames_equiv 8 \
    --square_size 576 \
    --patch_size 16 \
    --num_workers 8 \
    --keep_first_full_frame \
    --padding_policy zero
```

**Stage 2: Pack frames and generate training JSONL**

```bash
python Compressed_Video_Reader/tool/stage2.py \
    --mode pack \
    --input_dataset /path/to/raw_videos.json \
    --out_jsonl /path/to/training_codec.jsonl \
    --visidx_root /path/to/stage1_output \
    --out_image_root /path/to/stage2_images \
    --num_images 8 \
    --square_size 576 \
    --T 64 \
    --patch 16 \
    --write_positions \
    --num_workers 8 \
    --first_full
```

#### Pipeline Parameters

| Parameter | Stage | Description |
|-----------|-------|-------------|
| `--sequence_length` / `--T` | 1 & 2 | Number of frames for codec analysis (default: 64) |
| `--keep_frames_equiv` / `--num_images` | 1 & 2 | Number of output images per video (default: 8) |
| `--square_size` | 1 & 2 | Image size (default: 576) |
| `--patch_size` / `--patch` | 1 & 2 | Patch size for position encoding (default: 16) |
| `--keep_first_full_frame` | 1 | Keep first frame as complete I-frame (recommended) |
| `--padding_policy` | 1 | How to handle empty patches: `zero` or `repeat` |
| `--first_full` | 2 | Corresponding flag when using `--keep_first_full_frame` |
| `--write_positions` | 2 | Generate `positions_thw.npy` files |
| `--num_workers` | 1 & 2 | Parallel processing workers |

#### Output Structure

```
stage2_output/
├── sample_<id>__<video_stem>__<hash>/
│   ├── video_000.jpg ~ video_007.jpg
│   └── positions_thw.npy
└── ...
training_codec.jsonl
```

---

## Evaluation

### Preparing Offline Codec Assets for Evaluation

For video evaluations using **codec mode**, precompute visual assets before running evaluation.

#### Quick Start

```bash
cd llava_next

# Preprocess a single benchmark (auto-downloads data if needed)
bash scripts/precompute_codec_patch/preprocess_video_benchmark.sh videomme
bash scripts/precompute_codec_patch/preprocess_video_benchmark.sh mvbench
bash scripts/precompute_codec_patch/preprocess_video_benchmark.sh perceptiontest

# Or preprocess all supported benchmarks
bash scripts/precompute_codec_patch/preprocess_video_benchmark.sh all
```

#### Supported Benchmarks

| Task Name | lmms-eval Task | Description |
|-----------|----------------|-------------|
| `videomme` | videomme | Video-MME benchmark |
| `mvbench` | mvbench | MVBench benchmark |
| `perceptiontest` | perceptiontest_val_mc | PerceptionTest Val |
| `nextqa` | nextqa_mc_test | NExTQA benchmark |
| `temporalbench` | temporalbench_long_qa | TemporalBench |
| `video_mmmu` | video_mmmu | Video-MMMU |
| `tomato` | tomato | TOMATO benchmark |
| `longvideobench` | longvideobench_val_v | LongVideoBench |

#### Prerequisites (Gated Datasets)

Some datasets require HuggingFace authentication:

```bash
# Login to Hugging Face (one-time setup)
huggingface-cli login

# Accept dataset terms on HuggingFace website if required
```

#### Output Structure

```
.huggingface_cache/
├── mvbench_video/              # lmms-eval video cache (auto-downloaded)
│   └── *.mp4
├── mvbench_offline/            # Precomputed offline assets
│   ├── mvbench_videos.jsonl
│   └── assets/
│       └── <video_stem>/
│           ├── mosaic_000.jpg ~ mosaic_007.jpg
│           ├── positions_thw.npy
│           └── meta.json
```

### Running Evaluation

#### Using Local Script (Recommended)

The local eval script auto-detects offline assets based on task:

```bash
bash scripts/eval/local_eval_ov_encoder.sh
```

#### Manual Environment Variables

```bash
export LLAVA_CODEC_USE_OFFLINE=1
export LLAVA_CODEC_OFFLINE_ROOT=$(pwd)/.huggingface_cache/<task>_offline/assets
export LLAVA_CODEC_VISIDX_MODE=pack_topk
export LLAVA_CODEC_SEQ_LEN_FRAMES=64
export LLAVA_CODEC_NUM_IMAGES=8
export LLAVA_CODEC_SQUARE_SIZE=576
export LLAVA_CODEC_PATCH_SIZE=16

bash scripts/eval/eval_ov_encoder.sh
```

#### Codec Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `SEQ_LEN_FRAMES` | 64 | Number of frames for codec analysis |
| `NUM_IMAGES` | 8 | Number of output mosaic images per video |
| `SQUARE_SIZE` | 576 | Image size (576×576) |
| `PATCH_SIZE` | 16 | Patch size for position encoding |

### Troubleshooting

If evaluation shows `MISS` (fallback to frame extraction):

1. **Check offline root path**: `LLAVA_CODEC_OFFLINE_ROOT` should point to `assets/` directory
2. **Check video key matching**: The `<video_stem>` folder name must match what the model expects
3. **Verify files exist**: `mosaic_000.jpg`, `positions_thw.npy`, `meta.json` should be present
4. **Check codec parameters**: Ensure precompute and eval use the same parameters

### Manual Preprocessing (Advanced)

For custom datasets or fine-grained control:

```bash
# 1. Prepare input JSONL with video paths and unique keys
# Each line: {"video": "/path/to/video.mp4", "key": "unique_id", ...}

# 2. Run offline precompute
python Compressed_Video_Reader/tool/offline_precompute_llava_codec_assets.py \
    --jsonl path/to/eval_videos.jsonl \
    --out_root path/to/offline_root \
    --num_workers 8 \
    --seq_len_frames 64 \
    --num_images 8 \
    --square_size 576 \
    --patch_size 16

# Optional: sharding for large datasets
python Compressed_Video_Reader/tool/offline_precompute_llava_codec_assets.py \
    --jsonl path/to/eval_videos.jsonl \
    --out_root path/to/offline_root \
    --num_shards 8 --shard_id 0
```

---

## License

This project is licensed under the Apache 2.0 License.
