## Quick Start Guide

### 1.🐳 Docker (Recommended)

We strongly recommend using the docker environment for a seamless experience. The following instructions are tailored for the A100 80GB GPU environment.


```bash
# Clone repository
git clone https://github.com/EvolvingLMMs-Lab/OneVision-Encoder
cd OneVision-Encoder/llava_next

docker build -t ov_encoder_llava:26.01 .

# Run container with -w to set working directory directly to the mounted volume
docker run -it --gpus all \
    --ipc host --net host --privileged --cap-add IPC_LOCK \
    --ulimit memlock=-1 --ulimit stack=67108864 --rm \
    -v $(pwd):/workspace/OV-Encoder-Llava \
    -w /workspace/OV-Encoder-Llava \
    --name "ov_encoder_llava_container" \
    ov_encoder_llava:26.01 bash -c "service ssh restart; bash; "
```

---

### Preparing eval datasets and offline codec-patch assets (for video tasks)

Some video evaluations benefit from **offline precomputed visual assets** (mosaics + position indices) to avoid per-sample frame extraction during evaluation.

This repo provides:
- `Compressed_Video_Reader/tool/offline_precompute_llava_codec_assets.py` — precomputes offline assets from a jsonl list.
- `scripts/precompute_codec_patch/run_offline_codec_patch_for_eval.sh` — a runner script (recommended) with `path/to/...` placeholders.

#### 1) Create the input jsonl (video list)

Prepare a jsonl file where **each line** contains at least:
- `video`: absolute path to the video file
- `key`: a **stable unique id** used as the offline asset folder name

Recommended (matches the precompute script schema):
- `task`, `split`, `doc_id`, `n`, `exists`

Example line:
```json
{"task":"videomme","split":"test","doc_id":123,"n":0,"video":"path/to/video.mp4","key":"<unique_key>","exists":true}
```

**Important:** the `key` must be consistent between **precompute** and the model's **offline loader**. If the loader reports `MISS`, it usually means the `key` (or folder layout / filenames) doesn't match.

#### 2) Precompute offline assets

Run the precompute tool to generate assets under:
```
path/to/offline_root/assets/<key>/
  mosaic_000.jpg ... mosaic_007.jpg
  positions_thw.npy
  visible_indices.npy
  frame_ids.npy
  meta.json
```

Example command:
```bash
python Compressed_Video_Reader/tool/offline_precompute_llava_codec_assets.py \
  --jsonl path/to/eval_videos.jsonl \
  --out_root path/to/offline_root \
  --num_workers 8 \
  --seq_len_frames 64 \
  --num_images 8 \
  --square_size 576 \
  --patch_size 16
```

Optional sharding (useful for large datasets):
```bash
python Compressed_Video_Reader/tool/offline_precompute_llava_codec_assets.py \
  --jsonl path/to/eval_videos.jsonl \
  --out_root path/to/offline_root \
  --num_workers 8 \
  --num_shards 8 \
  --shard_id 0
```

**Do not rename** the expected output filenames (e.g. `mosaic_000.jpg`, `positions_thw.npy`). Renaming will cause `MISS` and trigger fallback to frame extraction.

#### 3) Run evaluation using offline assets

Set these environment variables before launching `lmms_eval`:
```bash
export LLAVA_CODEC_USE_OFFLINE=1
export LLAVA_CODEC_OFFLINE_ROOT=path/to/offline_root/assets

# Optional (recommended): keep configs aligned with your offline assets
export LLAVA_CODEC_VISIDX_MODE=pack_topk
export LLAVA_CODEC_SEQ_LEN_FRAMES=64
export LLAVA_CODEC_NUM_IMAGES=8
export LLAVA_CODEC_SQUARE_SIZE=576
export LLAVA_CODEC_PATCH_SIZE=16

# Debug (single switch)
export LLAVA_CODEC_DEBUG=1
```

If offline assets are found, logs will include `HIT` / `USING_OFFLINE`. If not found, you will see `MISS` and the pipeline will **fallback to frame extraction**.

Quick checklist when you see `MISS`:
- `LLAVA_CODEC_OFFLINE_ROOT` points to the directory that contains `assets/<key>/...`
- `<key>` matches the one produced in your input jsonl
- `mosaic_000.jpg ...` and `positions_thw.npy` exist under `assets/<key>/`
- your `seq_len_frames / num_images / square_size / patch_size` match between precompute and eval

---
