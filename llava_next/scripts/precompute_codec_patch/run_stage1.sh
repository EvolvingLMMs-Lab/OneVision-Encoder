#!/usr/bin/env bash
set -euo pipefail

# ====== Resolve repo-relative paths ======
# This script lives at: llava_next/scripts/precompute_codec_patch/run_stage1.sh
# Stage-1 python entry lives at: llava_next/Compressed_Video_Reader/tool/stage1.py
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAVA_NEXT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"  # -> llava_next
SCRIPT_PATH="${LLAVA_NEXT_ROOT}/Compressed_Video_Reader/tool/stage1.py"

# ====== User-configurable paths (replace `path/to/...` with your own) ======
DATASET_PATH="path/to/dataset_video.jsonl"

# Output root for this shard
OUT_ROOT="./stage1_out_kept_video8f/shard00"

# Logs
LOG_DIR="./logs"
FAIL_TXT="${LOG_DIR}/stage1_failed.kept_video8f.shard00of01.txt"
UNSUPPORTED_TXT="${LOG_DIR}/stage1_unsupported_codec.kept_video8f.shard00of01.txt"

# Create output directories
mkdir -p "${OUT_ROOT}" "${LOG_DIR}"

python -u "${SCRIPT_PATH}" \
  --dataset_path "${DATASET_PATH}" \
  --out_root "${OUT_ROOT}" \
  --sequence_length 64 \
  --patch_size 16 \
  --square_size 576 \
  --keep_frames_equiv 8 \
  --padding_policy zero \
  --keep_first_full_frame \
  --mv_compensate median \
  --num_workers 4 \
  --num_shards 1 \
  --shard_id 0 \
  --maxtasks_per_child 100 \
  --check_codec \
  --log_every 500 \
  --fail_txt "${FAIL_TXT}" \
  --unsupported_txt "${UNSUPPORTED_TXT}"