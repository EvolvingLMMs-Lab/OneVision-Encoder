#!/usr/bin/env bash
set -euo pipefail

# ====== Resolve repo-relative paths ======
# This script lives at: llava_next/scripts/precompute_codec_patch/run_stage2.sh
# Stage-2 python entry is expected at: llava_next/Compressed_Video_Reader/tool/stage2.py
# (If your repo uses a different entry, update STAGE2_ENTRY_REL below.)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAVA_NEXT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"  # -> llava_next

STAGE2_ENTRY_REL="Compressed_Video_Reader/tool/stage2.py"
SCRIPT_PATH="${LLAVA_NEXT_ROOT}/${STAGE2_ENTRY_REL}"

# ====== User-configurable paths (replace `path/to/...` with your own) ======
DATASET_PATH="path/to/dataset_video.jsonl"

# Stage-1 outputs (visidx) produced by run_stage1.sh
VISIDX_ROOT="./stage1_out_kept_video8f/shard00"

# Stage-2 outputs
OUT_ROOT="./stage2_pack_kept_video8f"
OUT_IMAGE_ROOT="${OUT_ROOT}/images/shard00"
OUT_JSONL="${OUT_ROOT}/packs.jsonl"

# Logs
LOG_DIR="${OUT_ROOT}/logs"
FAIL_TXT="${LOG_DIR}/stage2_pack_failed.kept_video8f.shard00of01.txt"
STDERR_LOG="${LOG_DIR}/stage2_pack_stderr.kept_video8f.shard00of01.log"

# Create output directories
mkdir -p "${OUT_IMAGE_ROOT}" "${LOG_DIR}"

python -u "${SCRIPT_PATH}" \
  --mode pack \
  --input_dataset "${DATASET_PATH}" \
  --visidx_root "${VISIDX_ROOT}" \
  --out_image_root "${OUT_IMAGE_ROOT}" \
  --out_jsonl "${OUT_JSONL}" \
  --force_shard_out \
  --square_size 576 --T 64 --patch 16 \
  --num_images 8 --layout time_spatial \
  --first_full --write_positions --skip_missing \
  --num_workers 4 --maxtasks_per_child 200 --log_every 200 \
  --num_shards 1 --shard_id 0 \
  --fail_txt "${FAIL_TXT}" \
  2>>"${STDERR_LOG}"
