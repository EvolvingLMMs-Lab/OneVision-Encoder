python ../../Compressed_Video_Reader/tool/offline_precompute_llava_codec_assets.py   --jsonl /video_vit/yunyaoyan/lmms/lmms-eval-main/tool/mvbench.json   --out_root ./mvbench   --num_workers 16   --seq_len_frames 64   --num_images 8   --square_size 576   --patch_size 16

#!/usr/bin/env bash
set -euo pipefail

# Run offline precompute to generate LLaVA codec mosaic assets for an eval jsonl.
#
# Usage:
#   bash run_offline_codec_patch_for_eval.sh \
#     --jsonl path/to/eval_dump.jsonl \
#     --out_root path/to/output_dir \
#     --num_workers 16 \
#     --seq_len_frames 64 \
#     --num_images 8 \
#     --square_size 576 \
#     --patch_size 16
#
# Examples:
#   bash run_offline_codec_patch_for_eval.sh --jsonl path/to/mvbench.jsonl --out_root path/to/mvbench_assets

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN=${PYTHON_BIN:-python3}

# Entry script lives in Compressed_Video_Reader/tool/
PRECOMPUTE_PY="${REPO_ROOT}/Compressed_Video_Reader/tool/offline_precompute_llava_codec_assets.py"

if [[ ! -f "${PRECOMPUTE_PY}" ]]; then
  echo "[error] Cannot find offline precompute script: ${PRECOMPUTE_PY}" >&2
  echo "[hint] Are you running this from the repo checkout?" >&2
  exit 1
fi

# If user didn't pass args, show a short help.
if [[ $# -eq 0 ]]; then
  echo "[usage]" >&2
  echo "  bash $(basename "$0") --jsonl path/to/eval_dump.jsonl --out_root path/to/output_dir [more args...]" >&2
  echo "" >&2
  echo "[note] You can pass any flags supported by offline_precompute_llava_codec_assets.py." >&2
  echo "" >&2
  "${PYTHON_BIN}" "${PRECOMPUTE_PY}" --help
  exit 2
fi

# Reasonable defaults (can be overridden by passing explicit flags).
DEFAULT_ARGS=(
  --num_workers 16
  --seq_len_frames 64
  --num_images 8
  --square_size 576
  --patch_size 16
)

# If the caller provides their own flags, they will override defaults because they appear later.
"${PYTHON_BIN}" "${PRECOMPUTE_PY}" "${DEFAULT_ARGS[@]}" "$@"