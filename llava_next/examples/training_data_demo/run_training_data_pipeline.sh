#!/bin/bash
# =============================================================================
# Training Data Preparation Pipeline for Codec Mode
# =============================================================================
# This script demonstrates how to convert raw video training data to codec format.
#
# Pipeline Overview:
#   Raw Video Data (JSON)
#         ↓
#   Stage1: Extract codec info (MV/Residual energy) → visidx_thw.npy, frame_ids.npy
#         ↓
#   Stage2: Pack frames into 8 images → positions_thw.npy, training.jsonl
# =============================================================================

set -e

# =============================================================================
# Configuration
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Input: raw video data
RAW_DATA="${SCRIPT_DIR}/raw_video_samples.json"

# Output directories
STAGE1_OUT="${SCRIPT_DIR}/output/stage1"
STAGE2_OUT="${SCRIPT_DIR}/output/stage2"
FINAL_JSONL="${SCRIPT_DIR}/output/training_codec.jsonl"

# Processing parameters
SEQUENCE_LENGTH=64      # Number of frames to sample for codec analysis
NUM_IMAGES=8            # Number of output images per video
SQUARE_SIZE=576         # Image size (576x576)
PATCH_SIZE=16           # Patch size for position encoding
NUM_WORKERS=2           # Number of parallel workers
KEEP_FIRST_FULL=true    # Keep first frame as full I-frame (not codec-packed)

# =============================================================================
# Step 0: Verify input data
# =============================================================================
echo "=============================================="
echo "Step 0: Verifying input data"
echo "=============================================="
if [ ! -f "$RAW_DATA" ]; then
    echo "Error: Raw data file not found: $RAW_DATA"
    exit 1
fi
echo "Input data: $RAW_DATA"
echo "Number of samples: $(python3 -c "import json; print(len(json.load(open('$RAW_DATA'))))")"

# Create output directories
mkdir -p "$STAGE1_OUT" "$STAGE2_OUT"

# =============================================================================
# Step 1: Stage1 - Extract codec information
# =============================================================================
echo ""
echo "=============================================="
echo "Step 1: Stage1 - Extract codec information"
echo "=============================================="
echo "This step analyzes video codec (MV/Residual) to determine patch importance."
echo ""

cd "$PROJECT_ROOT"

python Compressed_Video_Reader/tool/stage1.py \
    --dataset_path "$RAW_DATA" \
    --out_root "$STAGE1_OUT" \
    --sequence_length $SEQUENCE_LENGTH \
    --keep_frames_equiv $NUM_IMAGES \
    --square_size $SQUARE_SIZE \
    --patch_size $PATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --log_every 1 \
    ${KEEP_FIRST_FULL:+--keep_first_full_frame} \
    ${KEEP_FIRST_FULL:+--padding_policy zero}

echo ""
echo "Stage1 output structure:"
find "$STAGE1_OUT" -type f | head -20

# =============================================================================
# Step 2: Stage2 - Pack frames and generate training data
# =============================================================================
echo ""
echo "=============================================="
echo "Step 2: Stage2 - Pack frames into images"
echo "=============================================="
echo "This step packs selected patches into 8 images and generates training jsonl."
echo ""

python Compressed_Video_Reader/tool/stage2.py \
    --mode pack \
    --input_dataset "$RAW_DATA" \
    --out_jsonl "$FINAL_JSONL" \
    --visidx_root "$STAGE1_OUT" \
    --out_image_root "$STAGE2_OUT" \
    --num_images $NUM_IMAGES \
    --square_size $SQUARE_SIZE \
    --T $SEQUENCE_LENGTH \
    --patch $PATCH_SIZE \
    --write_positions \
    --num_workers $NUM_WORKERS \
    --log_every 1 \
    ${KEEP_FIRST_FULL:+--first_full}

echo ""
echo "=============================================="
echo "Pipeline Complete!"
echo "=============================================="
echo ""
echo "Output files:"
echo "  - Training JSONL: $FINAL_JSONL"
echo "  - Stage1 outputs: $STAGE1_OUT"
echo "  - Stage2 images:  $STAGE2_OUT"
echo ""
echo "Stage2 output structure:"
find "$STAGE2_OUT" -type f | head -20
echo ""
echo "Sample training data entry:"
head -1 "$FINAL_JSONL" | python3 -c "import json,sys; d=json.load(sys.stdin); print(json.dumps(d, indent=2))"
