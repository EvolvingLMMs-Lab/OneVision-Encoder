#!/bin/bash
# Universal Video Benchmark Preprocessing Script for Offline Codec Evaluation
#
# This script preprocesses video benchmarks for offline codec evaluation.
# Supported tasks: videomme, perceptiontest, mvbench, nextqa, temporalbench, video_mmmu
#
# Usage:
#   cd llava_next
#   bash scripts/precompute_codec_patch/preprocess_video_benchmark.sh videomme
#   bash scripts/precompute_codec_patch/preprocess_video_benchmark.sh mvbench
#   bash scripts/precompute_codec_patch/preprocess_video_benchmark.sh all  # Process all benchmarks

set -e

# ============ Configuration ============
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
CACHE_DIR="${REPO_DIR}/.huggingface_cache"
echo "Using huggingface cache directory: ${CACHE_DIR}"

# Set HF_HOME to the repo's cache directory so that lmms-eval cached videos are found
export HF_HOME="${CACHE_DIR}"

# Codec parameters (match with eval config)
SEQ_LEN_FRAMES=64
NUM_IMAGES=8
SQUARE_SIZE=576
PATCH_SIZE=16
NUM_WORKERS=32

# Task to process
TASK="${1:-}"

if [ -z "$TASK" ]; then
    echo "Usage: $0 <task>"
    echo ""
    echo "Supported tasks (use either name or lmms-eval task name):"
    echo "  videomme                  - Video-MME benchmark"
    echo "  perceptiontest[_val_mc]   - PerceptionTest Val benchmark"
    echo "  mvbench                   - MVBench benchmark"
    echo "  nextqa[_mc_test]          - NExTQA benchmark"
    echo "  temporalbench[_long_qa]   - TemporalBench benchmark"
    echo "  video_mmmu                - Video-MMMU benchmark"
    echo "  tomato                    - TOMATO benchmark"
    echo "  longvideobench[_val_v]    - LongVideoBench Val V benchmark"
    echo "  mlvu_dev                  - MLVU Dev benchmark"
    echo "  all                       - Process all benchmarks"
    echo ""
    echo "Example:"
    echo "  $0 videomme"
    echo "  $0 longvideobench_val_v"
    echo "  $0 all"
    exit 1
fi

# Define task configurations
# Format: HF_REPO|LOCAL_DIR|VIDEO_FIELD|KEY_FIELD|SPLIT|HAS_ZIP
declare -A TASK_CONFIGS
TASK_CONFIGS["videomme"]="lmms-lab/Video-MME|Video-MME|videoID|videoID|test|yes"
TASK_CONFIGS["perceptiontest"]="lmms-lab/PerceptionTest_Val|PerceptionTest_Val|video_id|video_id|validation|no"
TASK_CONFIGS["perceptiontest_val_mc"]="lmms-lab/PerceptionTest_Val|PerceptionTest_Val|video_id|video_id|validation|no"
TASK_CONFIGS["mvbench"]="OpenGVLab/MVBench|MVBench|video|video|train|no"
TASK_CONFIGS["nextqa"]="lmms-lab/NExTQA|NExTQA|video|video|test|no"
TASK_CONFIGS["nextqa_mc_test"]="lmms-lab/NExTQA|NExTQA|video|video|test|no"
TASK_CONFIGS["temporalbench"]="microsoft/TemporalBench|TemporalBench|video|video|test_long_qa|no"
TASK_CONFIGS["temporalbench_long_qa"]="microsoft/TemporalBench|TemporalBench|video|video|test_long_qa|no"
TASK_CONFIGS["video_mmmu"]="lmms-lab/VideoMMMU|VideoMMMU|video|id|test|no"
TASK_CONFIGS["tomato"]="lmms-lab/TOMATO|TOMATO|video|video|test|no"
TASK_CONFIGS["longvideobench"]="longvideobench/LongVideoBench|LongVideoBench|video|video|validation|no"
TASK_CONFIGS["longvideobench_val_v"]="longvideobench/LongVideoBench|LongVideoBench|video|video|validation|no"
TASK_CONFIGS["mlvu_dev"]="sy1998/MLVU_dev|mlvu|video_name|video_name|test|no"

# Function to process a single task
process_task() {
    local task=$1
    local config="${TASK_CONFIGS[$task]}"
    
    if [ -z "$config" ]; then
        echo "[ERROR] Unknown task: $task"
        echo "Supported tasks: ${!TASK_CONFIGS[@]}"
        return 1
    fi
    
    # Parse config
    IFS='|' read -r HF_REPO LOCAL_DIR VIDEO_FIELD KEY_FIELD SPLIT HAS_ZIP <<< "$config"
    
    local DATASET_DIR="${CACHE_DIR}/${LOCAL_DIR}"
    local OFFLINE_ROOT="${CACHE_DIR}/${task}_offline"
    local INPUT_JSONL="${OFFLINE_ROOT}/${task}_videos.jsonl"
    
    # lmms-eval caches videos to HF_HOME with specific folder names
    # We should prioritize these because they contain the ACTUAL videos used during evaluation
    local HF_HOME_DIR="${HF_HOME:-$HOME/.cache/huggingface}"
    
    # Define lmms-eval cache locations for each task
    # Format: task -> cache_subdir (relative to HF_HOME)
    declare -A LMMS_EVAL_CACHE_MAP
    LMMS_EVAL_CACHE_MAP["mvbench"]="mvbench_video"
    LMMS_EVAL_CACHE_MAP["videomme"]="videomme"
    LMMS_EVAL_CACHE_MAP["nextqa"]="nextqa"
    LMMS_EVAL_CACHE_MAP["nextqa_mc_test"]="nextqa"
    LMMS_EVAL_CACHE_MAP["tomato"]="TOMATO"
    LMMS_EVAL_CACHE_MAP["perceptiontest"]="PerceptionTest"
    LMMS_EVAL_CACHE_MAP["perceptiontest_val_mc"]="PerceptionTest"
    LMMS_EVAL_CACHE_MAP["temporalbench"]="TemporalBench"
    LMMS_EVAL_CACHE_MAP["temporalbench_long_qa"]="TemporalBench"
    LMMS_EVAL_CACHE_MAP["longvideobench"]="LongVideoBench"
    LMMS_EVAL_CACHE_MAP["longvideobench_val_v"]="LongVideoBench"
    LMMS_EVAL_CACHE_MAP["video_mmmu"]="video_mmmu"
    LMMS_EVAL_CACHE_MAP["mlvu_dev"]="mlvu"
    
    # Get the lmms-eval cache directory for this task
    local LMMS_CACHE_SUBDIR="${LMMS_EVAL_CACHE_MAP[$task]}"
    local LMMS_EVAL_CACHE=""
    if [ -n "$LMMS_CACHE_SUBDIR" ]; then
        LMMS_EVAL_CACHE="${HF_HOME_DIR}/${LMMS_CACHE_SUBDIR}"
    fi
    
    echo ""
    echo "=============================================="
    echo "Processing: ${task}"
    echo "=============================================="
    echo "HF Repo: ${HF_REPO}"
    echo "Local Dataset Dir: ${DATASET_DIR}"
    echo "Offline Root: ${OFFLINE_ROOT}"
    if [ -n "$LMMS_EVAL_CACHE" ]; then
        echo "lmms-eval Cache: ${LMMS_EVAL_CACHE}"
    fi
    echo ""
    
    # Priority order for video source:
    # 1. lmms-eval cache (HF_HOME/<task>) - contains actual videos used during evaluation
    # 2. Local dataset dir (CACHE_DIR/<task>) - may contain different video versions
    # 3. Trigger lmms-eval to download the dataset (this ensures we get the exact same videos as evaluation)
    
    local VIDEO_SOURCE=""
    
    # Check lmms-eval cache first (highest priority)
    if [ -n "$LMMS_EVAL_CACHE" ] && [ -d "$LMMS_EVAL_CACHE" ]; then
        # Check if it actually has video files (follow symlinks)
        local VIDEO_COUNT=$(find -L "$LMMS_EVAL_CACHE" -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.webm" -o -name "*.mkv" \) 2>/dev/null | wc -l)
        if [ "$VIDEO_COUNT" -gt 0 ]; then
            echo "[INFO] Found lmms-eval cache at: ${LMMS_EVAL_CACHE} (${VIDEO_COUNT} videos)"
            echo "[INFO] Using lmms-eval cache (recommended: contains actual evaluation videos)"
            VIDEO_SOURCE="$LMMS_EVAL_CACHE"
            DATASET_DIR="$LMMS_EVAL_CACHE"
        else
            echo "[INFO] lmms-eval cache exists but has no videos: ${LMMS_EVAL_CACHE}"
        fi
    fi
    
    # Check if lmms-eval cache is a symlink target in HF hub
    if [ -z "$VIDEO_SOURCE" ] && [ -n "$LMMS_CACHE_SUBDIR" ]; then
        # Try to find in HF hub datasets
        local HF_REPO_SLUG=$(echo "$HF_REPO" | tr '/' '--')
        local HUB_PATH="${HF_HOME_DIR}/hub/datasets--${HF_REPO_SLUG}"
        if [ -d "$HUB_PATH" ]; then
            local SNAPSHOT=$(find "$HUB_PATH" -type d -name "snapshots" -exec find {} -mindepth 1 -maxdepth 1 -type d \; 2>/dev/null | head -1)
            if [ -n "$SNAPSHOT" ] && [ -d "$SNAPSHOT" ]; then
                local VIDEO_COUNT=$(find -L "$SNAPSHOT" -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.webm" -o -name "*.mkv" \) 2>/dev/null | wc -l)
                if [ "$VIDEO_COUNT" -gt 0 ]; then
                    echo "[INFO] Found HF hub cache at: ${SNAPSHOT} (${VIDEO_COUNT} videos)"
                    VIDEO_SOURCE="$SNAPSHOT"
                    DATASET_DIR="$SNAPSHOT"
                fi
            fi
        fi
    fi
    
    # If no lmms-eval cache found, trigger lmms-eval to download the dataset
    # This ensures we get the EXACT same videos as used during evaluation
    if [ -z "$VIDEO_SOURCE" ]; then
        echo ""
        echo "[INFO] No lmms-eval video cache found. Triggering lmms-eval data download..."
        echo "[INFO] This will download the dataset and create the video cache that evaluation uses."
        echo ""
        
        # Use Python to trigger lmms-eval dataset loading
        # Note: Variables are passed via environment to avoid escaping issues
        PREPROCESS_TASK="${task}" \
        PREPROCESS_HF_HOME="${HF_HOME_DIR}" \
        PREPROCESS_CACHE_SUBDIR="${LMMS_CACHE_SUBDIR}" \
        python3 << 'LMMS_DOWNLOAD_EOF'
import os
import sys

task = os.environ.get("PREPROCESS_TASK", "")
hf_home = os.environ.get("PREPROCESS_HF_HOME", "")
lmms_cache_subdir = os.environ.get("PREPROCESS_CACHE_SUBDIR", "")

print(f"[lmms-eval download] Task: {task}")
print(f"[lmms-eval download] HF_HOME: {hf_home}")
print(f"[lmms-eval download] Expected cache dir: {lmms_cache_subdir}")

if not task:
    print("[lmms-eval download] ERROR: No task specified")
    sys.exit(1)

# Set HF_HOME environment variable
os.environ["HF_HOME"] = hf_home

try:
    # Import lmms_eval task loading
    from lmms_eval.tasks import get_task_dict
    
    # Map simplified task names to lmms-eval task names
    task_name_map = {
        "mvbench": "mvbench",
        "videomme": "videomme",
        "nextqa": "nextqa_mc_test",
        "nextqa_mc_test": "nextqa_mc_test",
        "tomato": "tomato",
        "perceptiontest": "perceptiontest_val_mc",
        "perceptiontest_val_mc": "perceptiontest_val_mc",
        "temporalbench": "temporalbench_long_qa",
        "temporalbench_long_qa": "temporalbench_long_qa",
        "longvideobench": "longvideobench_val_v",
        "longvideobench_val_v": "longvideobench_val_v",
        "video_mmmu": "video_mmmu",
        "mlvu_dev": "mlvu_dev",
    }
    
    lmms_task = task_name_map.get(task, task)
    print(f"[lmms-eval download] Loading task: {lmms_task}")
    
    # Load the task - this triggers dataset download
    task_dict = get_task_dict([lmms_task])
    
    # Access the dataset to trigger video download AND cache dataset metadata
    for task_name, task_obj in task_dict.items():
        print(f"[lmms-eval download] Processing task: {task_name}")
        if hasattr(task_obj, 'dataset'):
            ds = task_obj.dataset
            print(f"[lmms-eval download] Dataset loaded with {len(ds) if hasattr(ds, '__len__') else 'unknown'} samples")
        
        # Load ALL docs to ensure complete dataset caching (metadata + videos)
        # This is necessary for lmms-eval to work offline later
        if hasattr(task_obj, 'test_docs'):
            print(f"[lmms-eval download] Loading all test docs to cache dataset...")
            docs = list(task_obj.test_docs())  # Load ALL to trigger complete download
            print(f"[lmms-eval download] Loaded {len(docs)} test docs (complete)")
        elif hasattr(task_obj, 'validation_docs'):
            print(f"[lmms-eval download] Loading all validation docs to cache dataset...")
            docs = list(task_obj.validation_docs())  # Load ALL
            print(f"[lmms-eval download] Loaded {len(docs)} validation docs (complete)")
    
    print("[lmms-eval download] Dataset download complete!")
    
except ImportError as e:
    print(f"[lmms-eval download] WARNING: Could not import lmms_eval: {e}")
    print("[lmms-eval download] Falling back to direct HuggingFace download...")
    sys.exit(1)
except Exception as e:
    # Many exceptions happen AFTER videos are already downloaded
    # (e.g., token issues when loading task docs)
    # So we exit with 0 to let the script check if videos exist
    print(f"[lmms-eval download] WARNING: Task loading issue: {e}")
    print("[lmms-eval download] Videos may still have been downloaded. Checking...")
    sys.exit(0)  # Exit 0 so script continues to check for videos
LMMS_DOWNLOAD_EOF
        
        local DOWNLOAD_STATUS=$?
        
        # Check if videos exist in lmms-eval cache (regardless of download status)
        # Sometimes download succeeds but task loading fails (e.g., token issues)
        if [ -n "$LMMS_EVAL_CACHE" ] && [ -d "$LMMS_EVAL_CACHE" ]; then
            local VIDEO_COUNT=$(find -L "$LMMS_EVAL_CACHE" -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.webm" -o -name "*.mkv" \) 2>/dev/null | wc -l)
            if [ "$VIDEO_COUNT" -gt 0 ]; then
                echo "[INFO] Found ${VIDEO_COUNT} videos at lmms-eval cache: ${LMMS_EVAL_CACHE}"
                VIDEO_SOURCE="$LMMS_EVAL_CACHE"
                DATASET_DIR="$LMMS_EVAL_CACHE"
            fi
        fi
        
        # If no videos in lmms-eval cache, fall back to HuggingFace CLI
        if [ -z "$VIDEO_SOURCE" ]; then
            echo "[INFO] Falling back to direct HuggingFace download..."
            huggingface-cli download "${HF_REPO}" --repo-type dataset --local-dir "${CACHE_DIR}/${LOCAL_DIR}"
            VIDEO_SOURCE="${CACHE_DIR}/${LOCAL_DIR}"
            DATASET_DIR="${CACHE_DIR}/${LOCAL_DIR}"
            
            echo ""
            echo "[WARNING] =================================================="
            echo "[WARNING] Downloaded dataset directly from HuggingFace."
            echo "[WARNING] This may contain DIFFERENT videos than lmms-eval uses!"
            echo "[WARNING] For best results, run lmms-eval once to download the"
            echo "[WARNING] exact videos, then re-run this preprocessing script."
            echo "[WARNING] =================================================="
            echo ""
        fi
    fi
    
    echo "[INFO] Video source: ${VIDEO_SOURCE}"
    echo ""
    
    # Extract zip files if needed (e.g., videomme, mvbench)
    if [ "$HAS_ZIP" = "yes" ]; then
        echo "[Step 1/3] Extracting videos from zip files..."
        local VIDEOS_DIR="${DATASET_DIR}/videos"
        mkdir -p "${VIDEOS_DIR}"
        
        python3 << EOF
import zipfile
import os
from pathlib import Path

dataset_dir = "${DATASET_DIR}"
videos_dir = "${VIDEOS_DIR}"

# Search for zip files in root and subdirectories (e.g., video/*.zip for MVBench)
zip_files = sorted(Path(dataset_dir).glob("*.zip"))
zip_files += sorted(Path(dataset_dir).glob("video/*.zip"))
zip_files += sorted(Path(dataset_dir).glob("videos/*.zip"))
# Remove duplicates while preserving order
seen = set()
zip_files = [z for z in zip_files if not (z in seen or seen.add(z))]

print(f"Found {len(zip_files)} zip files to process")

extracted = 0
for zip_path in zip_files:
    if 'subtitle' in zip_path.name.lower():
        continue
    marker_file = Path(videos_dir) / f".extracted_{zip_path.stem}"
    
    if marker_file.exists():
        print(f"  {zip_path.name}: Already extracted, skipping...")
        continue
    
    print(f"  Extracting {zip_path.name}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(videos_dir)
        marker_file.touch()
        extracted += 1
    except Exception as e:
        print(f"  ERROR extracting {zip_path.name}: {e}")

print(f"Extracted {extracted} new zip files")
EOF
        echo "[Step 1/3] Done."
    else
        echo "[Step 1/3] No zip extraction needed for ${task}."
    fi
    
    # Generate input JSONL
    echo ""
    echo "[Step 2/3] Generating input JSONL for offline precompute..."
    mkdir -p "${OFFLINE_ROOT}"
    
    python3 << EOF
import json
import os
from pathlib import Path
import glob

task = "${task}"
dataset_dir = "${DATASET_DIR}"
output_jsonl = "${INPUT_JSONL}"
video_field = "${VIDEO_FIELD}"
key_field = "${KEY_FIELD}"
split = "${SPLIT}"
has_zip = "${HAS_ZIP}" == "yes"

print(f"Task: {task}")
print(f"Dataset dir: {dataset_dir}")
print(f"Video field: {video_field}")
print(f"Key field: {key_field}")

# Find parquet files
parquet_files = list(Path(dataset_dir).rglob("*.parquet"))
if not parquet_files:
    # Try to find in subdirectories with task name
    parquet_files = list(Path(dataset_dir).rglob(f"*{split}*.parquet"))

# Also check for arrow files
arrow_files = list(Path(dataset_dir).rglob("*.arrow"))

print(f"Found {len(parquet_files)} parquet files, {len(arrow_files)} arrow files")

# Universal direct video scan mode
# This is the most reliable approach: directly scan for video files
# instead of relying on dataset metadata which may have different paths
print(f"[{task}] Using direct video scan mode (most reliable)")

# Find all video files (.mp4, .avi, .mkv, .webm)
video_extensions = ['.mp4', '.avi', '.mkv', '.webm', '.mov']
video_files = []
for ext in video_extensions:
    video_files.extend(sorted(Path(dataset_dir).rglob(f"*{ext}")))

# Remove duplicates while preserving order
seen = set()
video_files = [v for v in video_files if not (v in seen or seen.add(v))]

print(f"[{task}] Found {len(video_files)} video files")

if video_files:
    count = 0
    with open(output_jsonl, 'w') as f:
        for video_path in video_files:
            # Use video stem as key (no task prefix needed since assets are in separate dirs)
            key = video_path.stem
            entry = {
                "video": str(video_path),
                "key": key
            }
            f.write(json.dumps(entry) + '\n')
            count += 1
    
    print(f"[{task}] Generated {output_jsonl} with {count} entries")
    exit(0)  # Skip the metadata-based processing
else:
    print(f"[{task}] No video files found in {dataset_dir}, falling back to metadata processing")

# Fallback: Try loading with pandas first (for datasets without direct video files)
data_loaded = False
df = None

if parquet_files:
    import pandas as pd
    # Concatenate all parquet files
    dfs = []
    for pf in parquet_files:
        try:
            dfs.append(pd.read_parquet(pf))
        except Exception as e:
            print(f"Warning: Failed to load {pf}: {e}")
    if dfs:
        df = pd.concat(dfs, ignore_index=True)
        data_loaded = True
        print(f"Loaded {len(df)} samples from parquet files")

if not data_loaded and arrow_files:
    import pyarrow as pa
    import pandas as pd
    dfs = []
    for af in arrow_files:
        try:
            table = pa.ipc.open_file(af).read_all()
            dfs.append(table.to_pandas())
        except Exception as e:
            print(f"Warning: Failed to load {af}: {e}")
    if dfs:
        df = pd.concat(dfs, ignore_index=True)
        data_loaded = True
        print(f"Loaded {len(df)} samples from arrow files")

if not data_loaded:
    # Try using datasets library
    try:
        from datasets import load_from_disk, load_dataset
        ds = load_from_disk(dataset_dir)
        if hasattr(ds, split):
            ds = ds[split]
        df = ds.to_pandas()
        data_loaded = True
        print(f"Loaded {len(df)} samples using datasets library")
    except Exception as e:
        print(f"Failed to load with datasets library: {e}")

if not data_loaded:
    # Try loading from JSON files (e.g., MVBench uses json_data/*.json)
    import pandas as pd
    json_files = list(Path(dataset_dir).rglob("*.json"))
    print(f"Found {len(json_files)} json files")
    
    all_data = []
    for jf in json_files:
        try:
            with open(jf, 'r') as f:
                data = json.load(f)
            if isinstance(data, list):
                all_data.extend(data)
            elif isinstance(data, dict):
                # Some datasets have nested structure
                for k, v in data.items():
                    if isinstance(v, list):
                        all_data.extend(v)
        except Exception as e:
            print(f"Warning: Failed to load {jf}: {e}")
    
    if all_data:
        df = pd.DataFrame(all_data)
        data_loaded = True
        print(f"Loaded {len(df)} samples from JSON files")

if not data_loaded or df is None:
    print("[ERROR] Could not load dataset")
    exit(1)

print(f"Columns: {list(df.columns)}")

# Find video directory
video_dirs = []
for root, dirs, files in os.walk(dataset_dir):
    mp4_files = [f for f in files if f.endswith('.mp4')]
    if mp4_files:
        video_dirs.append(root)
        
if video_dirs:
    print(f"Found video directories: {video_dirs[:3]}...")
else:
    print("Warning: No video directories found with .mp4 files")

# Build video path lookup
video_lookup = {}
for vdir in video_dirs:
    for f in os.listdir(vdir):
        if f.endswith('.mp4'):
            video_lookup[f] = os.path.join(vdir, f)
            # Also add without extension
            video_lookup[f.replace('.mp4', '')] = os.path.join(vdir, f)

print(f"Built lookup with {len(video_lookup)} video files")

# Generate jsonl
count = 0
missing = 0
processed_keys = set()

with open(output_jsonl, 'w') as f:
    for idx, row in df.iterrows():
        # Get video identifier
        video_val = row.get(video_field)
        if video_val is None:
            # Try alternative field names
            for alt in ['video', 'video_id', 'videoID', 'video_path', 'video_name']:
                if alt in row and row[alt] is not None:
                    video_val = row[alt]
                    break
        
        if video_val is None:
            continue
        
        # Get key for deduplication
        key_val = row.get(key_field, video_val)
        if key_val in processed_keys:
            continue
        
        # Determine video path
        video_path = None
        video_val_str = str(video_val)
        
        # Check if it's already an absolute path
        if os.path.isabs(video_val_str) and os.path.isfile(video_val_str):
            video_path = video_val_str
        else:
            # Try different variations
            candidates = [
                video_val_str,
                video_val_str + '.mp4' if not video_val_str.endswith('.mp4') else video_val_str,
                os.path.basename(video_val_str),
                os.path.basename(video_val_str).replace('.mp4', ''),
            ]
            
            for cand in candidates:
                if cand in video_lookup:
                    video_path = video_lookup[cand]
                    break
        
        if video_path is None or not os.path.isfile(video_path):
            missing += 1
            if missing <= 5:
                print(f"Warning: Video not found: {video_val_str}")
            continue
        
        processed_keys.add(key_val)
        
        # Use key_val directly (no task prefix needed since assets are in separate dirs)
        key = str(key_val)
        
        entry = {
            "task": task,
            "split": split,
            "doc_id": len(processed_keys) - 1,
            "n": 0,
            "video": video_path,
            "key": key,
            "exists": True
        }
        f.write(json.dumps(entry) + '\n')
        count += 1

print(f"Generated {count} unique video entries")
if missing > 0:
    print(f"Warning: {missing} videos not found")
print(f"Output: {output_jsonl}")
EOF

    if [ ! -f "${INPUT_JSONL}" ]; then
        echo "[ERROR] Failed to generate input jsonl"
        return 1
    fi
    
    local JSONL_COUNT=$(wc -l < "${INPUT_JSONL}")
    echo "[Step 2/3] Done. Generated ${JSONL_COUNT} entries in ${INPUT_JSONL}"
    
    if [ "$JSONL_COUNT" -eq 0 ]; then
        echo "[WARNING] No video entries generated. Skipping Step 3."
        return 0
    fi
    
    # Run offline precompute
    echo ""
    echo "[Step 3/3] Running offline codec precompute..."
    echo "This may take a while depending on dataset size and number of workers..."
    
    cd "${REPO_DIR}"
    
    python Compressed_Video_Reader/tool/offline_precompute_llava_codec_assets.py \
        --jsonl "${INPUT_JSONL}" \
        --out_root "${OFFLINE_ROOT}" \
        --num_workers ${NUM_WORKERS} \
        --seq_len_frames ${SEQ_LEN_FRAMES} \
        --num_images ${NUM_IMAGES} \
        --square_size ${SQUARE_SIZE} \
        --patch_size ${PATCH_SIZE}
    
    echo ""
    echo "[${task}] Preprocessing Complete!"
    echo "Offline assets location: ${OFFLINE_ROOT}/assets"
    echo ""
}

# Main execution
echo "=============================================="
echo "Video Benchmark Offline Codec Preprocessing"
echo "=============================================="
echo "Cache directory: ${CACHE_DIR}"
echo ""

if [ "$TASK" = "all" ]; then
    echo "Processing all supported benchmarks..."
    for task in "${!TASK_CONFIGS[@]}"; do
        process_task "$task" || echo "[WARNING] Failed to process $task"
    done
else
    process_task "$TASK"
fi

echo ""
echo "=============================================="
echo "All Done!"
echo "=============================================="
echo ""
echo "Offline assets are ready. To run evaluation:"
echo ""
echo "  cd ${REPO_DIR}"
echo "  bash scripts/eval/local_eval_ov_encoder.sh"
echo ""
echo "The eval script will automatically detect and use offline assets."
echo "Just set TASKS to your benchmark (e.g., TASKS=\"mvbench\") in the script."
echo ""
echo "Offline assets location: ${CACHE_DIR}/<task>_offline/assets"
echo ""
