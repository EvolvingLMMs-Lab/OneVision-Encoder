#!/usr/bin/env bash
set -euo pipefail

# 固定参数
MODEL_FAMILY=llava_vit
EMBEDDING_SIZE=768
NUM_FRAMES=64
PYTHONPATH=../
DATASETS=perception_test

# 需要运行的 NUM_TARGET 列表（可按需增删）
targets=(192 392 588 784 980 1176 1372 1568)


# 起始端口与 GPU
BASE_PORT=12350
GPU_START=0

# 可从环境传入，也可在此处直接赋值
MODEL_NAME=pretrain_encoder_base_patch16_224_v10_29_rms_head_ip 
CKPT_PATH=/video_vit/xiangan/checkpoint_llava_vit/continue_with_mlcd_1536_tokens_b16_mix_three_input/00078126/backbone.pt

for i in "${!targets[@]}"; do
  target="${targets[$i]}"
  port=$((BASE_PORT + i))
  gpu=$((GPU_START + i))
  output="output/three_input_b16_residual_multi_frames_${target}"

  mkdir -p "${output}"

  echo "Launching NUM_TARGET=${target} on GPU ${gpu}, PORT ${port}, OUTPUT=${output}"

  PORT="${port}" \
  OUTPUT="${output}" \
  NUM_TARGET="${target}" \
  CUDA_VISIBLE_DEVICES="${gpu}" \
  MODEL_FAMILY="${MODEL_FAMILY}" \
  EMBEDDING_SIZE="${EMBEDDING_SIZE}" \
  NUM_FRAMES="${NUM_FRAMES}" \
  PYTHONPATH="${PYTHONPATH}" \
  MODEL_NAME="${MODEL_NAME}" \
  DATASETS="${DATASETS}" \
  CKPT_PATH="${CKPT_PATH}" \
  bash video_attentive_probe_ip.sh > "${output}/${DATASETS}_test.log" &
  # 如果你想保持与原来一致，全部写同一个日志文件，请改为：
  # bash video_attentive_probe_ip.sh > output/three_input_b16_residual_multi_frames/perception_test.log &
done

wait
echo "All jobs launched."