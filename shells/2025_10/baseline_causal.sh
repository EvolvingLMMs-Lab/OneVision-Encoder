torchrun -m --nproc_per_node 8 training.train_univit \
  --list_batch_sizes 64 \
  --output ./output/baseline_causal \
  --model_name pretrain_encoder_small_patch16_224_v10_12_rms_unmask_with_head_causal