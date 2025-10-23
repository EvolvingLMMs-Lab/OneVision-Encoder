
# 例如设置最小丢弃 15%，第一帧不参与比例计算且不丢弃
export RES_MIN_DROP_RATIO=0.50

# 保存整段序列可视化
export VIZ_MASK=1
export VIZ_MASK_FRAMES=all
export VIZ_MASK_INTERVAL=1
export VIZ_MASK_SAMPLES=1
# export LLAVA_OUTPUT_DIR=/video_vit/yunyaoyan/Check_Code/LLaVA-ViT/checkpoints/mask
export UMT_HEVC_Y_ONLY=1 

# CUDA_VISIBLE_DEVICES=2 torchrun --master_port 29501 --nproc_per_node 1 -m training.train_univit_add_weight \
#   --model_name pretrain_encoder_small_patch16_224_v10_12_rms_mask05_head_ip \
#   --list_batch_sizes 64 64 \
#   --lr 1e-4 \
#   --list_datasets k710_ssv2_univit_pfs mlcd_coyo_laion \
#   --list_init_partial_fc_paths NULL /video_vit/pretrain_models/deepglint/mlcd_3drope/vit_s_16/coyo_laion_%03d.npy \
#   --init_backbone /vlm/xiangan/VideoMLCD/checkpoints/llava_vit_s_16.py/00190000/backbone.pt \
#   --output /video_vit/feilong/Check_Code/LLaVA-ViT/checkpoints/1021 \
#   --num_sampled_data 120000000
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --master_port 29501 --nproc_per_node 8 -m training.train_univit_add_weight \
  --model_name pretrain_encoder_small_patch16_224_v10_12_rms_mask05_head_ip \
  --list_batch_sizes 64 64 \
  --lr 1e-4 \
  --list_datasets k710_ssv2_univit_pfs mlcd_coyo_laion \
  --list_init_partial_fc_paths NULL /video_vit/pretrain_models/deepglint/mlcd_3drope/vit_s_16/coyo_laion_%03d.npy \
  --init_backbone /vlm/xiangan/VideoMLCD/checkpoints/llava_vit_s_16.py/00190000/backbone.pt \
  --output /video_vit/feilong/Check_Code/LLaVA-ViT/checkpoints/bidir_IP_10_23 \
  --num_sampled_data 120000000
  # --output /video_vit/yunyaoyan/Check_Code/LLaVA-ViT/checkpoints/1021 \
  # --num_sampled_data 120000000
  # --mask_debug_only 1 \
  # --max_debug_steps 1 \
  # --mask_log_topk 1

  #  mlcd_coyo_laion 
  #  /video_vit/pretrain_models/deepglint/mlcd_3drope/vit_s_16/coyo_laion_%03d.npy