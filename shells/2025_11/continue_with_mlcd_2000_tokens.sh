
# 例如设置最小丢弃 15%，第一帧不参与比例计算且不丢弃
export RES_MIN_DROP_RATIO=0.50

# 保存整段序列可视化
export HEVC_SHARDS=1
export VIZ_MASK=1
export VIZ_MASK_FRAMES=all
export VIZ_MASK_INTERVAL=1
export VIZ_MASK_SAMPLES=1
# export LLAVA_OUTPUT_DIR=/video_vit/yunyaoyan/Check_Code/LLaVA-ViT/checkpoints/mask
export UMT_HEVC_Y_ONLY=1 


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --master_port 29508 --nproc_per_node 8 -m training.train_univit_11_06_ip \
  --model_name pretrain_encoder_small_patch16_224_v10_29_rms_head_ip \
  --list_batch_sizes 64 64 \
  --lr 1e-4 \
  --list_datasets k710_ssv2_univit_pfs_fix_ip_fix_size mlcd_coyo_laion \
  --list_init_partial_fc_paths NULL /video_vit/pretrain_models/deepglint/mlcd_3drope/vit_s_16/coyo_laion_%03d.npy \
  --init_backbone /vlm/xiangan/VideoMLCD/checkpoints/llava_vit_s_16.py/00190000/backbone.pt \
  --output /video_vit/xiangan/checkpoint_llava_vit/`basename $0 .sh` \
  --num_sampled_data 120000000
