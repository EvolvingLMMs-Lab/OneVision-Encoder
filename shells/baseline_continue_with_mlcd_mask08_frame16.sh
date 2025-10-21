torchrun -m --nproc_per_node 8 training.train_univit \
  --model_name pretrain_encoder_small_patch16_224_v10_12_rms_mask08_head \
  --num_frames 16 \
  --list_batch_sizes 64 64 \
  --lr 1e-4 \
  --list_datasets k710_ssv2_univit_pfs mlcd_coyo_laion \
  --list_init_partial_fc_paths NULL /video_vit/pretrain_models/deepglint/mlcd_3drope/vit_s_16/coyo_laion_%03d.npy \
  --init_backbone /vlm/xiangan/VideoMLCD/checkpoints/llava_vit_s_16.py/00190000/backbone.pt \
  --output /video_vit/xiangan/checkpoint_llava_vit/`basename $0 .sh` \
  --num_sampled_data 120000000
