for i in {000..003}; do
    deepspeed --num_gpus=8 --num_nodes=12 --hostfile=hostfile step1_extract_video_features.py \
        --input /video_vit/dataset/clips_HowTo100M_meta_llava_vit/list_all_valid_part_${i} \
        --output /video_vit/dataset/clips_HowTo100M_meta_llava_vit/output_list_all_valid_part_${i} \
        --batch_size 32 \
        --num_frames 8
done