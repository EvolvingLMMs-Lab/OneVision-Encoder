# python process_test.py \
#     --data_set CHARADES \
#     --data_path /vlm/monash/feilong/OpenTAD-main/data/charades/raw_data/Charades_v1_480_30fps \
#     --save_path /vlm/monash/feilong/OpenTAD-main/data/charades/features/videomae_base \
#     --model_type videomae \
#     --ckpt_path MCG-NJU/videomae-base \
#     --world_size 8 \
#     --anno_file /vlm/monash/feilong/OpenTAD-main/data/charades/annotations/charades.json \


# python process_test.py \
#     --data_set CHARADES \
#     --data_path /vlm/monash/feilong/OpenTAD-main/data/charades/raw_data/Charades_v1_480_30fps \
#     --save_path /vlm/monash/feilong/OpenTAD-main/data/charades/features/videomae_large \
#     --model_type videomae \
#     --ckpt_path MCG-NJU/videomae-large \
#     --world_size 15 \
#     --anno_file /vlm/monash/feilong/OpenTAD-main/data/charades/annotations/charades.json \

# python process.py \
#     --data_set CHARADES \
#     --data_path /vlm/monash/feilong/OpenTAD-main/data/charades/raw_data/Charades_v1_480_30fps \
#     --save_path /vlm/monash/feilong/OpenTAD-main/data/charades/features/videomae_large_pth \
#     --model_name vit_large_patch16_224 \
#     --model_type videomae_v1 \
#     --ckpt_path /vlm/monash/feilong/OpenTAD-main/pretrained/checkpoint.pth \
#     --world_size 15 \
#     --anno_file /vlm/monash/feilong/OpenTAD-main/data/charades/annotations/charades.json \

python process_test.py \
    --data_set charades \
    --data_path /vlm/monash/feilong/OpenTAD-main/data/charades/raw_data/video \
    --save_path /vlm/monash/feilong/OpenTAD-main/data/charades/features/siglip_base \
    --model_type siglip \
    --ckpt_path google/siglip-base-patch16-224 \
    --world_size 4 \
    --anno_file /vlm/monash/feilong/OpenTAD-main/data/charades/annotations/charades.json \

python process_test.py \
    --data_set charades \
    --data_path /vlm/monash/feilong/OpenTAD-main/data/charades/raw_data/video \
    --save_path /vlm/monash/feilong/OpenTAD-main/data/charades/features/siglip_so \
    --model_type siglip \
    --ckpt_path google/siglip-so400m-patch14-224 \
    --world_size 4 \
    --anno_file /vlm/monash/feilong/OpenTAD-main/data/charades/annotations/charades.json \

python process_test.py \
    --data_set charades \
    --data_path /vlm/monash/feilong/OpenTAD-main/data/charades/raw_data/video \
    --save_path /vlm/monash/feilong/OpenTAD-main/data/charades/features/siglip_large \
    --model_type siglip \
    --ckpt_path google/siglip-large-patch16-256 \
    --world_size 4 \
    --anno_file /vlm/monash/feilong/OpenTAD-main/data/charades/annotations/charades.json \