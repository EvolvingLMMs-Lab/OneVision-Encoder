MODEL_NAME=pretrain_encoder_base_patch16_224_v10_29_rms_head_ip 
CKPT_PATH=/video_vit/xiangan/checkpoint_llava_vit/continue_with_mlcd_1536_tokens_b16_mix/00078126/backbone.pt


# PORT=12340 OUTPUT=output/b16_frames   CUDA_VISIBLE_DEVICES=0 MODEL_FAMILY=llava_vit_64frames EMBEDDING_SIZE=768 NUM_FRAMES=64 PYTHONPATH=../ MODEL_NAME=$MODEL_NAME DATASETS=ssv2 CKPT_PATH=$CKPT_PATH bash video_attentive_probe.sh > output/b16_frames/ssv2.log &
# PORT=12341 OUTPUT=output/b16_frames   CUDA_VISIBLE_DEVICES=1 MODEL_FAMILY=llava_vit_64frames EMBEDDING_SIZE=768 NUM_FRAMES=64 PYTHONPATH=../ MODEL_NAME=$MODEL_NAME DATASETS=ucf101 CKPT_PATH=$CKPT_PATH bash video_attentive_probe.sh > output/b16_frames/ucf101.log &
# PORT=12342 OUTPUT=output/b16_frames   CUDA_VISIBLE_DEVICES=2 MODEL_FAMILY=llava_vit_64frames EMBEDDING_SIZE=768 NUM_FRAMES=64 PYTHONPATH=../ MODEL_NAME=$MODEL_NAME DATASETS=perception_test CKPT_PATH=$CKPT_PATH bash video_attentive_probe.sh > output/b16_frames/perception_test.log &
# PORT=12343 OUTPUT=output/b16_frames   CUDA_VISIBLE_DEVICES=3 MODEL_FAMILY=llava_vit_64frames EMBEDDING_SIZE=768 NUM_FRAMES=64 PYTHONPATH=../ MODEL_NAME=$MODEL_NAME DATASETS=hmdb51 CKPT_PATH=$CKPT_PATH bash video_attentive_probe.sh > output/b16_frames/hmdb51.log &
# PORT=12344 OUTPUT=output/b16_residual CUDA_VISIBLE_DEVICES=4 MODEL_FAMILY=llava_vit          EMBEDDING_SIZE=768 NUM_FRAMES=64 PYTHONPATH=../ MODEL_NAME=$MODEL_NAME DATASETS=ssv2 CKPT_PATH=$CKPT_PATH bash video_attentive_probe_ip.sh > output/b16_residual/ssv2.log &
# PORT=12345 OUTPUT=output/b16_residual CUDA_VISIBLE_DEVICES=5 MODEL_FAMILY=llava_vit          EMBEDDING_SIZE=768 NUM_FRAMES=64 PYTHONPATH=../ MODEL_NAME=$MODEL_NAME DATASETS=ucf101 CKPT_PATH=$CKPT_PATH bash video_attentive_probe_ip.sh > output/b16_residual/ucf101.log &
# PORT=12346 OUTPUT=output/b16_residual CUDA_VISIBLE_DEVICES=6 MODEL_FAMILY=llava_vit          EMBEDDING_SIZE=768 NUM_FRAMES=64 PYTHONPATH=../ MODEL_NAME=$MODEL_NAME DATASETS=perception_test CKPT_PATH=$CKPT_PATH bash video_attentive_probe_ip.sh > output/b16_residual/perception_test.log &
# PORT=12347 OUTPUT=output/b16_residual CUDA_VISIBLE_DEVICES=7 MODEL_FAMILY=llava_vit          EMBEDDING_SIZE=768 NUM_FRAMES=64 PYTHONPATH=../ MODEL_NAME=$MODEL_NAME DATASETS=hmdb51 CKPT_PATH=$CKPT_PATH bash video_attentive_probe_ip.sh > output/b16_residual/hmdb51.log &


mkdir -p output/two_input_b16_frames
mkdir -p output/two_input_b16_residual
mkdir -p output/two_input_b16_tiling


PORT=12344 OUTPUT=output/two_input_b16_tiling CUDA_VISIBLE_DEVICES=4 MODEL_FAMILY=llava_vit_tiling EMBEDDING_SIZE=768 NUM_FRAMES=64 PYTHONPATH=../ MODEL_NAME=$MODEL_NAME DATASETS=ssv2 CKPT_PATH=$CKPT_PATH bash video_attentive_probe.sh            > output/two_input_b16_tiling/ssv2.log &
PORT=12345 OUTPUT=output/two_input_b16_tiling CUDA_VISIBLE_DEVICES=5 MODEL_FAMILY=llava_vit_tiling EMBEDDING_SIZE=768 NUM_FRAMES=64 PYTHONPATH=../ MODEL_NAME=$MODEL_NAME DATASETS=ucf101 CKPT_PATH=$CKPT_PATH bash video_attentive_probe.sh          > output/two_input_b16_tiling/ucf101.log &
PORT=12346 OUTPUT=output/two_input_b16_tiling CUDA_VISIBLE_DEVICES=6 MODEL_FAMILY=llava_vit_tiling EMBEDDING_SIZE=768 NUM_FRAMES=64 PYTHONPATH=../ MODEL_NAME=$MODEL_NAME DATASETS=perception_test CKPT_PATH=$CKPT_PATH bash video_attentive_probe.sh > output/two_input_b16_tiling/perception_test.log &
PORT=12347 OUTPUT=output/two_input_b16_tiling CUDA_VISIBLE_DEVICES=7 MODEL_FAMILY=llava_vit_tiling EMBEDDING_SIZE=768 NUM_FRAMES=64 PYTHONPATH=../ MODEL_NAME=$MODEL_NAME DATASETS=hmdb51 CKPT_PATH=$CKPT_PATH bash video_attentive_probe.sh          > output/two_input_b16_tiling/hmdb51.log &
