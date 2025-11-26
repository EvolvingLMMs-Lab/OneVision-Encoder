export PYTHONPATH=$(pwd)
export HF_HOME=/vlm/huggingface

model_path="/video_vit/xiangan/checkpoint_llava_next/llavanext-_video_vit_xiangan_checkpoint_llava_vit_2025_11_22_new_l14_continue_128gpus_how_to_100m_448px_224px_00148000_backbone_hevc_vit_hf-_vlm_pretrain_models_Qwen_Qwen2.5-7B-Instruct-mlp2x_gelu-pretrain_blip558k-finetune_llavanext780k-10nodes/checkpoint-2500/"

conv_template='qwen_1_5'
run_port=12444
model_name='llava_vit'


# infovqa,ai2d,gqa,livexiv_vqa_v1,pope,scienceqa_img,mmstar,realworldqa
# infovqa,ai2d,gqa,livexiv_vqa_v1,pope,scienceqa_img,mmstar,realworldqa
# chartqa,docvqa,textvqa,mmbench
# infovqa,chartqa,docvqa,textvqa,ocrbench,livexiv_vqa_v1,ai2d,gqa,mmbench,pope,scienceqa_img,mmstar,realworldqa

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m accelerate.commands.launch \
    --main_process_port=$run_port \
    --num_processes=8 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained=$model_path,conv_template=$conv_template \
    --batch_size 1 \
    --tasks mme,ocrbench \
    --log_samples \
    --log_samples_suffix $model_name \
    --output_path ./eval_log/ 


