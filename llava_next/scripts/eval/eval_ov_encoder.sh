export PYTHONPATH=$(pwd):$(pwd)/lmms-eval

MODEL_PATH="trained_model/must_contain_llava_in_name"
RUN_PORT=12453
MODEL_NAME='llava_ov_encoder'
CONV_TEMPLATE='qwen_1_5'

TASKS="ai2d,chartqa,infovqa_val,mmbench_en_dev,mmstar,realworldqa,ocrbench,docvqa_val,videomme,perceptiontest_val_mc,temporalbench_long_qa,video_mmmu,mvbench,nextqa_mc_test"

# Run the evaluation script with the specified parameters
python -m accelerate.commands.launch \
    --main_process_port=$RUN_PORT \
    --num_processes=8 \
    -m lmms_eval \
    --model llava_ov_encoder \
    --model_args pretrained=$MODEL_PATH,conv_template=$CONV_TEMPLATE \
    --tasks $TASKS \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix ${MODEL_NAME}_$(date +%Y%m%d) \
    --output_path ./eval_log/${MODEL_NAME}/
