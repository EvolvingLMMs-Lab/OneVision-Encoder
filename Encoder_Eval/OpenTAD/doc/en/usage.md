## Usage

By default, we use DistributedDataParallel (DDP) both in single GPU and multiple GPU cases for simplicity.

### Training

`torchrun --nnodes={num_node} --nproc_per_node={num_gpu} --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/train.py {config}`

- `num_node` is often set as 1 if all gpus are allocated in a single node. `num_gpu` is the number of used GPU.
- `config` is the path of the config file.

Example:

- Training feature-based ActionFormer on 1 GPU.
```bash
torchrun \
    --nnodes=1 \
    --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    tools/train.py configs/actionformer/thumos_i3d.py
```

- Training end-to-end-based AdaTAD on 4 GPUs within 1 node.
```bash
torchrun \
    --nnodes=1 \
    --nproc_per_node=4 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    tools/train.py configs/adatad/e2e_anet_videomae_s_adapter_frame768_img160.py
```

Note:
- **GPU number would affect the detection performance in most cases.** Since TAD dataset is small, and the number of ground truth actions per video differs dramatically in different videos. Therefore, the recommended setting for training feature-based TAD is 1 GPU, empirically.
- By default, evaluation is also conducted in the training, based on the argument in the config file. You can disable this, or increase the evaluation interval to speed up the training. 

### Inference and Evaluation

`torchrun --nnodes={num_node} --nproc_per_node={num_gpu} --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/test.py {config} --checkpoint {path}`

- if `checkpoint` is not specified, the `best.pth` in the config's result folder will be used.


Example:

- Inference and Evaluate ActionFormer on 1 GPU.
```bash
torchrun \
    --nnodes=1 \
    --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    tools/test.py \
    configs/actionformer/thumos_i3d.py \
    --checkpoint exps/thumos/actionformer_i3d/gpu1_id0/checkpoint/epoch_34.pth
```
