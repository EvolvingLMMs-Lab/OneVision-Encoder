<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="asset/logo_dark.png">
    <source media="(prefers-color-scheme: light)" srcset="asset/logo_light.png">
    <img alt="OneVision Encoder" src="output/logo.png" width="600" style="max-width: 100%;">
  </picture>
</p>

<p align="center">
  <strong>HEVC-Style Vision Transformer</strong>
</p>

## 📖 Table of Contents

- [Introduction](#-introduction)
- [Setup](#-setup)
- [Quick Start](#-quick-start)
- [Training](#-training)
- [Evaluation](#-evaluation)
- [Packing ViT Model](#-packing-vit-model)
- [Contributors](#-contributors)
- [License](#-license)

---

## 🔍 Introduction

Video understanding models face a fundamental trade-off: processing more frames captures richer temporal information but increases computation quadratically. Traditional approaches address this through sparse frame sampling, but this discards fine-grained motion dynamics and treats all spatial regions equally—wasting computation on static backgrounds.

We present OneVision Encoder, a vision transformer that resolves this trade-off using principles from HEVC video compression. Instead of sampling sparse frames densely (all patches from few frames), we sample dense frames sparsely (important patches from many frames). Our codec-style patch selection identifies temporally-salient regions—areas with motion, object interactions, or semantic changes—and processes only these informative patches.

Combined with global contrastive learning using a 2M concept bank, OneVision Encoder achieves state-of-the-art results on video benchmarks (MVBench, VideoMME, Perception Test) and image understanding tasks (DocVQA, ChartQA, OCRBench).

### Method Overview

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/anxiangsir/asset/main/OneVision/method_github_dark.png">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/anxiangsir/asset/main/OneVision/method_github_light.png">
    <img alt="OneVision Encoder Method Overview" src="https://raw.githubusercontent.com/anxiangsir/asset/main/OneVision/method_github_light.png" width="800" style="max-width: 100%;">
  </picture>
</p>

### Cluster Discrimination Visualization

Standard contrastive learning (e.g., CLIP) is limited by batch size—negative samples are drawn only from the current batch, typically 32K-64K examples. This creates a narrow view of the embedding space and leads to suboptimal representations. Our approach maintains a global concept bank of 2M clustered centers, enabling each training sample to contrast against a diverse, representative set of negatives regardless of batch composition. This produces more discriminative embeddings with better-separated semantic clusters.


<p align="center">
  <img src="pages/images/global_contrastive_comparison.gif" alt="Global Contrastive Comparison" width="800" style="max-width: 100%;">
</p>

### Video Processing Pipeline

The visualization below demonstrates our complete video processing pipeline. The animation shows four key stages: (1) Original Video - a continuous 64-frame stream capturing the full temporal context, (2) Uniform Frame Sampling - traditional approach selecting 4-8 evenly-spaced frames, which is simple but lossy and misses inter-frame motion, (3) Temporal Saliency Detection - analysis of all 64 frames to identify regions with high temporal information such as motion, appearance changes, and semantic events, and (4) Codec-Style Patch Extraction - extraction of only the salient patches in zigzag order, achieving 75-98% compression while preserving temporal dynamics.

<table>
  <tr>
    <td align="center">
      <img src="https://raw.githubusercontent.com/anxiangsir/asset/main/OneVision/case4.gif" alt="Case 4 Demonstration" width="800"><br>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="https://raw.githubusercontent.com/anxiangsir/asset/main/OneVision/case5.gif" alt="Case 5 Demonstration" width="800"><br>
    </td>
  </tr>
</table>

### Pre-training Tips

1. **Scale-up is the final step** - Maximize model capabilities before scaling, and ensure generalization phenomena emerge
2. **Avoid direct supervision from existing models** - Indirect usage is preferred over direct distillation, which may limit scaling capabilities
3. **Progressive training when resources are limited** - Start with low resolution/frame rate, then gradually fine-tune to higher settings (ref: CLIPA)

---

### Attentive Probe Results

Performance comparison of different vision encoders using Attentive Probe evaluation. Models are evaluated using single clip input and trained for 10 epochs across 8 action recognition datasets. Results show average performance and per-dataset scores for 8-frame and 16-frame configurations.

<p align="center">
  <img src="asset/result_ap.png" alt="AP" width="800" style="max-width: 100%;">
</p>

### LMM Probe Results

Training on a mixed dataset of 740K samples from LLaVA-OneVision and 800K samples from LLaVA-Video SFT. The training pipeline proceeds directly to Stage 2 fine-tuning. We adopt a streamlined native-resolution strategy inspired by LLaVA-OneVision: when the input frame resolution matches the model's native input size, it is fed directly—without tiling or cropping—to evaluate the ViT's native resolution capability.



## 🔧 Setup

### Prerequisites

- Docker with NVIDIA GPU support
- CUDA-compatible GPU(s)

### Mount Data Storage (Optional)

If using shared storage for datasets, mount your NFS/CFS volumes:

```bash
mkdir -p /video_vit /vlm
mount -t nfs4 -o minorversion=1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2,noresvport <your-nfs-server>:/ /video_vit
mount -t nfs4 -o minorversion=1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2,noresvport <your-nfs-server>:/ /vlm
```

> [!NOTE]
> Replace `<your-nfs-server>` with your actual storage endpoint. Internal users should refer to the internal documentation for specific mount configurations.

### Docker Build

#### Option 1: Build from Dockerfile

```bash
docker build -t llava_vit:25.11 .
```

#### Option 2: Load Pre-built Docker Image

```bash
docker load -i /video_vit/docker_images/llava_vit_tag_25.11.22.tar && \
docker tag $(docker images -q | head -n 1) llava_vit:25.11.22
```

### Running the Container

#### Single Node

```bash
docker run -it --gpus all --ipc host --net host --privileged \
    -v "$(pwd)":/workspace/OneVision-Encoder \
    -w /workspace/OneVision-Encoder \
    llava_vit:25.11.22 bash
```

#### Multi Node

> [!IMPORTANT]
> 多机必须使用预编译的镜像，且镜像必须一致

```bash
docker run -it --gpus all --ipc host --net host --privileged --cap-add IPC_LOCK \
    --ulimit memlock=-1 --ulimit stack=67108864 --rm \
    -v "$(pwd)":/workspace/OneVision-Encoder \
    -v /train_tmp:/train_tmp \
    -v /vlm:/vlm -v /video_vit:/video_vit -v /rice_ocr:/rice_ocr \
    -v /data_0:/data_0 -v /data_1:/data_1 -v /data_2:/data_2 -v /data_3:/data_3 \
    -w /workspace/OneVision-Encoder \
    -e NCCL_TIMEOUT=1800 \
    -e CUDA_DEVICE_MAX_CONNECTIONS=1 \
    -e NCCL_SOCKET_IFNAME=eth0 \
    -e NCCL_IB_GID_INDEX=3 \
    -e NCCL_IB_DISABLE=0 \
    -e NCCL_IB_HCA="mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_1" \
    -e NCCL_NET_GDR_LEVEL=2 \
    -e NCCL_IB_QPS_PER_CONNECTION=4 \
    -e NCCL_IB_TC=160 \
    -e NCCL_IB_TIMEOUT=22 \
    -e NCCL_CROSS_NIC=1 \
    -e NCCL_MIN_NCHANNELS=8 \
    -e NCCL_MAX_NCHANNELS=16 \
    llava_vit:25.11.22 bash -c "service ssh restart; bash"
```

### Install Package

Inside the container, install the package in editable mode:

```bash
pip install -e .
```

---

## ⚡ Quick Start

> **Note:** This model supports native resolution input. For optimal performance:
> - **Image**: 448×448 resolution (pre-trained)
> - **Video**: 224×224 resolution with 256 tokens per frame (pre-trained)
>
> Use CLIP preprocessing from the [model repository](https://huggingface.co/lmms-lab/onevision-encoder-large).

```python
from transformers import AutoModel, AutoImageProcessor
from PIL import Image
import torch

# Load model and preprocessor
model = AutoModel.from_pretrained(
    "lmms-lab/onevision-encoder-large",
    trust_remote_code=True,
    attn_implementation="flash_attention_2"
).to("cuda").eval()

preprocessor = AutoImageProcessor.from_pretrained(
    "lmms-lab/onevision-encoder-large",
    trust_remote_code=True
)

# Image inference: [B, C, H, W]
image = Image.open("path/to/your/image.jpg")  # Replace with your image path
pixel_values = preprocessor(images=image, return_tensors="pt")["pixel_values"].to("cuda")
with torch.no_grad():
    outputs = model(pixel_values)
    # outputs.last_hidden_state: [B, num_patches, hidden_size]
    # outputs.pooler_output: [B, hidden_size]

# Video inference: [B, C, T, H, W] with visible_indices
num_frames, frame_tokens, target_frames = 16, 256, 64
# Load video frames and preprocess each frame (replace with your video frame paths)
frames = [Image.open(f"path/to/frame_{i}.jpg") for i in range(num_frames)]
video_pixel_values = preprocessor(images=frames, return_tensors="pt")["pixel_values"]
# Reshape from [T, C, H, W] to [B, C, T, H, W]
video = video_pixel_values.unsqueeze(0).permute(0, 2, 1, 3, 4).to("cuda")

# Build visible_indices for temporal sampling
frame_pos = torch.linspace(0, target_frames - 1, num_frames).long().cuda()
visible_indices = (frame_pos.unsqueeze(-1) * frame_tokens + torch.arange(frame_tokens).cuda()).reshape(1, -1)
# visible_indices example (with 256 tokens per frame):
#   Frame 0 (pos=0):  indices [0, 1, 2, ..., 255]
#   Frame 1 (pos=4):  indices [1024, 1025, 1026, ..., 1279]
#   Frame 2 (pos=8):  indices [2048, 2049, 2050, ..., 2303]
#   ...
#   Frame 15 (pos=63): indices [16128, 16129, ..., 16383]

with torch.no_grad():
    outputs = model(video, visible_indices=visible_indices)
```

### Codec Input

> **TODO:** Add codec-style input documentation for temporal saliency-based patch selection.

---

## 🚀 Training

### Single Node

```bash
torchrun --nproc_per_node 8 -m training.train_univit \
    --list_batch_size 64 \
    --output ./output/baseline
```

### Multi Node

For multi-node distributed training, configure your training script according to your cluster setup. See example scripts in the `shells/` directory.

---

## 📊 Evaluation

### Attentive Probe Evaluation

```bash
torchrun --nproc_per_node 8 --master_port 15555 \
    eval_encoder/attentive_probe.py \
    --eval_freq 1 \
    --default_lr_list 0.0003 \
    --batch_size 16 \
    --default_weight_decay 0 \
    --dali_py_num_workers 8 \
    --model_family llava_vit_sampling \
    --dataset ssv2
```

### Supported Evaluation Datasets

- SSv2 (Something-Something v2)
- UCF101
- And more...

---

## 📦 Packing ViT Model

To package a trained ViT model for distribution or deployment:

```bash
python -m tools.pack_model \
    --checkpoint ./output/baseline/checkpoint.pt \
    --output ./output/packed_model
```

The packed model can be loaded directly with HuggingFace Transformers:

```python
from onevision_encoder import OneVisionEncoderModel

model = OneVisionEncoderModel.from_pretrained("./output/packed_model")
```

---

## 👥 Contributors

<!-- Add contributor list here -->

---

## 📄 License

This project is released under the Apache 2.0 License.
