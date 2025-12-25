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
- [Documentation](#-documentation)

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
    <img alt="OneVision Encoder Method Overview" src="https://raw.githubusercontent.com/anxiangsir/asset/main/OneVision/method_github_light.png" width="900" style="max-width: 100%;">
  </picture>
</p>

### Video Processing Pipeline

The visualization below demonstrates our complete video processing pipeline. The animation shows four key stages: (1) Original Video - a continuous 64-frame stream capturing the full temporal context, (2) Uniform Frame Sampling - traditional approach selecting 4-8 evenly-spaced frames, which is simple but lossy and misses inter-frame motion, (3) Temporal Saliency Detection - analysis of all 64 frames to identify regions with high temporal information such as motion, appearance changes, and semantic events, and (4) Codec-Style Patch Extraction - extraction of only the salient patches in zigzag order, achieving 75-98% compression while preserving temporal dynamics.

<div align="center">
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
</div>

### Cluster Discrimination Visualization

Standard contrastive learning (e.g., CLIP) is limited by batch size—negative samples are drawn only from the current batch, typically 32K-64K examples. This creates a narrow view of the embedding space and leads to suboptimal representations. Our approach maintains a global concept bank of 2M clustered centers, enabling each training sample to contrast against a diverse, representative set of negatives regardless of batch composition. This produces more discriminative embeddings with better-separated semantic clusters.


<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/anxiangsir/asset/main/OneVision/loss_github_dark.gif">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/anxiangsir/asset/main/OneVision/loss_github_light.gif">
    <img alt="Training Loss Visualization" src="https://raw.githubusercontent.com/anxiangsir/asset/main/OneVision/loss_github_light.gif" width="800" style="max-width: 100%;">
  </picture>
</p>

### Pre-training Tips

1. **Scale-up is the final step** - Maximize model capabilities before scaling, and ensure generalization phenomena emerge
2. **Avoid direct supervision from existing models** - Indirect usage is preferred over direct distillation, which may limit scaling capabilities
3. **Progressive training when resources are limited** - Start with low resolution/frame rate, then gradually fine-tune to higher settings (ref: CLIPA)

---


### LMM Probe Results

Training on a mixed dataset of 740K samples from LLaVA-OneVision and 800K samples from LLaVA-Video SFT. The training pipeline proceeds directly to Stage 2 fine-tuning. We adopt a streamlined native-resolution strategy inspired by LLaVA-OneVision: when the input frame resolution matches the model's native input size, it is fed directly—without tiling or cropping—to evaluate the ViT's native resolution capability.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/anxiangsir/asset/main/OneVision/probe_lmm_github_dark_fixed.png">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/anxiangsir/asset/main/OneVision/probe_lmm_github_light.png">
    <img alt="LMM Probe Results" src="https://raw.githubusercontent.com/anxiangsir/asset/main/OneVision/probe_lmm_github_light.png" width="800" style="max-width: 100%;">
  </picture>
</p>

### Attentive Probe Results

Performance comparison of different vision encoders using Attentive Probe evaluation. Models are evaluated using single clip input and trained for 10 epochs across 8 action recognition datasets. Results show average performance and per-dataset scores for 8-frame and 16-frame configurations.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/anxiangsir/asset/main/OneVision/fix_00_probe_video_github_dark.png">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/anxiangsir/asset/main/OneVision/fix_00_probe_video_github_light.png">
    <img alt="LMM Probe Results" src="https://raw.githubusercontent.com/anxiangsir/asset/main/OneVision/probe_lmm_github_light.png" width="900" style="max-width: 100%;">
  </picture>
</p>





## 🔧 Setup

### Prerequisites

- Docker with NVIDIA GPU support
- CUDA-compatible GPU(s)

### Docker Build

#### Option 1: Build from Dockerfile

```bash
docker build -t ov_encoder:25.12 .
```

### Running the Container

#### Single Node

```bash
docker run -it --gpus all --ipc host --net host --privileged \
    -v "$(pwd)":/workspace/OneVision-Encoder \
    -w /workspace/OneVision-Encoder \
    ov_encoder:25.12 bash
```

#### Multi Node

> [!IMPORTANT]
> 多机必须使用预编译的镜像，且镜像必须一致

```bash
docker run -it --gpus all --ipc host --net host --privileged --cap-add IPC_LOCK \
    --ulimit memlock=-1 --ulimit stack=67108864 --rm \
    -v "$(pwd)":/workspace/OneVision-Encoder \
    -w /workspace/OneVision-Encoder \
    -e NCCL_TIMEOUT=1800 \
    -e CUDA_DEVICE_MAX_CONNECTIONS=1 \
    ov_encoder:25.12 bash -c "service ssh restart; bash"
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

Add codec-style input documentation for temporal saliency-based patch selection.

---

## 🚀 Training

### Single Node & Multi Node

Training configurations and hyperparameters will be documented soon.  For now, please refer to `--help` for available options.

## 📊 Evaluation

### Attentive Probe Evaluation


#### Chunk-wise Sampling Evaluation

To evaluate the encoder with uniform frame sampling, first navigate to the evaluation directory:

```bash
cd eval_encoder
```

Then run the following command:

```bash
torchrun --nproc_per_node=8 --master_port=29507 attentive_probe.py \
  --eval_freq 1 \
  --default_lr_list 0.0001 \
  --batch_size 32 \
  --default_weight_decay 0 \
  --dali_py_num_workers 8 \
  --model_family llava_vit_sampling \
  --dataset diving48 \
  --num_frames 8 \
  --model_weight lmms-lab/onevision-encoder-large \
  --model_name hf_llava_vit_large_ln \
  --embedding_size 1024 \
  --frames_token_num 256
```

**Sampling-Specific Parameters:**
- `frames_token_num`: Number of tokens per frame (e.g., 256 tokens for standard sampling).

#### OV-Encoder Codec Evaluation

To evaluate the encoder with codec-style patch selection, first navigate to the evaluation directory:

```bash
cd eval_encoder
```

Then run the following command:

```bash
torchrun --nproc_per_node=8 --master_port=29512 attentive_prob_codec.py \
  --eval_freq 1 \
  --default_lr_list 0.0001 \
  --batch_size 4 \
  --default_weight_decay 0 \
  --dali_py_num_workers 8 \
  --model_family llava_vit_codec \
  --dataset diving48 \
  --num_frames 64 \
  --model_weight lmms-lab/onevision-encoder-large \
  --model_name hf_llava_vit_large_ln \
  --embedding_size 1024 \
  --default_epoch 30 \
  --data_root /path/to/your/data_attentive_probe/ \
  --cache_dir /path/to/your/cache_residuals/ \
  --K_keep 2048 \
  --mv_compensate median
```

**Codec-Specific Parameters:**
- `cache_dir`: Directory for cached codec patches. This is where the codec-selected patches will be stored/loaded.
- `K_keep`: Number of patches to keep. For example, 256 patches per frame × 8 frames = 2048 total patches. Adjust based on your frame count and desired compression ratio.
- `mv_compensate`: Motion vector compensation method (e.g., `median`).

#### Shared Parameters

The following parameters are common to both evaluation methods:

- `dataset`: Dataset to evaluate on (e.g., `diving48`, `ssv2`, `kinetics400`). Prepare the dataset according to the Attentive Probe format.
- `num_frames`: Total number of frames in the video sequence (e.g., 8 for sampling, 64 for codec).
- `model_weight`: Path to the pre-trained model. Use `lmms-lab/onevision-encoder-large` to load directly from HuggingFace, or provide a local path.
- `model_name`: Model architecture name (e.g., `hf_llava_vit_large_ln`).
- `embedding_size`: Size of the embedding dimension (e.g., 1024).
- `batch_size`: Training batch size (varies by evaluation type).
- `default_lr_list`: Learning rate for the probe training.
- `default_weight_decay`: Weight decay for optimization.
- `eval_freq`: Evaluation frequency during training.
- `dali_py_num_workers`: Number of DALI data loading workers.
- `data_root`: Root directory containing your prepared dataset (codec evaluation only).



## 👥 Contributors

<!-- Add contributor list here -->

---

## 📄 License

This project is released under the Apache 2.0 License.

---

## 📚 Documentation

- [Model Card](docs/model_card.md) - Detailed documentation for OneVision Encoder Large model
- [Data Card](docs/datacard.md) - Training dataset information and statistics
