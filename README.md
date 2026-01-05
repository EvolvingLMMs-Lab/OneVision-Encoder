<!-- <p align="center">
  <img alt="OneVision Encoder" src="asset/onevision_encoder.png" width="1200" style="max-width: 100%;">
</p> -->
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="asset/logo_dark.png">
  <source media="(prefers-color-scheme: light)" srcset="asset/logo_light.png">
  <img alt="OneVision Encoder" src="output/logo.png" width="1200" style="max-width: 100%;">
</picture>

<p align="center">
  <strong>The first HEVC style Vision Transformer with advanced multimodal capabilities</strong>
</p>

<div align="center">

📝 **[Homepage](https://www.lmms-lab.com/onevision-encoder/index.html)**
🤗 **[Models](https://huggingface.co/lmms-lab-encoder/onevision-encoder-large)** |
📄 **[Tech Report (coming)]()** |
📋 **[Model Card](docs/model_card.md)** |
📊 **[Data Card](docs/data_card.md)**

</div>

## 📖 Table of Contents

- [Introduction](#-introduction)
- [Setup](#-setup)
- [Quick Start](#-quick-start)
- [Training](#-training)
- [Evaluation](#-evaluation)
- [Contributors](#-contributors)
- [License](#-license)
- [Documentation](#-documentation)

---

## 🔍 Introduction

Video understanding models face a fundamental trade-off: incorporating more frames enables richer temporal reasoning but increases computational cost quadratically.
Conventional approaches mitigate this by sparsely sampling frames, however, this strategy discards fine-grained motion dynamics and treats all spatial regions uniformly, resulting in wasted computation on static content.

We introduce OneVision Encoder, a vision transformer that resolves this trade-off by drawing inspiration from HEVC (High-Efficiency Video Coding). Rather than densely processing all patches from a few frames, OneVision Encoder sparsely selects informative patches from many frames. This codec-inspired patch selection mechanism identifies temporally salient regions (e.g., motion, object interactions, and semantic changes) and allocates computation exclusively to these informative areas.

Coupled with global contrastive learning over a 2M-scale concept memory bank, OneVision Encoder achieves state-of-the-art performance across major video benchmarks (MVBench, VideoMME, Perception Test), while also delivering strong results on image understanding tasks (DocVQA, ChartQA, and OCRBench).


### Key Features

- **Unified Vision Foundation**: A single base model for consistent understanding of images, videos, and OCR.
- **Codec-Style Patch Selection**: Instead of sampling sparse frames densely (all patches from few frames), OneVision Encoder samples dense frames sparsely (important patches from many frames).
- **3D Rotary Position Embedding**: Uses a 4:6:6 split for temporal, height, and width dimensions to capture spatiotemporal relationships.
- **Global Contrastive Learning**: Trained with a 2M concept bank for better-separated semantic clusters.
- **Native Resolution Support**: Supports native resolution input without tiling or cropping.
- **Open Training Data & Pipeline**: In addition to the model and code, we will open-source the curated training dataset and the full data processing pipeline.

### Method Overview

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/anxiangsir/asset/main/OneVision/method_github_dark.png">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/anxiangsir/asset/main/OneVision/method_github_light.png">
    <img alt="OneVision Encoder Method Overview" src="https://raw.githubusercontent.com/anxiangsir/asset/main/OneVision/method_github_light.png" width="900" style="max-width: 100%;">
  </picture>
</p>

### Video Processing Pipeline

The visualization below illustrates four different video processing pipelines.

**1. Original Video**: a continuous 64-frame sequence that preserves the complete temporal context.

**2. Uniform Frame Sampling**: a conventional strategy that selects 4–8 evenly spaced frames; while simple and efficient, it is inherently lossy and fails to capture fine-grained inter-frame motion.

**3. Temporal Saliency Detection**: a global analysis of all 64 frames to identify regions rich in temporal information, including motion patterns, appearance variations, and semantic events.

**4. Codec-Style Patch Extraction**: selective extraction of the temporally salient patches in a zigzag order, achieving 75–98% compression while retaining critical temporal dynamics.

<div align="center">
<table style="width: 100%; max-width: 1200px; table-layout: fixed;">
  <tr>
    <th style="width: 25%;">(1) </th>
    <th style="width: 25%;">(2) </th>
    <th style="width: 25%;">(3) </th>
    <th style="width: 25%;">(4) </th>
  </tr>

  <tr>
    <td colspan="4" align="center">
      <img src="https://raw.githubusercontent.com/anxiangsir/asset/main/OneVision/gifs/case4.gif" alt="Case 4 Demonstration" width="800"><br>
    </td>
  </tr>
  <tr>
    <td colspan="4" align="center">
      <img src="https://raw.githubusercontent.com/anxiangsir/asset/main/OneVision/gifs/case6.gif" alt="Case 6 Demonstration" width="800"><br>
    </td>
  </tr>
</table>
</div>

### Cluster Discrimination Visualization

Standard contrastive learning methods (e.g., CLIP) are fundamentally constrained by batch size, as negative samples are drawn only from the current batch, typically limited to 32K–64K examples. This restriction yields a narrow and incomplete view of the embedding space, often resulting in suboptimal representation learning. In contrast, our approach maintains a global concept bank comprising 2M clustered centers, allowing each training sample to contrast against a diverse and representative set of negatives independent of batch composition. This global contrasting mechanism leads to more discriminative embeddings and well-separated semantic clusters.


<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/anxiangsir/asset/main/OneVision/loss_github_dark.gif">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/anxiangsir/asset/main/OneVision/loss_github_light.gif">
    <img alt="Training Loss Visualization" src="https://raw.githubusercontent.com/anxiangsir/asset/main/OneVision/loss_github_light.gif" width="800" style="max-width: 100%;">
  </picture>
</p>


---


### LMM Probe Results

We train the model on a mixed dataset comprising 740K samples from LLaVA-OneVision and 800K samples from LLaVA-Video SFT, proceeding directly to Stage-2 fine-tuning. Following a streamlined native-resolution strategy inspired by LLaVA-OneVision, input frames that match the model’s native resolution are fed directly into the network without tiling or cropping, allowing us to fully evaluate the ViT’s native-resolution modeling capability.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/anxiangsir/asset/main/OneVision/probe_lmm_github_dark_fixed.png">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/anxiangsir/asset/main/OneVision/probe_lmm_github_light.png">
    <img alt="LMM Probe Results" src="https://raw.githubusercontent.com/anxiangsir/asset/main/OneVision/probe_lmm_github_light.png" width="800" style="max-width: 100%;">
  </picture>
</p>

### Attentive Probe Results

We compare the performance of different vision encoders using the Attentive Probe evaluation protocol. All models are evaluated with a single-clip input and trained for 10 epochs across eight action recognition datasets. The results report both the average accuracy and per-dataset performance under 8-frame and 16-frame input configurations.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/anxiangsir/asset/main/OneVision/fix_00_probe_video_github_dark.png">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/anxiangsir/asset/main/OneVision/fix_00_probe_video_github_light.png">
    <img alt="LMM Probe Results" src="https://raw.githubusercontent.com/anxiangsir/asset/main/OneVision/probe_lmm_github_light.png" width="900" style="max-width: 100%;">
  </picture>
</p>



## ⚡ Quick Start

> [!IMPORTANT]
> **Transformers Version Compatibility:**
> - ✅ **`transformers==4.53.1`** (Recommended): Works with `AutoModel.from_pretrained()` 
> - ⚠️ **`transformers>=5.0.0`**: Not currently supported. We are actively working on a fix.


> **Note:** This model supports native resolution input. For optimal performance:
> - **Image**: 448×448 resolution (pre-trained)
> - **Video**: 224×224 resolution with 256 tokens per frame (pre-trained)
>
> Use CLIP preprocessing from the [model repository](https://huggingface.co/lmms-lab-encoder/onevision-encoder-large).

### Using AutoModel (Recommended: transformers==4.53.1)

```python
from transformers import AutoModel, AutoImageProcessor
from PIL import Image
import torch

# Load model and preprocessor
model = AutoModel.from_pretrained(
    "lmms-lab-encoder/onevision-encoder-large",
    trust_remote_code=True,
    attn_implementation="flash_attention_2"
).to("cuda").eval()

preprocessor = AutoImageProcessor.from_pretrained(
    "lmms-lab-encoder/onevision-encoder-large",
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

### Loading from Source Code  

```bash
git clone https://github.com/EvolvingLMMs-Lab/OneVision-Encoder.git
cd OneVision-Encoder
pip install -e .
```

```python
from onevision_encoder import OneVisionEncoderModel, OneVisionEncoderConfig
from transformers import AutoImageProcessor
model = OneVisionEncoderModel.from_pretrained(
    "lmms-lab-encoder/onevision-encoder-large",
    trust_remote_code=True,
    attn_implementation="flash_attention_2"
).to("cuda").eval()
preprocessor = AutoImageProcessor.from_pretrained(
    "lmms-lab-encoder/onevision-encoder-large",
    trust_remote_code=True
)
```

### Codec Input

Add codec-style input documentation for temporal saliency-based patch selection.

---

## 🚀 Training

You can set up the environment using **one of the following two methods**:

### Option 1 (Conda + Pip)

```bash
conda env create -f environment.yml -n ov_encoder
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu118
pip install --extra-index-url https://pypi.nvidia.com --upgrade nvidia-dali-cuda110
pip install -r requirements.txt
```

### Option 2 (Docker)


```bash
docker build -t onevision-encoder:2601 .

docker run -it --rm --gpus all --ipc host --net host --privileged \
    -v "$(pwd)":/workspace/OneVision-Encoder \
    -w /workspace/OneVision-Encoder \
    onevision-encoder:2601 bash
```


### Install Package

Inside the container, install the package in editable mode:

```bash
pip install -e .
```

### Single Node Dry Run To Test Setup

```bash
bash shells/ov_encoder_base_stage1_si_dry_run.sh
```

### Single Node Stage-1 Single Image

```bash
bash shells/ov_encoder_base_stage1_si.sh
```

### Single Node Stage-2 Video Contine Pretraining

Download the Stage-1 checkpoint from HuggingFace:

```bash
git clone https://huggingface.co/lmms-lab-encoder/onevision-encoder-large-si
```

Download the pretraining data and prepare the data directory as per the instructions in `data/README.md`.


More documentation will be added soon.

```bash
bash shells/ov_encoder_large_stage2_residual_8gpus.sh
```


Training configurations and hyperparameters will be documented soon. For now, please refer to `--help` for available options.

## 📊 Evaluation

### Attentive Probe Evaluation


#### Chunk-wise Sampling Evaluation

To evaluate the encoder with uniform frame sampling, first navigate to the evaluation directory:

```bash
pip install -e .
cd eval_encoder
```

Then run the following command:

```bash
bash shells_eval_ap/eval_ov_encoder_large_16frames.sh
```

**Sampling-Specific Parameters:**
- `frames_token_num`: Number of tokens per frame (e.g., 256 tokens for standard sampling).

#### OV-Encoder Codec Evaluation

We release the attentive probing artifacts for our codec-based model across multiple video understanding benchmarks. For each dataset, we provide the codec-derived patch indices, training logs, model checkpoints, and final evaluation results.

> 📦 **Artifacts Repository**: All codec evaluation artifacts (codec indices, training logs, and checkpoints) are available on HuggingFace:  
> 🤗 [lmms-lab-encoder/onevision-encoder-codec-eval](https://huggingface.co/datasets/lmms-lab-encoder/onevision-encoder-codec-eval)

**Evaluation Results Summary:**

| Dataset | SSv2 | Diving48 | Perception Test | CharadesEgo | Epic-Verb | Epic-Noun | Kinetics-400 | HMDB51 |
|---------|------|----------|-----------------|-------------|-----------|-----------|--------------|--------|
| **Accuracy** | 58.4% | 66.0% | 59.7% | 12.1% | 62.2% | 53.9% | 84.3% | 83.5% |

**Available Artifacts per Dataset:**

For each dataset listed above, we provide:
- **Codec Index** (`*.json`): Pre-computed temporally salient patch indices for efficient codec-style evaluation
- **Training Logs** (`*.log`): Complete training logs including loss curves and intermediate metrics
- **Checkpoints** (`*.pt`): Fine-tuned attentive probe checkpoints for each dataset

To download artifacts for a specific dataset:

```bash
# Install huggingface-hub if needed
pip install huggingface-hub

# Download all codec indices
huggingface-cli download lmms-lab-encoder/onevision-encoder-codec-eval \
  --repo-type dataset \
  --include "codec_index/*" \
  --local-dir ./codec_artifacts

# Download logs for a specific dataset (e.g., ssv2)
huggingface-cli download lmms-lab-encoder/onevision-encoder-codec-eval \
  --repo-type dataset \
  --include "logs/ssv2.log" \
  --local-dir ./codec_artifacts

# Download checkpoint for a specific dataset (e.g., ssv2)
huggingface-cli download lmms-lab-encoder/onevision-encoder-codec-eval \
  --repo-type dataset \
  --include "checkpoints/ssv2.pt" \
  --local-dir ./codec_artifacts
```

Available datasets: ssv2, diving48, perception_test, charadesego, epic_verb, epic_noun, k400, hmdb51


**Running Codec-Style Evaluation:**

To evaluate the encoder with codec-style patch selection:

```bash
# Navigate to evaluation directory
cd eval_encoder

# Run codec evaluation with 2K patches (recommended)
# First positional argument is the model weight path
bash shells_eval_ap/eval_ov_encoder_large_2kpatches_codec.sh lmms-lab-encoder/onevision-encoder-large

# Or with 4K patches for higher quality
bash shells_eval_ap/eval_ov_encoder_large_4kpatches_codec.sh lmms-lab-encoder/onevision-encoder-large
```

**Note:** 
- The model weight path is a required positional argument. You can use `lmms-lab-encoder/onevision-encoder-large` to load directly from HuggingFace, or provide a local path to your model checkpoint.
- The evaluation scripts are configured for 8 GPUs by default. Adjust `CUDA_VISIBLE_DEVICES` in the shell script if you have a different GPU configuration.

**Codec-Specific Parameters:**
- `K_keep`: Number of patches to keep (e.g., 2048 for 2K patches, 4096 for 4K patches)
- `num_frames`: Total number of frames in the video sequence (typically 64 for codec evaluation)
- `frames_token_num`: Number of tokens per frame (e.g., 256 tokens)
- `cache_dir` (optional): Directory for cached codec patches. Use this to specify where codec-selected patches are stored/loaded when you want to persist or reuse them

**Using Pre-computed Codec Indices:**

To reproduce our exact results using the pre-computed codec indices:

```bash
# Download the codec indices from HuggingFace
huggingface-cli download lmms-lab-encoder/onevision-encoder-codec-eval \
  --repo-type dataset \
  --include "codec_index/*" \
  --local-dir ./codec_artifacts

# Run evaluation with the cache directory by running the Python script directly
# Example for SSv2 dataset with 2K patches (requires 8 GPUs)
cd eval_encoder
torchrun --nproc_per_node 8 --master_port 15555 \
  attentive_probe_codec.py \
  --model_family ov_encoder_codec \
  --model_name ov_encoder_large \
  --model_weight lmms-lab-encoder/onevision-encoder-large \
  --dataset ssv2 \
  --num_frames 64 \
  --frames_token_num 256 \
  --embedding_size 1024 \
  --K_keep 2048 \
  --batch_size 4 \
  --default_lr_list 0.0001 \
  --default_epoch 10 \
  --default_weight_decay 0 \
  --cache_dir ../codec_artifacts/codec_index \
  --save_report ./results/ssv2

# Note: Adjust --nproc_per_node based on your available GPUs
```

#### Shared Parameters

The following parameters are common to both evaluation methods:

- `dataset`: Dataset to evaluate on (e.g., `diving48`, `ssv2`, `kinetics400`). Prepare the dataset according to the Attentive Probe format.
- `num_frames`: Total number of frames in the video sequence (e.g., 8 for sampling, 64 for codec).
- `model_weight`: Path to the pre-trained model. Use `lmms-lab-encoder/onevision-encoder-large` to load directly from HuggingFace, or provide a local path.
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

## 🔗 Related Projects

- [nano-hevc](https://github.com/Luodian/nano-hevc) – A minimal and educational HEVC (H.265) encoder written in Python, designed to expose the full encoding pipeline and core design principles.
