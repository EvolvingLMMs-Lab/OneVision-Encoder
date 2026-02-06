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
🤗 **[Models](https://huggingface.co/collections/lmms-lab-encoder/onevision-encoder)** |
📄 **[Tech Report (coming)]()** |
📋 **[Model Card](docs/model_card.md)** |
📊 **[Data Card](docs/data_card.md)**

</div>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/anxiangsir/asset/main/OneVision/method_github_dark.png">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/anxiangsir/asset/main/OneVision/method_github_light.png">
    <img alt="OneVision Encoder Method Overview" src="https://raw.githubusercontent.com/anxiangsir/asset/main/OneVision/method_github_light.png" width="900" style="max-width: 100%;">
  </picture>
</p>

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
- **3D Rotary Position && Native Resolution**: Uses a 4:6:6 split for temporal, height, and width dimensions to capture spatiotemporal relationships. Supports native resolution input without tiling or cropping.
- **Global Contrastive Learning**: Trained with a 2M concept bank for better-separated semantic clusters.

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

## ⚡ Quick Start

> [!IMPORTANT]
> **Transformers Version Compatibility:**
>
> - ✅ **`transformers==4.57.3`** (Recommended): Works with `AutoModel.from_pretrained()`
> - ⚠️ **`transformers>=5.0.0`**: Not currently supported. We are actively working on a fix.

> **Note:** This model supports native resolution input. For optimal performance:
>
> - **Image**: 448×448 resolution (pre-trained)
> - **Video**: 224×224 resolution with 256 tokens per frame (pre-trained)
>
> Use CLIP preprocessing from the [model repository](https://huggingface.co/lmms-lab-encoder/onevision-encoder-large).

### Using AutoModel (Recommended: transformers==4.57.3)

<details>
<summary>Click to expand code example</summary>

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

# Video inference: [B, C, T, H, W] with patch_positions
num_frames, target_frames = 16, 64
patch_size = 14
# Load video frames and preprocess each frame (replace with your video frame paths)
frames = [Image.open(f"path/to/frame_{i}.jpg") for i in range(num_frames)]
video_pixel_values = preprocessor(images=frames, return_tensors="pt")["pixel_values"]
# Reshape from [T, C, H, W] to [B, C, T, H, W]
video = video_pixel_values.unsqueeze(0).permute(0, 2, 1, 3, 4).to("cuda")

# Build patch_positions for temporal sampling: [B, num_frames * frame_tokens, 3]
frame_pos = torch.linspace(0, target_frames - 1, num_frames).long().cuda()  # [T]
grid_h, grid_w = video.shape[-2] // patch_size, video.shape[-1] // patch_size  # patch grid
frame_tokens = grid_h * grid_w

t_positions = frame_pos[:, None].repeat(1, frame_tokens).reshape(-1)  # [T * frame_tokens]
h_positions = torch.arange(grid_h, device="cuda").repeat_interleave(grid_w)
h_positions = h_positions.repeat(num_frames)  # [T * frame_tokens]
w_positions = torch.arange(grid_w, device="cuda").repeat(grid_h)
w_positions = w_positions.repeat(num_frames)  # [T * frame_tokens]

patch_positions = torch.stack([t_positions, h_positions, w_positions], dim=-1).unsqueeze(0)
# patch_positions example (256 tokens per frame, 16x16 patch grid):
#   Each row is [t, h, w].
#   First 4 patches of frame 0 (t=0):
#     patch_positions[0, 0:4, :] -> [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3]]
#   First 4 patches of frame 1 (t=4):
#     patch_positions[0, 256:260, :] -> [[4, 0, 0], [4, 0, 1], [4, 0, 2], [4, 0, 3]]

with torch.no_grad():
    outputs = model(video, patch_positions=patch_positions)
```

</details>

### Loading from Source Code  

<details>
<summary>Click to expand installation and usage code</summary>

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

</details>

### Codec Input

Add codec-style input documentation for temporal saliency-based patch selection.

---

## 🚀 Training

You can set up the environment using **one of the following two methods**:

### Option 1 (Conda + Pip)

<details>
<summary>Click to expand setup commands</summary>

```bash
conda env create -f environment.yml -n ov_encoder
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu118
pip install --extra-index-url https://pypi.nvidia.com --upgrade nvidia-dali-cuda110
pip install -r requirements.txt
```

</details>

### Option 2 (Docker)

<details>
<summary>Click to expand Docker commands</summary>

```bash
docker build -t onevision-encoder:2601 .

docker run -it --rm --gpus all --ipc host --net host --privileged \
    -v "$(pwd)":/workspace/OneVision-Encoder \
    -w /workspace/OneVision-Encoder \
    onevision-encoder:2601 bash
```

</details>

### Install Package

Inside the container, install the package in editable mode:

<details>
<summary>Click to expand install command</summary>

```bash
pip install -e .
```

</details>

### Single Node Dry Run To Test Setup

<details>
<summary>Click to expand dry run command</summary>

```bash
bash shells/ov_encoder_base_stage1_si_dry_run.sh
```

</details>

### Single Node Stage-1 Single Image

<details>
<summary>Click to expand training command</summary>

```bash
bash shells/ov_encoder_base_stage1_si.sh
```

</details>

### Single Node Stage-2 Video Contine Pretraining

Download the Stage-1 checkpoint from HuggingFace:

<details>
<summary>Click to expand download and training commands</summary>

```bash
git clone https://huggingface.co/lmms-lab-encoder/onevision-encoder-large-si
```

Download the pretraining data and prepare the data directory as per the instructions in `data/README.md`.

More documentation will be added soon.

```bash
bash shells/ov_encoder_large_stage2_residual_8gpus.sh
```

</details>

Training configurations and hyperparameters will be documented soon. For now, please refer to `--help` for available options.

## 📊 Evaluation

### LLaVA-NeXT Evaluation

To evaluate the OneVision Encoder as a vision backbone for LLaVA-NeXT multimodal models, we use the lmms-eval framework with various vision-language benchmarks.

#### Setup

Navigate to the llava_next directory and follow the setup instructions:

<details>
<summary>Click to expand LLaVA-NeXT evaluation setup</summary>

```bash
cd llava_next

# Using Docker (recommended)
docker build -t ov_encoder_llava:26.01 .
docker run -it --gpus all --ipc host --net host --privileged \
    -v "$(pwd)":/workspace/OV-Encoder-Llava \
    -w /workspace/OV-Encoder-Llava \
    ov_encoder_llava:26.01 bash
```

</details>

#### Running Evaluation

For image benchmarks (ChartQA, DocVQA, AI2D, OCRBench, etc.):

<details>
<summary>Click to expand evaluation commands</summary>

```bash
# Evaluate on image benchmarks
TASKS="ai2d,chartqa,docvqa_val" bash scripts/eval/eval_ov_encoder.sh
```

</details>

For video benchmarks (VideoMME, MVBench, PerceptionTest, etc.), run each benchmark separately:

<details>
<summary>Click to expand video evaluation commands</summary>

```bash
# Preprocess video benchmark (one-time setup)
bash scripts/precompute_codec_patch/preprocess_video_benchmark.sh videomme

# Run evaluation
TASKS="videomme" bash scripts/eval/eval_ov_encoder.sh
```

</details>

For more details, refer to the [LLaVA-NeXT documentation](llava_next/README.md).

### Attentive Probe Evaluation

#### Chunk-wise Sampling Evaluation

To evaluate the encoder with uniform frame sampling, first navigate to the evaluation directory:

<details>
<summary>Click to expand evaluation commands</summary>

```bash
pip install -e .
cd eval_encoder
```

Then run the following command:

```bash
bash shells_eval_ap/eval_ov_encoder_large_16frames.sh
```

</details>

**Sampling-Specific Parameters:**

- `frames_token_num`: Number of tokens per frame (e.g., 256 tokens for standard sampling).

#### OV-Encoder Codec Evaluation

To evaluate the encoder with codec-style patch selection, first navigate to the evaluation directory:

<details>
<summary>Click to expand codec evaluation commands</summary>

```bash
cd eval_encoder
```

Then run the following command:

```bash
bash shells_eval_ap/eval_ov_encoder_large_2kpatches_codec.sh
```

</details>

## 👥 Contributors

<!-- Add contributor list here -->

---

## 📄 License

This project is released under the Apache 2.0 License.



## 🔗 Related Projects

- [nano-hevc](https://github.com/Luodian/nano-hevc) – A minimal and educational HEVC (H.265) encoder written in Python, designed to expose the full encoding pipeline and core design principles.
