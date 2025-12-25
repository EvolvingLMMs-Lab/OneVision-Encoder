# Model Card: OneVision Encoder Large

## Model Overview

**OneVision Encoder Large** is a vision transformer that resolves the fundamental trade-off in video understanding: processing more frames captures richer temporal information but increases computation quadratically. Using principles from HEVC video compression, it implements codec-style patch selection that identifies temporally-salient regions—areas with motion, object interactions, or semantic changes—and processes only these informative patches.

### Model Details

| Property | Value |
|----------|-------|
| **Model Type** | Vision Transformer (ViT) |
| **Architecture** | HEVC-Style Vision Transformer |
| **Hidden Size** | 1024 |
| **Intermediate Size** | 4096 |
| **Number of Layers** | 24 |
| **Number of Attention Heads** | 16 |
| **Patch Size** | 16 |
| **Image Resolution** | 448×448 (pre-trained) |
| **Video Resolution** | 224×224 with 256 tokens per frame |
| **Positional Encoding** | 3D RoPE (4:6:6 split for T:H:W) |
| **Normalization** | Layer Normalization |
| **Activation Function** | GELU |
| **Attention Implementation** | Flash Attention 2 |
| **License** | Apache 2.0 |

## Key Features

- **Codec-Style Patch Selection**: Instead of sampling sparse frames densely (all patches from few frames), OneVision Encoder samples dense frames sparsely (important patches from many frames).
- **3D Rotary Position Embedding**: Uses a 4:6:6 split for temporal, height, and width dimensions to capture spatiotemporal relationships.
- **Global Contrastive Learning**: Trained with a 2M concept bank for better-separated semantic clusters.
- **Native Resolution Support**: Supports native resolution input without tiling or cropping.
- **Flash Attention 2**: Efficient attention implementation for improved performance and memory efficiency.

## Intended Use

### Primary Use Cases

- **Video Understanding**: Action recognition, video captioning, video question answering
- **Image Understanding**: Document understanding (DocVQA), chart understanding (ChartQA), OCR tasks
- **Vision-Language Models**: As the vision encoder backbone for multimodal large language models

### Downstream Tasks

- Video benchmarks: MVBench, VideoMME, Perception Test
- Image understanding: DocVQA, ChartQA, OCRBench
- Action recognition: SSv2, UCF101, Kinetics

## Quick Start

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
image = Image.open("path/to/your/image.jpg")
pixel_values = preprocessor(images=image, return_tensors="pt")["pixel_values"].to("cuda")
with torch.no_grad():
    outputs = model(pixel_values)
    # outputs.last_hidden_state: [B, num_patches, hidden_size]
    # outputs.pooler_output: [B, hidden_size]

# Video inference: [B, C, T, H, W] with visible_indices
num_frames, frame_tokens, target_frames = 16, 256, 64
frames = [Image.open(f"path/to/frame_{i}.jpg") for i in range(num_frames)]
video_pixel_values = preprocessor(images=frames, return_tensors="pt")["pixel_values"]
video = video_pixel_values.unsqueeze(0).permute(0, 2, 1, 3, 4).to("cuda")

# Build visible_indices for temporal sampling
frame_pos = torch.linspace(0, target_frames - 1, num_frames).long().cuda()
visible_indices = (frame_pos.unsqueeze(-1) * frame_tokens + torch.arange(frame_tokens).cuda()).reshape(1, -1)

with torch.no_grad():
    outputs = model(video, visible_indices=visible_indices)
```


## Evaluation Results

### Attentive Probe Results

Performance evaluated using Attentive Probe evaluation with single clip input, trained for 10 epochs across 8 action recognition datasets.

### LMM Probe Results

Training on a mixed dataset of 740K samples from LLaVA-OneVision and 800K samples from LLaVA-Video SFT. The training pipeline proceeds directly to Stage 2 fine-tuning with native-resolution strategy.

## Limitations

- The model is pre-trained at specific resolutions (448×448 for images, 224×224 for video)
- Performance may vary on domains significantly different from training data
- Video processing requires proper temporal sampling configuration

## Citation

```bibtex
@misc{onevision-encoder,
  title={OneVision Encoder: HEVC-Style Vision Transformer},
  author={EvolvingLMMs-Lab},
  year={2024},
  url={https://github.com/EvolvingLMMs-Lab/OneVision-Encoder}
}
```

## Contact

For questions and issues, please open an issue on the [GitHub repository](https://github.com/EvolvingLMMs-Lab/OneVision-Encoder).
