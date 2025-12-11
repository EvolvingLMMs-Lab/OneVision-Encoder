<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="asset/llava_vit_white.png">
    <source media="(prefers-color-scheme: light)" srcset="asset/llava_vit_white.png">
    <img alt="LLaVA-ViT" src="asset/llava_vit_white.png" width="600" style="max-width: 100%;">
  </picture>
</p>

<p align="center">
  <b>LLaVA-ViT: A Vision Transformer for Large Language-and-Vision Assistant</b>
</p>

---

## 📖 Table of Contents

- [Introduction](#-introduction)
- [Setup](#-setup)
- [Training](#-training)
- [Evaluation](#-evaluation)
- [Packing ViT Model](#-packing-vit-model)
- [Contributors](#-contributors)
- [License](#-license)

---

## 🔍 Introduction

LLaVA-ViT is a vision encoder designed for multimodal large language models, featuring efficient video representation with sparse video input. This project provides training code, data processing tools, and model evaluation utilities.

### Input Method Comparison

<table>
  <caption style="caption-side: top; text-align: center; font-weight: bold; margin-bottom: 10px;">Comparison of Frame Sampling Input vs Codec Input</caption>
  <tr>
    <td align="center">
      <img src="pages/images/example.gif" alt="Animated demonstration of traditional uniform frame sampling method for video processing" width="400"><br>
      <b>抽帧输入 (Frame Sampling Input)</b><br>
      Traditional uniform frame sampling approach
    </td>
    <td align="center">
      <img src="pages/images/example_codec_input.gif" alt="Animated demonstration of efficient codec-based input decomposition with I-frames and P-frames" width="400"><br>
      <b>Codec Input</b><br>
      Our efficient codec-based input decomposition
    </td>
  </tr>
</table>

### Cluster Discrimination Visualization

<p align="center">
  <img src="pages/images/global_contrastive_comparison.gif" alt="Global Contrastive Comparison" width="800" style="max-width: 100%;">
</p>

### Pre-training Tips

1. **Scale-up is the final step** - Maximize model capabilities before scaling, and ensure generalization phenomena emerge
2. **Avoid direct supervision from existing models** - Indirect usage is preferred over direct distillation, which may limit scaling capabilities
3. **Progressive training when resources are limited** - Start with low resolution/frame rate, then gradually fine-tune to higher settings (ref: CLIPA)

---

## 🔧 Setup

### Prerequisites

- Docker with NVIDIA GPU support
- CUDA-compatible GPU(s)

### Mount NFS

```bash
mkdir -p /video_vit
mount -t nfs4 -o minorversion=1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2,noresvport cfs-iyHiNUmePn.lb-0a25b0a7.cfs.bj.baidubce.com:/ /video_vit

mkdir -p /vlm
mount -t nfs4 -o minorversion=1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2,noresvport cfs-xvbkSb1zPT.lb-563926be.cfs.bj.baidubce.com:/ /vlm
```

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
    -v "$(pwd)":/workspace/LLaVA-ViT \
    -w /workspace/LLaVA-ViT \
    llava_vit:25.11.22 bash
```

#### Multi Node

> [!IMPORTANT]
> 多机必须使用预编译的镜像，且镜像必须一致

```bash
docker run -it --gpus all --ipc host --net host --privileged --cap-add IPC_LOCK \
    --ulimit memlock=-1 --ulimit stack=67108864 --rm \
    -v "$(pwd)":/workspace/LLaVA-ViT -v /train_tmp:/train_tmp \
    -v /vlm:/vlm -v /video_vit:/video_vit -v /rice_ocr:/rice_ocr \
    -v /data_0:/data_0 -v /data_1:/data_1 -v /data_2:/data_2 -v /data_3:/data_3 \
    -w /workspace/LLaVA-ViT/ \
    -e NCCL_TIMEOUT=1800 -e CUDA_DEVICE_MAX_CONNECTIONS=1 -e NCCL_SOCKET_IFNAME=eth0 -e NCCL_IB_GID_INDEX=3 -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA="mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_1" -e NCCL_NET_GDR_LEVEL=2 -e NCCL_IB_QPS_PER_CONNECTION=4 -e NCCL_IB_TC=160 -e NCCL_IB_TIMEOUT=22 -e NCCL_CROSS_NIC=1 -e NCCL_MIN_NCHANNELS=8 -e NCCL_MAX_NCHANNELS=16 \
    -e http_proxy=http://172.16.5.77:8889 -e https_proxy=http://172.16.5.77:8889 \
    llava_vit:25.11.22 bash -c "service ssh restart; bash"
```

### Install Package

Inside the container, install the package in editable mode:

```bash
pip install -e .
```

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

LLaVA-ViT provides a packing model (`LlavaViTPackingModel`) for efficient variable-length sequence processing with FlashAttention support, similar to Qwen2VL's vision encoder.

> **Detailed documentation**: See [`model_factory/README_PACKING.md`](model_factory/README_PACKING.md) for complete usage guide.

### Requirements

```bash
# FlashAttention 2 is required
pip install flash-attn --no-build-isolation
```

### Understanding `patch_positions`

The `patch_positions` parameter allows you to explicitly specify the RoPE (Rotary Position Embedding) positions for each patch. This is essential for:
- Achieving consistent outputs between the source model and the packing model
- Processing videos with non-uniform frame sampling (e.g., uniform sampling from long videos)
- Enabling flexible spatial-temporal position encoding

#### `patch_positions` Format

`patch_positions` is a tensor of shape `(seq_len, 3)` where each row contains `[t, h, w]`:
- `t`: Temporal position (frame index)
- `h`: Height position (patch row index)
- `w`: Width position (patch column index)

### How to Prepare `patch_positions`

#### Method 1: Using `compute_patch_positions_from_grid_thw` (Recommended for Images)

For simple image processing where patches are arranged sequentially:

```python
from model_factory.vit_preview_v0_packing_hf import (
    LlavaViTPackingModel,
    compute_patch_positions_from_grid_thw,
)
import torch

# For a 224x224 image with patch_size=16
# h_patches = w_patches = 224 // 16 = 14
grid_thw = torch.tensor([[1, 14, 14]], dtype=torch.long, device='cuda')  # [t=1, h=14, w=14]

# Compute patch positions automatically
patch_positions = compute_patch_positions_from_grid_thw(grid_thw)
# Shape: (196, 3) for 14*14=196 patches
# Values: [[0, 0, 0], [0, 0, 1], ..., [0, 13, 13]]
#         [t, h, w] for each patch

# Forward pass
outputs = model(
    hidden_states=hidden_states,
    grid_thw=grid_thw,
    patch_positions=patch_positions,
)
```

#### Method 2: Using Interpolated Temporal Positions (For Video)

For video with uniform frame sampling (e.g., 8 frames from a 64-frame context):

```python
from model_factory.convert_llava_vit_packing_to_hf import (
    interpolate_frame_indices,
    compute_patch_positions_with_interpolated_temporal,
)
import torch

# Example: 8 frames uniformly sampled from 64-frame context
num_frames = 8
target_frames = 64  # The source model's expected temporal context
h_patches, w_patches = 14, 14  # For 224x224 image with patch_size=16

# Step 1: Compute interpolated frame indices
frame_indices = torch.arange(num_frames).unsqueeze(0).cuda()  # [1, 8] = [[0,1,2,3,4,5,6,7]]
total_frames = torch.tensor([num_frames]).cuda()  # [8]

interpolated_indices = interpolate_frame_indices(frame_indices, total_frames, target_frames)
# Result: [[0, 9, 18, 27, 36, 45, 54, 63]] - evenly spaced in 64-frame context

# Step 2: Compute patch positions with interpolated temporal positions
patch_positions = compute_patch_positions_with_interpolated_temporal(
    interpolated_indices, h_patches, w_patches, device='cuda'
)
# Shape: (num_frames * h_patches * w_patches, 3) = (8*14*14, 3) = (1568, 3)
# Each row: [t_interpolated, h, w]
# The temporal values are 0, 9, 18, 27, 36, 45, 54, 63 (interpolated to 64-frame context)

# Create grid_thw for actual frames
grid_thw = torch.tensor([[num_frames, h_patches, w_patches]], dtype=torch.long, device='cuda')

# Forward pass
outputs = model(
    hidden_states=hidden_states,
    grid_thw=grid_thw,
    patch_positions=patch_positions,
)
```

#### Method 3: Manual Construction (Advanced)

For custom spatial-temporal positions:

```python
import torch

def manual_patch_positions(t_frames, h_patches, w_patches, device='cuda'):
    """
    Manually construct patch_positions tensor.
    
    Patch ordering: [frame_0_patches, frame_1_patches, ..., frame_t_patches]
    Within each frame: row-major order (h varies slower than w)
    """
    positions = []
    for t in range(t_frames):
        for h in range(h_patches):
            for w in range(w_patches):
                positions.append([t, h, w])
    return torch.tensor(positions, dtype=torch.long, device=device)

# Example: 8 frames at 14x14 patches
patch_positions = manual_patch_positions(8, 14, 14)
# Shape: (1568, 3)
# Values: [[0,0,0], [0,0,1], ..., [0,13,13], [1,0,0], ..., [7,13,13]]
```

### Complete Example: Image Processing

```python
import torch
from PIL import Image
import torchvision.transforms as T
from model_factory.vit_preview_v0_packing_hf import (
    LlavaViTPackingModel,
    compute_patch_positions_from_grid_thw,
)

# Load model
model = LlavaViTPackingModel.from_pretrained("path/to/model", torch_dtype=torch.bfloat16)
model = model.cuda().eval()

# Prepare image
patch_size = 16
image = Image.open("image.jpg").resize((448, 448))
transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]),
])
pixel_tensor = transform(image)  # (3, 448, 448)

# Calculate patch dimensions
channels, height, width = pixel_tensor.shape
h_patches = height // patch_size  # 28
w_patches = width // patch_size   # 28

# Reshape to patches: (C, H, W) -> (seq_len, patch_dim)
patches = pixel_tensor.view(channels, h_patches, patch_size, w_patches, patch_size)
patches = patches.permute(1, 3, 0, 2, 4).contiguous()  # (h, w, C, pH, pW)
hidden_states = patches.view(h_patches * w_patches, patch_size * patch_size * channels)
hidden_states = hidden_states.cuda().bfloat16()

# Prepare grid_thw and patch_positions
grid_thw = torch.tensor([[1, h_patches, w_patches]], dtype=torch.long, device='cuda')
patch_positions = compute_patch_positions_from_grid_thw(grid_thw)

# Forward pass
with torch.no_grad():
    outputs = model(
        hidden_states=hidden_states,
        grid_thw=grid_thw,
        patch_positions=patch_positions,
    )

print(f"Output shape: {outputs.last_hidden_state.shape}")  # (784, hidden_size)
print(f"Pooler shape: {outputs.pooler_output.shape}")      # (1, hidden_size)
```

### Complete Example: Video Processing

```python
import torch
from PIL import Image
import torchvision.transforms as T
from model_factory.vit_preview_v0_packing_hf import LlavaViTPackingModel
from model_factory.convert_llava_vit_packing_to_hf import (
    interpolate_frame_indices,
    compute_patch_positions_with_interpolated_temporal,
)

# Load model
model = LlavaViTPackingModel.from_pretrained("path/to/model", torch_dtype=torch.bfloat16)
model = model.cuda().eval()

# Video parameters
patch_size = 16
num_frames = 8
frame_size = 224
target_frames = 64  # Source model's temporal context

# Load video frames (example: list of PIL Images)
frames = [Image.open(f"frame_{i}.jpg").resize((frame_size, frame_size)) for i in range(num_frames)]

transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]),
])

# Calculate patch dimensions
h_patches = frame_size // patch_size  # 14
w_patches = frame_size // patch_size  # 14

# Process frames and reshape to patches
all_patches = []
for frame in frames:
    pixel_tensor = transform(frame)  # (3, 224, 224)
    channels = pixel_tensor.shape[0]
    patches = pixel_tensor.view(channels, h_patches, patch_size, w_patches, patch_size)
    patches = patches.permute(1, 3, 0, 2, 4).contiguous()  # (h, w, C, pH, pW)
    frame_patches = patches.view(h_patches * w_patches, patch_size * patch_size * channels)
    all_patches.append(frame_patches)

hidden_states = torch.cat(all_patches, dim=0)  # (num_frames * h * w, patch_dim)
hidden_states = hidden_states.cuda().bfloat16()

# Compute interpolated temporal positions for video
frame_indices = torch.arange(num_frames).unsqueeze(0).cuda()
total_frames_tensor = torch.tensor([num_frames]).cuda()
interpolated_indices = interpolate_frame_indices(frame_indices, total_frames_tensor, target_frames)

# Compute patch_positions with interpolated temporal values
patch_positions = compute_patch_positions_with_interpolated_temporal(
    interpolated_indices, h_patches, w_patches, device='cuda'
)

# grid_thw uses actual frame count
grid_thw = torch.tensor([[num_frames, h_patches, w_patches]], dtype=torch.long, device='cuda')

# Forward pass
with torch.no_grad():
    outputs = model(
        hidden_states=hidden_states,
        grid_thw=grid_thw,
        patch_positions=patch_positions,
    )

print(f"Output shape: {outputs.last_hidden_state.shape}")  # (1568, hidden_size)
print(f"Pooler shape: {outputs.pooler_output.shape}")      # (1, hidden_size)
```

### Model Conversion

Convert weights from source model to packing model format:

```bash
python model_factory/convert_llava_vit_packing_to_hf.py \
    llava_vit_large_ln \
    /path/to/backbone.pt \
    --output_dir /path/to/output
```

The conversion script automatically verifies both image and video consistency between the source and packing models.

---

## 👥 Contributors

Thanks so much to all of our amazing contributors!

<!-- readme: collaborators,contributors -start -->
<table>
	<tbody>
		<tr>
            <td align="center">
                <a href="https://github.com/GeoffreyChen777">
                    <img src="https://avatars.githubusercontent.com/u/14183213?v=4" width="80;" alt="GeoffreyChen777"/>
                    <br />
                    <sub><b>GeoffreyChen777</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/Luodian">
                    <img src="https://avatars.githubusercontent.com/u/15847405?v=4" width="80;" alt="Luodian"/>
                    <br />
                    <sub><b>Luodian</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/anxiangsir">
                    <img src="https://avatars.githubusercontent.com/u/31175974?v=4" width="80;" alt="anxiangsir"/>
                    <br />
                    <sub><b>anxiangsir</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/yiyexy">
                    <img src="https://avatars.githubusercontent.com/u/35927125?v=4" width="80;" alt="yiyexy"/>
                    <br />
                    <sub><b>yiyexy</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/manyuan97">
                    <img src="https://avatars.githubusercontent.com/u/70136737?v=4" width="80;" alt="manyuan97"/>
                    <br />
                    <sub><b>manyuan97</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/YunyaoYan">
                    <img src="https://avatars.githubusercontent.com/u/109638667?v=4" width="80;" alt="YunyaoYan"/>
                    <br />
                    <sub><b>YunyaoYan</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/FeilongTangmonash">
                    <img src="https://avatars.githubusercontent.com/u/152372878?v=4" width="80;" alt="FeilongTangmonash"/>
                    <br />
                    <sub><b>FeilongTangmonash</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/wkzhang636">
                    <img src="https://avatars.githubusercontent.com/u/194186498?v=4" width="80;" alt="wkzhang636"/>
                    <br />
                    <sub><b>wkzhang636</b></sub>
                </a>
            </td>
		</tr>
	<tbody>
</table>
<!-- readme: collaborators,contributors -end -->

---

## 📄 License

This project is open source.
