# 上传 LlavaViT 模型到 HuggingFace Hub

[English](#english) | [中文](#中文)

---

## 中文

本指南说明如何将 LlavaViT 模型上传到 HuggingFace Hub，使其可以通过 `AutoModel.from_pretrained()` 直接加载。

### 快速开始

#### 1. 安装依赖

```bash
pip install huggingface_hub transformers timm torch
```

#### 2. 获取 HuggingFace Token

访问 https://huggingface.co/settings/tokens 创建一个新的 token（需要写入权限）。

#### 3. 上传模型

```bash
python model_factory/upload_llava_vit_to_hf.py \
    --model_name hf_llava_vit_large_ln \
    --weight_path /path/to/your/checkpoint.pth \
    --repo_id your-username/llava-vit-large \
    --token YOUR_HF_TOKEN
```

### 支持的模型架构

脚本支持以下预定义的模型架构：

| 模型名称 | Hidden Size | Layers | Heads | Patch Size |
|---------|------------|--------|-------|------------|
| `hf_llava_vit_small_ln` | 384 | 6 | 6 | 16 |
| `hf_llava_vit_base_ln` | 768 | 12 | 12 | 16 |
| `hf_llava_vit_large_ln` | 1024 | 24 | 16 | 14 |
| `hf_llava_vit_huge_ln` | 1536 | 27 | 24 | 14 |
| `hf_llava_vit_giant_ln` | 1536 | 40 | 16 | 14 |

### 使用示例

#### 基础使用

```python
from transformers import AutoModel, CLIPImageProcessor
import torch
from PIL import Image
import requests

# 加载模型和处理器
model = AutoModel.from_pretrained(
    "your-username/llava-vit-large", 
    trust_remote_code=True
)
processor = CLIPImageProcessor.from_pretrained("your-username/llava-vit-large")

# 加载图片
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# 处理图片
inputs = processor(images=image, return_tensors="pt")

# 获取嵌入
with torch.no_grad():
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state  # [batch_size, num_patches, hidden_size]
    pooled = outputs.pooler_output          # [batch_size, hidden_size]

print(f"Embeddings shape: {embeddings.shape}")
print(f"Pooled output shape: {pooled.shape}")
```

#### 视频输入

```python
import torch

# 视频输入: [batch_size, channels, num_frames, height, width]
video_input = torch.randn(1, 3, 8, 448, 448)

with torch.no_grad():
    outputs = model(pixel_values=video_input)
    video_embeddings = outputs.last_hidden_state
    
print(f"Video embeddings: {video_embeddings.shape}")
```

#### 使用可见索引（Masking）

```python
import torch

pixel_values = torch.randn(1, 3, 448, 448)
num_patches = (448 // 14) ** 2  # 对于 patch_size=14

# 只使用前 75% 的 patches
visible_indices = torch.arange(int(num_patches * 0.75)).unsqueeze(0)

with torch.no_grad():
    outputs = model(
        pixel_values=pixel_values, 
        visible_indices=visible_indices
    )
    embeddings = outputs.last_hidden_state
```

### 命令行参数

| 参数 | 必需 | 描述 |
|------|------|------|
| `--model_name` | ✅ | 模型架构名称（见上表） |
| `--repo_id` | ✅ | HuggingFace 仓库 ID (格式: "username/model-name") |
| `--weight_path` | ❌ | 模型权重文件路径 (.pth)，不提供则使用随机初始化 |
| `--token` | ❌ | HuggingFace API token（也可以通过 HF_TOKEN 环境变量设置） |
| `--private` | ❌ | 创建私有仓库 |
| `--commit_message` | ❌ | 上传的 commit 信息 |

### 完整示例

#### 上传已训练模型

```bash
python model_factory/upload_llava_vit_to_hf.py \
    --model_name hf_llava_vit_large_ln \
    --weight_path /path/to/trained_model.pth \
    --repo_id myusername/llava-vit-large-trained \
    --token hf_xxxxxxxxxxxxx
```

#### 上传模型到私有仓库

```bash
python model_factory/upload_llava_vit_to_hf.py \
    --model_name hf_llava_vit_huge_ln \
    --weight_path /path/to/checkpoint.pth \
    --repo_id mycompany/private-llava-vit \
    --token hf_xxxxxxxxxxxxx \
    --private
```

#### 使用环境变量

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxx

python model_factory/upload_llava_vit_to_hf.py \
    --model_name hf_llava_vit_base_ln \
    --weight_path /path/to/weights.pth \
    --repo_id myusername/llava-vit-base
```

### 上传后文件结构

上传完成后，你的 HuggingFace 仓库将包含以下文件：

```
your-repo/
├── config.json                    # 模型配置
├── pytorch_model.bin              # 模型权重
├── configuration_llava_vit.py     # 配置类定义
├── modeling_llava_vit.py          # 模型类定义
├── preprocessor_config.json       # 图像处理器配置
├── README.md                      # 自动生成的模型卡片
└── example_usage.py               # 使用示例代码
```

### 模型特性

- ✅ **3D 旋转位置编码 (RoPE)**: 支持时空理解，T、H、W 维度采用 4:6:6 分割
- ✅ **Flash Attention 2**: 优化的注意力实现，提升性能
- ✅ **多头注意力池化**: PMA 风格池化，生成丰富的表示
- ✅ **灵活输入**: 支持 2D（图像）和 3D（视频）输入
- ✅ **Masking 支持**: 兼容 MAE 风格的可见索引

### 输出说明

模型返回 `BaseModelOutputWithPooling`，包含：

- `last_hidden_state`: Token 嵌入，形状 (batch_size, sequence_length, hidden_size)
- `pooler_output`: 池化表示，形状 (batch_size, hidden_size)
- `hidden_states`: （可选）所有层的隐藏状态
- `attentions`: （可选）所有层的注意力权重

### 注意事项

1. **Trust Remote Code**: 加载模型时需要设置 `trust_remote_code=True`
2. **Flash Attention**: 需要安装 `flash_attn` 包以获得最佳性能
3. **精度**: 建议使用 bfloat16 或 float16 精度以获得最佳性能
4. **图像大小**: 默认为 448x448，但可以处理其他尺寸

### 故障排除

#### 1. Token 错误

```
❌ Error: HuggingFace token required!
```

**解决方法**: 确保提供了有效的 HuggingFace token，并且具有写入权限。

#### 2. 权重加载失败

```
❌ Failed to load weights
```

**解决方法**: 检查权重文件路径是否正确，以及权重是否与选择的模型架构匹配。

#### 3. 上传失败

如果上传失败，文件会保存在 `/tmp/llava_vit_upload_*` 目录，你可以手动上传到 HuggingFace。

---

## English

This guide explains how to upload a LlavaViT model to HuggingFace Hub so it can be loaded directly using `AutoModel.from_pretrained()`.

### Quick Start

#### 1. Install Dependencies

```bash
pip install huggingface_hub transformers timm torch
```

#### 2. Get HuggingFace Token

Visit https://huggingface.co/settings/tokens to create a new token (write permission required).

#### 3. Upload Model

```bash
python model_factory/upload_llava_vit_to_hf.py \
    --model_name hf_llava_vit_large_ln \
    --weight_path /path/to/your/checkpoint.pth \
    --repo_id your-username/llava-vit-large \
    --token YOUR_HF_TOKEN
```

### Supported Model Architectures

The script supports the following predefined model architectures:

| Model Name | Hidden Size | Layers | Heads | Patch Size |
|-----------|------------|--------|-------|------------|
| `hf_llava_vit_small_ln` | 384 | 6 | 6 | 16 |
| `hf_llava_vit_base_ln` | 768 | 12 | 12 | 16 |
| `hf_llava_vit_large_ln` | 1024 | 24 | 16 | 14 |
| `hf_llava_vit_huge_ln` | 1536 | 27 | 24 | 14 |
| `hf_llava_vit_giant_ln` | 1536 | 40 | 16 | 14 |

### Usage Examples

#### Basic Usage

```python
from transformers import AutoModel, CLIPImageProcessor
import torch
from PIL import Image
import requests

# Load model and processor
model = AutoModel.from_pretrained(
    "your-username/llava-vit-large", 
    trust_remote_code=True
)
processor = CLIPImageProcessor.from_pretrained("your-username/llava-vit-large")

# Load image
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# Process image
inputs = processor(images=image, return_tensors="pt")

# Get embeddings
with torch.no_grad():
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state  # [batch_size, num_patches, hidden_size]
    pooled = outputs.pooler_output          # [batch_size, hidden_size]

print(f"Embeddings shape: {embeddings.shape}")
print(f"Pooled output shape: {pooled.shape}")
```

#### Video Input

```python
import torch

# Video input: [batch_size, channels, num_frames, height, width]
video_input = torch.randn(1, 3, 8, 448, 448)

with torch.no_grad():
    outputs = model(pixel_values=video_input)
    video_embeddings = outputs.last_hidden_state
    
print(f"Video embeddings: {video_embeddings.shape}")
```

#### Using Visible Indices (Masking)

```python
import torch

pixel_values = torch.randn(1, 3, 448, 448)
num_patches = (448 // 14) ** 2  # For patch_size=14

# Use only first 75% of patches
visible_indices = torch.arange(int(num_patches * 0.75)).unsqueeze(0)

with torch.no_grad():
    outputs = model(
        pixel_values=pixel_values, 
        visible_indices=visible_indices
    )
    embeddings = outputs.last_hidden_state
```

### Command Line Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--model_name` | ✅ | Model architecture name (see table above) |
| `--repo_id` | ✅ | HuggingFace repository ID (format: "username/model-name") |
| `--weight_path` | ❌ | Path to model weights file (.pth), random init if not provided |
| `--token` | ❌ | HuggingFace API token (can also set HF_TOKEN env var) |
| `--private` | ❌ | Create private repository |
| `--commit_message` | ❌ | Commit message for the upload |

### Complete Examples

#### Upload Trained Model

```bash
python model_factory/upload_llava_vit_to_hf.py \
    --model_name hf_llava_vit_large_ln \
    --weight_path /path/to/trained_model.pth \
    --repo_id myusername/llava-vit-large-trained \
    --token hf_xxxxxxxxxxxxx
```

#### Upload to Private Repository

```bash
python model_factory/upload_llava_vit_to_hf.py \
    --model_name hf_llava_vit_huge_ln \
    --weight_path /path/to/checkpoint.pth \
    --repo_id mycompany/private-llava-vit \
    --token hf_xxxxxxxxxxxxx \
    --private
```

#### Using Environment Variable

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxx

python model_factory/upload_llava_vit_to_hf.py \
    --model_name hf_llava_vit_base_ln \
    --weight_path /path/to/weights.pth \
    --repo_id myusername/llava-vit-base
```

### Repository Structure After Upload

After successful upload, your HuggingFace repository will contain:

```
your-repo/
├── config.json                    # Model configuration
├── pytorch_model.bin              # Model weights
├── configuration_llava_vit.py     # Configuration class definition
├── modeling_llava_vit.py          # Model class definition
├── preprocessor_config.json       # Image processor config
├── README.md                      # Auto-generated model card
└── example_usage.py               # Example usage code
```

### Model Features

- ✅ **3D Rotary Position Embeddings (RoPE)**: Supports spatial-temporal understanding with 4:6:6 split for T, H, W dimensions
- ✅ **Flash Attention 2**: Optimized attention implementation for better performance
- ✅ **Multi-head Attention Pooling**: PMA-style pooling for rich representation
- ✅ **Flexible Input**: Supports both 2D (images) and 3D (videos) inputs
- ✅ **Masking Support**: Compatible with MAE-style visible indices

### Output Specifications

The model returns `BaseModelOutputWithPooling` containing:

- `last_hidden_state`: Token embeddings of shape (batch_size, sequence_length, hidden_size)
- `pooler_output`: Pooled representation of shape (batch_size, hidden_size)
- `hidden_states`: (optional) Hidden states from all layers
- `attentions`: (optional) Attention weights from all layers

### Important Notes

1. **Trust Remote Code**: You need to set `trust_remote_code=True` when loading the model
2. **Flash Attention**: Requires `flash_attn` package for optimal performance
3. **Precision**: Recommended to use bfloat16 or float16 for best performance
4. **Image Size**: Default is 448x448, but can handle other sizes

### Troubleshooting

#### 1. Token Error

```
❌ Error: HuggingFace token required!
```

**Solution**: Make sure you provide a valid HuggingFace token with write permissions.

#### 2. Weight Loading Failure

```
❌ Failed to load weights
```

**Solution**: Check that the weight file path is correct and weights match the selected model architecture.

#### 3. Upload Failure

If upload fails, files are saved locally in `/tmp/llava_vit_upload_*` directory. You can manually upload them to HuggingFace.

### Additional Resources

- HuggingFace Documentation: https://huggingface.co/docs
- Model Hub: https://huggingface.co/models
- Transformers Library: https://github.com/huggingface/transformers

### License

Apache 2.0
