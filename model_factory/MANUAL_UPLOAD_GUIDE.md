# 手动上传 LlavaViT 到 HuggingFace 指南

## 文件说明

已为您创建了两个独立的文件用于 HuggingFace 上传：

- `configuration_llava_vit.py` (4.8KB) - 模型配置类
- `modeling_llava_vit.py` (28KB) - 模型实现

## 快速上传步骤

### 1. 保存模型权重

```python
import timm
import torch

# 创建并加载模型
model = timm.create_model('hf_llava_vit_large_ln', pretrained=False)
checkpoint = torch.load('your_checkpoint.pth', map_location='cpu')
model.load_state_dict(checkpoint.get('model', checkpoint), strict=False)

# 保存为 HuggingFace 格式
output_dir = "llava_vit_output"
model.save_pretrained(output_dir)
```

### 2. 复制必要文件

```bash
# 复制配置和模型文件到输出目录
cp model_factory/configuration_llava_vit.py llava_vit_output/
cp model_factory/modeling_llava_vit.py llava_vit_output/
```

### 3. 修改 config.json

在 `llava_vit_output/config.json` 中添加 `auto_map` 字段：

```json
{
  "architectures": ["LlavaViTModel"],
  "model_type": "llava_vit",
  "hidden_size": 1024,
  "num_hidden_layers": 24,
  ...其他配置...,
  "auto_map": {
    "AutoConfig": "configuration_llava_vit.LlavaViTConfig",
    "AutoModel": "modeling_llava_vit.LlavaViTModel"
  }
}
```

### 4. 创建 README.md

在 `llava_vit_output/README.md` 中添加模型说明：

```markdown
---
license: apache-2.0
tags:
- vision
- image-classification
- video-understanding
library_name: transformers
---

# LlavaViT Model

## Usage

```python
from transformers import AutoModel
import torch

model = AutoModel.from_pretrained("your-username/model-name", trust_remote_code=True)
pixel_values = torch.randn(1, 3, 448, 448)
outputs = model(pixel_values=pixel_values)
```
```

### 5. 上传到 HuggingFace Hub

使用命令行上传：

```bash
# 安装 huggingface_hub
pip install huggingface_hub

# 登录
huggingface-cli login

# 上传
huggingface-cli upload your-username/model-name llava_vit_output/ --repo-type model
```

或使用 Python：

```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="llava_vit_output",
    repo_id="your-username/model-name",
    repo_type="model"
)
```

## 完整的目录结构

上传后的目录应该包含：

```
your-repo/
├── config.json                    # 模型配置（包含 auto_map）
├── pytorch_model.bin              # 模型权重
├── configuration_llava_vit.py     # 配置类
├── modeling_llava_vit.py          # 模型类
└── README.md                      # 模型说明
```

## 使用上传的模型

```python
from transformers import AutoModel, AutoConfig
import torch

# 加载配置
config = AutoConfig.from_pretrained("your-username/model-name", trust_remote_code=True)
print(f"Model type: {config.model_type}")
print(f"Hidden size: {config.hidden_size}")

# 加载模型
model = AutoModel.from_pretrained("your-username/model-name", trust_remote_code=True)

# 使用模型
pixel_values = torch.randn(1, 3, 448, 448)
outputs = model(pixel_values=pixel_values)

print(f"Last hidden state shape: {outputs.last_hidden_state.shape}")
print(f"Pooled output shape: {outputs.pooler_output.shape}")
```

## 支持的输入格式

### 图像输入
```python
# 4D 张量: (batch_size, channels, height, width)
images = torch.randn(2, 3, 448, 448)
outputs = model(pixel_values=images)
```

### 视频输入
```python
# 5D 张量: (batch_size, channels, num_frames, height, width)
videos = torch.randn(1, 3, 8, 448, 448)
outputs = model(pixel_values=videos)
```

### 使用 Masking
```python
# 只处理部分 patches
pixel_values = torch.randn(1, 3, 448, 448)
num_patches = (448 // 14) ** 2
visible_indices = torch.arange(num_patches // 2).unsqueeze(0)

outputs = model(pixel_values=pixel_values, visible_indices=visible_indices)
```

## 注意事项

1. **trust_remote_code=True 是必需的**，因为代码在仓库中而不是 transformers 库里
2. **Flash Attention**: 如果安装了 `flash_attn`，会自动使用；否则会降级到 eager attention
3. **模型精度**: 建议使用 `torch_dtype=torch.bfloat16` 或 `torch.float16` 以提高性能
4. **默认图像大小**: 448x448，但可以处理其他尺寸

## 常见问题

### Q: 如何更新已上传的模型？

A: 只需重新运行上传命令，会覆盖现有版本。

### Q: 如何创建私有仓库？

A: 在 HuggingFace 网站创建仓库时选择 "Private"，或在上传时添加 `--private` 参数。

### Q: Flash Attention 是否必需？

A: 不是。如果未安装，模型会自动使用 eager attention（标准注意力实现）。

### Q: 如何添加图像处理器？

A: 使用 `CLIPImageProcessor`：

```python
from transformers import CLIPImageProcessor

processor = CLIPImageProcessor(
    size={"height": 448, "width": 448},
    image_mean=[0.48145466, 0.4578275, 0.40821073],
    image_std=[0.26862954, 0.26130258, 0.27577711],
)
processor.save_pretrained("llava_vit_output")
```

## 自动化上传（推荐）

如果您想自动化整个流程，可以使用我们提供的脚本：

```bash
python model_factory/upload_llava_vit_to_hf.py \
    --model_name hf_llava_vit_large_ln \
    --weight_path your_checkpoint.pth \
    --repo_id your-username/model-name \
    --token YOUR_HF_TOKEN
```

该脚本会自动处理所有步骤，包括：
- 加载和转换权重
- 创建配置和模型文件
- 添加 auto_map
- 生成 README
- 上传到 HuggingFace

## 需要帮助？

参考完整文档：
- `README_UPLOAD_TO_HF.md` - 详细的双语文档
- `QUICK_START_CN.md` - 中文快速指南
- `AUTOMODEL_IMPLEMENTATION_SUMMARY.md` - 技术实现细节
