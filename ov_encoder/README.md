# LlavaViT - HuggingFace 兼容模型包

这个目录包含用于 HuggingFace Hub 上传的 LlavaViT 模型实现。

## 文件说明

- `configuration_llava_vit.py` - 模型配置类
- `modeling_llava_vit.py` - 模型实现
- `__init__.py` - 包初始化文件

## 与 model_factory 的关系

这个目录中的代码与 `model_factory/vit_preview_v0_hf.py` 保持架构一致，但为 HuggingFace 上传进行了优化：

- 独立的配置文件
- 使用相对导入
- Flash Attention 自动降级
- 完整的 HuggingFace 兼容性

## 验证架构一致性

使用根目录下的验证脚本：

```bash
python verify_architecture_alignment.py
```

或使用详细模式：

```bash
python verify_architecture_alignment.py --verbose
```

## 使用方法

### 作为 Python 包导入

```python
from llavavit import LlavaViTConfig, LlavaViTModel

# 创建配置
config = LlavaViTConfig(
    hidden_size=1024,
    num_hidden_layers=24,
    num_attention_heads=16,
    image_size=448,
)

# 创建模型
model = LlavaViTModel(config)
```

### 上传到 HuggingFace

1. 保存模型：
```python
model.save_pretrained("output_dir")
```

2. 将此目录的文件复制到输出目录：
```bash
cp llavavit/configuration_llava_vit.py output_dir/
cp llavavit/modeling_llava_vit.py output_dir/
```

3. 修改 `output_dir/config.json` 添加 `auto_map`：
```json
{
  "auto_map": {
    "AutoConfig": "configuration_llava_vit.LlavaViTConfig",
    "AutoModel": "modeling_llava_vit.LlavaViTModel"
  }
}
```

4. 上传到 HuggingFace Hub

### 从 HuggingFace 加载

```python
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "your-username/model-name",
    trust_remote_code=True
)
```

## 特性

- ✅ 完整的 AutoModel 支持
- ✅ 图像和视频输入
- ✅ Masking / 可见索引支持
- ✅ Flash Attention 2（可选）
- ✅ 3D RoPE 位置编码
- ✅ 多头注意力池化

## 模型架构

支持的模型大小：

| 模型 | Hidden Size | Layers | Heads | Patch Size |
|------|------------|--------|-------|------------|
| Small | 384 | 6 | 6 | 16 |
| Base | 768 | 12 | 12 | 16 |
| Large | 1024 | 24 | 16 | 14 |
| Huge | 1536 | 27 | 24 | 14 |
| Giant | 1536 | 40 | 16 | 14 |

## 开发

修改此目录中的代码后，运行验证脚本确保与 model_factory 保持一致：

```bash
python verify_architecture_alignment.py
```

## 许可证

Apache 2.0
