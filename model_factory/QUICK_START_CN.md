# LlavaViT 上传到 HuggingFace 快速指南

## 一键上传

```bash
# 基础命令
python model_factory/upload_llava_vit_to_hf.py \
    --model_name hf_llava_vit_large_ln \
    --weight_path /path/to/checkpoint.pth \
    --repo_id your-username/llava-vit-large \
    --token YOUR_HF_TOKEN
```

## 上传后使用

```python
from transformers import AutoModel

# 加载模型（需要设置 trust_remote_code=True）
model = AutoModel.from_pretrained(
    "your-username/llava-vit-large", 
    trust_remote_code=True
)

# 使用模型
import torch
pixel_values = torch.randn(1, 3, 448, 448)
outputs = model(pixel_values=pixel_values)
```

## 支持的模型

- `hf_llava_vit_small_ln` - 小型 (384 hidden, 6 layers)
- `hf_llava_vit_base_ln` - 基础 (768 hidden, 12 layers)
- `hf_llava_vit_large_ln` - 大型 (1024 hidden, 24 layers)
- `hf_llava_vit_huge_ln` - 超大型 (1536 hidden, 27 layers)
- `hf_llava_vit_giant_ln` - 巨型 (1536 hidden, 40 layers)

## 核心功能

上传脚本会自动：

1. ✅ 配置 `auto_map` - 让 AutoModel 能识别你的模型
2. ✅ 创建 `configuration_llava_vit.py` - 独立的配置类文件
3. ✅ 创建 `modeling_llava_vit.py` - 独立的模型类文件
4. ✅ 保存图像处理器配置 - CLIP 预处理参数
5. ✅ 生成模型卡片 (README.md) - 详细的使用说明
6. ✅ 创建示例代码 (example_usage.py) - 开箱即用的例子

## 重要配置说明

### auto_map 是什么？

`auto_map` 是 HuggingFace 的一个机制，让 `AutoModel.from_pretrained()` 知道去哪里找你的模型类。

脚本会自动在 `config.json` 中添加：

```json
{
  "auto_map": {
    "AutoConfig": "configuration_llava_vit.LlavaViTConfig",
    "AutoModel": "modeling_llava_vit.LlavaViTModel"
  }
}
```

### trust_remote_code 是什么？

因为你的模型代码不在 transformers 库里，而是在你的 HuggingFace 仓库中，所以用户加载时需要：

```python
model = AutoModel.from_pretrained(
    "your-repo", 
    trust_remote_code=True  # 必须设置！
)
```

这告诉 transformers："我信任这个仓库的代码，可以执行它"。

## 常见问题

### Q1: 为什么需要上传两个 Python 文件？

A: 
- `configuration_llava_vit.py` - 定义模型配置
- `modeling_llava_vit.py` - 定义模型结构

这是 HuggingFace 的标准做法，让你的模型可以被 AutoModel 识别。

### Q2: 权重文件怎么准备？

A: 你的 checkpoint.pth 应该是一个包含模型权重的文件，通常是：

```python
# 保存时
torch.save({
    'model': model.state_dict(),
    'epoch': epoch,
    # ... 其他信息
}, 'checkpoint.pth')

# 或者直接保存
torch.save(model.state_dict(), 'checkpoint.pth')
```

### Q3: 可以不提供权重吗？

A: 可以！不提供 `--weight_path` 时，会上传随机初始化的模型。这对于：
- 测试上传流程
- 分享模型架构
- 后续再更新权重

都很有用。

### Q4: 如何更新已上传的模型？

A: 直接用相同的 `--repo_id` 再次运行脚本，会覆盖之前的版本：

```bash
python model_factory/upload_llava_vit_to_hf.py \
    --model_name hf_llava_vit_large_ln \
    --weight_path /path/to/new_checkpoint.pth \
    --repo_id your-username/llava-vit-large \
    --token YOUR_HF_TOKEN
```

### Q5: 私有仓库怎么创建？

A: 添加 `--private` 参数：

```bash
python model_factory/upload_llava_vit_to_hf.py \
    --model_name hf_llava_vit_large_ln \
    --weight_path /path/to/checkpoint.pth \
    --repo_id your-username/private-model \
    --token YOUR_HF_TOKEN \
    --private
```

## 完整工作流

### 1. 训练模型

```python
import timm

# 创建并训练你的模型
model = timm.create_model('hf_llava_vit_large_ln', pretrained=False)
# ... 训练代码 ...

# 保存权重
torch.save({
    'model': model.state_dict(),
    'epoch': final_epoch,
}, 'trained_model.pth')
```

### 2. 上传到 HuggingFace

```bash
export HF_TOKEN=hf_your_token_here

python model_factory/upload_llava_vit_to_hf.py \
    --model_name hf_llava_vit_large_ln \
    --weight_path trained_model.pth \
    --repo_id your-username/my-awesome-vit
```

### 3. 在其他地方使用

```python
from transformers import AutoModel, CLIPImageProcessor
import torch
from PIL import Image

# 加载模型
model = AutoModel.from_pretrained(
    "your-username/my-awesome-vit",
    trust_remote_code=True
)
processor = CLIPImageProcessor.from_pretrained("your-username/my-awesome-vit")

# 加载并处理图片
image = Image.open("your_image.jpg")
inputs = processor(images=image, return_tensors="pt")

# 推理
with torch.no_grad():
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state
    pooled = outputs.pooler_output

print(f"Got embeddings: {embeddings.shape}")
```

## 高级用法

### 使用视频输入

```python
# 视频: [batch, channels, frames, height, width]
video = torch.randn(1, 3, 8, 448, 448)
outputs = model(pixel_values=video)
```

### 使用 Masking（提高效率）

```python
# 只处理部分 patches
pixel_values = torch.randn(1, 3, 448, 448)
num_patches = (448 // 14) ** 2
visible_indices = torch.arange(num_patches // 2).unsqueeze(0)  # 只用一半

outputs = model(
    pixel_values=pixel_values,
    visible_indices=visible_indices
)
```

### 批量推理

```python
# 批量处理多张图片
images = [Image.open(f"image_{i}.jpg") for i in range(10)]
inputs = processor(images=images, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    # outputs.last_hidden_state 的 batch_size 将是 10
```

## 性能优化建议

### 1. 使用半精度

```python
model = AutoModel.from_pretrained(
    "your-repo",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16  # 或 torch.float16
).cuda()
```

### 2. 使用 Flash Attention

确保安装了 `flash_attn`：

```bash
pip install flash-attn --no-build-isolation
```

### 3. 批量处理

```python
# 不好：逐个处理
for img in images:
    output = model(processor(img, return_tensors="pt"))

# 好：批量处理
inputs = processor(images=images, return_tensors="pt")
outputs = model(**inputs)
```

## 测试你的上传

使用提供的测试脚本：

```bash
python model_factory/test_automodel_loading.py your-username/llava-vit-large
```

这会自动测试：
- ✅ 配置加载
- ✅ 模型加载
- ✅ 图像输入
- ✅ 视频输入
- ✅ Masking 功能

## 需要帮助？

1. 查看详细文档：`model_factory/README_UPLOAD_TO_HF.md`
2. 查看示例代码：上传后的 `example_usage.py`
3. 测试加载：`test_automodel_loading.py`

## 检查清单

上传前确保：

- [ ] 已安装依赖：`pip install huggingface_hub transformers timm torch`
- [ ] 已获取 HF Token：https://huggingface.co/settings/tokens
- [ ] 权重文件路径正确
- [ ] 选择了正确的 model_name
- [ ] repo_id 格式正确：`username/model-name`

上传后验证：

- [ ] 访问 `https://huggingface.co/your-username/model-name` 检查文件
- [ ] README.md 显示正常
- [ ] 运行测试脚本验证加载
- [ ] 尝试实际推理

完成！🎉
