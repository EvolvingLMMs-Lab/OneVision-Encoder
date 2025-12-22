# LlavaViT 架构一致性验证指南

## 概述

此文档说明如何使用新创建的 `llavavit/` 目录和架构验证程序。

## 目录结构

```
LLaVA-ViT/
├── llavavit/                           ← 新建：用于 HuggingFace 上传
│   ├── __init__.py                     
│   ├── configuration_llava_vit.py      
│   ├── modeling_llava_vit.py           
│   └── README.md                       
├── model_factory/                      
│   ├── vit_preview_v0_hf.py           ← 原始实现
│   ├── configuration_llava_vit.py      ← 参考副本
│   ├── modeling_llava_vit.py           ← 参考副本
│   └── ...
└── verify_architecture_alignment.py    ← 新建：验证工具
```

## 为什么有两个位置？

1. **`llavavit/`** - 根目录下的独立包
   - 用于上传到 HuggingFace Hub
   - 作为标准 Python 包可以直接导入
   - 与 `model_factory` 保持架构一致

2. **`model_factory/`** - 模型工厂目录
   - 包含原始实现 (`vit_preview_v0_hf.py`)
   - 保留配置和模型文件的副本供参考
   - 包含其他模型变体和工具

## 使用架构验证程序

### 基本使用

验证 `llavavit/` 和 `model_factory/vit_preview_v0_hf.py` 的架构一致性：

```bash
python verify_architecture_alignment.py
```

### 详细模式

查看所有参数的详细比较：

```bash
python verify_architecture_alignment.py --verbose
```

### 验证内容

程序会检查：

1. **配置对齐** - 比较所有配置参数
   - hidden_size, num_layers, attention_heads 等
   - 确保两边使用相同的默认值

2. **结构对齐** - 验证模型结构
   - 参数名称是否一致
   - 参数形状是否匹配
   - 层数和组件是否相同

3. **前向传播对齐** - 测试推理
   - 使用相同输入
   - 验证输出形状一致
   - 确保前向传播正常工作

### 输出示例

```
================================================================================
🚀 开始架构一致性验证 / Starting Architecture Alignment Verification
================================================================================
✅ 成功导入 model_factory 模型 / Successfully imported model_factory model
✅ 成功导入 llavavit 模型 / Successfully imported llavavit model

📝 使用配置 / Using configuration:
   hidden_size: 768
   num_hidden_layers: 12
   num_attention_heads: 12
   ...

================================================================================
📋 配置对齐检查 / Configuration Alignment Check
================================================================================
  ✅ 所有配置参数一致 / All config parameters match

================================================================================
🏗️  模型结构对齐检查 / Model Structure Alignment Check
================================================================================
  ✅ 模型结构完全一致 / Model structures are identical
     共有参数 / Total parameters: 147

================================================================================
🔄 前向传播对齐检查 / Forward Pass Alignment Check
================================================================================
  ✅ 输出形状一致 / Output shapes match
  ✅ Pooler 输出形状一致 / Pooler output shapes match

================================================================================
📊 验证总结 / Verification Summary
================================================================================
  配置对齐 / Config Alignment              : ✅ 通过 / PASS
  结构对齐 / Structure Alignment           : ✅ 通过 / PASS
  前向传播对齐 / Forward Pass Alignment    : ✅ 通过 / PASS
================================================================================

🎉 所有检查通过！架构完全一致！
   All checks passed! Architectures are fully aligned!
```

## 使用 llavavit/ 上传到 HuggingFace

### 方法 1: 使用 llavavit 作为包

```python
# 直接从 llavavit 导入
from llavavit import LlavaViTConfig, LlavaViTModel
import torch

# 创建配置
config = LlavaViTConfig(
    hidden_size=1024,
    num_hidden_layers=24,
    num_attention_heads=16,
    image_size=448,
)

# 创建模型
model = LlavaViTModel(config)

# 加载权重（如果有）
# checkpoint = torch.load('checkpoint.pth')
# model.load_state_dict(checkpoint, strict=False)

# 保存为 HuggingFace 格式
output_dir = "hf_output"
model.save_pretrained(output_dir)
config.save_pretrained(output_dir)
```

### 方法 2: 复制文件到输出目录

```bash
# 1. 保存模型后，复制必要文件
cp llavavit/configuration_llava_vit.py hf_output/
cp llavavit/modeling_llava_vit.py hf_output/

# 2. 编辑 hf_output/config.json，添加 auto_map
# {
#   "auto_map": {
#     "AutoConfig": "configuration_llava_vit.LlavaViTConfig",
#     "AutoModel": "modeling_llava_vit.LlavaViTModel"
#   }
# }

# 3. 上传到 HuggingFace
huggingface-cli upload your-username/model-name hf_output/ --repo-type model
```

### 方法 3: 使用自动化脚本

```bash
python model_factory/upload_llava_vit_to_hf.py \
    --model_name hf_llava_vit_large_ln \
    --weight_path checkpoint.pth \
    --repo_id your-username/llava-vit-large \
    --token YOUR_HF_TOKEN
```

## 维护架构一致性

### 何时运行验证？

在以下情况下运行验证程序：

1. 修改 `llavavit/` 中的代码后
2. 更新 `model_factory/vit_preview_v0_hf.py` 后
3. 提交 PR 之前
4. 发布新版本之前

### 如果验证失败怎么办？

1. **配置不匹配** - 检查默认参数是否一致
2. **结构不匹配** - 确保层定义相同
3. **前向传播失败** - 检查前向传播逻辑

通常需要同步更新两个位置的代码。

## 开发工作流

### 修改模型代码

```bash
# 1. 修改 model_factory/vit_preview_v0_hf.py
vim model_factory/vit_preview_v0_hf.py

# 2. 同步修改到 llavavit/
# 手动编辑或使用脚本

# 3. 运行验证
python verify_architecture_alignment.py

# 4. 如果通过，提交更改
git add llavavit/ model_factory/vit_preview_v0_hf.py
git commit -m "Update model architecture"
```

## 常见问题

### Q: 为什么要在两个地方都保存文件？

A: 
- `model_factory/` 是开发和训练环境
- `llavavit/` 是为 HuggingFace 部署优化的独立包
- 保持两者一致确保部署的模型与训练的完全相同

### Q: 我应该修改哪个文件？

A:
- 如果是开发/训练: 修改 `model_factory/vit_preview_v0_hf.py`
- 如果只是上传: 使用 `llavavit/` 中的文件
- 重大修改: 两边都需要更新并运行验证

### Q: 验证程序报错怎么办？

A: 检查错误信息：
- Import 错误: 确保依赖已安装
- 结构不匹配: 检查代码是否同步
- 前向传播错误: 检查逻辑是否一致

### Q: 可以只使用 llavavit/ 吗？

A: 可以，但建议：
- 开发时使用 `model_factory/` 的完整环境
- 部署时使用 `llavavit/` 的独立包
- 定期运行验证确保一致性

## 总结

- ✅ `llavavit/` 用于 HuggingFace 上传和部署
- ✅ `model_factory/` 用于开发和训练
- ✅ `verify_architecture_alignment.py` 确保一致性
- ✅ 修改代码后始终运行验证
- ✅ 两个位置保持同步很重要

## 参考文档

- `llavavit/README.md` - llavavit 包的使用说明
- `model_factory/MANUAL_UPLOAD_GUIDE.md` - 手动上传指南
- `model_factory/README_UPLOAD_TO_HF.md` - 完整上传文档
- `model_factory/QUICK_START_CN.md` - 快速开始指南
