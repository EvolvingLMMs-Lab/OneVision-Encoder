## UNIT: Unifying Image and Text Recognition in One Vision Encoder

- Paper: [Arxiv](https://arxiv.org/abs/2409.04095) [NeurIPS 2024]
- Model: [UNIT_600M](https://huggingface.co/yeeaa/UNIT_600M/tree/main), [UNIT_1B](https://huggingface.co/yeeaa/UNIT_1B/tree/main)


## Install
This project supports both NVIDIA and Ascend GPUs.

- Dependencies & Environment
  - Python >= 3.9
  - NVIDIA GPU, CUDA >= 11.7
  - ASCEND NPU (Recommend to use 910B), CANN 8.0.0 RC1, torch-npu = 2.1.0  

- Install pytorch packages
```Shell
pip install torch==2.1.0 
pip install timm==0.9.12 
pip install transformers==4.32.1
```


## Usage

```Python
import torch
from PIL import Image
from transformers import CLIPImageProcessor

from unit import UNITModel

### uncomment to use Ascend NPU
# import torch_npu
# from torch_npu.npu import amp 
# from torch_npu.contrib import transfer_to_npu

# use UNIT_600M model
model_path = "/path/to/UNIT_600M/"
### uncomment to use UNIT_1B model
# model_path = "/path/to/UNIT_1B/"

model = UNITModel.from_pretrained(model_path)

model.to(device='cuda')
model.eval()

image_processor = CLIPImageProcessor.from_pretrained(model_path)

image = Image.open("test.jpg").convert('RGB')

image_input = image_processor(image)['pixel_values'][0]
image_tensor = torch.tensor(image_input).unsqueeze(0).to(torch.bfloat16).cuda()

with torch.set_grad_enabled(False):
    cls_tokens, spatial_tokens = model(image_tensor)

### Note: Applying a LayerNorm layer to these tokens is crucial before feeding them into LLMs.
```

## Results
- MLLM downstrean tasks
  
| Method  |  GQA         | OKVQA |  ChartQA | DocVQA | InfoVQA | OCRBench | POPE | MME | SEED-Image | MathVista | 
| ---- | ---- | ---- | ---- | ---- | ---- | ---- |  ---- | ---- | ---- | ---- | 
| CLIP-L                   |        62.34 | 56.97 |  51.96 |      57.19 |       29.31 | 382  | 84.67 | 1503.60| 69.79 | 42.7 | 
| SigLIP                    |        63.02 | 61.06 |  56.48 |      61.97 |       29.70 | 429 | 85.93 |  1489.37 | 71.63 |  44.2 | 
| UNIT-600M                    |       63.89 | 61.52|  61.0 |      65.49 |       31.92 | 480|  85.81 |  1529.76 | 72.81 |  44.6 |  
| UNIT-1B                    |       64.90 | 56.78 |  66.64 |      71.34 |       34.81 | 540 |   87.54 |  1531.92 | 73.15 |  44.3 | 

## Citation
If you use the code in your research, please cite:

```bib
@INPROCEEDINGS{Zhu2024UNIT,
    author = {Zhu, Yi and Zhou, Yanpeng and Wang, Chunwei and Cao, Yang and Han, Jianhua and Hou, Lu and Xu, Hang.},
    title = {UNIT: Unifying Image and Text Recognition in One Vision Encoder},
    booktitle = {The Thirty-eighth Annual Conference on Neural Information Processing Systems (NeurIPS)},
    year = {2024}
}
```
