# LLaVA-Next-2401


## Results

已按每个指标列的最高分加粗：

| Data (PT) | Data (IT) | Model            | MMMU  | Math-Vista | MMB-ENG | MMB-CN | MM-Vet | LLaVA-Wild | SEED-IMG |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| N/A  | N/A   | GPT-4V          | 56.8 | 49.9 | 75.8 | 73.9 | **67.6** | -   | 71.6 |
| N/A  | N/A   | Gemini Ultra    | **59.4** | **53** | -    | -    | -    | -   | -    |
| N/A  | N/A   | Gemini Pro      | 47.9 | 45.2 | 73.6 | 74.3 | 64.3 | -   | 70.7 |
| 1.4B | 50M   | Qwen-VL-Plus    | 45.2 | 43.3 | -    | -    | 55.7 | -   | 65.7 |
| 1.5B | 5.12M | CogVLM-30B      | 32.1 | -    | -    | -    | 56.8 | -   | -    |
| 125M | ~1M   | Yi-VL-34B       | 45.9 | -    | -    | -    | -    | -   | -    |
| 558K | 665K  | LLaVA-1.5-13B   | 36.4 | 27.6 | 67.8 | 63.3 | 36.3 | 72.5 | 68.2 |
| 558K | 760K  | LLaVA-NeXT-34B  | 51.1 | 46.5 | **79.3** | **79** | 57.4 | **89.6** | **75.9** |

## Model Card


| Name | LLaVA-NeXT-7B | LLaVA-NeXT-13B | LLaVA-NeXT-34B |
|---|---:|---:|---:|
| Model Size — Total | 7.06B | 13.35B | 34.75B |
| Model Size — Vision Encoder | 303.5M | 303.5M | 303.5M |
| Model Size — Connector | 21M | 31.5M | 58.7M |
| Model Size — LLM | 6.74B | 13B | 34.39B |
| Resolution | 336 x [(2,2), (1,2), (2,1), (1,3), (3,1), (1,4), (4,1)] |  |  |
| Stage‑1 — Training Data | 558K |  |  |
| Stage‑1 — Trainable Module | Connector |  |  |
| Stage‑2 — Training Data | 760K |  |  |
| Stage‑2 — Trainable Module | Full model |  |  |
| Compute (#GPU x #Hours) | 8x20 | 16x24 | 32x30 |
| Training Data (#Samples) | 1318K |  |  |


# LLaVA-NeXT-2405

| Datasets | GPT4-V | Qwen1.5-110B | Qwen1.5-72B | LLaMA3-8B | Yi-34B | Vicuna-1.5-13B | Vicuna-1.5-7B | Mistral-7B |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AI2D* | 78.2 | **80.4** | 77.4 | 71.6 | 74.9 | 70.0 | 66.6 | 60.8 |
| ChartQA* | 78.5 | **79.7** | 77.0 | 69.5 | 68.7 | 62.2 | 54.8 | 38.8 |
| DocVQA* | - | **85.7** | 84.4 | 78.2 | 84.0 | 77.5 | 74.4 | 72.2 |
| MathVista | **49.9** | 49.0 | 46.6 | 37.5 | 46.0 | 35.1 | 34.4 | 37.4 |
| MMBench | 75.0 | **80.5** | **80.5** | 72.1 | 79.3 | - | - | - |
| MMMU | **56.8** | 50.1 | 49.9 | 41.7 | 49.7 | 35.9 | 35.1 | 33.4 |
| RealWorldQA | 61.4 | 63.1 | **65.4** | 60.0 | 61.0 | - | - | 54.4 |
| LLaVA‑W** | **98.0** | 90.4 | 89.2 | 80.1 | 88.8 | 72.3 | 72.3 | 71.7 |
| LLaVA‑Bench (Wilder) Small | **71.5** | 70.5 | 71.2 | 62.5 | - | - | - | - |
| LLaVA‑Bench (Wilder) Medium | **78.5** | 72.5 | 73.4 | 63.1 | - | - | - | - |
| MME‑Cognition | **517** | 454 | 460 | 368 | 397 | 317 | 323 | 324 |
| MME‑Perception | 1409 | **1747** | 1699 | 1604 | 1633 | 1575 | 1519 | 1501 |

*Train split observed during SFT stage.
**We report the evaluation results with GPT‑4‑0613 on LLaVA‑W.


## Model Card

| Name | L3‑LLaVA‑NeXT‑8B | LLaVA‑NeXT‑72B | LLaVA‑NeXT‑110B |
|---|---:|---:|---:|
| Size — Total | 8.35B | 72.7B | 111.5B |
| Size — Vision Encoder | 303.5M | 303.5M | 303.5M |
| Size — Connector | 20.0M | 72.0M | 72.0M |
| Size — LLM | 8.03B | 72.3B | 111.0B |
| Resolution | — | 336×(2x2;1x2;2x1;1x3;3x1) | — |
| Stage‑1 — Data | — | 558K | — |
| Stage‑1 — Module | — | Connector | — |
| Stage‑2 — Data | — | ~790K | — |
| Stage‑2 — Module | — | Full model | — |
| Compute (GPU×h) | 16 A100‑80G × 15–20 | 128 A100‑80G × ~18 | 128 H800‑80G × ~18 |
| Total Data (#) | — | 1348K | — |


# LLaVA-NeXT-2406

| Vision Encoder | Model size | Res. | Visual Tokens | Pretrained Source | Pretrained Amount | Seen Samples | Time Cost | Avg. | AI2D  | ChartQA  | DocVQA  | MathVista  | MME | MMMU (dev) | LLaVA‑W | ScienceQA  | Image‑DC (EN‑100) |
|---|---:|---:|---:|---|---:|---:|---:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| CLIP‑L   | 0.3B | 224 | 256 × 5  | WIT            | 0.4B | 13B | ~12H | 63.4 | 67.0 | 60.3 | 62.2 | 33.5 | **78.8** | **38.2** | 71.7 | 71.9 | 86.7 |
| CLIP‑L   | 0.3B | 336 | 576 × 5  | WIT            | 0.4B | 13B | ~30H | 65.3 | 67.4 | **65.2** | **74.5** | **35.4** | 77.3 | 36.6 | 72.6 | 71.0 | 87.6 |
| EVA‑02‑E | 4.7B | 224 | 256 × 5  | LAION          | 2B   | 9B  | ~30H | 61.0 | 66.9 | 42.4 | 65.4 | 33.5 | 77.5 | 33.6 | 73.9 | 69.5 | 85.9 |
| EVA‑8B   | 8B   | 224 | 256 × 5  | LAION + COYO   | 2B   | 9B  | ~24H | 63.3 | 67.8 | 56.0 | 66.3 | 32.1 | 77.1 | 35.0 | 75.9 | 71.5 | 88.0 |
| EVA‑8B   | 8B   | 448 | 1024 × 5 | LAION + COYO   | 2B   | 9B  | ~75H | 64.4 | 68.4 | 59.7 | 69.8 | 33.4 | 77.3 | 34.6 | 74.4 | 71.9 | **90.2** |
| SO400M   | 0.4B | 384 | 729 × 5  | WebLI          | 10B  | 40B | ~36H | **66.4** | **69.4** | 62.7 | 72.5 | 35.1 | 76.5 | 34.8 | **85.8** | **72.4** | 88.8 |


### Configurations

| Section | Key | Value |
|---|---|---|
| Architecture | Image Encoder | Google SO4000M (384×384) |
| Architecture | Connector | 2‑Layer ReLU MLP |
| Architecture | LLM | Qwen‑1.5 0.5B |
| — | # Total parameters | 0.9B |
| Visual Representations | Pattern | Dynamic: 336 × {2×2, 1×{2,3}, {2,3}×1} |
| Stage‑1 | Training Data | 558K |
| Stage‑1 | Trainable Module | Connector |
| Stage‑2 | Training Data | 790K |
| Stage‑2 | Trainable Module | Full model |
| Training Data | # samples | 1348K = 558K + 790K |
| Training Schedule | Learning rate | LLM: 2e‑5 / Vision: 2e‑6 |
| Training Schedule | Batch Size | 64 |

### Thresholded Bilinear Interpolation

For AnyRes with a grid configuration of width $a$, height $b$, and $\#\text{token}$ $T$ per grid, the total number of visual tokens is
$$
L = (a \times b + 1) \times T.
$$

We consider a threshold $\tau$, and reduce the tokens per grid using bilinear interpolation if needed:
$$
T_{\text{new}} =
\begin{cases}
\dfrac{\tau}{a \times b + 1}, & \text{if } L > \tau,\\[6pt]
T, & \text{if } L \le \tau.
\end{cases}
$$


### Impact on Max. #Grids in Anyres and Max. #Tokens.
| Max. #Grids | Max. #Tokens | Training Time | Interpolation | AI2D  | ChartQA  | DocVQA  | InfoVQA  | Image‑DC  | Video‑DC  | SynDOG  | OK‑VQA  | POPE  | ScienceQA  | VizWiz‑VQA  | MMMU  |
|---|---|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2×2 | `(4+1)*729` | 6H30M | FALSE | 51.1 | 49.2 | 58.8 | 25.7 | 71.1 | 64.1 | 425.7 | 36.5 | 85.4 | **59.6** | 29.2 | 28.2 |
| 4×4 | `(4+1)*729` | 7H30M | TRUE | **52.8** | 49.4 | 58.1 | 26.0 | 69.9 | 63.5 | 433.6 | 36.0 | 85.8 | 57.9 | 31.0 | 28.6 |
| 5×5 | `(4+1)*729` | 7H50M | TRUE | 52.4 | 49.6 | 57.6 | 26.9 | **72.9** | 63.8 | 435.6 | 36.5 | 86.1 | 58.5 | 28.7 | 28.4 |
| 6×6 | `(4+1)*729` | 8H05M | TRUE | 52.7 | 50.1 | 56.7 | **27.1** | 71.0 | 64.2 | 437.2 | 35.9 | 85.9 | 58.4 | 32.2 | 28.3 |
| 6×6 | `(9+1)*729` | 11H14M | TRUE | 52.7 | 55.8 | **62.7** | 26.7 | 71.7 | 64.6 | 438.9 | 42.0 | 86.1 | 58.7 | **34.7** | **29.3** |
| 6×6 | `(16+1)*729` | 13H10M | TRUE | 52.7 | **56.1** | 62.2 | **27.1** | 70.2 | **65.2** | **443.5** | **42.5** | **87.4** | 58.2 | 32.8 | 27.4 |


### Effectiveness with LLM Scaling
| LLM (Qwen‑1.5) | Max. #Grids | Max. #Tokens | Interp. | AI2D  | ChartQA  | DocVQA  | InfoVQA  | Image‑DC  | Video‑DC  | SynDOG  | OKVQA  | POPE  | ScienceQA  | VizWiz‑VQA  | MMMU  |
|---|---|---|:---:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 0.5B | 2×2 | `(4+1)*729` | FALSE | 51.1 | 49.2 | 58.8 | 25.7 | 71.1 | **62.4** | 418.5 | 36.5 | 85.1 | **59.5** | 28.8 | 28.2 |
| 0.5B | 6×6 | `(9+1)*729` | TRUE  | **52.7** | **55.8** | **62.7** | **26.7** | **71.7** | **62.4** | **443.5** | **42.0** | **86.1** | 58.7 | **34.7** | **29.3** |
| 1.8B | 2×2 | `(4+1)*729` | FALSE | **61.9** | 56.2 | 66.0 | 30.5 | 80.1 | 70.2 | 447.1 | 43.6 | **86.9** | 63.7 | **51.0** | 32.0 |
| 1.8B | 6×6 | `(9+1)*729` | TRUE  | 60.9 | **56.7** | **67.5** | **31.3** | **82.0** | **71.0** | **459.1** | **46.5** | **86.9** | **64.4** | 48.8 | **32.6** |
| 4B   | 2×2 | `(4+1)*729` | FALSE | **71.5** | **65.0** | 73.8 | 34.8 | 84.2 | 74.5 | 456.7 | 47.5 | **87.1** | **71.1** | **58.7** | **34.4** |
| 4B   | 6×6 | `(9+1)*729` | TRUE  | 70.2 | **65.0** | **77.2** | **41.1** | **86.3** | **76.4** | **467.7** | **50.6** | 86.3 | 70.1 | 58.0 | 32.0 |
| 7B   | 2×2 | `(4+1)*729` | FALSE | **72.9** | 66.3 | 75.5 | **36.9** | **87.9** | 69.8 | 458.2 | **50.2** | 86.9 | **71.2** | **61.4** | **37.2** |
| 7B   | 6×6 | `(9+1)*729` | TRUE  | 71.7 | **69.5** | **79.0** | 36.4 | 86.4 | **71.4** | **467.1** | 47.9 | **87.3** | 70.2 | 57.4 | **37.2** |
| 14B  | 2×2 | `(4+1)*729` | FALSE | **77.6** | 72.2 | 80.0 | 44.4 | **89.6** | 74.2 | 460.8 | **57.7** | 87.3 | **78.9** | **64.2** | **44.2** |
| 14B  | 6×6 | `(9+1)*729` | TRUE  | 76.1 | **74.0** | **83.6** | **46.9** | 87.8 | **78.1** | **470.4** | 53.2 | **87.9** | 76.7 | 61.5 | 40.3 |

已按“每列取最大值”加粗（并列同样加粗）。

已按“跨4个表格的全局最大值”加粗（仅对数值列；本数据无并列最大值）。

### Baseline
| Stage‑1 | Stage‑1.5 | Stage‑2 | Avg. | AI2D  | ChartQA  | DocVQA  | InfoVQA  | MathVista  | MME | LLaVA‑W | ScienceQA  | Image‑DC  |
|---|---|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 558K | — | 790K | 67.4 | 67.4 | 65.2 | 74.5 | 34.5 | 35.4 | 65.5 | 72.6 | 70.8 | 87.5 |

### High‑Quality Knowledge: Detailed Re‑Captioning（Stage‑1.5 + Stage‑2=790K）
| Stage‑1 | Stage‑1.5 | Stage‑2 | Avg. | AI2D  | ChartQA  | DocVQA  | InfoVQA  | MathVista  | MME | LLaVA‑W | ScienceQA  | Image‑DC  |
|---|---|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 558K | 118K (ReCap) | 790K | 68.6 | 66.9 | 66.0 | 75.5 | 36.0 | 36.2 | 65.7 | 79.7 | 71.0 | 87.6 |
| 558K | 558K (ReCap) | 790K | 69.4 | 70.1 | 67.8 | 76.9 | **39.4** | 36.2 | 65.1 | 79.4 | 71.5 | 88.2 |
| 558K | 3M (ReCap) | 790K | **70.7** | **72.7** | **68.3** | 77.7 | 38.1 | **38.6** | 65.7 | 80.1 | 72.0 | **90.4** |
| 558K | COCO118K | 790K | 67.4 | 66.1 | 65.7 | 73.7 | 35.1 | 35.5 | 65.8 | 75.9 | 70.1 | 86.2 |
| 558K | BLIP558K | 790K | 68.3 | 67.3 | 66.1 | 75.4 | 36.8 | 35.8 | 66.6 | 77.6 | 70.9 | 86.6 |
| 558K | CC3M | 790K | 68.7 | 67.5 | 66.3 | 77.0 | 38.1 | 34.9 | **66.8** | 79.6 | 71.0 | 86.5 |

### High‑Quality Knowledge: New Domain Knowledge
| Stage‑1 | Stage‑1.5 | Stage‑2 | Avg. | AI2D  | ChartQA  | DocVQA  | InfoVQA  | MathVista  | MME | LLaVA‑W | ScienceQA  | Image‑DC  |
|---|---|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 558K | UReader 100K | 790K | 67.2 | 66.2 | 67.2 | 77.6 | **36.9** | 34.2 | 63.9 | 70.7 | 71.9 | 86.1 |
| 558K | ShareGPT4V Chinese Caption 100K | 790K | 68.7 | 68.7 | 67.1 | 75.1 | **36.9** | 36.3 | 64.4 | 78.1 | 72.2 | 87.4 |
| 558K | SynDOG 1M | 790K | 66.3 | 66.4 | 62.0 | 72.9 | 36.7 | 31.6 | 65.8 | 76.9 | **72.5** | 82.3 |

### Mixed Data
| Stage‑1 | Stage‑1.5 | Stage‑2 | Avg. | AI2D  | ChartQA  | DocVQA  | InfoVQA  | MathVista  | MME | LLaVA‑W | ScienceQA  | Image‑DC  |
|---|---|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 558K | 118K (ReCap) + UReader | 790K | 68.9 | 66.9 | 68.1 | **79.2** | 37.8 | 36.0 | 64.2 | 77.4 | 71.0 | 88.5 |
| 558K | 118K (ReCap) + UReader + Evol‑143K | 790K | 69.4 | 66.2 | 67.7 | 78.5 | 38.1 | 36.2 | 66.1 | **81.4** | 71.3 | 88.1 |




下面把四个表格合并为一个表格（已按“跨四个表格的全局最大值”加粗，仅数值列参与比较）。

| Stage‑1 | Stage‑1.5 | Stage‑2 | Avg. | AI2D | ChartQA | DocVQA | InfoVQA | MathVista | MME | LLaVA‑W | ScienceQA | Image‑DC |
|---|---|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|**baseline**||
| 558K | — | 790K | 67.4 | 67.4 | 65.2 | 74.5 | 34.5 | 35.4 | 65.5 | 72.6 | 70.8 | 87.5 |
| **高质量知识，详尽的描述**||
| 558K | 118K (ReCap) | 790K | 68.6 | 66.9 | 66.0 | 75.5 | 36.0 | 36.2 | 65.7 | 79.7 | 71.0 | 87.6 |
| 558K | 558K (ReCap) | 790K | 69.4 | 70.1 | 67.8 | 76.9 | **39.4** | 36.2 | 65.1 | 79.4 | 71.5 | 88.2 |
| 558K | 3M (ReCap) | 790K | **70.7** | **72.7** | **68.3** | 77.7 | 38.1 | **38.6** | 65.7 | 80.1 | 72.0 | **90.4** |
| 558K | COCO118K | 790K | 67.4 | 66.1 | 65.7 | 73.7 | 35.1 | 35.5 | 65.8 | 75.9 | 70.1 | 86.2 |
| 558K | BLIP558K | 790K | 68.3 | 67.3 | 66.1 | 75.4 | 36.8 | 35.8 | 66.6 | 77.6 | 70.9 | 86.6 |
| 558K | CC3M | 790K | 68.7 | 67.5 | 66.3 | 77.0 | 38.1 | 34.9 | **66.8** | 79.6 | 71.0 | 86.5 |
| **高质量知识，新的域知识**||
| 558K | UReader 100K | 790K | 67.2 | 66.2 | 67.2 | 77.6 | 36.9 | 34.2 | 63.9 | 70.7 | 71.9 | 86.1 |
| 558K | ShareGPT4V Chinese Caption 100K | 790K | 68.7 | 68.7 | 67.1 | 75.1 | 36.9 | 36.3 | 64.4 | 78.1 | 72.2 | 87.4 |
| 558K | SynDOG 1M | 790K | 66.3 | 66.4 | 62.0 | 72.9 | 36.7 | 31.6 | 65.8 | 76.9 | **72.5** | 82.3 |
| **混合数据**||
| 558K | 118K (ReCap) + UReader | 790K | 68.9 | 66.9 | 68.1 | **79.2** | 37.8 | 36.0 | 64.2 | 77.4 | 71.0 | 88.5 |
| 558K | 118K (ReCap) + UReader + Evol‑143K | 790K | 69.4 | 66.2 | 67.7 | 78.5 | 38.1 | 36.2 | 66.1 | **81.4** | 71.3 | 88.1 |

需要我把它导出为 CSV/Excel 或者生成一个 Markdown 文件吗？

# Section 3.1 - Language-Image Alignment

We considered two groups of data to align the image features into the text embedding space:

1. Public Data: BLIP558K, CC3M, and CC12M.
2. Web Data: to avoid the limitations imposed by the quantity of existing public data, we consider multimodal image-text data from the internet at similar scales. We applied quality control measures to filter this data to match public data at similar scales of 0.6M, 3M and 12M. The well-trained projector is used directly to run full model tuning with visual instructions, and the results are reported below. With tuning the projector only, the data scaling is less effective with public raw data, while more effective with top-quality data mixture, followed by the randomly selected data mixture from the same web dataset.

## Public Data


| Stage-1 Data | Quality Measure | Avg. | AI2D* (test) | ChartQA* (test) | DocVQA (val) | MathVista  | MME | LLaVA-W | ScienceQA (IMG) | Image-DC (EN) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 558K  | N/A | **67.4** | **67.4** | **65.2** | **74.5** | **35.4** | 65.54 | 72.6 | **71.0** | **87.6** |
| CC3M  | N/A | 67.2 | 66.0 | 62.4 | 73.7 | **35.4** | **66.60** | **79.9** | 69.5 | 84.3 |
| CC12M | N/A | 66.4 | 66.8 | 58.9 | 72.5 | 34.7 | 64.14 | 79.6 | 69.7 | 85.1 |

## Web Dataset*

| Stage-1 Data | Quality Measure | Avg. | AI2D* (test) | ChartQA* (test) | DocVQA (val) | MathVista  | MME | LLaVA-W | ScienceQA (IMG) | Image-DC (EN) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Web 0.6M | Top Quality | 68.2 | 67.8 | **64.8** | 74.2 | 35.2 | 66.61 | 80.1 | **71.4** | **85.4** |
| Web 0.6M | Random      | 67.7 | 68.0 | 64.4 | 73.7 | 34.4 | 65.83 | 80.6 | 70.8 | 83.7 |
| Web 3M   | Top Quality | 68.4 | 67.8 | 62.9 | 73.8 | 34.1 | 67.05 | **86.4** | 70.3 | 84.5 |
| Web 3M   | Random      | 67.6 | 68.2 | 62.8 | 73.2 | 33.4 | 66.00 | 83.0 | 70.1 | 84.3 |
| Web 12M  | Top Quality | **69.3** | **68.6** | 64.5 | **74.9** | **35.8** | **69.34** | 85.2 | 71.0 | 85.1 |
| Web 12M  | Random      | 68.2 | 68.2 | 62.4 | 73.4 | 34.1 | 66.87 | 85.6 | 70.9 | 83.8 |



# Section 4 - Training Techniques

We illustrate two key techniques for training LLaVA‑NExT‑Interleave using the M4‑Instruct dataset, and provide ablation studies for analysis.

## Continue training from single-image model

To better leverage the pre‑trained vision‑language alignment, we adopt an off‑the‑shelf LLaVA‑NExT‑Image as the base model, which has gone through stage‑1 558K image‑caption pre‑training and stage‑2 760K single‑image fine‑tuning. On top of this checkpoint, we perform the multi‑image instruction tuning with the M4‑Instruct dataset.

As shown by the ablation below, compared to the training based on the stage‑1 pre‑training, the “continue training from stage 2” scheme performs better. In this way, we can inherit the instruction‑following capabilities in single‑image tasks, and better extend the scope to multi‑image, video, and 3D scenarios. In addition, single‑image performance cannot be maintained if training directly from stage 1.

### Ablation: Stage‑1 vs Stage‑2 as the starting point

| Continue training | Mantis‑Eval | BLINK | Q‑Bench | NLVR2 | ScanQA | ai2d | chartqa | docvqa | MME* | pope | SCI-Img | Act‑QA | MVBench | VDD | VGT| Detail | Context | Temporal | Consistency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| stage‑1 | 41.0 | 37.6 | 47.0 | 54.0 | 27.7 | 46.3 | 38.3 | 47.5 | 47.1 | 85.4 | 59.4 | 44.7/12.17 | 43.0 | 2.96 | 2.97 | 2.87 | 3.49 | 2.42 | 3.14 |
| stage‑2 | 45.6 | 39.2 | 52.0 | 67.8 | 29.3 | 52.2 | 52.2 | 59.2 | 52.0 | 86.8 | 60.6 | 48.0/2.84 | 45.6 | 3.25 | 3.12 | 2.97 | 3.62 | 2.36 | 3.27 |

## Mix training for in‑the‑front and interleaved formats

For interleaved multi‑image input, we have two format choices for the positions of image tokens during training. The first is to place all the image tokens in front of the text and refer to each image in the sentence with a special token, aligning with the format in the single‑image model. The second preserves the interleaved instruction to put image tokens in the place they are originally in, which extends models to real interleaved data format.

In the below ablation, we train on the multi‑image data with different formats. The results indicate that mixing two strategies during training leads to higher performance in both two inference schemes, which provides more flexibility for multi‑image input format by users.

### Ablation: training format × inference setting (in‑domain evaluation)

| Training Setting | Inference Setting | Avg | Spot the Difference | Visual Story Telling | Text‑rich VQA | Q‑Bench |
|---|---|---:|---:|---:|---:|---:|
| In‑the‑front format | Interleaved format | 52.88 | 36.8 | 30.5 | 70.1 | 74.0 |
| In‑the‑front format | In‑the‑front format | 54.27 | 36.6 | 32.8 | 74.7 | 75.3 |
| Interleaved format | Interleaved format | 55.38 | 37.8 | 32.9 | 76.2 | 76.0 |
| Interleaved format | In‑the‑front format | 52.35 | 36.1 | 29.0 | 72.9 | 71.8 |
| Mix format | Interleaved format | 56.96 | 38.3 | 32.5 | 78.1 | 76.9 |
| Mix format | In‑the‑front format | 56.62 | 37.9 | 32.5 | 78.4 | 76.3 |

## Training strategy comparison on video (Pooling vs not pooling)

In this ablation, we study the impact of image token pooling. We train and infer our model under two settings: pooling to 1/4 and not pooling with ShareGPTVideo-Caption+QA(255K) data. Pooling to a 1/4 setting is similar to LLaVA‑NEXT‑Video, which uses the pooling technique to trade off between the number of image tokens and the number of frames. In our experiment, we find that not pooling yields better performance under similar #image tokens.

During training, we sample 10 frames for videos. In this table, we also observe that adding more frames (from 10 to 16) during inference improves performance.

| Train Setting | Inference Setting | Inference #frames | # Image tokens | ActivityNet‑QA (Acc/Score) | Avg | VDD | VideoChat‑GPT · Correctness | Detail | Context | Temporal | Consistency |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Pooling 1/4 | Pooling 1/4 | 40 | 40×729×1/4=10×729 | 52.75/3.53 | 3.35 | 3.38 | 3.46 | 3.25 | 3.87 | 2.59 | 3.57 |
| Pooling 1/4 | Pooling 1/4 | 64 | 64×729×1/4=16×729 | 52.7/3.53 | 3.33 | 3.38 | 3.45 | 3.23 | 3.86 | 2.49 | 3.55 |
| Not Pooling | Not Pooling | 10 | 10×729 | 52.9/3.48 | 3.38 | 3.46 | 3.43 | 3.26 | 3.85 | 2.64 | 3.61 |
| Not Pooling | Not Pooling | 16 | 16×729 | 54.4/3.51 | 3.41 | 3.46 | 3.48 | 3.28 | 3.87 | 2.74 | 3.62 |

## Effects of adding more tasks and data

To study the impact of adding more data, we conduct experiments under different data settings and evaluate the video benchmark. As we gradually add more data, performance consistently improves. VDD means Video Detailed Description.

| Data | Next‑QA | Avg | VDD | VideoChat‑GPT · Correctness | Detail | Context | Temporal | Consistency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| video | 42.60 | 3.40 | 3.46 | 3.47 | 3.27 | 3.87 | 2.74 | 3.61 |
| video + single‑image | 67.70 | 3.40 | 3.49 | 3.46 | 3.30 | 3.85 | 2.71 | 3.60 |
| video + multi‑image | 77.70 | 3.42 | 3.50 | 3.50 | 3.31 | 3.90 | 2.70 | 3.63 |
| video + both | 78.20 | 3.45 | 3.58 | 3.50 | 3.27 | 3.87 | 2.77 | 3.68 |



下面给出基于你提供的上下文，、更清晰且中文化的表格与说明（专业名词保留英文）：

### MetaCLIP 元数据（queries/entries）构成与阈值

| 组件 | 来源 | 选择规则 | 阈值类型 | 阈值 | 条目数 | 备注 |
|---|---|---|---|---:|---:|---|
| (1) WordNet synsets | WordNet | 将所有尚未包含在列表中的 synsets 加入 | N/A | [ALL]（follow CLIP） | 86,654 | 取全部可用 synsets，去重后计数 |
| (2) Wiki uni-gram | Wikipedia（英文版） | 词频筛选 | Count | 100（follow CLIP） | 251,465 | 仅保留出现频次≥100的 uni-grams |
| (3) Wiki bi-gram | Wikipedia（英文版） | 关联度筛选 | Pointwise Mutual Information (PMI) | 30（estimated） | 100,646 | 选择 PMI≥30 的 bi-grams，使规模≈100k |
| (4) Wiki titles | Wikipedia（英文版） | 热度/频率筛选 | Search/View Frequency | 70（estimated） | 61,235 | 选取热度≥阈值的词条标题，用于补足总量 |
| 合计 | — | — | — | — | 500,000 | 满足整体预算 500k entries |

补充说明
- 构成与顺序
  - 基础列表为 Wikipedia 英文版中出现至少 100 次的 uni-grams；
  - 其后加入高 PMI 的 bi-grams；
  - 再加入达到一定搜索/浏览热度的 Wikipedia titles；
  - 最后补入尚未包含的 WordNet synsets。
- 阈值来源
  - “follow CLIP”表示沿用 CLIP 的设定（如 uni-gram 的出现次数阈值 100）。
  - “estimated”表示根据总预算（500k entries）反推得到的估计阈值：PMI 取 30 以得到约 100k bi-grams；剩余名额由 Wikipedia titles 以热度阈值约 70 补齐到 500k。
- 术语提示
  - Pointwise Mutual Information (PMI): PMI(x, y) = log(p(x,y) / (p(x)p(y)))，用于衡量 bi-gram 的相关性。
  - Search/View Frequency：正文使用“search volume”，表格列名为“View Frequency”。两者在复现实验时通常可用 Wikipedia 的 pageview 或搜索热度统计近似实现；阈值 70 为与规模目标匹配的经验估计。
- 去重规则
  - 追加时对条目进行去重；表中计数为最终列表中的独立 entries 数量。



# Performance Optimizations

| Feature                                  | Flag                         | Benefit                                   |
|------------------------------------------|------------------------------|-------------------------------------------|
| FlashAttention                           | `--attention-backend`        | Faster attention and lower memory usage   |
| FP8 Training                             | `--fp8-hybrid`               | Faster training                           |
| Activation Checkpointing                 | `--recompute-activations`    | Reduced memory usage                      |
| Data Parallelism Communication Overlap   | `--overlap-grad-reduce`      | Faster distributed training               |
| Distributed Optimizer                    | `--use-distributed-optimizer`| Reduced checkpointing time                |



| 时间     | 工作项                   | 关键说明 |
|----------|--------------------------|----------|
| 2025-10 | LLaVA-OneVision-1.5-RL         | 全部透明经验分享，如何 build出更适合满思考的MLLM基座 |
| 2025-11 | 全新的 LLaVA-ViT         | **稀疏视频输入**，高效视频表征；训练代码、数据、模型全部开源 |
| 2025-12 | 全新的 LLaVA-OneVision-2.0 | 更强的具身能力与视频能力；**支持全帧率输入**的高效LLM；训练代码、数据、模型全部开源 |