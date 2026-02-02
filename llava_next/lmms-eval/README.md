<p align="center" width="70%">
<img src="https://i.postimg.cc/KvkLzbF9/WX20241212-014400-2x.png">
</p>

# The Evaluation Suite of Large Multimodal Models

🌐 **English** | [简体中文](docs/i18n/README_zh-CN.md) | [繁體中文](docs/i18n/README_zh-TW.md) | [日本語](docs/i18n/README_ja.md) | [한국어](docs/i18n/README_ko.md) | [Español](docs/i18n/README_es.md) | [Français](docs/i18n/README_fr.md) | [Deutsch](docs/i18n/README_de.md) | [Português](docs/i18n/README_pt-BR.md) | [Русский](docs/i18n/README_ru.md) | [Italiano](docs/i18n/README_it.md) | [Nederlands](docs/i18n/README_nl.md) | [Polski](docs/i18n/README_pl.md) | [Türkçe](docs/i18n/README_tr.md) | [العربية](docs/i18n/README_ar.md) | [हिन्दी](docs/i18n/README_hi.md) | [Tiếng Việt](docs/i18n/README_vi.md) | [Indonesia](docs/i18n/README_id.md) 

[![PyPI](https://img.shields.io/pypi/v/lmms-eval)](https://pypi.org/project/lmms-eval)
![PyPI - Downloads](https://img.shields.io/pypi/dm/lmms-eval)
[![GitHub contributors](https://img.shields.io/github/contributors/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/graphs/contributors)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)
[![open issues](https://img.shields.io/github/issues-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)

> Accelerating the development of large multimodal models (LMMs) with `lmms-eval`. We support most text, image, video and audio tasks.

🏠 [LMMs-Lab Homepage](https://www.lmms-lab.com/) | 🤗 [Huggingface Datasets](https://huggingface.co/lmms-lab) | <a href="https://emoji.gg/emoji/1684-discord-thread"><img src="https://cdn3.emoji.gg/emojis/1684-discord-thread.png" width="14px" height="14px" alt="Discord_Thread"></a> [discord/lmms-eval](https://discord.gg/zdkwKUqrPy)

📖 [Supported Tasks (100+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/current_tasks.md) | 🌟 [Supported Models (30+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/lmms_eval/models) | 📚 [Documentation](docs/README.md)

---

## What's New

Evaluating multimodal models is harder than it looks. We have hundreds of benchmarks, but no standard way to run them. Results vary between labs. Comparisons become unreliable. We've been working to address this - not through heroic effort, but through systematic process.

**January 2026** - We recognized that spatial and compositional reasoning remained blind spots in existing benchmarks. We added [CaptionQA](https://captionqa.github.io/), [SpatialTreeBench](https://github.com/THUNLP-MT/SpatialTreeBench), [SiteBench](https://sitebench.github.io/), and [ViewSpatial](https://github.com/ViewSpatial/ViewSpatial). For teams running remote evaluation pipelines, we introduced an HTTP eval server (#972). For those who need statistical rigor, we added CLT and clustered standard error estimation (#989).

**October 2025 (v0.5)** - Audio had been a gap. Models could hear, but we had no consistent way to test them. This release added comprehensive audio evaluation, response caching for efficiency, and 50+ benchmark variants spanning audio, vision, and reasoning. [Release notes](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/lmms-eval-0.5.md).

<details>
<summary>Below is a chronological list of recent tasks, models, and features added by our amazing contributors. </summary>

- [2025-01] 🎓🎓 We have released our new benchmark: [Video-MMMU: Evaluating Knowledge Acquisition from Multi-Discipline Professional Videos](https://arxiv.org/abs/2501.13826). Please refer to the [project page](https://videommmu.github.io/) for more details.
- [2024-12] 🎉🎉 We have presented [MME-Survey: A Comprehensive Survey on Evaluation of Multimodal LLMs](https://arxiv.org/pdf/2411.15296), jointly with [MME Team](https://github.com/BradyFU/Video-MME) and [OpenCompass Team](https://github.com/open-compass).
- [2024-11] 🔈🔊 The `lmms-eval/v0.3.0` has been upgraded to support audio evaluations for audio models like Qwen2-Audio and Gemini-Audio across tasks such as AIR-Bench, Clotho-AQA, LibriSpeech, and more. Please refer to the [blog](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/lmms-eval-0.3.md) for more details!
- [2024-10] 🎉🎉 We welcome the new task [NaturalBench](https://huggingface.co/datasets/BaiqiL/NaturalBench), a vision-centric VQA benchmark (NeurIPS'24) that challenges vision-language models with simple questions about natural imagery.
- [2024-10] 🎉🎉 We welcome the new task [TemporalBench](https://huggingface.co/datasets/microsoft/TemporalBench) for fine-grained temporal understanding and reasoning for videos, which reveals a huge (>30%) human-AI gap.
- [2024-10] 🎉🎉 We welcome the new tasks [VDC](https://rese1f.github.io/aurora-web/) for video detailed captioning, [MovieChat-1K](https://rese1f.github.io/MovieChat/) for long-form video understanding, and [Vinoground](https://vinoground.github.io/), a temporal counterfactual LMM benchmark composed of 1000 short natural video-caption pairs. We also welcome the new models: [AuroraCap](https://github.com/rese1f/aurora) and [MovieChat](https://github.com/rese1f/MovieChat).
- [2024-09] 🎉🎉 We welcome the new tasks [MMSearch](https://mmsearch.github.io/) and [MME-RealWorld](https://mme-realworld.github.io/) for inference acceleration
- [2024-09] ⚙️️⚙️️️️ We upgrade `lmms-eval` to `0.2.3` with more tasks and features. We support a compact set of language tasks evaluations (code credit to [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)), and we remove the registration logic at start (for all models and tasks) to reduce the overhead. Now `lmms-eval` only launches necessary tasks/models. Please check the [release notes](https://github.com/EvolvingLMMs-Lab/lmms-eval/releases/tag/v0.2.3) for more details.
- [2024-08] 🎉🎉 We welcome the new model [LLaVA-OneVision](https://huggingface.co/papers/2408.03326), [Mantis](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/162), new tasks [MVBench](https://huggingface.co/datasets/OpenGVLab/MVBench), [LongVideoBench](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/117), [MMStar](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/158). We provide new feature of SGlang Runtime API for llava-onevision model, please refer the [doc](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/commands.md) for inference acceleration
- [2024-07] 👨‍💻👨‍💻 The `lmms-eval/v0.2.1` has been upgraded to support more models, including [LongVA](https://github.com/EvolvingLMMs-Lab/LongVA), [InternVL-2](https://github.com/OpenGVLab/InternVL), [VILA](https://github.com/NVlabs/VILA), and many more evaluation tasks, e.g. [Details Captions](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/136), [MLVU](https://arxiv.org/abs/2406.04264), [WildVision-Bench](https://huggingface.co/datasets/WildVision/wildvision-arena-data), [VITATECS](https://github.com/lscpku/VITATECS) and [LLaVA-Interleave-Bench](https://llava-vl.github.io/blog/2024-06-16-llava-next-interleave/).
- [2024-07] 🎉🎉 We have released the [technical report](https://arxiv.org/abs/2407.12772) and [LiveBench](https://huggingface.co/spaces/lmms-lab/LiveBench)! 
- [2024-06] 🎬🎬 The `lmms-eval/v0.2.0` has been upgraded to support video evaluations for video models like LLaVA-NeXT Video and Gemini 1.5 Pro across tasks such as EgoSchema, PerceptionTest, VideoMME, and more. Please refer to the [blog](https://lmms-lab.github.io/posts/lmms-eval-0.2/) for more details!
- [2024-03] 📝📝 We have released the first version of `lmms-eval`, please refer to the [blog](https://lmms-lab.github.io/posts/lmms-eval-0.1/) for more details!

</details>

## Why `lmms-eval`?

We are in the middle of something that feels like the 1960s space race - except this time, the destination is artificial general intelligence. Large multimodal models are our rockets. They can see, hear, read, and reason. The progress is real and accelerating.

But here is the problem: our measurement systems have not kept pace with our ambitions.

We have benchmarks - hundreds of them. But they are scattered across Google Drive folders, Dropbox links, university websites, and lab servers. Each benchmark has its own data format, its own evaluation script, its own quirks. When two teams report results on the same benchmark, they often get different numbers. Not because their models differ, but because their evaluation pipelines differ.

Imagine if, during the space race, every country measured distance in different units and never shared their conversion tables. That is roughly where we are today with multimodal evaluation.

This is not a minor inconvenience. It is a systemic failure. Without consistent measurement, we cannot know which models are actually better. We cannot reproduce results. We cannot build on each other's work.

For language models, this problem was largely solved by [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness). It provides unified data loading, standardized evaluation, and reproducible results. It powers the [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard). It has become infrastructure.

We built `lmms-eval` to do the same for multimodal models. Same principle: one framework, consistent interfaces, reproducible numbers. The moonshot needs a reliable ruler.

## Installation

### Using uv (Recommended for consistent environments)

We use `uv` for package management to ensure all developers use exactly the same package versions. First, install uv:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

For development with consistent environment:
```bash
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval
cd lmms-eval
# Recommend
uv pip install -e ".[all]"
# If you want to use uv sync
# uv sync  # This creates/updates your environment from uv.lock
```

To run commands:
```bash
uv run python -m lmms_eval --help  # Run any command with uv run
```

To add new dependencies:
```bash
uv add <package>  # Updates both pyproject.toml and uv.lock
```

### Alternative Installation

For direct usage from Git:
```bash
uv venv eval
uv venv --python 3.12
source eval/bin/activate
# You might need to add and include your own task yaml if using this installation
uv pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git
```

<details>
<summary>Reproduction of LLaVA-1.5's paper results</summary>

You can check the [environment install script](miscs/repr_scripts.sh) and [torch environment info](miscs/repr_torch_envs.txt) to **reproduce LLaVA-1.5's paper results**. We found torch/cuda versions difference would cause small variations in the results, we provide the [results check](miscs/llava_result_check.md) with different environments.

</details>

If you want to test on caption dataset such as `coco`, `refcoco`, and `nocaps`, you will need to have `java==1.8.0` to let pycocoeval api to work. If you don't have it, you can install by using conda
```
conda install openjdk=8
```
you can then check your java version by `java -version` 


<details>
<summary>Comprehensive Evaluation Results of LLaVA Family Models</summary>
<br>

As demonstrated by the extensive table below, we aim to provide detailed information for readers to understand the datasets included in lmms-eval and some specific details about these datasets (we remain grateful for any corrections readers may have during our evaluation process).

We provide a Google Sheet for the detailed results of the LLaVA series models on different datasets. You can access the sheet [here](https://docs.google.com/spreadsheets/d/1a5ImfdKATDI8T7Cwh6eH-bEsnQFzanFraFUgcS9KHWc/edit?usp=sharing). It's a live sheet, and we are updating it with new results.

<p align="center" width="100%">
<img src="https://i.postimg.cc/jdw497NS/WX20240307-162526-2x.png"  width="100%" height="80%">
</p>

We also provide the raw data exported from Weights & Biases for the detailed results of the LLaVA series models on different datasets. You can access the raw data [here](https://docs.google.com/spreadsheets/d/1AvaEmuG4csSmXaHjgu4ei1KBMmNNW8wflOD_kkTDdv8/edit?usp=sharing).

</details>
<br>

If you want to test [VILA](https://github.com/NVlabs/VILA), you should install the following dependencies:

```bash
pip install s2wrapper@git+https://github.com/bfshi/scaling_on_scales
```

Our Development will be continuing on the main branch, and we encourage you to give us feedback on what features are desired and how to improve the library further, or ask questions, either in issues or PRs on GitHub.

## Web UI

LMMS-Eval includes an optional Web UI for interactive evaluation configuration.

### Requirements

- Node.js 18+ (for building the frontend, auto-built on first run)

### Usage

```bash
# Start the Web UI (opens browser automatically)
uv run lmms-eval-ui

# Custom port
LMMS_SERVER_PORT=3000 uv run lmms-eval-ui
```

The web UI provides:
- Model selection from all available models
- Task selection with search/filter
- Real-time command preview
- Live evaluation output streaming
- Start/Stop evaluation controls

For more details, see [Web UI README](lmms_eval/tui/README.md).

## Usages

> More examples can be found in [examples/models](examples/models)

**Evaluation of OpenAI-Compatible Model**

```bash
bash examples/models/openai_compatible.sh
bash examples/models/xai_grok.sh
```

**Evaluation of vLLM**

```bash
bash examples/models/vllm_qwen2vl.sh
```

**Evaluation of LLaVA-OneVision**

```bash
bash examples/models/llava_onevision.sh
```

**Evaluation of LLaVA-OneVision1_5**

```bash
bash examples/models/llava_onevision1_5.sh
```

**Evaluation of LLaMA-3.2-Vision**

```bash
bash examples/models/llama_vision.sh
```

**Evaluation of Qwen2-VL**

```bash
bash examples/models/qwen2_vl.sh
bash examples/models/qwen2_5_vl.sh
```

**Evaluation of LLaVA on MME**

If you want to test LLaVA 1.5, you will have to clone their repo from [LLaVA](https://github.com/haotian-liu/LLaVA) and

```bash
bash examples/models/llava_next.sh
```

**Evaluation with tensor parallel for bigger model (llava-next-72b)**

```bash
bash examples/models/tensor_parallel.sh
```

**Evaluation with SGLang for bigger model (llava-next-72b)**

```bash
bash examples/models/sglang.sh
```


**Evaluation with vLLM for bigger model (llava-next-72b)**

```bash
bash examples/models/vllm_qwen2vl.sh
```

---

### Preparing eval datasets and offline codec-patch assets (for video tasks)

Some video evaluations benefit from **offline precomputed visual assets** (mosaics + position indices) to avoid per-sample frame extraction during evaluation.

This repo provides:
- `Compressed_Video_Reader/tool/offline_precompute_llava_codec_assets.py` — precomputes offline assets from a jsonl list.
- `scripts/precompute_codec_patch/run_offline_codec_patch_for_eval.sh` — a runner script (recommended) with `path/to/...` placeholders.

#### 1) Create the input jsonl (video list)

Prepare a jsonl file where **each line** contains at least:
- `video`: absolute path to the video file
- `key`: a **stable unique id** used as the offline asset folder name

Recommended (matches the precompute script schema):
- `task`, `split`, `doc_id`, `n`, `exists`

Example line:
```json
{"task":"videomme","split":"test","doc_id":123,"n":0,"video":"path/to/video.mp4","key":"<unique_key>","exists":true}
```

**Important:** the `key` must be consistent between **precompute** and the model's **offline loader**. If the loader reports `MISS`, it usually means the `key` (or folder layout / filenames) doesn't match.

#### 2) Precompute offline assets

Run the precompute tool to generate assets under:
```
path/to/offline_root/assets/<key>/
  mosaic_000.jpg ... mosaic_007.jpg
  positions_thw.npy
  visible_indices.npy
  frame_ids.npy
  meta.json
```

Example command:
```bash
python Compressed_Video_Reader/tool/offline_precompute_llava_codec_assets.py \
  --jsonl path/to/eval_videos.jsonl \
  --out_root path/to/offline_root \
  --num_workers 8 \
  --seq_len_frames 64 \
  --num_images 8 \
  --square_size 576 \
  --patch_size 16
```

Optional sharding (useful for large datasets):
```bash
python Compressed_Video_Reader/tool/offline_precompute_llava_codec_assets.py \
  --jsonl path/to/eval_videos.jsonl \
  --out_root path/to/offline_root \
  --num_workers 8 \
  --num_shards 8 \
  --shard_id 0
```

**Do not rename** the expected output filenames (e.g. `mosaic_000.jpg`, `positions_thw.npy`). Renaming will cause `MISS` and trigger fallback to frame extraction.

#### 3) Run evaluation using offline assets

Set these environment variables before launching `lmms_eval`:
```bash
export LLAVA_CODEC_USE_OFFLINE=1
export LLAVA_CODEC_OFFLINE_ROOT=path/to/offline_root/assets

# Optional (recommended): keep configs aligned with your offline assets
export LLAVA_CODEC_VISIDX_MODE=pack_topk
export LLAVA_CODEC_SEQ_LEN_FRAMES=64
export LLAVA_CODEC_NUM_IMAGES=8
export LLAVA_CODEC_SQUARE_SIZE=576
export LLAVA_CODEC_PATCH_SIZE=16

# Debug (single switch)
export LLAVA_CODEC_DEBUG=1
```

If offline assets are found, logs will include `HIT` / `USING_OFFLINE`. If not found, you will see `MISS` and the pipeline will **fallback to frame extraction**.

Quick checklist when you see `MISS`:
- `LLAVA_CODEC_OFFLINE_ROOT` points to the directory that contains `assets/<key>/...`
- `<key>` matches the one produced in your input jsonl
- `mosaic_000.jpg ...` and `positions_thw.npy` exist under `assets/<key>/`
- your `seq_len_frames / num_images / square_size / patch_size` match between precompute and eval
