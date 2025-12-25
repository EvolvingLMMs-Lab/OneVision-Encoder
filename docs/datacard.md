# Data Card: OneVision Encoder Training Data

## Overview

This document describes the datasets used for training OneVision Encoder. The training data consists of both image and video datasets, totaling approximately 754 million samples.

## Dataset Summary

| Category | Total Samples |
|----------|---------------|
| **Image** | ~694M |
| **Video** | ~100M+ |
| **Total** | ~794M+ |

---

## Image Datasets

| Dataset | Samples | Description |
|---------|---------|-------------|
| **LAION-400M** | 250M | Large-scale image-text dataset curated from Common Crawl, filtered for high-quality image-text pairs |
| **COYO-700M** | 400M | Comprehensive image-text dataset with diverse web-sourced content |
| **OBELICS** | 15M | Interleaved image-text documents for multimodal understanding |
| **Zero250M** | 15M | High-quality image dataset for visual representation learning |
| **ImageNet-21K** | 14M | Large-scale hierarchical image dataset covering 21,841 synsets |

### Image Dataset Details

#### LAION-400M (250M samples used)
- **Source**: Common Crawl web data
- **Content**: Diverse web images with associated alt-text captions
- **Usage**: Pre-training for general visual understanding

#### COYO-700M (400M samples used)
- **Source**: Web-crawled image-text pairs
- **Content**: Large-scale diverse visual content
- **Usage**: Pre-training for broad visual coverage

#### OBELICS (15M samples)
- **Source**: Curated multimodal documents
- **Content**: Interleaved image-text documents
- **Usage**: Learning from contextual image-text relationships

#### Zero250M (15M samples used)
- **Source**: Curated image collection
- **Content**: High-quality images for representation learning
- **Usage**: Visual representation pre-training

#### ImageNet-21K (14M samples)
- **Source**: ImageNet project
- **Content**: Hierarchically organized images across 21,841 categories
- **Usage**: Fine-grained visual recognition pre-training

---

## Video Datasets

| Dataset | Samples | Description |
|---------|---------|-------------|
| **HowTo100M** | 50M | Instructional videos with narrated activities |
| **Panda-70M** | 50M | Large-scale video-text dataset with high-quality captions |
| **Kinetics-710** | - | Human action recognition benchmark (for evaluation/fine-tuning) |
| **Something-Something V2 (SSv2)** | - | Fine-grained temporal reasoning benchmark (for evaluation/fine-tuning) |

### Video Dataset Details

#### HowTo100M
- **Source**: YouTube instructional videos
- **Content**: How-to videos with automatic speech recognition transcripts
- **Usage**: Learning temporal dynamics and action understanding

#### Panda-70M
- **Source**: Curated video-text pairs
- **Content**: High-quality video clips with detailed captions
- **Usage**: Video-language alignment pre-training

#### Kinetics-710 (K710)
- **Source**: YouTube videos of human actions
- **Content**: Human action video clips
- **Usage**: Action recognition evaluation and fine-tuning

#### Something-Something V2 (SSv2)
- **Source**: Crowdsourced human actions
- **Content**: Fine-grained hand-object interactions
- **Usage**: Temporal reasoning evaluation and fine-tuning

---

## Data Processing

### Image Processing
- Native resolution support up to 448×448
- CLIP-style preprocessing
- No tiling or cropping for native resolution matching

### Video Processing
- Frame sampling with temporal saliency detection
- Codec-style patch extraction for efficient processing
- Support for dense temporal sampling (up to 64 frames)

---

## Data Licensing

Please refer to the original dataset licenses for usage terms:

- **LAION-400M**: CC-BY 4.0
- **COYO-700M**: CC-BY 4.0
- **OBELICS**: Various (see original source)
- **ImageNet-21K**: ImageNet License
- **HowTo100M**: Various (YouTube content)
- **Panda-70M**: Various (see original source)
- **Kinetics-710**: Various (YouTube content)
- **Something-Something V2**: Non-commercial research use

---

## Citation

If you use this data configuration, please cite the original dataset papers:

```bibtex
@article{schuhmann2021laion,
  title={LAION-400M: Open Dataset of CLIP-Filtered 400 Million Image-Text Pairs},
  author={Schuhmann, Christoph and others},
  year={2021}
}

@article{kakaobrain2022coyo-700m,
  title={COYO-700M: Image-Text Pair Dataset},
  author={Kakao Brain},
  year={2022}
}

@article{miech19howto100m,
  title={HowTo100M: Learning a Text-Video Embedding by Watching Hundred Million Narrated Video Clips},
  author={Miech, Antoine and others},
  year={2019}
}

@article{chen2024panda70m,
  title={Panda-70M: Captioning 70M Videos with Multiple Cross-Modality Teachers},
  author={Chen, Tsai-Shien and others},
  year={2024}
}
```
