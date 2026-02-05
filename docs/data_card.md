# Data Card: OneVision Encoder Training Data

## Overview

This document describes the datasets used for training OneVision Encoder. The pretraining corpus combines large-scale image and video datasets for unified visual representation learning.

## OneVision-Encoder Pretraining Dataset

| Source | Samples | Type | Modality | Temporal | Curation |
|--------|---------|------|----------|----------|----------|
| **LAION-400M** | 250M | WebImages | Image | -- | Yes |
| **COYO-700M** | 400M | WebImages | Image | -- | Yes |
| **OBELICS** | 15M | Documents | Image | -- | Yes |
| **Zero250M** | 15M | CuratedImages | Image | -- | Yes |
| **ImageNet-21K** | 14M | Images | Image | -- | Yes |
| **HowTo100M** | 50M | ExoVideo | Video | Short | No |
| **Panda-70M** | 50M | ExoVideo | Video | Long | Yes |
| **Kinetics-710** | 658K | ActionVideo | Video | Short | Yes |
| **SSV2** | 221K | ActionVideo | Video | Short | Yes |

### Dataset Summary

| Category | Total Samples |
|----------|---------------|
| **Image** | ~694M |
| **Video** | ~100M+ |
| **Total** | ~794M+ |

## Image Data Annotation

For image data, we primarily process LAION-400M and COYO-700M with the following pipeline:

**Deduplication:** We employ a Union-Find algorithm to strictly deduplicate the dataset.

**Clustering and Multi-label Annotation:** We utilize the metaclip-h14-fullcc2.5b model to extract image features and cluster all images into 2 million classes. Based on this clustering, each image sample is annotated with the nearest Top-10 class centers as its multi-label supervision signal.

**OCR-based Fine-grained Tagging:** Furthermore, we incorporate the OBELICS and Zero250M datasets. We utilize PaddleOCR to recognize text within images and perform word segmentation on the recognized content; the resulting vocabulary is used as multi-labels to construct a supervision signal containing exactly 100 fine-grained tags per image.
