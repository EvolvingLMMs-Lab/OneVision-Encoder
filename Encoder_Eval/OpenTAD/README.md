# OpenTAD
# OpenTAD (CUDA 12.5 Compatible)

> **Status**: *alpha* — installation verified on CUDA 12.5; dataset/model pipelines still untested (see **TODO**).

---

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [Dataset Layout](#dataset-layout)
4. [`video_TAD.sh` Arguments](#video_TADsh-arguments)
5. [TODO](#todo)

---

## Installation

### 1. Create the environment & install **PyTorch**

```bash
conda create -n opentad python=3.10.12
conda activate opentad

# CUDA 12.4 wheels also work on 12.5
pip install torch==2.2.2 torchvision==0.17.2 \
  --extra-index-url https://download.pytorch.org/whl/cu124
```
> Python < = 3.9 or Python > = 3.11 may fail to install the environment of OpenTAD
### 2. Install **MMCV** & **MMAction2**

```bash
pip install openmim
mim install mmcv==2.1.0
mim install mmaction2==1.2.0
```

> **Heads‑up 📌** `mmaction2 == 1.2.0` may raise an `import drn` error. Fix:
>
> 1. Clone `https://github.com/open-mmlab/mmaction2` (matching tag).
> 2. Copy the folder `mmaction/models/localizers/drn` into the same path inside your **conda** site‑packages for `mmaction2`.

### 3. Install **OpenTAD**

```bash
git clone git@github.com:sming256/OpenTAD.git
cd OpenTAD
pip install -r requirements.txt
```

---

## Usage

The project is wrapped by a single entry‑point script:

```bash
bash video_TAD.sh
```

This will perform:

1. **Feature extraction** (Hugging Face or local `.pth` backbones)
2. **Training / inference** with an Action Detection model

---

## Dataset Layout

```
<DATA_PATH>
└── <dataset_name>/
    ├── raw_data/
    │   └── video/          #   *.mp4 | *.avi
    ├── feature/            #   extracted *.npy features
    └── annotations/        #   *.json or *.csv labels
```
如果数据集需要重新下载，或者 annotation 缺失，可以通过 [Encoder_TAD 数据下载指南](https://github.com/FeilongTangmonash/Encoder_TAD/blob/41f101281c6c1259e5a38f8f642e539d0861932e/doc/en/data.md) 来查看如何下载数据。


---

## `video_TAD.sh` Arguments

| Variable              | Description                              | Example                             |
| --------------------- | ---------------------------------------- | ----------------------------------- |
| `DATA_PATH`           | Root folder of the dataset (see above)   | `/data/charades`                    |
| `CONFIG_PATH`         | Path to the model config you want to run | `configs/charades/temporalmaxer.py` |
| `CHECKPOINT_PATH`     | Where to save / load model checkpoints   | `./work_dirs`                       |
| **Hugging Face mode** |                                          |                                     |
| `MODEL_NAME`          | HF *short* model id                      | `videomae-base`                     |
| `CKPT`                | Full HF repo path                        | `facebook/videomae-base`            |
| `MODEL_TYPE`          | Backbone family name                     | `videomae` / `internvideo`          |
| **Local `.pth` mode** |                                          |                                     |
| `MODEL_NAME`          | Name accepted by `timm.create_model`     | `internvideo2_tem_dense_urope_tube_small_patch16_224_fc_512_v1`          |
| `CKPT`                | `.pth` checkpoint path                   | `~/checkpoints/backbone_tube248_dense_moreepoch.pt`     |
| `MODEL_TYPE`          | Custom family name                       | `univit`                            |

---

## TODO

* [ ] Validate **dataset preprocessing** scripts on target datasets
* [ ] Benchmark **model training** & ensure checkpoints load correctly
* [ ] Add CI workflow for CUDA 12.5 container build

---

## License

This fork inherits the original [OpenTAD license](LICENSE) unless otherwise noted.

---

*Enjoy Temporal Action Detection!* 🚀

