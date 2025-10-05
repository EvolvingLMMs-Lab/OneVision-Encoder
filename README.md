<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="asset/llava_vit_white.png">
    <source media="(prefers-color-scheme: light)" srcset="asset/llava_vit_white.png">
    <img alt="LLaVA-OneVision 1.5" src="output/llava_onevision_white.png" width="600" style="max-width: 100%;">
  </picture>
</p>

---


# LLaVA-ViT

## 🔧 Setup

```shell
# Mount NFS

mkdir -p /video_vit
mount -t nfs4 -o minorversion=1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2,noresvport cfs-iyHiNUmePn.lb-0a25b0a7.cfs.bj.baidubce.com:/ /video_vit
```

### 1. Optional: Using Dockerfile for Environment Setup
```bash
docker build -t llava_vit:25.10 .
```

### 1. Optional: Load Docker Image
```bash
docker load -i /video_vit/docker_images/llava_vit_tag_25.10.tar && \
docker tag $(docker images -q | head -n 1) llava_vit:25.10
```

### 2. Run
```
# Run container with -w to set working directory directly to the mounted volume
docker run -it --gpus all \
    --ipc host --net host --privileged --cap-add IPC_LOCK \
    --ulimit memlock=-1 --ulimit stack=67108864 --rm \
    -v "$(pwd)":/workspace/LLaVA-ViT \
    -v /video_vit:/video_vit \
    -w /workspace/LLaVA-ViT/ \
    --name "llava_vit__container" \
    llava_vit:25.10 /bin/bash

# Inside the container, install the package in editable mode
pip install -e .
```

## 🚀 Training

```bash
# Example command to start training
 training/train_predict_10_04.py \
    --model_name_or_path "google/vit-base-patch16-224-in21k" \
    --data_path /video_vit/dataset/ssv2_tmpfs.txt \
    --output_dir output_onevision_vit_base_10_04 \
    --num_frames 8 \
```

## 🚀 Evaluation
```bash
DATASETS=ssv2 \
MODEL_FAMILY=llava_vit \
MODEL_NAME=pretrain_encoder_small_patch16_224_v10_03 \
CKPT_PATH=/video_vit/xiangan/checkpoint_llava_vit/date_25_10_05_first_success_training/encoder_checkpoint_125000.pt \
EMBEDDING_SIZE=384 \
NUM_EPOCH=100 \
bash video_attentive_probe.sh
```

## Contributors
Thanks so much to all of our amazing contributors!

<!-- readme: collaborators,contributors -start -->
<table>
	<tbody>
		<tr>
            <td align="center">
                <a href="https://github.com/Luodian">
                    <img src="https://avatars.githubusercontent.com/u/15847405?v=4" width="80;" alt="Luodian"/>
                    <br />
                    <sub><b>Luodian</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/anxiangsir">
                    <img src="https://avatars.githubusercontent.com/u/31175974?v=4" width="80;" alt="anxiangsir"/>
                    <br />
                    <sub><b>anxiangsir</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/wideyard">
                    <img src="https://avatars.githubusercontent.com/u/101321826?v=4" width="80;" alt="wideyard"/>
                    <br />
                    <sub><b>wideyard</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/YunyaoYan">
                    <img src="https://avatars.githubusercontent.com/u/109638667?v=4" width="80;" alt="YunyaoYan"/>
                    <br />
                    <sub><b>YunyaoYan</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/FeilongTangmonash">
                    <img src="https://avatars.githubusercontent.com/u/152372878?v=4" width="80;" alt="FeilongTangmonash"/>
                    <br />
                    <sub><b>FeilongTangmonash</b></sub>
                </a>
            </td>
		</tr>
	<tbody>
</table>
<!-- readme: collaborators,contributors -end -->

