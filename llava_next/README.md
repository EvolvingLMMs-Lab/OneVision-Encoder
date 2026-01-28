## Quick Start Guide

### 1.🐳 Docker (Recommended)

We strongly recommend using the docker environment for a seamless experience. The following instructions are tailored for the A100 80GB GPU environment.


```bash
# Clone repository
git clone https://github.com/EvolvingLMMs-Lab/OneVision-Encoder
cd OneVision-Encoder/llava_next

docker build -t ov_encoder_llava:26.01 .

# Run container with -w to set working directory directly to the mounted volume
docker run -it --gpus all \
    --ipc host --net host --privileged --cap-add IPC_LOCK \
    --ulimit memlock=-1 --ulimit stack=67108864 --rm \
    -v $(pwd):/workspace/OV-Encoder-Llava \
    -w /workspace/OV-Encoder-Llava \
    --name "ov_encoder_llava_container" \
    ov_encoder_llava:26.01 bash -c "service ssh restart; bash; "
```
