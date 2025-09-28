# Installation

**Step 1.** Install PyTorch=2.0.1, Python=3.10.12

```
conda create -n opentad python=3.10.12
source activate opentad
conda install pytorch=2.0.1 torchvision=0.15.2 pytorch-cuda=11.8 -c pytorch -c nvidia
```

**Step 2.** Install mmaction2 for end-to-end training
```
pip install openmim
mim install mmcv==2.0.1
mim install mmaction2==1.1.0
```

**Step 3.** Install OpenTAD
```
git clone git@github.com:sming256/OpenTAD.git
cd OpenTAD

pip install -r requirements.txt
```

The code is tested with Python 3.10.12, PyTorch 2.0.1, CUDA 11.8, and gcc 11.3.0 on Ubuntu 20.04, other versions might also work.
