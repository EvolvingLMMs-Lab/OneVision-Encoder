# CausalTAD

> [Harnessing Temporal Causality for Advanced Temporal Action Detection](https://arxiv.org/abs/2407.17792)  
> Shuming Liu, Lin Sui, Chen-Lin Zhang, Fangzhou Mu, Chen Zhao, Bernard Ghanem

<!-- [ALGORITHM] -->

## Abstract

As a fundamental task in long-form video understanding, temporal action detection (TAD) aims to capture inherent temporal relations in untrimmed videos and identify candidate actions with precise boundaries. Over the years, various networks, including convolutions, graphs, and transformers, have been explored for effective temporal modeling for TAD. However, these modules typically treat past and future information equally, overlooking the crucial fact that changes in action boundaries are essentially causal events. 
Inspired by this insight, we propose leveraging the temporal causality of actions to enhance TAD representation by restricting the model's access to only past or future context. We introduce CausalTAD, which combines causal attention and causal Mamba to achieve state-of-the-art performance on multiple benchmarks. Notably, with CausalTAD, we ranked 1st in the Action Recognition, Action Detection, and Audio-Based Interaction Detection tracks at the EPIC-Kitchens Challenge 2024, as well as 1st in the Moment Queries track at the Ego4D Challenge 2024.

## Results and Models

**ActivityNet-1.3**

| Features | Classifier | mAP@0.5 | mAP@0.75 | mAP@0.95 | ave. mAP |        Config         |                                                                                          Download                                                                                          |
| :------: | :--------: | :-----: | :------: | :------: | :------: | :-------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|   TSP    |    CUHK    |  55.62  |  38.51   |   9.40   |  37.46   | [config](anet_tsp.py) | [model](https://drive.google.com/file/d/1s9mzuOqc-KSoBg6xyEZgb-X5SzR5Y6i-/view?usp=sharing)   \| [log](https://drive.google.com/file/d/12hej5jRg_FX9v-ShT2epAvG83lxBv8NR/view?usp=sharing) |


**THUMOS-14**

| Features | mAP@0.3 | mAP@0.4 | mAP@0.5 | mAP@0.6 | mAP@0.7 | ave. mAP |         Config          |                                                                                          Download                                                                                          |
| :------: | :-----: | :-----: | :-----: | :-----: | :-----: | :------: | :---------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|   I3D    |  84.43  |  80.75  |  73.57  |  62.70  |  47.33  |  69.75   | [config](thumos_i3d.py) | [model](https://drive.google.com/file/d/1P7O8RJp-gX_gBY2RQgYk9CZwHbdY2ef4/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1ImdvPnX56npu-ZqHFvm_Fxf7bY9x6Z-1/view?usp=sharing) |


**Epic-Kitchens-100**

| Subset | Features | mAP@0.1 | mAP@0.2 | mAP@0.3 | mAP@0.4 | mAP@0.5 | ave. mAP |             Config              |                                                                                          Download                                                                                          |
| :----: | :------: | :-----: | :-----: | :-----: | :-----: | :-----: | :------: | :-----------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|  Noun  | SlowFast |  28.13  |  26.79  |  25.16  |  22.63  |  18.70  |  24.28   | [config](epic_slowfast_noun.py) | [model](https://drive.google.com/file/d/186JsNtmsSYOe_HFIG6UvhOfAEt_xpeg5/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1k5lU-ArJ1h5Vnvz05azlMdJ0MpgBKv03/view?usp=sharing) |
|  Verb  | SlowFast |  29.62  |  28.69  |  27.16  |  25.24  |  21.44  |  26.43   | [config](epic_slowfast_verb.py) | [model](https://drive.google.com/file/d/1icQcNStbjRvdiu71JpgU_h5WMgymbNNS/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1muFQYyB1__3NZrODwtu5fHAGkisspJa0/view?usp=sharing) |

**Ego4D-MQ**

|   Features   | mAP@0.1 | mAP@0.2 | mAP@0.3 | mAP@0.4 | mAP@0.5 | ave. mAP |             Config              |                                                                                          Download                                                                                          |
| :----------: | :-----: | :-----: | :-----: | :-----: | :-----: | :------: | :-----------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| InternVideo1 |  37.68  |  35.28  |  32.23  |  29.49  |  26.29  |  32.19   | [config](ego4d_internvideo1.py) | [model](https://drive.google.com/file/d/1SC3XFSSwguJG8_8DhdYi8doB6W6Ayfne/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1BLTbyw_lSnWtjHZY1tZO_laF_Chgye_h/view?usp=sharing) |
| InternVideo2 |  39.01  |  36.05  |  33.06  |  30.45  |  26.70  |  33.05   | [config](ego4d_internvideo2.py) | [model](https://drive.google.com/file/d/1U2k9RLHNiCDSlppAPUl5GADYmfKtQlZ0/view?usp=sharing)   \| [log](https://drive.google.com/file/d/14D-q6N7RiCgmRexFPiozjpa0BiaQGnlI/view?usp=sharing) |

For our solution to Ego4D Challenge 2024 and EPIC-Kitchens Challenge 2024, please refer to [here](egovis_challenge_2024/README.md), including detailed challenge config and ensemble strategy.


## Train

You can use the following command to train a model.

```shell
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train CausalTAD on THUMOS dataset.

```shell
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/train.py configs/causaltad/thumos_i3d.py
```

For more details, you can refer to the Training part in the [Usage](../../docs/en/usage.md).

## Test

You can use the following command to test a model.

```shell
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/test.py ${CONFIG_FILE} --checkpoint ${CHECKPOINT_FILE} [optional arguments]
```

Example: test CausalTAD on THUMOS dataset.

```shell
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/test.py configs/causaltad/thumos_i3d.py --checkpoint exps/thumos/causal_i3d/gpu1_id0/checkpoint/epoch_38.pth
```

For more details, you can refer to the Test part in the [Usage](../../docs/en/usage.md).

## Citation

```latex
@article{liu2024harnessing,
  title={Harnessing Temporal Causality for Advanced Temporal Action Detection},
  author={Liu, Shuming and Sui, Lin and Zhang, Chen-Lin and Mu, Fangzhou and Zhao, Chen and Ghanem, Bernard},
  journal={arXiv preprint arXiv:2407.17792},
  year={2024}
}
```
