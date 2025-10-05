# TemporalMaxer

> [TemporalMaxer: Maximize Temporal Context with only Max Pooling for Temporal Action Localization](https://arxiv.org/abs/2303.09055)  
> Tuan N. Tang, Kwonyoung Kim, Kwanghoon Sohn

<!-- [ALGORITHM] -->

## Abstract

Temporal Action Localization (TAL) is a challenging task in video understanding that aims to identify and localize actions within a video sequence. Recent studies have emphasized the importance of applying long-term temporal context modeling (TCM) blocks to the extracted video clip features such as employing complex self-attention mechanisms. In this paper, we present the simplest method ever to address this task and argue that the extracted video clip features are already informative to achieve outstanding performance without sophisticated architectures. To this end, we introduce TemporalMaxer, which minimizes long-term temporal context modeling while maximizing information from the extracted video clip features with a basic, parameter-free, and local region operating max-pooling block. Picking out only the most critical information for adjacent and local clip embeddings, this block results in a more efficient TAL model. We demonstrate that TemporalMaxer outperforms other state-of-the-art methods that utilize long-term TCM such as self-attention on various TAL datasets while requiring significantly fewer parameters and computational resources. The code for our approach is publicly available at https://github.com/TuanTNG/TemporalMaxer.

## Results and Models

**Note:** TemporalMaxer needs more than 40 GB GPU memory when training on Multi-THUMOS and Epic-kitchen.

**THUMOS-14**

| Features | mAP@0.3 | mAP@0.4 | mAP@0.5 | mAP@0.6 | mAP@0.7 | ave. mAP |         Config          |                                                                                          Download                                                                                          |
| :------: | :-----: | :-----: | :-----: | :-----: | :-----: | :------: | :---------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|   I3D    |  83.17  |  79.09  |  71.66  |  61.72  |  46.00  |  68.33   | [config](thumos_i3d.py) | [model](https://drive.google.com/file/d/1SabUwj3ztoR3_o8MHM3YAu05YTxA0dwj/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1t_pxF14q-og2DV0ESxBAHTjEwl3ytgpF/view?usp=sharing) |

**MultiTHUMOS**

|    Features    | mAP@0.2 | mAP@0.5 | mAP@0.7 | ave. mAP (0.1:0.9:0.1) |              Config              |                                                                                          Download                                                                                          |
| :------------: | :-----: | :-----: | :-----: | :--------------------: | :------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|   I3D (rgb)    |  53.57  |  38.39  |  20.61  |         33.92          | [config](multithumos_i3d_rgb.py) | [model](https://drive.google.com/file/d/1bMaLgSShB7G8DEkbqK4zzOFBisX7N4zN/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1E2xIdrCz30CEr1XU6uKa_inRTd1EGqmL/view?usp=sharing) |
| I3D (rgb+flow) |  60.05  |  44.86  |  25.59  |         39.26          |   [config](multithumos_i3d.py)   | [model](https://drive.google.com/file/d/1YJ4WYG9XySq45yLXxMxgEEcJ3qv777pE/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1I213ECnp3m4GW3dXVSThnIEJeXsLocuM/view?usp=sharing) |

**Epic-Kitchens-100**

| Subset | Features | mAP@0.1 | mAP@0.2 | mAP@0.3 | mAP@0.4 | mAP@0.5 | ave. mAP |                  Config                  |                                                                                          Download                                                                                          |
| :----: | :------: | :-----: | :-----: | :-----: | :-----: | :-----: | :------: | :--------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|  Noun  | SlowFast |  25.46  |  24.35  |  22.55  |  20.32  |  17.08  |  21.96   | [config](epic_kitchens_slowfast_noun.py) | [model](https://drive.google.com/file/d/1mghPd-Y08n_fzXlXAAZWNASGEw-0xpe6/view?usp=sharing)   \| [log](https://drive.google.com/file/d/17_UmbdTEGsYWClgEvi3lROBPDpes43ck/view?usp=sharing) |
|  Verb  | SlowFast |  28.64  |  27.84  |  25.59  |  23.60  |  19.69  |  25.07   | [config](epic_kitchens_slowfast_verb.py) | [model](https://drive.google.com/file/d/1HMukmE3IR0Lr4EgEYIUOEXpNwKBizAOi/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1QKto-5vwW4zbr7Xn11PDxv4Q1aEhReMu/view?usp=sharing) |


## Train

You can use the following command to train a model.

```shell
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train TemporalMaxer on THUMOS dataset.

```shell
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/train.py configs/temporalmaxer/thumos_i3d.py
```

For more details, you can refer to the Training part in the [Usage](../../docs/en/usage.md).

## Test

You can use the following command to test a model.

```shell
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/test.py ${CONFIG_FILE} --checkpoint ${CHECKPOINT_FILE} [optional arguments]
```

Example: test TemporalMaxer on THUMOS dataset.

```shell
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/test.py configs/temporalmaxer/thumos_i3d.py --checkpoint exps/thumos/temporalmaxer_i3d/gpu1_id0/checkpoint/epoch_45.pth
```

For more details, you can refer to the Test part in the [Usage](../../docs/en/usage.md).

## Citation

```latex
@article{tang2023temporalmaxer,
  title={TemporalMaxer: Maximize Temporal Context with only Max Pooling for Temporal Action Localization},
  author={Tang, Tuan N and Kim, Kwonyoung and Sohn, Kwanghoon},
  journal={arXiv preprint arXiv:2303.09055},
  year={2023}
}
```