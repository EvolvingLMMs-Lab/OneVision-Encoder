# ActionFormer

> [ActionFormer: Localizing Moments of Actions with Transformers](https://arxiv.org/abs/2202.07925)  
> Chen-Lin Zhang, Jianxin Wu, Yin Li

<!-- [ALGORITHM] -->

## Abstract

Self-attention based Transformer models have demonstrated impressive results for image classification and object detection, and more recently for video understanding. Inspired by this success, we investigate the application of Transformer networks for temporal action localization in videos. To this end, we present ActionFormer -- a simple yet powerful model to identify actions in time and recognize their categories in a single shot, without using action proposals or relying on pre-defined anchor windows. ActionFormer combines a multiscale feature representation with local self-attention, and uses a light-weighted decoder to classify every moment in time and estimate the corresponding action boundaries. We show that this orchestrated design results in major improvements upon prior works. Without bells and whistles, ActionFormer achieves 71.0% mAP at tIoU=0.5 on THUMOS14, outperforming the best prior model by 14.1 absolute percentage points. Further, ActionFormer demonstrates strong results on ActivityNet 1.3 (36.6% average mAP) and EPIC-Kitchens 100 (+13.5% average mAP over prior works).

## Results and Models

**ActivityNet-1.3** with CUHK classifier.

| Features | mAP@0.5 | mAP@0.75 | mAP@0.95 | ave. mAP |        Config         |                                                                                          Download                                                                                          |
| :------: | :-----: | :------: | :------: | :------: | :-------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|   TSP    |  55.08  |  38.27   |   8.91   |  37.07   | [config](anet_tsp.py) | [model](https://drive.google.com/file/d/1CY-BAKjjYAt7t0-OASm66jPBkUzJnN6u/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1YkbeU0ldfJ0fSJGe70mM1GCfAzNv-Jmy/view?usp=sharing) |

**THUMOS-14**

| Features | mAP@0.3 | mAP@0.4 | mAP@0.5 | mAP@0.6 | mAP@0.7 | ave. mAP |         Config          |                                                                                          Download                                                                                          |
| :------: | :-----: | :-----: | :-----: | :-----: | :-----: | :------: | :---------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|   I3D    |  83.78  |  80.06  |  73.16  |  60.46  |  44.72  |  68.44   | [config](thumos_i3d.py) | [model](https://drive.google.com/file/d/1zTWLAerk5lZscOE-RZN9vuZ47MJCDno8/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1nYNK5WKkrbeQfJXdA2iKiCs5hWl1sd97/view?usp=sharing) |

**HACS**

| Features | mAP@0.5 | mAP@0.75 | mAP@0.95 | ave. mAP |           Config           |                                                                                          Download                                                                                          |
| :------: | :-----: | :------: | :------: | :------: | :------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| SlowFast |  56.18  |  37.97   |  11.05   |  37.71   | [config](hacs_slowfast.py) | [model](https://drive.google.com/file/d/1gga4G65qpnHbTuQtIleIZlMmn9sy8Kxq/view?usp=sharing)   \| [log](https://drive.google.com/file/d/15pJOJB6OjQ7PzEOBGEagcNXI5HFfycSM/view?usp=sharing) |

**Epic-Kitchens-100**

| Subset | Features | mAP@0.1 | mAP@0.2 | mAP@0.3 | mAP@0.4 | mAP@0.5 | ave. mAP |                  Config                  |                                                                                          Download                                                                                          |
| :----: | :------: | :-----: | :-----: | :-----: | :-----: | :-----: | :------: | :--------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|  Noun  | SlowFast |  25.78  |  24.73  |  22.83  |  20.84  |  17.45  |  22.33   | [config](epic_kitchens_slowfast_noun.py) | [model](https://drive.google.com/file/d/1TRXPMfTAEOR9Cl1rPyYTPN-uhly1b7M2/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1Kz00NHSLCEWEG6WLtoxwdnjpxqm61Fy0/view?usp=sharing) |
|  Verb  | SlowFast |  27.68  |  26.79  |  25.62  |  24.06  |  20.48  |  24.93   | [config](epic_kitchens_slowfast_verb.py) | [model](https://drive.google.com/file/d/1nox7XT6wEOciwwMxPBu91jgKgmyedMv-/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1gDwUMebJ_wO8n9gAlR07L-pMI6wnO3eG/view?usp=sharing) |

**Ego4D-MQ**

|  Features   | mAP@0.1 | mAP@0.2 | mAP@0.3 | mAP@0.4 | mAP@0.5 | ave. mAP |             Config             |                                                                                          Download                                                                                          |
| :---------: | :-----: | :-----: | :-----: | :-----: | :-----: | :------: | :----------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|  SlowFast   |  20.90  |  18.12  |  15.81  |  14.25  |  12.21  |  16.26   |  [config](ego4d_slowfast.py)   | [model](https://drive.google.com/file/d/1Z9xJbehxnt5-PWL3NMtTlINpLmY1vh5b/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1A3guHt6cxYLQC-99a4jQW3TzvLs9IwSJ/view?usp=sharing) |
|   EgoVLP    |  27.79  |  24.97  |  22.37  |  19.25  |  16.25  |  22.13   |   [config](ego4d_egovlp.py)    | [model](https://drive.google.com/file/d/1ZYYi2si_YOZoZpquPJwD3-w-RizALInQ/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1VMBWxOC-Z0UwRB9vGnY9g0uyclya90Jf/view?usp=sharing) |
| InternVideo |  32.59  |  30.28  |  27.53  |  25.09  |  22.13  |  27.52   | [config](ego4d_internvideo.py) | [model](https://drive.google.com/file/d/1QEgzjdxCdh1dNm7qBWexhKtxsIgW8kfv/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1mm1sviTNeLimJYijNZhSEPPZQiz2msSh/view?usp=sharing) |


**MultiTHUMOS**

|    Features    | mAP@0.2 | mAP@0.5 | mAP@0.7 | ave. mAP (0.1:0.9:0.1) |              Config              |                                                                                          Download                                                                                          |
| :------------: | :-----: | :-----: | :-----: | :--------------------: | :------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|   I3D (rgb)    |  53.52  |  39.05  |  19.69  |         34.02          | [config](multithumos_i3d_rgb.py) | [model](https://drive.google.com/file/d/1h2zizpltrQ5SvYN5q8o8RXf9lfMabAKk/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1NkEs_QUNZk-wEUqz28KskMyRrzHhF1K-/view?usp=sharing) |
| I3D (rgb+flow) |  60.18  |  45.01  |  24.56  |         39.19          |   [config](multithumos_i3d.py)   | [model](https://drive.google.com/file/d/1h2zizpltrQ5SvYN5q8o8RXf9lfMabAKk/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1NkEs_QUNZk-wEUqz28KskMyRrzHhF1K-/view?usp=sharing) |

**Charades**

|  Features  | mAP@0.2 | mAP@0.5 | mAP@0.7 | ave. mAP (0.1:0.9:0.1) |              Config              |                                                                                          Download                                                                                          |
| :--------: | :-----: | :-----: | :-----: | :--------------------: | :------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| I3D (rgb)  |  31.33  |  23.07  |  13.60  |         20.60          |  [config](charades_i3d_rgb.py)   | [model](https://drive.google.com/file/d/1mWH4qEStWL6GNbvN7yA4wuVryyqkgA1W/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1Lzhs3XVaOeyBDReBfqReQapraKLnbWvD/view?usp=sharing) |
| VideoMAE-L |  38.87  |  29.67  |  17.52  |         26.04          | [config](charades_videomae_l.py) | [model](https://drive.google.com/file/d/1Ci6eCNyziwEdHbAcrxiJO3Dj7IabS7Hh/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1EwoCachEPjuVVQQvHUPSWAfcxcBVLst5/view?usp=sharing) |

**FineAction** with InternVideo classifier

|     Features      | mAP@0.5 | mAP@0.75 | mAP@0.95 | ave. mAP |                Config                |                                                                                         Download                                                                                          |
| :---------------: | :-----: | :------: | :------: | :------: | :----------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|  VideoMAE_H_K700  |  29.44  |  19.46   |   5.06   |  19.32   |  [config](fineaction_videomae_h.py)  | [model](https://drive.google.com/file/d/1MpFPQ_2Uu4ksEtHF1ScDAyyMPdU4Icfe/view?usp=sharing)  \| [log](https://drive.google.com/file/d/13EGNhiEMGCjb0I26_XoBOx-3OLF6KwP6/view?usp=sharing) |
| VideoMAEv2_g_K710 |  29.85  |  19.72   |   5.17   |  19.62   | [config](fineaction_videomaev2_g.py) | [model](https://drive.google.com/file/d/1yss4xbieHjon5EA_wk91z90W65bmXoag/view?usp=sharing)  \| [log](https://drive.google.com/file/d/1lucPvEHh_asvMZQOzShq8tdOpC6cAak_/view?usp=sharing) |


## Train

You can use the following command to train a model.

```shell
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train ActionFormer on ActivityNet dataset.

```shell
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/train.py configs/actionformer/anet_tsp.py
```

For more details, you can refer to the Training part in the [Usage](../../docs/en/usage.md).

## Test

You can use the following command to test a model.

```shell
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/test.py ${CONFIG_FILE} --checkpoint ${CHECKPOINT_FILE} [optional arguments]
```

Example: test ActionFormer on ActivityNet dataset.

```shell
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/test.py configs/actionformer/anet_tsp.py --checkpoint exps/anet/actionformer_tsp/gpu1_id0/checkpoint/epoch_14.pth
```

For more details, you can refer to the Test part in the [Usage](../../docs/en/usage.md).

## Citation

```latex
@inproceedings{zhang2022actionformer,
  title={Actionformer: Localizing moments of actions with transformers},
  author={Zhang, Chen-Lin and Wu, Jianxin and Li, Yin},
  booktitle={Computer Vision--ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23--27, 2022, Proceedings, Part IV},
  pages={492--510},
  year={2022},
  organization={Springer}
}
```