# AdaTAD

> [End-to-End Temporal Action Detection with 1B Parameters Across 1000 Frames](https://arxiv.org/abs/2311.17241)  
> Shuming Liu, Chen-Lin Zhang, Chen Zhao, Bernard Ghanem

<!-- [ALGORITHM] -->

## Abstract

Recently, temporal action detection (TAD) has seen significant performance improvement with end-to-end training. However, due to the memory bottleneck, only models with limited scales and limited data volumes can afford end-to-end training, which inevitably restricts TAD performance. In this paper, we reduce the memory consumption for end-to-end training, and manage to scale up the TAD backbone to 1 billion parameters and the input video to 1,536 frames, leading to significant detection performance. The key to our approach lies in our proposed temporal-informative adapter (TIA), which is a novel lightweight module that reduces training memory. Using TIA, we free the humongous backbone from learning to adapt to the TAD task by only updating the parameters in TIA. TIA also leads to better TAD representation by temporally aggregating context from adjacent frames throughout the backbone. We evaluate our model across four representative datasets. Owing to our efficient design, we are able to train end-to-end on VideoMAEv2-giant and achieve 75.4% mAP on THUMOS14, being the first end-to-end model to outperform the best feature-based methods.

## Prepare the pretrained VideoMAE checkpoints

Before running the experiments, please download the pretrained VideoMAE model weights (converted from original repo), and put them under the path `./pretrained/`.

|    Model     | Pretrain Dataset | Finetune Dataset |                                           Original Link                                           |                                                                  Converted Checkpoints                                                                   |
| :----------: | :--------------: | :--------------: | :-----------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------: |
|  VideoMAE-S  |       K400       |       K400       | [Url](https://github.com/MCG-NJU/VideoMAE/blob/main/MODEL_ZOO.md#:~:text=/log/-,checkpoint,-79.0) |                            [Google Drive](https://drive.google.com/file/d/1BH5BZmdImaZesUfqtW23eBGC341Gui1D/view?usp=sharing)                            |
|  VideoMAE-B  |       K400       |       K400       | [Url](https://github.com/MCG-NJU/VideoMAE/blob/main/MODEL_ZOO.md#:~:text=/log/-,checkpoint,-81.5) | [mmaction2](https://download.openmmlab.com/mmaction/v1.0/recognition/videomae/vit-base-p16_videomae-k400-pre_16x4x1_kinetics-400_20221013-860a3cd3.pth)  |
|  VideoMAE-L  |       K400       |       K400       | [Url](https://github.com/MCG-NJU/VideoMAE/blob/main/MODEL_ZOO.md#:~:text=/log/-,checkpoint,-85.2) | [mmaction2](https://download.openmmlab.com/mmaction/v1.0/recognition/videomae/vit-large-p16_videomae-k400-pre_16x4x1_kinetics-400_20221013-229dbb03.pth) |
|  VideoMAE-H  |       K400       |       K400       | [Url](https://github.com/MCG-NJU/VideoMAE/blob/main/MODEL_ZOO.md#:~:text=/log/-,checkpoint,-86.6) |                            [Google Drive](https://drive.google.com/file/d/1wWXs7xpkVkQ2cJnvRVKnWZ86QswZC1UI/view?usp=sharing)                            |
| VideoMAEv2-g |      Hybrid      |       K710       |           [Url](https://github.com/OpenGVLab/VideoMAEv2/blob/master/docs/MODEL_ZOO.md)            |                                                                       Not Provided                                                                       |

- Note that we are not allowed to redistribute VideoMAEv2's checkpoints. You can fill out the official [request form](https://github.com/OpenGVLab/VideoMAEv2/blob/master/docs/MODEL_ZOO.md#model-weight-links), then convert the checkpoint by the following command.

```bash
python tools/model_converters/convert_videomaev2.py \
    vit_g_hybrid_pt_1200e_k710_ft.pth pretrained/vit-giant-p14_videomaev2-hybrid_pt_1200e_k710_ft_my.pth
```

## ActivityNet Results

Please refer to [README.md](../../tools/prepare_data/activitynet/README.md#download-raw-videos) to prepare the raw video of ActivityNet.

|   Backbone   | GPUs  | Setting | Frames | Img Size |  Classifier  | mAP@0.5 | mAP@0.75 | mAP@0.95 | ave. mAP |                             Config                             |                                                                                          Download                                                                                          |
| :----------: | :---: | :-----: | :----: | :------: | :----------: | :-----: | :------: | :------: | :------: | :------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|  VideoMAE-S  |   4   | AdaTAD  |  768   |   160    |     CUHK     |  56.23  |  38.90   |   8.88   |  37.81   |    [config](anet/e2e_anet_videomae_s_192x4_160_adapter.py)     | [model](https://drive.google.com/file/d/1gncN-xjArNtgVoBKCwCJCH4ISA3yVqIU/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1kEjIu0NXiixW8jJEUG2q7gehE5EBh1pM/view?usp=sharing) |
|  VideoMAE-B  |   4   | AdaTAD  |  768   |   160    |     CUHK     |  56.72  |  39.44   |   9.54   |  38.35   |    [config](anet/e2e_anet_videomae_b_192x4_160_adapter.py)     | [model](https://drive.google.com/file/d/1tePHMitdwUrWax1nYlbucaqI5LbvZhZo/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1O_Z2F6PMOq_892P_uvFOSRZT6MT8ToAE/view?usp=sharing) |
|  VideoMAE-L  |   4   | AdaTAD  |  768   |   160    |     CUHK     |  57.73  |  40.53   |   9.96   |  39.21   |    [config](anet/e2e_anet_videomae_l_192x4_160_adapter.py)     | [model](https://drive.google.com/file/d/1GxwNLc1rRp6x5ug1zd1r_1DmYCZD_tw5/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1cM1SwVOB_uCYvp870_iowaJ8bzubk4CE/view?usp=sharing) |
|  VideoMAE-H  |   4   | AdaTAD  |  768   |   160    |     CUHK     |  57.77  |  40.60   |   9.78   |  39.31   |    [config](anet/e2e_anet_videomae_h_192x4_160_adapter.py)     | [model](https://drive.google.com/file/d/1Hqpdq7Qclf0-1oF25tWwZLI8Ranp-uBv/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1Uf4uexTXcOXTqWcFyI7-q-mzznbwxKTg/view?usp=sharing) |
| VideoMAEV2-g |   4   | AdaTAD  |  768   |   160    |     CUHK     |  58.42  |  40.89   |  10.01   |  39.77   |   [config](anet/e2e_anet_videomaev2_g_192x4_160_adapter.py)    | [model](https://drive.google.com/file/d/1lfWyWrt1gJOm7YfwCdXi7HiNomHPGvna/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1NlDvNCb7AGmvdu9t7F80QcXb6BVLzv6C/view?usp=sharing) |
| VideoMAEV2-g |   8   | AdaTAD  |  768   |   224    |     CUHK     |  58.57  |  41.19   |  10.27   |  39.86   |       [config](e2e_anet_videomaev2_g_192x4_224_adapter)        | [model](https://drive.google.com/file/d/1x4BxA_EA1F1zACBATuiNh13HcFgyS_45/view?usp=sharing)   \| [log](https://drive.google.com/file/d/10vIP9hA98y8l24KPFiPyisCrmCm2GpFA/view?usp=sharing) |
| VideoMAEV2-g |   8   | AdaTAD  |  768   |   224    | InternVideo  |  61.74  |  43.17   |  10.68   |  41.85   | [config](e2e_anet_videomaev2_g_192x4_224_adapter_internvideo)  |                                                 [log](https://drive.google.com/file/d/1NQwJg4VbageCN0QYylWX9I_mSDE9wKa-/view?usp=sharing)                                                  |
| VideoMAEV2-g |   8   | AdaTAD  |  768   |   224    | InternVideo2 |  63.59  |  44.31   |  10.66   |  42.90   | [config](e2e_anet_videomaev2_g_192x4_224_adapter_internvideo2) |                                                 [log](https://drive.google.com/file/d/1DQquCFhNNRcK8dAsOT81dsuM4UGZ6HiJ/view?usp=sharing)                                                  |

- To train the model on ActivityNet, you can run the following command.

```bash
torchrun --nnodes=1 --nproc_per_node=4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/train.py configs/adatad/anet/e2e_anet_videomae_s_192x4_160_adapter.py
```

- To use the same checkpoint but test with another classifier, you can run the following command.

```bash
torchrun --nnodes=1 --nproc_per_node=4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/test.py configs/adatad/anet/e2e_anet_videomaev2_g_192x4_224_adapter_internvideo2.py --checkpoint epoch_10_cba1017a.pth
```

**[NEW]** We provide the following checkpoints which does not require external classifier but directly trains 200 classification head, for the convenience of zero-shot inference.
|  Backbone  | GPUs  | Setting | Frames | Img Size | Classifier | mAP@0.5 | mAP@0.75 | mAP@0.95 | ave. mAP |                           Config                            |                                                                                          Download                                                                                          |
| :--------: | :---: | :-----: | :----: | :------: | :--------: | :-----: | :------: | :------: | :------: | :---------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| VideoMAE-L |   4   | AdaTAD  |  768   |   224    |     x      |  59.00  |  39.96   |   9.15   |  39.15   | [config](anet/e2e_anet_videomae_l_192x4_224_adapter_cls.py) | [model](https://drive.google.com/file/d/1VYAvDrc7O7W4hDmUjjE6y32WmVNQ4ZR_/view?usp=sharing)   \| [log](https://drive.google.com/file/d/12BNxMKnigssPvzfX2s8skkVCCgvomwVE/view?usp=sharing) |

## THUMOS-14 Results

Please refer to [README.md](../../tools/prepare_data/thumos/README.md#download-raw-videos) to prepare the raw video of THUMOS.

|   Backbone   | GPUs  | Setting | Frames | Img Size | mAP@0.3 | mAP@0.4 | mAP@0.5 | mAP@0.6 | mAP@0.7 | ave. mAP |                             Config                              |                                                                                          Download                                                                                          |
| :----------: | :---: | :-----: | :----: | :------: | :-----: | :-----: | :-----: | :-----: | :-----: | :------: | :-------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|  VideoMAE-S  |   2   | AdaTAD  |  768   |   160    |  83.90  |  79.01  |  72.38  |  61.57  |  48.27  |  69.03   |   [config](thumos/e2e_thumos_videomae_s_768x1_160_adapter.py)   | [model](https://drive.google.com/file/d/1_noys-dyuQFGlXigGzW_sLB7I4Q4-zst/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1esvBEVunTnK87Kh2g1dFZO-38zOGvn4-/view?usp=sharing) |
|  VideoMAE-B  |   2   | AdaTAD  |  768   |   160    |  85.95  |  81.86  |  75.02  |  63.29  |  49.56  |  71.14   |   [config](thumos/e2e_thumos_videomae_b_768x1_160_adapter.py)   | [model](https://drive.google.com/file/d/1KpZjPDb7W3rTIRkOYbe_ViQNElVI7ZO9/view?usp=sharing)   \| [log](https://drive.google.com/file/d/17MMuKh-N_dRp7JijmrcLVkYBguIgVb41/view?usp=sharing) |
|  VideoMAE-L  |   2   | AdaTAD  |  768   |   160    |  87.17  |  83.58  |  76.88  |  66.81  |  53.13  |  73.51   |   [config](thumos/e2e_thumos_videomae_l_768x1_160_adapter.py)   | [model](https://drive.google.com/file/d/1yijRNA4zeoYnFtTk3N7NyCClihvXFAYK/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1gPZdHeZlg_H7XlmTMu0vGxhZMs9pf9PL/view?usp=sharing) |
|  VideoMAE-H  |   2   | AdaTAD  |  768   |   160    |  88.42  |  84.63  |  78.72  |  69.04  |  53.95  |  74.95   |   [config](thumos/e2e_thumos_videomae_h_768x1_160_adapter.py)   | [model](https://drive.google.com/file/d/1Tpl6GTuWhclZ9b5t6Kswori7az51-DFQ/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1WYGnUAtD0UWfM_rgGyGvrd9CXu_EGQr8/view?usp=sharing) |
| VideoMAEV2-g |   2   | AdaTAD  |  768   |   160    |  88.63  |  85.39  |  79.17  |  68.34  |  53.79  |  75.06   |  [config](thumos/e2e_thumos_videomaev2_g_768x1_160_adapter.py)  | [model](https://drive.google.com/file/d/1fpac_NXmQBR-g2Lau5FE49LBePPprgMY/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1HrXpqU1-LCDODC92WyG3bA_W2mm_ps_z/view?usp=sharing) |
| VideoMAEV2-g |   2   | AdaTAD  |  1536  |   224    |  89.93  |  86.83  |  81.24  |  69.97  |  57.36  |  77.07   |  [config](thumos/e2e_thumos_videomaev2_g_768x2_224_adapter.py)  | [model](https://drive.google.com/file/d/1_pfJRKnjkLzW4EOQNpms1bMJIeMJXpMC/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1BZFajKXUHlZA2qf7nOhoEJrnONcVybPH/view?usp=sharing) |
| VideoMAEV2-g |   2   | AdaTAD† |  1536  |   224    |  88.43  |  84.72  |  77.88  |  68.51  |  53.72  |  74.65   | [config](thumos/e2e_thumos_videomaev2_g_768x2_224_side_2e-4.py) | [model](https://drive.google.com/file/d/1sIH9D4hpqoUiwNSMa16JoHwn1KZG_IIc/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1LXwxH28tXgNXMpYRrHMgGAzty-5_eazi/view?usp=sharing) |

- To train the model on THUMOS, you can run the following command.

```bash
torchrun --nnodes=1 --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/train.py configs/adatad/thumos/e2e_thumos_videomae_s_768x1_160_adapter.py
```

- To search the adapter's learning rate, or change other hyper-parameters, you can run the following command.

```bash
torchrun --nnodes=1 --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/train.py configs/adatad/thumos/e2e_thumos_videomae_s_768x1_160_adapter.py \ 
  --cfg-options optimizer.backbone.custom.0.lr=1e-4 --id 1
```

## EPIC-KITCHENS Results


Before running the experiments, please download the EPIC-pretrained VideoMAE's weights, and put them under the path `./pretrained/`.

|   Model    | Pretrain Dataset | Finetune Dataset |                                            Checkpoints                                             |
| :--------: | :--------------: | :--------------: | :------------------------------------------------------------------------------------------------: |
| VideoMAE-L |   InternVideo1   |    EPIC-Noun     | [Google Drive](https://drive.google.com/file/d/1nRuzJI4ej90vFsKCBSugRVOmxrR8urwW/view?usp=sharing) |
| VideoMAE-L |   InternVideo1   |    EPIC-Verb     | [Google Drive](https://drive.google.com/file/d/1h7oLiNN5LTXau4HWmmzS_ekvuNdZkp-b/view?usp=sharing) |

Please refer to [README.md](../../tools/prepare_data/epic/README.md#download-raw-videos) to prepare the raw video of EPIC-Kitchens.

| Subset |   Backbone    | GPUs  | Setting | Frames | Img Size | mAP@0.1 | mAP@0.2 | mAP@0.3 | mAP@0.4 | mAP@0.5 | ave. mAP |                             Config                              |                                                                                          Download                                                                                          |
| :----: | :-----------: | :---: | :-----: | :----: | :------: | :-----: | :-----: | :-----: | :-----: | :-----: | :------: | :-------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|  Noun  | VideoMAE-Noun |   2   | AdaTAD  | 768x8  |   160    |  33.88  |  32.41  |  30.58  |  27.66  |  22.67  |  29.44   | [config](epic/e2e_epic_videomae_l_ft_768x8_160_adapter_noun.py) | [model](https://drive.google.com/file/d/17k3f6wirqniLTjKOsIXbfqJPA_iLb88E/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1vNsPuw_XRjcnI3F4UQ8hQ2iz-o_7X3lO/view?usp=sharing) |
|  Verb  | VideoMAE-Verb |   2   | AdaTAD  | 768x8  |   160    |  33.02  |  32.43  |  30.51  |  27.80  |  24.69  |  29.69   | [config](epic/e2e_epic_videomae_l_ft_768x8_160_adapter_verb.py) | [model](https://drive.google.com/file/d/16Hq3sHu0S97Ge2AewHT6DOaHSo0TqIlx/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1VaH7VVguBrXuZHlLq1mmKSLlIqV05Kt8/view?usp=sharing) |

- To train the model on EPIC-Kitchens, you can run the following command.

```bash
torchrun --nnodes=1 --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/train.py configs/adatad/epic/e2e_epic_videomae_l_ft_768x8_160_adapter_noun.py
```

## Ego4D-MQ Results


Before running the experiments, please download the InternVideo1-MQ weights, and put them under the path `./pretrained/`.

|      Model      | Pretrain Dataset  |   Finetune Dataset    |                                                Original Link                                                |                                       Converted Checkpoints                                        |
| :-------------: | :---------------: | :-------------------: | :---------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------: |
| InternVideo1-MQ | InternVideo1-K700 | Ego4D-Verb + Ego4D-MQ | [Url](https://github.com/OpenGVLab/ego4d-eccv2022-solutions/tree/main#:~:text=%2D-,Download,-UniFormer%2DB) | [Google Drive](https://drive.google.com/file/d/1MYyXdBCxbGsHWiRPf1gWSjLc56Tsk3cZ/view?usp=sharing) |

Please refer to [README.md](../../tools/prepare_data/ego4d/README.md#download-raw-videos) to prepare the raw video of Ego4D-MQ.

|    Backbone     | GPUs  | Setting | Frames | Img Size | mAP@0.1 | mAP@0.2 | mAP@0.3 | mAP@0.4 | mAP@0.5 | ave. mAP |                              Config                               |                                                                                          Download                                                                                          |
| :-------------: | :---: | :-----: | :----: | :------: | :-----: | :-----: | :-----: | :-----: | :-----: | :------: | :---------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| InternVideo1-MQ |   2   | AdaTAD  | 1800x4 |   192    |  33.69  |  31.19  |  28.37  |  26.12  |  22.67  |  28.41   | [config](epic/e2e_ego4d_internvideo_1800x4_192_adapter_lr4e-4.py) | [model](https://drive.google.com/file/d/15Y_Qe2ksUQ0zmgUTCzMWsWsAmNTFN0RF/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1N9EaoWjYrI_V2KDdFAHneSRZ4maRIkj7/view?usp=sharing) |

- To train the model on Ego4D-MQ, you can run the following command.

```bash
torchrun --nnodes=1 --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/train.py configs/adatad/ego4d/e2e_ego4d_internvideo_1800x4_192_adapter_lr4e-4.py
```

## Multi-THUMOS Results

Please refer to [README.md](../../tools/prepare_data/multi-thumos/README.md#download-raw-videos) to prepare the raw video of Multi-THUMOS.

|   Backbone   | GPUs  | Setting | Frames | Img Size | mAP@0.2 | mAP@0.5 | mAP@0.7 | ave. mAP (0.1:0.9:0.1) |                                  Config                                  |                                                                                          Download                                                                                          |
| :----------: | :---: | :-----: | :----: | :------: | :-----: | :-----: | :-----: | :--------------------: | :----------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|  VideoMAE-S  |   2   | AdaTAD  |  768   |   160    |  61.34  |  46.74  |  26.88  |         40.77          |  [config](multi_thumos/e2e_multithumos_videomae_s_768x1_160_adapter.py)  | [model](https://drive.google.com/file/d/1rpNh10Atpb6pNyltcm-1CoKh9uZHCaua/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1403HdIJvPDY_jrSYNhv5Chx8bZiUllHX/view?usp=sharing) |
|  VideoMAE-B  |   2   | AdaTAD  |  768   |   160    |  63.90  |  48.74  |  28.72  |         42.76          |  [config](multi_thumos/e2e_multithumos_videomae_b_768x1_160_adapter.py)  | [model](https://drive.google.com/file/d/17-TxzyppIsi2J9NeYQNgJVqKMWM_E1jk/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1Ue_jZbcAdYVvv73d6MAidY4bxI3apene/view?usp=sharing) |
|  VideoMAE-L  |   2   | AdaTAD  |  768   |   160    |  66.06  |  51.80  |  31.73  |         45.15          |  [config](multi_thumos/e2e_multithumos_videomae_l_768x1_160_adapter.py)  | [model](https://drive.google.com/file/d/1Kvywb_gOy8CLLFqLHTWklp677vXM7GFo/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1h6cpJV7UBV8sIWMuHKGOq5cOOcp5oJx-/view?usp=sharing) |
|  VideoMAE-H  |   2   | AdaTAD  |  768   |   160    |  67.20  |  52.99  |  32.70  |         46.02          |  [config](multi_thumos/e2e_multithumos_videomae_h_768x1_160_adapter.py)  | [model](https://drive.google.com/file/d/1YkufQ9BJs-CYOdwd2j8A9t4PfX5wIQtG/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1ZqShnZBGv0pWDjS0iz8Hsz9FQMMA92Ec/view?usp=sharing) |
| VideoMAEV2-g |   2   | AdaTAD  |  768   |   160    |  68.23  |  53.87  |  33.03  |         46.74          | [config](multi_thumos/e2e_multithumos_videomaev2_g_768x1_160_adapter.py) | [model](https://drive.google.com/file/d/1XLo9HreqOICgXwK42HWagecgLV_tDmuW/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1yK4UCPeo0F6KfofPVg321_2PSjHF1OS8/view?usp=sharing) |
| VideoMAEV2-g |   2   | AdaTAD  |  1536  |   224    |  71.11  |  55.83  |  34.86  |         48.73          | [config](multi_thumos/e2e_multithumos_videomaev2_g_768x2_224_adapter.py) | [model](https://drive.google.com/file/d/1o-w_JYJf-EY1zxKpYIRf_9NT5IYEQa70/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1THvzglg8Bi1skHDnPsYzW3K4yOOnUGjZ/view?usp=sharing) |

- To train the model on Multi-THUMOS, you can run the following command.

```bash
torchrun --nnodes=1 --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/train.py configs/adatad/multi_thumos/e2e_multithumos_videomae_s_768x1_160_adapter.py
```

## Charades Results

Please refer to [README.md](../../tools/prepare_data/charades/README.md#download-raw-videos) to prepare the raw video of Charades.

|   Backbone   | GPUs  | Setting | Frames | Img Size | mAP@0.2 | mAP@0.5 | mAP@0.7 | ave. mAP (0.1:0.9:0.1) |                              Config                               |                                                                                          Download                                                                                          |
| :----------: | :---: | :-----: | :----: | :------: | :-----: | :-----: | :-----: | :--------------------: | :---------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|  VideoMAE-S  |   2   | AdaTAD  |  512   |   160    |  35.89  |  27.43  |  16.35  |         24.14          |  [config](charades/e2e_charades_videomae_s_512x1_160_adapter.py)  | [model](https://drive.google.com/file/d/1ylK8qkX1-Vz5aiU7yiatGGUETam_4qvb/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1cJMXV9oxVIbVo164DnVqOPKiK4sLrBgp/view?usp=sharing) |
|  VideoMAE-B  |   2   | AdaTAD  |  512   |   160    |  40.84  |  31.75  |  20.05  |         27.99          |  [config](charades/e2e_charades_videomae_b_512x1_160_adapter.py)  | [model](https://drive.google.com/file/d/1OZ0-89fs4429CDjatxzx0Uoa021hxjAY/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1cbvtUiHajdxdqY_W_8XRSdg1fT5FMvMB/view?usp=sharing) |
|  VideoMAE-L  |   2   | AdaTAD  |  512   |   160    |  47.00  |  37.01  |  23.05  |         32.31          |  [config](charades/e2e_charades_videomae_l_512x1_160_adapter.py)  | [model](https://drive.google.com/file/d/1U6xSSatrRziA1CqzevYSR0-4L0nrbKFG/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1yiDl6Be6IOKIm8wzER7oiXkQXpqj_Md-/view?usp=sharing) |
|  VideoMAE-H  |   2   | AdaTAD  |  512   |   160    |  48.76  |  38.80  |  24.85  |         33.94          |  [config](charades/e2e_charades_videomae_h_512x1_160_adapter.py)  | [model](https://drive.google.com/file/d/1WyxiJm3s0UbfLPso6XQpRtqZDxQL0XAZ/view?usp=sharing)   \| [log](https://drive.google.com/file/d/1Ote-lURDkj979Ib3moiampElW25HWI3w/view?usp=sharing) |
| VideoMAEV2-g |   4   | AdaTAD  |  512   |   160    |  53.72  |  42.91  |  27.69  |         37.56          | [config](charades/e2e_charades_videomaev2_g_512x1_160_adapter.py) | [model](https://drive.google.com/file/d/1agp_aTzqpMq4CQUtSIPSLlY2PSfDxoI-/view?usp=sharing)   \| [log](https://drive.google.com/file/d/10Yd6Wy2OTh9rPn8QxAn75tLqj3p4SRaF/view?usp=sharing) |

- To train the model on Charades, you can run the following command.

```bash
torchrun --nnodes=1 --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/train.py configs/adatad/charades/e2e_charades_videomae_s_512x1_160_adapter.py
```


## Citation

```latex
@InProceedings{Liu_2024_CVPR,
    author    = {Liu, Shuming and Zhang, Chen-Lin and Zhao, Chen and Ghanem, Bernard},
    title     = {End-to-End Temporal Action Detection with 1B Parameters Across 1000 Frames},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {18591-18601}
}
```