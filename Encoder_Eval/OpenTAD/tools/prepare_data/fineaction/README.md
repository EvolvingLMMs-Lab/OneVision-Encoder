# Data Preparation for FineAction-1.3

## Download Annotations

You can download the annotation by
```bash
bash download_annotation.sh
```
The annotation should be in `data/fineaction/annotations/`.


## Download Pre-extracted Features

Please put the downloaded feature under the path: `data/fineaction/features/`.


We provide the following pre-extracted features:

|     Feature     |                                                                          Url                                                                           |         Backbone         |             Feature Extraction Setting              |
| :-------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------: | :-------------------------------------------------: |
|   videomae_h    | [InternVideo Repo](https://github.com/OpenGVLab/InternVideo/tree/main/Downstream/Temporal-Action-Localization#to-reproduce-our-results-of-internvideo) |  VideoMAE-H-K700-16x4x1  | snippet_stride=16, clip_length=16, frame_interval=1 |
|  videomaev2_g   |                                   [VideoMAEv2 Repo](https://github.com/OpenGVLab/VideoMAEv2/blob/master/docs/TAD.md)                                   | VideoMAEv2-g-K710-16x4x1 | snippet_stride=16, clip_length=16, frame_interval=1 |
| internvideo2_1b |           [Official Repo](https://github.com/OpenGVLab/video-mamba-suite/blob/main/video-mamba-suite/temporal-action-localization/README.md)           |     InternVideo2-6B      | snippet_stride=4, clip_length=16, frame_interval=1  |

## Download Raw Videos

Please put the downloaded video under the path: `data/fineaction/raw_data/`.

You can download the raw video from [official website](https://github.com/Richard-61/FineAction).

## Citation

```BibTeX
@article{liu2022fineaction,
  title={Fineaction: A fine-grained video dataset for temporal action localization},
  author={Liu, Yi and Wang, Limin and Wang, Yali and Ma, Xiao and Qiao, Yu},
  journal={IEEE Transactions on Image Processing},
  year={2022},
  publisher={IEEE}
}
```