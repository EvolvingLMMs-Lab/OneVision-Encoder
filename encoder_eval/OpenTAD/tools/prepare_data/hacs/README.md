# Data Preparation for HACS

## Download Annotations

You can download the annotation by
```bash
bash download_annotation.sh
```
The annotation should be in `data/hacs-1.1.1/annotations/`.


## Download Pre-extracted Features

Please put the downloaded feature under the path: `data/hacs-1.1.1/features/`.


We provide the following pre-extracted features for HACS:

|     Feature     |                                                                          Url                                                                           |            Backbone            |                           Feature Extraction Setting                           |
| :-------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------: | :----------------------------------------------------------------------------: |
|    slowfast     |                           [Google Drive](https://drive.google.com/file/d/1kDQAhpPI7zsYlRBDd0wjuvpBdfCthxoO/view?usp=sharing)                           | SlowFast-R101 (HACS Finetuned) | 15fps, snippet_stride=8, clip_length=32, frame_interval=1, extracted by TCANet |
|    videomae     | [InternVideo Repo](https://github.com/OpenGVLab/InternVideo/tree/main/Downstream/Temporal-Action-Localization#to-reproduce-our-results-of-internvideo) |        VideoMAE-L-K700         |              snippet_stride=16, clip_length=16, frame_interval=1               |
| internvideo2_6b |           [Official Repo](https://github.com/OpenGVLab/video-mamba-suite/blob/main/video-mamba-suite/temporal-action-localization/README.md)           |        InternVideo2-6B         |               snippet_stride=8, clip_length=16, frame_interval=1               |

## Download Raw Videos

Please put the downloaded video under the path: `data/hacs/raw_data/`.

You can download the raw video from [official website](https://github.com/hangzhaomit/HACS-dataset).

## Citation

```BibTeX
@article{zhao2019hacs,
  title={HACS: Human Action Clips and Segments Dataset for Recognition and Temporal Localization},
  author={Zhao, Hang and Yan, Zhicheng and Torresani, Lorenzo and Torralba, Antonio},
  journal={arXiv preprint arXiv:1712.09374},
  year={2019}
}
```