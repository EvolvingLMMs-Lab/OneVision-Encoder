# Data Preparation for ActivityNet-1.3

## Download Annotations

You can download the annotation by
```bash
bash download_annotation.sh
```
The annotation should be in `data/activitynet-1.3/annotations/`.  

- Since some videos in the validation set are no longer exiting in YouTube, therefore BMN/AFSD/RTD-Action/TALLFormer and other following works choose to ignore these videos during evaluation and report the performance. We follow this evaluation protocol in this codebase for fair comparison. The blocked videos are recorded in the `blocked.json`, and there are total 4,728 videos left in validation subset. Therefore, the performance of ActionFormer/VideoMambaSuite could be slightly higher than their paper reported.

## Download Pre-extracted Features

Please put the downloaded feature under the path: `data/activitynet-1.3/features/`.


We provide the following pre-extracted features for ActivityNet:

|     Feature     |                                                                Url                                                                 |       Backbone        |                  Feature Extraction Setting                  |
| :-------------: | :--------------------------------------------------------------------------------------------------------------------------------: | :-------------------: | :----------------------------------------------------------: |
|  tsp_unresize   |                 [Google Drive](https://drive.google.com/file/d/1lJZh9H6U7fSPhQkx8aNDLJ1ddqfBfSwp/view?usp=sharing)                 |   TSP (r2plus1d-34)   | 15 fps,  snippet_stride=16, clip_length=16, frame_interval=1 |
|  tsn_unresize   |                 [Google Drive](https://drive.google.com/file/d/1PHWBVJrUAMUiHHd7SKQCezpp_i7gAHR3/view?usp=sharing)                 |   TSN (two stream)    |                                                              |
|  slowfast_r50   |                 [Google Drive](https://drive.google.com/file/d/1iHFZwRfmb0hZ0NPvoEFQsHsg81bkn23E/view?usp=sharing)                 |  SlowFast-R50-8x8x1   |      snippet_stride=8, clip_length=32, frame_interval=1      |
|  slowfast_r101  |                 [Google Drive](https://drive.google.com/file/d/1cXARVJKZNk6QCwjR3ZZJ1TgkNTZBC-b8/view?usp=sharing)                 |  SlowFast-R101-8x8x1  |      snippet_stride=8, clip_length=32, frame_interval=1      |
|   videomae_b    |                 [Google Drive](https://drive.google.com/file/d/1y_FwHz-YwoymaPZOxszjrkkVXZXC5nxs/view?usp=sharing)                 | VideoMAE-Base-16x4x1  |      snippet_stride=8, clip_length=16, frame_interval=4      |
|   videomae_l    |                 [Google Drive](https://drive.google.com/file/d/1YhldR2ruNljpZ3EDLYrzIc5gRrz4UbuW/view?usp=sharing)                 | VideoMAE-Large-16x4x1 |      snippet_stride=8, clip_length=16, frame_interval=4      |
| internvideo2_6b | [Official Repo](https://github.com/OpenGVLab/video-mamba-suite/blob/main/video-mamba-suite/temporal-action-localization/README.md) |    InternVideo2-6B    |      snippet_stride=8, clip_length=16, frame_interval=1      |

## Download Raw Videos

Please put the downloaded video under the path: `data/activitynet-1.3/raw_data/`.

You can download the raw video from [official website](https://docs.google.com/forms/d/e/1FAIpQLSdxhNVeeSCwB2USAfeNWCaI9saVT6i2hpiiizVYfa3MsTyamg/viewform), which provides 7-day access for downloading.

**[Update]** We have recently added a processed version of the ActivityNet-v1.3 videos to the folders above, named `Anet_videos_15fps_short256.zip`. The video has been converted by ffmpeg to 15 fps, and the shorter side of the video is resized to 256 pixels. In this codebase, all end-to-end ActivityNet experiments are based on this data.

## Citation

```BibTeX
@article{Heilbron2015ActivityNetAL,
  title={ActivityNet: A large-scale video benchmark for human activity understanding},
  author={Fabian Caba Heilbron and Victor Escorcia and Bernard Ghanem and Juan Carlos Niebles},
  journal={2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2015},
  pages={961-970}
}
```
