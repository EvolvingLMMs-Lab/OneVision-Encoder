# Data Preparation for EPIC-SOUNDS

## Download Annotations

You can download the annotation by
```bash
bash download_annotation.sh
```
The annotation should be in `data/epic_sounds/annotations/`.

-  You can also download original csv annotations from [official website](https://github.com/epic-kitchens/epic-sounds-annotations), and use `convert_epic_sounds_anno.py` to generate the annotations that suitable for the codebase.

## Download Pre-extracted Features

Please put the downloaded feature under the path: `data/epic_sounds/features/`.


We provide the following pre-extracted features for EPIC-SOUNDS:

|      Feature      |                                                Url                                                 |              Backbone              |                Feature Extraction Setting                 |
| :---------------: | :------------------------------------------------------------------------------------------------: | :--------------------------------: | :-------------------------------------------------------: |
| Auditory SlowFast | [Google Drive](https://drive.google.com/file/d/1hTgzhsRzp1E4iRCUXdqXDdytEbz4VzmN/view?usp=sharing) | Auditory SlowFast (Epic Finetuned) | 30fps, snippet_stride=6, clip_length=32, frame_interval=1 |

## Citation

```BibTeX
@inproceedings{EPICSOUNDS2023,
           title={{EPIC-SOUNDS}: {A} {L}arge-{S}cale {D}ataset of {A}ctions that {S}ound},
           author={Huh, Jaesung and Chalk, Jacob and Kazakos, Evangelos and Damen, Dima and Zisserman, Andrew},
           booktitle   = {IEEE International Conference on Acoustics, Speech, & Signal Processing (ICASSP)},
           year      = {2023}
} 
```