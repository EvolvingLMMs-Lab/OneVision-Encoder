# Prepare the Annotation and Data

The following table lists the supported datasets and provides links to the corresponding data preparation instructions.

| Dataset                                                    | Description                                                                                   |
| :--------------------------------------------------------- | :-------------------------------------------------------------------------------------------- |
| [ActivityNet](/tools/prepare_data/activitynet/README.md)   | A Large-Scale Video Benchmark for Human Activity Understanding with 19,994 videos.            |
| [THUMOS14](/tools/prepare_data/thumos/README.md)           | Consists of 413 videos with temporal annotations.                                             |
| [EPIC-KITCHENS](/tools/prepare_data/epic/README.md)        | Large-scale dataset in first-person (egocentric) vision. Latest version is EPIC-KITCHENS-100. |
| [EPIC-Sounds](/tools/prepare_data/epic_sounds/README.md)    | A large scale dataset of audio annotations capturing temporal extents and class labels.       |
| [Ego4D-MQ](/tools/prepare_data/ego4d/README.md)            | Ego4D is the world's largest egocentric video dataset. MQ refers to its moment query task.    |
| [HACS](/tools/prepare_data/hacs/README.md)                 | The same action taxonomy with ActivityNet, but consists of around 50K videos.                 |
| [FineAction](/tools/prepare_data/fineaction/README.md)     | Contains 103K temporal instances of 106 action categories, annotated in 17K untrimmed videos. |
| [Multi-THUMOS](/tools/prepare_data/multi-thumos/README.md) | Dense, multilabel action annotations of THUMOS14.                                             |
| [Charades](/tools/prepare_data/charades/README.md)         | Contains dense-labeled 9,848 annotated videos of daily activities.                            |


## FAQ

1. If you meet `FileNotFoundError: [Errno 2] No such file or directory: 'xxx/missing_files.txt'`
- It means you may need to generate a `missing_files.txt`, which should record the missing features compared to all the videos in the annotation files. You can use `python tools/prepare_data/generate_missing_list.py annotation.json feature_folder` to generate the txt file.
- eg. `python tools/prepare_data/generate_missing_list.py data/fineaction/annotations/annotations_gt.json  data/fineaction/features/fineaction_mae_g`
- In the provided feature from this codebase, we have already included this txt in the zip file.
