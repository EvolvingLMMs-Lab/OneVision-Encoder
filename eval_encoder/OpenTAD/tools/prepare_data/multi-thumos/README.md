# Data Preparation for Multi-THUMOS14

## Download Annotations

You can download the annotation by
```bash
bash download_annotation.sh
```
The annotation should be in `data/multithumos/annotations/`.

## Download Pre-extracted Features and Raw Videos

Note that the MultiTHUMOS dataset contains dense, multilabel, frame-level action annotations for 30 hours across 413 videos in the THUMOS'14 action detection dataset. Therefore, they share the same raw videos and pre-extracted features. Please refer to [THUMOS14 page](/tools/prepare_data/thumos/README.md) for more details.

## Citation

```BibTeX
@article{yeung2018every,
  title={Every moment counts: Dense detailed labeling of actions in complex videos},
  author={Yeung, Serena and Russakovsky, Olga and Jin, Ning and Andriluka, Mykhaylo and Mori, Greg and Fei-Fei, Li},
  journal={International Journal of Computer Vision},
  volume={126},
  pages={375--389},
  year={2018},
  publisher={Springer}
}
```