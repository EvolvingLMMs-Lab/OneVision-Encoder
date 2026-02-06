# Compressed Video Reader

The Compressed Video Reader is designed to read motion vectors and residuals from H.264/H.265 encoded videos.

## Installation

To install the reader, you can run the installation script located in the project root:

```shell
bash install.sh
```

The script will perform the following tasks:

1. Download the source code of FFmpeg
2. Apply patches to the source code
3. Configure and compile the FFmpeg package
4. Build and install the reader

To test if the reader has been successfully installed, run the following command:

```bash
# Test if the reader is installed successfully.
cv_reader -h || echo "Installation failed!"
```

## Python API

```python
import cv_reader
video_frames = cv_reader.read_video(video_path=path_to_video, with_residual=True)
```

## CLI Interface

You can use the following command to extract motion vectors and residuals from a compressed video:

```text
$ cv_reader -h
usage: Compressed Video Reader [-h] video output

positional arguments:
  video       Path to h.264/h.265 video file
  output      Path to save extracted motion vectors and residuals

optional arguments:
  -h, --help  show this help message and exit
```

To run the extraction process on the example video, execute the following command:

```bash
python debug_vis_mvres.py --video ../test_videos/h264_sample.mp4 --num_frames 16 --out_dir ./h264_debug
python debug_vis_mvres.py --video ../test_videos/h265_sample.mp4 --num_frames 16 --out_dir ./h265_debug
```