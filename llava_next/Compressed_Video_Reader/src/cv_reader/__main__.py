import math

import cv_reader

import os
import argparse

import flow_vis
import cv2


def main():
    parser = argparse.ArgumentParser("Compressed Video Reader")
    parser.add_argument('video', help='Path to h.264 video file')
    parser.add_argument('output', help='Path to save extracted motion vectors and residuals')

    args = parser.parse_args()

    video_frames = cv_reader.read_video(video_path=args.video, with_residual=True)

    if len(video_frames) > 0:
        print(f"Number of frames: {len(video_frames)}")
        print(f"Size: {video_frames[0]['width']}x{video_frames[0]['height']}")
        print(f"Frame type: {''.join([f['pict_type'] for f in video_frames])}")

        os.makedirs(args.output, exist_ok=True)

        num_digits = math.ceil(math.log10(len(video_frames)))

        for frame_idx, frame in enumerate(video_frames):
            print(f"\rWriting: {frame_idx + 1}/{len(video_frames)}", flush=True,
                  end="" if frame_idx + 1 != len(video_frames) else "\n")
            cv2.imwrite(
                os.path.join(args.output,
                             f"{str(frame_idx).rjust(num_digits, '0')}_{frame['pict_type']}_mv_pre.jpg"),
                flow_vis.flow_to_color(frame["motion_vector"][..., :2])
            )
            cv2.imwrite(
                os.path.join(args.output,
                             f"{str(frame_idx).rjust(num_digits, '0')}_{frame['pict_type']}_mv_post.jpg"),
                flow_vis.flow_to_color(frame["motion_vector"][..., 2:4])
            )
            cv2.imwrite(
                os.path.join(args.output,
                             f"{str(frame_idx).rjust(num_digits, '0')}_{frame['pict_type']}_res.jpg"),
                frame["residual"]
            )
        print(f"Motion vectors and residuals are written to '{args.output}'.")
    else:
        print("Video is empty.")


if __name__ == '__main__':
    main()
