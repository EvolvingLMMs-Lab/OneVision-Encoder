import argparse
import os
import json
import tqdm


def resize_and_chunk_video(src_video_path, dst_video_path, short_size, start_second, end_second):
    # check whether is stereo video if the width is smaller than height
    result = os.popen(
        f"ffprobe -hide_banner -loglevel error -select_streams v:0 -show_entries stream=width,height,duration -of csv=p=0 {src_video_path}"
    )
    w, h, duration = [d for d in result.readline().rstrip().split(",")]
    w, h, duration = int(w), int(h), float(duration)

    if h > w:
        if h == 2880 and w == 1440:
            print(f"This is a stereo video: {src_video_path}, width: {w}, height: {h}. Chunk half and resize.")
            os.system(
                f"ffmpeg -hide_banner -loglevel error -i {src_video_path} -c:v libx264 "
                f"-vf scale=1136:320,crop=iw/2:ih:0:0 -aspect 4:3 "
                f"-ss {start_second} -to {end_second} "
                f"-an {dst_video_path} -y"
            )
        else:
            print(f"{src_video_path}: height: {h} is larger than width: {w}. Resize the width to {short_size}.")
            os.system(
                f"ffmpeg -hide_banner -loglevel error -i {src_video_path} -c:v libx264 "
                f"-ss {start_second} -to {end_second} "
                f"-vf scale={short_size}:-2 -an {dst_video_path} -y"
            )
    else:
        os.system(
            f"ffmpeg -hide_banner -loglevel error -i {src_video_path} -c:v libx264 "
            f"-ss {start_second} -to {end_second} "
            f"-vf scale=-2:{short_size} -an {dst_video_path} -y"
        )


def parse_annotation(anno_folder):
    train_path = os.path.join(anno_folder, "moments_train.json")
    val_path = os.path.join(anno_folder, "moments_val.json")
    test_path = os.path.join(anno_folder, "moments_test_unannotated.json")

    train_anno = json.load(open(train_path, "r"))
    val_anno = json.load(open(val_path, "r"))
    test_anno = json.load(open(test_path, "r"))
    annos = train_anno["videos"] + val_anno["videos"] + test_anno["videos"]

    clip_list = []
    for video in tqdm.tqdm(annos):
        video_uid = video["video_uid"]

        for clip in video["clips"]:
            clip_uid = clip["clip_uid"]
            start_second = max(float(clip["video_start_sec"]), 0)
            end_second = float(clip["video_end_sec"])
            clip_list.append([video_uid, clip_uid, start_second, end_second])
    return clip_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("anno_dir", metavar="FILE", type=str, help="path to annotation folder")
    parser.add_argument("data_dir", metavar="FILE", type=str, help="path to raw video folder")
    parser.add_argument("save_path", metavar="FILE", type=str, help="path to save")
    parser.add_argument("--short_size", type=int, default=320, help="resize the short side of the video")
    parser.add_argument("--part", type=int, default=0, help="data[part::total]")
    parser.add_argument("--total", type=int, default=1, help="how many parts exist")
    args = parser.parse_args()

    data_list = parse_annotation(args.anno_dir)
    sample_list = data_list[args.part :: args.total]
    print(
        f"Total clip number: {len(data_list)}, processing part {args.part} out of {args.total} parts, "
        f"which has {len(sample_list)} clips."
    )

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    for sample in tqdm.tqdm(sample_list):
        video_uid, clip_uid, start_second, end_second = sample
        src_video_path = os.path.join(args.data_dir, f"{video_uid}.mp4")
        dst_video_path = os.path.join(args.save_path, f"{clip_uid}.mp4")

        if os.path.exists(dst_video_path):
            print(f"{dst_video_path} exists, skip")
        else:
            resize_and_chunk_video(
                src_video_path,
                dst_video_path,
                args.short_size,
                start_second,
                end_second,
            )

    print("Done!")

    # python tools/prepare_data/ego4d/accurate_trim_MQ.py ego4d_data/v2/annotations/ ego4d_data/v1/full_scale data/ego4d/raw_data/MQ_data/mq_videos_short320/
