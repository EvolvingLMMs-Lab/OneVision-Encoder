import argparse
import os
import pandas as pd
import json
import tqdm


def parse_annotation(args):
    if os.path.exists(args.save_path) is False:
        os.makedirs(args.save_path)

    train_path = os.path.join(args.csv_dir, "EPIC_Sounds_train.csv")
    val_path = os.path.join(args.csv_dir, "EPIC_Sounds_validation.csv")
    test_path = os.path.join(args.csv_dir, "EPIC_Sounds_detection_test_videos.csv")
    duration_path = os.path.join(args.csv_dir, "EPIC_100_video_info.csv")

    train_anno = pd.read_csv(train_path)
    val_anno = pd.read_csv(val_path)
    test_anno = pd.read_csv(test_path)
    duration_anno = pd.read_csv(duration_path)

    def _parse_subset(anno, video_info, subset, save_path):
        print(f"Processing {subset} subset...")
        database = {}
        all_video_id = anno["video_id"].unique()

        for video_id in tqdm.tqdm(all_video_id):
            video_duration = video_info[video_info["video_id"] == video_id]["duration"].values[0]
            filter_anno = anno[anno["video_id"] == video_id]

            if subset == "test":
                database[video_id] = dict(
                    subset=subset,
                    duration=video_duration,
                )
            else:
                annotations = []
                for _, row in filter_anno.iterrows():
                    # convert timestamp to seconds
                    def _convert_timestamp_to_seconds(timestamp):
                        timestamp = timestamp.split(":")
                        timestamp = [float(t) for t in timestamp]
                        return timestamp[0] * 3600 + timestamp[1] * 60 + timestamp[2]

                    start_time = _convert_timestamp_to_seconds(row["start_timestamp"])
                    end_time = _convert_timestamp_to_seconds(row["stop_timestamp"])

                    annotations.append(
                        dict(
                            segment=[round(start_time, 2), round(end_time, 2)],
                            label=f"id_{row['class_id']:03d}",
                        )
                    )

                database[video_id] = dict(
                    subset=subset,
                    duration=video_duration,
                    annotations=annotations,
                )
        return database

    database = {}
    database.update(_parse_subset(train_anno, duration_anno, "train", args.save_path))
    database.update(_parse_subset(val_anno, duration_anno, "val", args.save_path))
    database.update(_parse_subset(test_anno, duration_anno, "test", args.save_path))

    out = dict(version="epic-kitchens-100", database=database)
    json_path = os.path.join(args.save_path, "epic_kitchens_sound.json")
    with open(json_path, "w") as f:
        json.dump(out, f)
    print(f"The full annotation is saved at {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("csv_dir", metavar="FILE", type=str, help="path to original annotation")
    parser.add_argument("save_path", metavar="FILE", type=str, help="path to save")
    args = parser.parse_args()

    parse_annotation(args)

    # python tools/prepare_data/convert_epic_sound_anno.py data/epic_sounds/annotation data/epic_sounds/annotations
