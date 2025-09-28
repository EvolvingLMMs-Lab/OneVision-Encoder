import argparse
import os
import json
import tqdm


def parse_annotation(args):
    train_path = os.path.join(args.data_dir, "moments_train.json")
    val_path = os.path.join(args.data_dir, "moments_val.json")
    test_path = os.path.join(args.data_dir, "moments_test_unannotated.json")

    train_anno = json.load(open(train_path, "r"))
    val_anno = json.load(open(val_path, "r"))
    test_anno = json.load(open(test_path, "r"))
    annos = train_anno["videos"] + val_anno["videos"] + test_anno["videos"]

    database = {}
    for video in tqdm.tqdm(annos):
        subset = video["split"]

        clips = video["clips"]
        for clip in clips:
            cid = clip["clip_uid"]
            start = max(int(clip["video_start_frame"]), 0)
            end = int(clip["video_end_frame"])
            start_second = max(float(clip["video_start_sec"]), 0)
            end_second = float(clip["video_end_sec"])
            duration = round(end_second - start_second, 4)
            num_frame = end - start
            fps = num_frame / duration
            if fps < 10 or fps > 100:
                print("Abnormal fps: ", fps)
                continue

            if subset == "test":
                database[cid] = dict(duration=duration, frame=num_frame, subset=subset)
            else:
                annotations = []

                # parse annotations from different annotators
                annotators = clip["annotations"]
                for annotator in annotators:
                    # parse action items
                    items = annotator["labels"]
                    for item in items:
                        # skip items not from primary categories
                        if not item["primary"]:
                            continue

                        ssi = item["video_start_time"] - start_second
                        esi = item["video_end_time"] - start_second

                        label = item["label"]
                        annotations += [dict(segment=[round(ssi, 4), round(esi, 4)], label=label)]

                if len(annotations) == 0:
                    continue

                database[cid] = dict(duration=duration, frame=num_frame, subset=subset, annotations=annotations)

    out = dict(version=train_anno["version"], date=train_anno["date"], database=database)

    # check the output folder
    if not os.path.exists(os.path.dirname(args.save_path)):
        os.makedirs(os.path.dirname(args.save_path))
    with open(args.save_path, "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("data_dir", metavar="FILE", type=str, help="path to data folder")
    parser.add_argument("save_path", metavar="FILE", type=str, help="path to save")
    args = parser.parse_args()

    parse_annotation(args)

    # python tools/prepare_data/ego4d/convert_ego4d_anno.py ego4d/v2/annotations data/ego4d/annotations/ego4d_v2_220429.json
