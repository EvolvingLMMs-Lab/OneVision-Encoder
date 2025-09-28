import argparse
import json
import os
import tqdm


def main(args):
    missing_list = []

    anno_database = json.load(open(args.anno_file))["database"]
    for video_name in tqdm.tqdm(list(anno_database.keys())):
        file_path = os.path.join(args.data_dir, f"{args.prefix}{video_name}{args.suffix}.{args.ext}")

        if not os.path.exists(file_path):
            missing_list.append(video_name)

    saved_path = os.path.join(f"{args.data_dir}", "missing_files.txt")
    with open(saved_path, "w") as f:
        f.write("\n".join(missing_list))

    print(
        f"Total {len(anno_database.keys())} videos/features in dataset, "
        f"missing {len(missing_list)} videos/features."
    )
    print(f"Missing file has been saved in {saved_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Temporal Action Detector")
    parser.add_argument("anno_file", metavar="FILE", type=str, help="path to annotation")
    parser.add_argument("data_dir", metavar="FILE", type=str, help="path to data folder")
    parser.add_argument("--prefix", type=str, default="")
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--ext", type=str, default="npy")
    args = parser.parse_args()

    main(args)
