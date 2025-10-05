import argparse
import subprocess

import torch


def process_checkpoint(in_file, out_file):
    checkpoint = torch.load(in_file, map_location="cpu")

    # only keep `epoch` and `state_dict`/`state_dict_ema`` for smaller file size
    ckpt_keys = list(checkpoint.keys())
    save_keys = ["meta", "epoch"]
    if "state_dict_ema" in ckpt_keys:
        save_keys.append("state_dict_ema")
    else:
        save_keys.append("state_dict")

    for k in ckpt_keys:
        if k not in save_keys:
            print(f"Key `{k}` will be removed because it is not in save_keys.")
            checkpoint.pop(k, None)

    # if it is necessary to remove some sensitive data in checkpoint['meta'],
    # add the code here.
    torch.save(checkpoint, out_file)
    sha = subprocess.check_output(["sha256sum", out_file]).decode()
    if out_file.endswith(".pth"):
        out_file_name = out_file[:-4]
    else:
        out_file_name = out_file
    final_file = out_file_name + f"_{sha[:8]}.pth"
    subprocess.Popen(["mv", out_file, final_file])
    print(f"The published model is saved at {final_file}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a checkpoint to be published")
    parser.add_argument("in_file", help="input checkpoint filename")
    parser.add_argument("out_file", help="output checkpoint filename")
    args = parser.parse_args()

    process_checkpoint(args.in_file, args.out_file)
