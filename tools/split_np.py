import argparse

import numpy as np
import torch


parser = argparse.ArgumentParser()
parser.add_argument("list")
parser.add_argument("num_split", type=int)
parser.add_argument("output")
args = parser.parse_args()

if __name__ == "__main__":
    all_center = []
    for x in open(args.list):
        x = x.strip()
        print(x)
        if x.endswith(".npy"):
            all_center.append(np.load(x.strip()))
        elif x.endswith(".pt"):
            state_dict = torch.load(x.strip(), map_location="cpu")
            weight = state_dict["weight"].cpu().numpy()
            all_center.append(weight)
        else:
            raise ValueError(f"Unknown file type: {x.strip()}")

    center = np.concatenate(all_center, axis=0)
    num_classes = center.shape[0]
    world_size = args.num_split
    for rank in range(world_size):
        num_local = num_classes // world_size + int(rank < num_classes % world_size)
        class_start = num_classes // world_size * rank + min(rank, num_classes % world_size)
        weight_split = center[class_start: class_start + num_local]
        np.save(args.output % rank, weight_split)
