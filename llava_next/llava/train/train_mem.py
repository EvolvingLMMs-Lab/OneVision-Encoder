from llava.train.train import train
import os
os.environ["WANDB_MODE"] = "offline"

if __name__ == "__main__":
    train()
