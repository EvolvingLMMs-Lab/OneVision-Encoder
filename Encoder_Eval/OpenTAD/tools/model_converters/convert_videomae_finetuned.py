import argparse
import torch
import torch.nn.functional as F
from mmaction.registry import MODELS
from mmaction.utils import register_all_modules
from mmengine.runner import save_checkpoint

register_all_modules()


def process_checkpoint(in_path, out_path, arch):
    # rebuild model
    if arch == "small":
        model_cfg = dict(
            type="Recognizer3D",
            backbone=dict(
                type="VisionTransformer",
                img_size=224,
                patch_size=16,
                embed_dims=384,
                depth=12,
                num_heads=6,
                mlp_ratio=4,
                qkv_bias=True,
                num_frames=16,
                norm_cfg=dict(type="LN", eps=1e-6),
            ),
            cls_head=dict(
                type="TimeSformerHead",
                num_classes=400,
                in_channels=384,
                average_clips="prob",
            ),
            data_preprocessor=dict(
                type="ActionDataPreprocessor",
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                format_shape="NCTHW",
            ),
        )
    elif arch == "base":
        model_cfg = dict(
            type="Recognizer3D",
            backbone=dict(
                type="VisionTransformer",
                img_size=224,
                patch_size=16,
                embed_dims=768,
                depth=12,
                num_heads=12,
                mlp_ratio=4,
                qkv_bias=True,
                num_frames=16,
                norm_cfg=dict(type="LN", eps=1e-6),
            ),
            cls_head=dict(
                type="TimeSformerHead",
                num_classes=400,
                in_channels=768,
                average_clips="prob",
            ),
            data_preprocessor=dict(
                type="ActionDataPreprocessor",
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                format_shape="NCTHW",
            ),
        )
    elif arch == "large":

        model_cfg = dict(
            type="Recognizer3D",
            backbone=dict(
                type="VisionTransformer",
                img_size=224,
                patch_size=16,
                embed_dims=1024,
                depth=24,
                num_heads=16,
                mlp_ratio=4,
                qkv_bias=True,
                num_frames=16,
                norm_cfg=dict(type="LN", eps=1e-6),
            ),
            cls_head=dict(
                type="TimeSformerHead",
                num_classes=400,
                in_channels=1024,
                average_clips="prob",
            ),
            data_preprocessor=dict(
                type="ActionDataPreprocessor",
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                format_shape="NCTHW",
            ),
        )
    elif arch == "huge":
        model_cfg = dict(
            type="Recognizer3D",
            backbone=dict(
                type="VisionTransformer",
                img_size=224,
                patch_size=16,
                embed_dims=1280,
                depth=32,
                num_heads=16,
                mlp_ratio=4,
                qkv_bias=True,
                num_frames=16,
                norm_cfg=dict(type="LN", eps=1e-6),
            ),
            cls_head=dict(
                type="TimeSformerHead",
                num_classes=400,
                in_channels=1280,
                average_clips="prob",
            ),
            data_preprocessor=dict(
                type="ActionDataPreprocessor",
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                format_shape="NCTHW",
            ),
        )
    elif arch == "giant":
        model_cfg = dict(
            type="Recognizer3D",
            backbone=dict(
                type="VisionTransformer",
                img_size=224,
                patch_size=14,
                embed_dims=1408,
                depth=40,
                num_heads=16,
                mlp_ratio=48 / 11,
                qkv_bias=True,
                num_frames=16,
                norm_cfg=dict(type="LN", eps=1e-6),
            ),
            cls_head=dict(
                type="TimeSformerHead",
                num_classes=710,
                in_channels=1408,
                average_clips="prob",
            ),
            data_preprocessor=dict(
                type="ActionDataPreprocessor",
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                format_shape="NCTHW",
            ),
        )

    model = MODELS.build(model_cfg)

    # checkpoint model
    video_state_dict = torch.load(in_path, map_location="cpu")["module"]

    new_state_dict = {}
    print("\nThe following keys exist in checkpoint.pth but not in the model_cfg:")
    for key, value in video_state_dict.items():
        # convert keys
        if "fc1" in key:
            key = key.replace("fc1", "layers.0.0")
        elif "fc2" in key:
            key = key.replace("fc2", "layers.1")
        elif "patch_embed.proj" in key:
            key = key.replace("patch_embed.proj", "patch_embed.projection")
        elif "head" in key:
            key = key.replace("head", "cls_head.fc_cls")

        if "backbone." + key in model.state_dict().keys():  # blocks.0.xxx
            new_state_dict["backbone." + key] = value
        elif key.startswith("cls_head") and key in model.state_dict().keys():
            new_state_dict[key] = value
        else:
            print(key)

    print("\nThe following keys exist in model_cfg but not in the checkpoint.pth:")
    for key, value in model.state_dict().items():
        if key not in new_state_dict.keys():
            print(key)

    model.load_state_dict(new_state_dict, strict=False)
    save_checkpoint(model.state_dict(), out_path)
    print(f"Converted checkpoint saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert VideoMAE finetuned checkpoint")
    parser.add_argument("in_file", help="input checkpoint path")
    parser.add_argument("out_file", help="output checkpoint path")
    parser.add_argument("--arch", default="small", choices=["small", "base", "large", "huge", "giant"])
    args = parser.parse_args()

    process_checkpoint(args.in_file, args.out_file, args.arch)

"""example
python tools/model_converters/convert_videomae_finetuned.py \
    videomae_release_checkpoint.pth vit-base-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth --arch base
"""
