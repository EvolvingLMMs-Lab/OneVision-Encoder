_base_ = ["e2e_thumos_videomae_s_768x1_160_frozen.py"]

model = dict(
    backbone=dict(
        backbone=dict(embed_dims=768, depth=12, num_heads=12),
        custom=dict(pretrain="pretrained/vit-base-p16_videomae-k400-pre_16x4x1_kinetics-400_20221013-860a3cd3.pth"),
    ),
    projection=dict(in_channels=768),
)

work_dir = "exps/thumos/adatad/e2e_actionformer_videomae_b_768x1_160_frozen"
