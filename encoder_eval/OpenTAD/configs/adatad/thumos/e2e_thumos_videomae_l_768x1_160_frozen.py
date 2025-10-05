_base_ = ["e2e_thumos_videomae_s_768x1_160_frozen.py"]

model = dict(
    backbone=dict(
        backbone=dict(embed_dims=1024, depth=24, num_heads=16),
        custom=dict(pretrain="pretrained/vit-large-p16_videomae-k400-pre_16x4x1_kinetics-400_20221013-229dbb03.pth"),
    ),
    projection=dict(in_channels=1024),
)

work_dir = "exps/thumos/adatad/e2e_actionformer_videomae_l_768x1_160_frozen"
