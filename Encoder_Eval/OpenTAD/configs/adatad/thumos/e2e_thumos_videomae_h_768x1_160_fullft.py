_base_ = ["e2e_thumos_videomae_s_768x1_160_fullft.py"]

model = dict(
    backbone=dict(
        backbone=dict(embed_dims=1280, depth=32, num_heads=16),
        custom=dict(pretrain="pretrained/vit-huge-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth"),
    ),
    projection=dict(in_channels=1280),
)

work_dir = "exps/thumos/adatad/e2e_actionformer_videomae_h_768x1_160_fullft"
