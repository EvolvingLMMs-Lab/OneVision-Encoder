_base_ = ["e2e_thumos_videomae_s_768x1_160_fullft.py"]

model = dict(
    backbone=dict(
        backbone=dict(patch_size=14, embed_dims=1408, depth=40, num_heads=16, mlp_ratio=48 / 11),
        custom=dict(pretrain="pretrained/vit-giant-p14_videomaev2-hybrid_pt_1200e_k710_ft_my.pth"),
    ),
    projection=dict(in_channels=1408),
)

work_dir = "exps/thumos/adatad/e2e_actionformer_videomaev2_g_768x1_160_fullft"
