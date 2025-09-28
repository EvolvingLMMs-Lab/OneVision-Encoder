_base_ = ["e2e_anet_videomae_s_192x4_160_adapter.py"]

model = dict(
    backbone=dict(
        backbone=dict(
            embed_dims=1024,
            depth=24,
            num_heads=16,
            adapter_index=list(range(24)),
        ),
        custom=dict(pretrain="pretrained/vit-large-p16_videomae-k400-pre_16x4x1_kinetics-400_20221013-229dbb03.pth"),
    ),
    projection=dict(in_channels=1024),
)

work_dir = "exps/anet/adatad/e2e_actionformer_videomae_l_192x4_160_adapter"
