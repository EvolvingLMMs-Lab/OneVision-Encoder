_base_ = ["e2e_charades_videomae_l_512x1_160_adapter.py"]

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

optimizer = dict(backbone=dict(custom=[dict(name="adapter", lr=2e-4, weight_decay=0.05)]))

work_dir = "exps/thumos/temporalmaxer_videomae"
