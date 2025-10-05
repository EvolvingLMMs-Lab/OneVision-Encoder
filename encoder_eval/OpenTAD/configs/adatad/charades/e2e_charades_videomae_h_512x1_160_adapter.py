_base_ = ["e2e_charades_videomae_s_512x1_160_adapter.py"]

model = dict(
    backbone=dict(
        backbone=dict(
            embed_dims=1280,
            depth=32,
            num_heads=16,
            adapter_index=list(range(32)),
        ),
        custom=dict(pretrain="pretrained/vit-huge-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth"),
    ),
    projection=dict(in_channels=1280),
)

optimizer = dict(backbone=dict(custom=[dict(name="adapter", lr=1e-4, weight_decay=0.05)]))

work_dir = "exps/charades/adatad/e2e_actionformer_videomae_h_512x1_160_adapter"
