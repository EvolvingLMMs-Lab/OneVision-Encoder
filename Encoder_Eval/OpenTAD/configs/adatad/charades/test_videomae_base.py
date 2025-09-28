_base_ = ["test_videomae_base.py"]

model = dict(
    backbone=dict(
        backbone=dict(embed_dims=384, depth=12, num_heads=6),
        custom=dict(pretrain="pretrained/backbone_tube248_dense_moreepoch.pt"),
    ),
    projection=dict(in_channels=384),
)

optimizer = dict(backbone=dict(custom=[dict(name="adapter", lr=4e-4, weight_decay=0.05)]))

work_dir = "exps/charades/adatad/test_videomae_base"
