_base_ = ["e2e_anet_videomaev2_g_192x4_224_adapter.py"]

post_processing = dict(
    external_cls=dict(
        type="StandardClassifier",
        path="data/activitynet-1.3/classifiers/anet_UMTv2_6B_k710+K40_f16_frozenTuning.json_converted.json",
        topk=2,
    ),
)

work_dir = "exps/anet/adatad/e2e_actionformer_videomaev2_g_192x4_224_adapter_internvideo2"
