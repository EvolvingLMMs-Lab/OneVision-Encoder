_base_ = ["ego4d_internvideo1_internvideo2.py"]

pred0 = "exps/ego4d/causal_internvideo1_internvideo2/gpu1_id0/outputs"
pred1 = "exps/ego4d/causal_internvideo1_internvideo2/gpu1_id1/outputs"
pred2 = "exps/ego4d/causal_internvideo1_internvideo2/gpu1_id2/outputs"

inference = dict(load_from_raw_predictions=True, fuse_list=[pred0, pred1, pred2])

solver = dict(test=dict(batch_size=1, num_workers=1))

post_processing = dict(nms=dict(sigma=2.0), save_dict=True)

work_dir = "exps/ego4d/causal_internvideo1_internvideo2_ensemble"
