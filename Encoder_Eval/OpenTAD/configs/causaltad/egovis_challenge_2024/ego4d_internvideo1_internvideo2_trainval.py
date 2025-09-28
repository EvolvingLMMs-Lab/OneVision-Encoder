_base_ = ["ego4d_internvideo1_internvideo2.py"]

dataset = dict(train=dict(subset_name=["train", "val"]), test=dict(subset_name="test"))

workflow = dict(
    logging_interval=200,
    checkpoint_interval=1,
    val_loss_interval=-1,
    val_eval_interval=-1,
    end_epoch=20,
)

work_dir = "exps/ego4d/causal_internvideo1_internvideo2_trainval"
