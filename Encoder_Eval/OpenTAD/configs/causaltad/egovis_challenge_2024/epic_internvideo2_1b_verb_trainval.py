_base_ = ["epic_internvideo2_1b_verb.py"]

dataset = dict(train=dict(subset_name=["train", "val"]), test=dict(subset_name="test"))

workflow = dict(
    logging_interval=200,
    checkpoint_interval=1,
    val_loss_interval=-1,
    val_eval_interval=-1,
)

work_dir = "exps/epic_kitchens/causal_internvideo2_1b_verb_trainval"
