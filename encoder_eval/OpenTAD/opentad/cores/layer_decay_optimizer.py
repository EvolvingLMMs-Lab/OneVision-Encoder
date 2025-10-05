import torch


def get_layer_id_for_vit(var_name, max_layer_id):
    """Get the layer id to set the different learning rates in ``layer_wise``
    decay_type.

    Args:
        var_name (str): The key of the model.
        max_layer_id (int): Maximum layer id.
    Returns:
        int: The id number corresponding to different learning rate in
        ``LayerDecayOptimizerConstructor``.
    """
    if var_name.startswith("model.backbone"):
        if "patch_embed" in var_name or "pos_embed" in var_name:
            return 0
        elif ".blocks." in var_name:
            layer_id = int(var_name.split(".")[3]) + 1
            return layer_id
        else:
            return max_layer_id + 1
    else:
        return max_layer_id + 1


def build_vit_optimizer(cfg, model, logger):
    assert model.module.backbone.freeze_backbone == False, "The backbone should not be frozen."

    num_layers = cfg["num_layers"] + 2
    decay_rate = cfg["layer_decay_rate"]
    logger.info(f"Build LayerDecayOptimizer: layer_decay={decay_rate}, num_layers={num_layers}")
    base_lr = cfg["lr"]
    base_wd = cfg["weight_decay"]

    parameter_groups = {}

    # loop the backbone's parameters
    for name, param in model.module.backbone.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if name.startswith("model.backbone.blocks") and "norm" in name:
            group_name = "no_decay"
            this_weight_decay = 0.0
        elif "pos_embed" in name:
            group_name = "no_decay_pos_embed"
            this_weight_decay = 0
        else:
            group_name = "decay"
            this_weight_decay = base_wd

        layer_id = get_layer_id_for_vit(name, cfg["num_layers"])
        logger.info(f"set param {name} as id {layer_id}")

        group_name = f"layer_{layer_id}_{group_name}"

        if group_name not in parameter_groups:
            scale = decay_rate ** (num_layers - 1 - layer_id)

            parameter_groups[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "param_names": [],
                "lr_scale": scale,
                "group_name": group_name,
                "lr": scale * base_lr,
            }

        parameter_groups[group_name]["params"].append(param)
        parameter_groups[group_name]["param_names"].append(name)

    # set the detector's optim_groups: SHOULD NOT CONTAIN BACKBONE PARAMS
    # here, if each method want their own paramwise config, eg. to specify the learning rate,
    # weight decay for a certain layer, the model should have a function called get_optim_groups
    if "paramwise" in cfg.keys() and cfg["paramwise"]:
        cfg.pop("paramwise")
        det_optim_groups = model.module.get_optim_groups(cfg)
    else:
        # optim_groups that does not contain backbone params
        detector_params = []
        for name, param in model.module.named_parameters():
            # exclude the backbone
            if name.startswith("backbone"):
                continue
            detector_params.append(param)
        det_optim_groups = [dict(params=detector_params)]

    # merge the optim_groups
    optim_groups = []
    optim_groups.extend(parameter_groups.values())
    optim_groups = optim_groups + det_optim_groups
    optimizer = torch.optim.AdamW(optim_groups)
    return optimizer
