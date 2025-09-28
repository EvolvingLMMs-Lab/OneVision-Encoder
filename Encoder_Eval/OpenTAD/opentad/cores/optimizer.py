import torch
from .layer_decay_optimizer import build_vit_optimizer


def build_optimizer(cfg, model, logger):
    optimizer_type = cfg["type"]
    cfg.pop("type")

    if optimizer_type == "LayerDecayAdamW":
        return build_vit_optimizer(cfg, model, logger)

    # set the backbone's optim_groups: SHOULD ONLY CONTAIN BACKBONE PARAMS
    if hasattr(model.module, "backbone"):  # if backbone exists
        if model.module.backbone.freeze_backbone == False:  # not frozen
            assert (
                "backbone" in cfg.keys()
            ), "Freeze_backbone is set to False, but backbone parameters is not provided in the optimizer config."
            backbone_cfg = cfg["backbone"]
            cfg.pop("backbone")
            backbone_optim_groups = get_backbone_optim_groups(backbone_cfg, model, logger)

        else:  # frozen backbone
            backbone_optim_groups = []
            logger.info(f"Freeze the backbone...")
    else:
        backbone_optim_groups = []

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
    optim_groups = backbone_optim_groups + det_optim_groups

    if optimizer_type == "AdamW":
        optimizer = torch.optim.AdamW(optim_groups, **cfg)
    elif optimizer_type == "Adam":
        optimizer = torch.optim.Adam(optim_groups, **cfg)
    elif optimizer_type == "SGD":
        optimizer = torch.optim.SGD(optim_groups, **cfg)
    else:
        raise f"Optimizer {optimizer_type} is not supported so far."

    return optimizer


def get_backbone_optim_groups(cfg, model, logger):
    """Example:
    backbone = dict(
        lr=1e-5,
        weight_decay=1e-4,
        custom=[dict(name="residual", lr=1e-3, weight_decay=1e-4)],
        exclude=[],
    )
    """

    # custom_name_list
    if "custom" in cfg.keys():
        custom_name_list = [d["name"] for d in cfg["custom"]]
        custom_params_list = [[] for _ in custom_name_list]
    else:
        custom_name_list = []

    # exclude_name_list
    if "exclude" in cfg.keys():
        exclude_name_list = cfg["exclude"]
    else:
        exclude_name_list = []

    # rest_params_list
    rest_params_list = []

    name_list = []
    # split the backbone parameters into different groups
    for name, param in model.module.backbone.named_parameters():
        # loop the exclude_name_list
        is_exclude = False
        if len(exclude_name_list) > 0:
            for exclude_name in exclude_name_list:
                if exclude_name in name:
                    is_exclude = True
                    break

        # loop through the custom_name_list
        is_custom = False
        if len(custom_name_list) > 0:
            for i, custom_name in enumerate(custom_name_list):
                if custom_name in name:
                    custom_params_list[i].append(param)
                    name_list.append(name)
                    is_custom = True
                    break

        # if is_custom, we have already appended the param to the custom_params_list
        # if is _exclude, we do not need to append the param to the rest_params_list
        if is_exclude or is_custom:
            continue

        # this is a rest parameter without special treatment
        if not is_custom:
            # this is the rest backbone parameters
            rest_params_list.append(param)
            name_list.append(name)

    for name in name_list:
        logger.info(f"Backbone parameter: {name}")
    # add params to optim_groups
    backbone_optim_groups = []

    if len(rest_params_list) > 0:
        backbone_optim_groups.append(
            dict(
                params=rest_params_list,
                lr=cfg["lr"],
                weight_decay=cfg["weight_decay"],
            )
        )

    if len(custom_name_list) > 0:
        for i, custom_name in enumerate(custom_name_list):
            backbone_optim_groups.append(
                dict(
                    params=custom_params_list[i],
                    lr=cfg["custom"][i]["lr"],
                    weight_decay=cfg["custom"][i]["weight_decay"],
                )
            )
    return backbone_optim_groups
