import os
import logging
import torch
import glob
import numpy as np
import shutil

rank = int(os.getenv("RANK", "0"))
local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "1"))


def unwrap_module(model):
    """
    递归地解包模型中的任何封装（DDP, torch.compile等）
    
    Args:
        model: 可能被封装的模型
        
    Returns:
        解包后的原始模型
    """
    # 定义要检查的封装（属性名 -> getter函数）
    wrappers = {
        "_orig_mod": lambda m: getattr(m, "_orig_mod", None),
        "module": lambda m: getattr(m, "module", None)
    }
    
    # 尝试用每个封装解包模型
    for _, getter in wrappers.items():
        unwrapped = getter(model)
        if unwrapped is not None:
            # 递归解包结果
            return unwrap_module(unwrapped)
    
    # 如果没有更多封装，返回模型
    return model


def save_checkpoint(
    output_dir,
    backbone,
    pfc_modules,
    lr_scheduler,
    amp,
    global_step,
    list_head_names,
    keep_num=5):
    """
    保存训练状态，每个步骤创建一个单独的文件夹，每个PFC头部创建子文件夹
    
    Args:
        output_dir: 基础输出目录
        backbone: 主干网络模型
        pfc_modules: PFC模块列表
        lr_scheduler: 学习率调度器
        amp: 自动混合精度对象
        global_step: 当前全局步数
        list_head_names: 头部名称列表
        keep_num: 保留的checkpoint数量
    """
    # 创建当前步骤的文件夹
    step_dir = os.path.join(output_dir, f"{global_step:08d}")
    os.makedirs(step_dir, exist_ok=True)
    
    # 保存backbone模型和优化器状态（只在rank 0上保存）
    if rank == 0:
        # 保存backbone模型 (移动到CPU)
        backbone_path = os.path.join(step_dir, "backbone.pt")
        backbone_state_dict = {k: v.cpu() for k, v in unwrap_module(backbone).state_dict().items()}
        torch.save(backbone_state_dict, backbone_path)
        
        # 保存学习率调度器状态
        scheduler_path = os.path.join(step_dir, "scheduler.pt")
        torch.save(lr_scheduler.state_dict(), scheduler_path)
        
        # 保存AMP状态（如果可用）
        if amp is not None:
            amp_path = os.path.join(step_dir, "amp.pt")
            torch.save(amp.state_dict(), amp_path)
        
        logging.info(f"Backbone, scheduler saved at step {global_step}")

    if isinstance(pfc_modules, list):
        # 每个rank保存自己的PFC模块
        for head_id, (head_name, pfc) in enumerate(zip(list_head_names, pfc_modules)):
            if isinstance(pfc, list):
                for i in range(len(pfc)):
                    # print(pfc)
                    # print(len(pfc))
                    # new_pfc, pfc_type = pfc[i]
                    head_dir = os.path.join(step_dir, f"{head_name}_{i:02d}")
                    os.makedirs(head_dir, exist_ok=True)
                    # 保存PFC模型状态（在头部文件夹中）- 带上名称并移动到CPU
                    pfc_path = os.path.join(head_dir, f"{head_name}_{rank:03d}.pt")
                    # pfc_state_dict, pfc_type = pfc
                    pfc_state_dict = {k: v.cpu() for k, v in pfc[i][0].state_dict().items()}  # 移动到CPU
                    torch.save(pfc_state_dict, pfc_path)
            elif isinstance(pfc, torch.nn.Module):
                # 为当前PFC头部创建单独的文件夹
                head_dir = os.path.join(step_dir, head_name)
                os.makedirs(head_dir, exist_ok=True)
                
                # 保存PFC模型状态（在头部文件夹中）- 带上名称并移动到CPU
                pfc_path = os.path.join(head_dir, f"{head_name}_{rank:03d}.pt")
                pfc_state_dict = {k: v.cpu() for k, v in pfc.state_dict().items()}  # 移动到CPU
                torch.save(pfc_state_dict, pfc_path)

    elif isinstance(pfc_modules, dict):
        # 每个rank保存自己的PFC模块
        for head_name, pfc in pfc_modules.items():
            if isinstance(pfc, list):
                for i in range(len(pfc)):
                    head_dir = os.path.join(step_dir, f"{head_name}_{i:02d}")
                    os.makedirs(head_dir, exist_ok=True)
                    # 保存PFC模型状态（在头部文件夹中）- 带上名称并移动到CPU
                    pfc_path = os.path.join(head_dir, f"{head_name}_{rank:03d}.pt")
                    pfc_state_dict = {k: v.cpu() for k, v in pfc[i].state_dict().items()}
                    torch.save(pfc_state_dict, pfc_path)
            elif isinstance(pfc, torch.nn.Module):
                # 为当前PFC头部创建单独的文件夹
                head_dir = os.path.join(step_dir, head_name)
                os.makedirs(head_dir, exist_ok=True)
                
                # 保存PFC模型状态（在头部文件夹中）- 带上名称并移动到CPU
                pfc_path = os.path.join(head_dir, f"{head_name}_{rank:03d}.pt")
                pfc_state_dict = {k: v.cpu() for k, v in pfc.state_dict().items()}
                torch.save(pfc_state_dict, pfc_path)
    else:
        raise ValueError("pfc_modules should be a list or a dict")

    # 清理旧的检查点文件夹
    if rank == 0:
        clean_old_checkpoints(output_dir, keep_num)
    
    logging.info(f"Rank {rank}: PFC modules saved at step {global_step}")


def clean_old_checkpoints(output_dir, keep_num=5):
    """
    删除旧的检查点文件夹，只保留最新的keep_num个
    """
    # 获取所有检查点文件夹
    checkpoint_dirs = []
    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        # 只考虑格式为8位数字的文件夹
        if os.path.isdir(item_path) and item.isdigit() and len(item) == 8:
            checkpoint_dirs.append(item_path)
    
    # 按修改时间排序（或者可以按文件夹名称数字排序）
    checkpoint_dirs.sort(key=lambda x: int(os.path.basename(x)))
    
    # 如果超出保留数量，删除最旧的
    if len(checkpoint_dirs) > keep_num:
        dirs_to_remove = checkpoint_dirs[:-keep_num]
        for dir_path in dirs_to_remove:
            try:
                shutil.rmtree(dir_path)
                logging.info(f"Removed old checkpoint: {dir_path}")
            except Exception as e:
                logging.warning(f"Failed to remove {dir_path}: {e}")


def load_checkpoint(output_dir, step, backbone, pfc_modules, lr_scheduler, 
                  amp, list_head_names):
    """
    从指定步骤的检查点文件夹加载训练状态
    
    Args:
        output_dir: 基础输出目录
        step: 要加载的步骤(如果为None则加载最新步骤)
        backbone: 主干网络模型
        pfc_modules: PFC模块列表
        lr_scheduler: 学习率调度器
        amp: 自动混合精度对象
        list_head_names: 头部名称列表
    
    Returns:
        dict: 包含恢复的全局步骤信息
    """
    # 如果未指定步骤，查找最新的
    if step is None:
        # 查找所有检查点文件夹
        checkpoint_dirs = []
        for item in os.listdir(output_dir):
            item_path = os.path.join(output_dir, item)
            if os.path.isdir(item_path) and item.isdigit() and len(item) == 8:
                checkpoint_dirs.append(int(item))
        
        if not checkpoint_dirs:
            logging.warning(f"No checkpoint directories found in {output_dir}")
            return None
        
        step = max(checkpoint_dirs)
    
    # 构建步骤文件夹路径
    step_dir = os.path.join(output_dir, f"{step:08d}")
    if not os.path.isdir(step_dir):
        logging.warning(f"Checkpoint directory not found: {step_dir}")
        return None
    
    # 加载backbone
    backbone_file = os.path.join(step_dir, "backbone.pt")
    if not os.path.exists(backbone_file):
        logging.warning(f"Backbone file not found: {backbone_file}")
        return None
    
    backbone_state = torch.load(backbone_file, )
    unwrap_module(backbone).load_state_dict(backbone_state)
    
    if isinstance(pfc_modules, list):
        # 加载PFC模块
        for head_id, (head_name, pfc) in enumerate(zip(list_head_names, pfc_modules)):
            if isinstance(pfc, list):
                for i in range(len(pfc)):
                    head_dir = os.path.join(step_dir, f"{head_name}_{i:02d}")
                    # PFC文件在头部文件夹中
                    pfc_file = os.path.join(head_dir, f"{head_name}_{rank:03d}.pt")
                    if os.path.exists(pfc_file):
                        pfc_state = torch.load(pfc_file, )
                        pfc[i].load_state_dict(pfc_state)
                        logging.info(f"Rank {rank}: Loaded PFC weights for {head_name}")
                    else:
                        logging.warning(f"Rank {rank}: PFC file not found: {pfc_file}")
            elif isinstance(pfc, torch.nn.Module):
                # PFC文件在头部文件夹中
                head_dir = os.path.join(step_dir, head_name)
                if not os.path.isdir(head_dir):
                    logging.warning(f"Head directory not found: {head_dir}")
                    continue
                    
                pfc_file = os.path.join(head_dir, f"{head_name}_{rank:03d}.pt")
                if os.path.exists(pfc_file):
                    pfc_state = torch.load(pfc_file, )
                    pfc.load_state_dict(pfc_state)
                    logging.info(f"Rank {rank}: Loaded PFC weights for {head_name}")
                else:
                    logging.warning(f"Rank {rank}: PFC file not found: {pfc_file}")
    elif isinstance(pfc_modules, dict):
        # 加载PFC模块
        for head_name, pfc in pfc_modules.items():
            # PFC文件在头部文件夹中
            if isinstance(pfc, list):
                for i in range(len(pfc)):
                    head_dir = os.path.join(step_dir, f"{head_name}_{i:02d}")
                    pfc_file = os.path.join(head_dir, f"{head_name}_{rank:03d}.pt")
                    if os.path.exists(pfc_file):
                        pfc_state = torch.load(pfc_file, )
                        pfc[i].load_state_dict(pfc_state)
                        logging.info(f"Rank {rank}: Loaded PFC weights for {head_name}")
                    else:
                        logging.warning(f"Rank {rank}: PFC file not found: {pfc_file}")
            elif isinstance(pfc, torch.nn.Module):
                head_dir = os.path.join(step_dir, head_name)
                if not os.path.isdir(head_dir):
                    logging.warning(f"Head directory not found: {head_dir}")
                    continue
                    
                pfc_file = os.path.join(head_dir, f"{head_name}_{rank:03d}.pt")
                if os.path.exists(pfc_file):
                    pfc_state = torch.load(pfc_file, )
                    pfc.load_state_dict(pfc_state)
                    logging.info(f"Rank {rank}: Loaded PFC weights for {head_name}")
                else:
                    logging.warning(f"Rank {rank}: PFC file not found: {pfc_file}")
    else:
        raise ValueError("pfc_modules should be a list or a dict")

    # 加载学习率调度器
    scheduler_file = os.path.join(step_dir, "scheduler.pt")
    if os.path.exists(scheduler_file):
        lr_scheduler.load_state_dict(torch.load(scheduler_file, ))
    else:
        logging.warning(f"Scheduler state file not found: {scheduler_file}")
    
    # 加载AMP状态
    if amp is not None:
        amp_file = os.path.join(step_dir, "amp.pt")
        if os.path.exists(amp_file):
            amp.load_state_dict(torch.load(amp_file, ))
        else:
            logging.warning(f"AMP state file not found: {amp_file}")
    
    return {
        'global_step': step
    }
