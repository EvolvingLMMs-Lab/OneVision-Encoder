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
    Recursively unwrap any wrappers from the model (DDP, torch.compile, etc.)
    
    Args:
        model: Possibly wrapped model
        
    Returns:
        Unwrapped original model
    """
    # Define wrappers to check (attribute name -> getter function)
    wrappers = {
        "_orig_mod": lambda m: getattr(m, "_orig_mod", None),
        "module": lambda m: getattr(m, "module", None)
    }
    
    # Try to unwrap model with each wrapper
    for _, getter in wrappers.items():
        unwrapped = getter(model)
        if unwrapped is not None:
            # Recursively unwrap the result
            return unwrap_module(unwrapped)
    
    # If no more wrappers, return model
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
    Save training state, create a separate folder for each step, and subfolders for each PFC head
    
    Args:
        output_dir: Base output directory
        backbone: Backbone network model
        pfc_modules: PFC module list
        lr_scheduler: Learning rate scheduler
        amp: Automatic mixed precision object
        global_step: Current global step count
        list_head_names: List of head names
        keep_num: Number of checkpoints to keep
    """
    # Create folder for current step
    step_dir = os.path.join(output_dir, f"{global_step:08d}")
    os.makedirs(step_dir, exist_ok=True)
    
    # Save backbone model and optimizer state (only on rank 0)
    if rank == 0:
        # Save backbone model (move to CPU)
        backbone_path = os.path.join(step_dir, "backbone.pt")
        backbone_state_dict = {k: v.cpu() for k, v in unwrap_module(backbone).state_dict().items()}
        torch.save(backbone_state_dict, backbone_path)
        
        # Save learning rate scheduler state
        scheduler_path = os.path.join(step_dir, "scheduler.pt")
        torch.save(lr_scheduler.state_dict(), scheduler_path)
        
        # Save AMP state (if available)
        if amp is not None:
            amp_path = os.path.join(step_dir, "amp.pt")
            torch.save(amp.state_dict(), amp_path)
        
        logging.info(f"Backbone, scheduler saved at step {global_step}")

    if isinstance(pfc_modules, list):
        # Each rank saves its own PFC module
        for head_id, (head_name, pfc) in enumerate(zip(list_head_names, pfc_modules)):
            if isinstance(pfc, list):
                for i in range(len(pfc)):
                    # print(pfc)
                    # print(len(pfc))
                    # new_pfc, pfc_type = pfc[i]
                    head_dir = os.path.join(step_dir, f"{head_name}_{i:02d}")
                    os.makedirs(head_dir, exist_ok=True)
                    # Save PFC model state (in head folder) - with name and move to CPU
                    pfc_path = os.path.join(head_dir, f"{head_name}_{rank:03d}.pt")
                    # pfc_state_dict, pfc_type = pfc
                    pfc_state_dict = {k: v.cpu() for k, v in pfc[i][0].state_dict().items()}  # Move to CPU
                    torch.save(pfc_state_dict, pfc_path)
            elif isinstance(pfc, torch.nn.Module):
                # Create separate folder for current PFC head
                head_dir = os.path.join(step_dir, head_name)
                os.makedirs(head_dir, exist_ok=True)
                
                # Save PFC model state (in head folder) - with name and move to CPU
                pfc_path = os.path.join(head_dir, f"{head_name}_{rank:03d}.pt")
                pfc_state_dict = {k: v.cpu() for k, v in pfc.state_dict().items()}  # Move to CPU
                torch.save(pfc_state_dict, pfc_path)

    elif isinstance(pfc_modules, dict):
        # Each rank saves its own PFC module
        for head_name, pfc in pfc_modules.items():
            if isinstance(pfc, list):
                for i in range(len(pfc)):
                    head_dir = os.path.join(step_dir, f"{head_name}_{i:02d}")
                    os.makedirs(head_dir, exist_ok=True)
                    # Save PFC model state (in head folder) - with name and move to CPU
                    pfc_path = os.path.join(head_dir, f"{head_name}_{rank:03d}.pt")
                    pfc_state_dict = {k: v.cpu() for k, v in pfc[i].state_dict().items()}
                    torch.save(pfc_state_dict, pfc_path)
            elif isinstance(pfc, torch.nn.Module):
                # Create separate folder for current PFC head
                head_dir = os.path.join(step_dir, head_name)
                os.makedirs(head_dir, exist_ok=True)
                
                # Save PFC model state (in head folder) - with name and move to CPU
                pfc_path = os.path.join(head_dir, f"{head_name}_{rank:03d}.pt")
                pfc_state_dict = {k: v.cpu() for k, v in pfc.state_dict().items()}
                torch.save(pfc_state_dict, pfc_path)
    else:
        raise ValueError("pfc_modules should be a list or a dict")

    # Clean old checkpoint folders
    if rank == 0:
        clean_old_checkpoints(output_dir, keep_num)
    
    logging.info(f"Rank {rank}: PFC modules saved at step {global_step}")


def clean_old_checkpoints(output_dir, keep_num=5):
    """
    Delete old checkpoint folders, keeping only the latest keep_num
    """
    # Get all checkpoint folders
    checkpoint_dirs = []
    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        # Only consider folders with 8-digit format
        if os.path.isdir(item_path) and item.isdigit() and len(item) == 8:
            checkpoint_dirs.append(item_path)
    
    # Sort by modification time (or can sort by folder name numerically)
    checkpoint_dirs.sort(key=lambda x: int(os.path.basename(x)))
    
    # If exceeding retention count, delete oldest
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
    Load training state from checkpoint folder at specified step
    
    Args:
        output_dir: Base output directory
        step: Step to load (if None, load latest step)
        backbone: Backbone network model
        pfc_modules: PFC module list
        lr_scheduler: Learning rate scheduler
        amp: Automatic mixed precision object
        list_head_names: List of head names
    
    Returns:
        dict: Contains restored global step information
    """
    # If step not specified, find the latest
    if step is None:
        # Find all checkpoint folders
        checkpoint_dirs = []
        for item in os.listdir(output_dir):
            item_path = os.path.join(output_dir, item)
            if os.path.isdir(item_path) and item.isdigit() and len(item) == 8:
                checkpoint_dirs.append(int(item))
        
        if not checkpoint_dirs:
            logging.warning(f"No checkpoint directories found in {output_dir}")
            return None
        
        step = max(checkpoint_dirs)
    
    # Build step folder path
    step_dir = os.path.join(output_dir, f"{step:08d}")
    if not os.path.isdir(step_dir):
        logging.warning(f"Checkpoint directory not found: {step_dir}")
        return None
    
    # Load backbone
    backbone_file = os.path.join(step_dir, "backbone.pt")
    if not os.path.exists(backbone_file):
        logging.warning(f"Backbone file not found: {backbone_file}")
        return None
    
    backbone_state = torch.load(backbone_file, )
    unwrap_module(backbone).load_state_dict(backbone_state)
    
    if isinstance(pfc_modules, list):
        # Load PFC modules
        for head_id, (head_name, pfc) in enumerate(zip(list_head_names, pfc_modules)):
            if isinstance(pfc, list):
                for i in range(len(pfc)):
                    head_dir = os.path.join(step_dir, f"{head_name}_{i:02d}")
                    # PFC file is in head folder
                    pfc_file = os.path.join(head_dir, f"{head_name}_{rank:03d}.pt")
                    if os.path.exists(pfc_file):
                        pfc_state = torch.load(pfc_file, )
                        pfc[i].load_state_dict(pfc_state)
                        logging.info(f"Rank {rank}: Loaded PFC weights for {head_name}")
                    else:
                        logging.warning(f"Rank {rank}: PFC file not found: {pfc_file}")
            elif isinstance(pfc, torch.nn.Module):
                # PFC file is in head folder
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
        # Load PFC modules
        for head_name, pfc in pfc_modules.items():
            # PFC file is in head folder
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

    # Load learning rate scheduler
    scheduler_file = os.path.join(step_dir, "scheduler.pt")
    if os.path.exists(scheduler_file):
        lr_scheduler.load_state_dict(torch.load(scheduler_file, ))
    else:
        logging.warning(f"Scheduler state file not found: {scheduler_file}")
    
    # Load AMP state
    if amp is not None:
        amp_file = os.path.join(step_dir, "amp.pt")
        if os.path.exists(amp_file):
            amp.load_state_dict(torch.load(amp_file, ))
        else:
            logging.warning(f"AMP state file not found: {amp_file}")
    
    return {
        'global_step': step
    }
