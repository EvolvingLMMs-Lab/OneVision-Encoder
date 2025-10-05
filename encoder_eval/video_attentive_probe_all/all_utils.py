import torch
from torch import nn
import torch.nn.functional as F

import torch.distributed as dist
import os
import numpy as np
import random
from collections import OrderedDict
import time
from collections import defaultdict
from collections import deque
import datetime


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
    
def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def setup_seed(seed, cuda_deterministic=True):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if cuda_deterministic:  # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:  # faster, less reproducible
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def load_state_dict(model,
                    state_dict,
                    prefix='',
                    ignore_missing="relative_position_index"):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(state_dict, prefix, local_metadata, True,
                                     missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(ignore_missing_keys) > 0:
        print(
            "Ignored weights of {} not initialized from pretrained model: {}".
            format(model.__class__.__name__, ignore_missing_keys))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))


def load_finetune_checkpoint(args, video_model):
    checkpoint = torch.load(args.finetune, map_location='cpu')
    print("Load ckpt from %s" % args.finetune)
    if args.model_name == 'internvideo_v1':
        if args.finetune:
            checkpoint = torch.load(args.finetune, map_location='cpu')
            if 'state_dict' in checkpoint.keys():
                checkpoint = checkpoint['state_dict']

            print("\nLoad ckpt from %s" % args.finetune)
            checkpoint_model = None
            for model_key in args.model_key.split('|'):
                if model_key in checkpoint:
                    checkpoint_model = checkpoint[model_key]
                    print("Load state_dict by model_key = %s" % model_key)
                    break
            if checkpoint_model is None:
                checkpoint_model = checkpoint
            state_dict = base_model.state_dict()
            for k in ['head.weight', 'head.bias']:
                if k in checkpoint_model and checkpoint_model[
                        k].shape != state_dict[k].shape:
                    if checkpoint_model[k].shape[0] == 710:
                        if args.data_set == 'k400':
                            print('Convert to k400 class head')
                            checkpoint_model[k] = checkpoint_model[k][:400]
                        elif args.data_set == 'k600':
                            print('Convert to k600 class head')
                            label_map_path = '/mnt/petrelfs/huangbingkun/data/mix_kinetics/label_mixto600.json'
                            label_map = json.load(open(label_map_path))
                            checkpoint_model[k] = checkpoint_model[k][label_map]
                        elif args.data_set == 'k700':
                            print('Convert to k700 class head')
                            label_map_path = '/mnt/petrelfs/huangbingkun/data/mix_kinetics/label_mixto700.json'
                            label_map = json.load(open(label_map_path))
                            checkpoint_model[k] = checkpoint_model[k][label_map]
                        else:
                            print(f"Removing key {k} from pretrained checkpoint")
                            del checkpoint_model[k]
                    else:
                        print(f"Removing key {k} from pretrained checkpoint")
                        del checkpoint_model[k]

            all_keys = list(checkpoint_model.keys())
            new_dict = OrderedDict()
            for key in all_keys:
                if key.startswith('backbone.'):
                    new_dict[key[9:]] = checkpoint_model[key]
                elif key.startswith('encoder.'):
                    new_dict[key[8:]] = checkpoint_model[key]
                else:
                    new_dict[key] = checkpoint_model[key]
            checkpoint_model = new_dict

            # interpolate position embedding
            if 'pos_embed' in checkpoint_model:
                pos_embed_checkpoint = checkpoint_model['pos_embed']
                embedding_size = pos_embed_checkpoint.shape[-1]  # channel dim
                num_patches = base_model.patch_embed.num_patches  #
                num_extra_tokens = base_model.pos_embed.shape[-2] - num_patches  # 0/1

                # height (== width) for the checkpoint position embedding
                orig_size = int(
                    ((pos_embed_checkpoint.shape[-2] - num_extra_tokens) //
                    (args.num_frames // base_model.patch_embed.tubelet_size))**0.5)
                # height (== width) for the new position embedding
                new_size = int(
                    (num_patches //
                    (args.num_frames // base_model.patch_embed.tubelet_size))**0.5)
                # class_token and dist_token are kept unchanged
                if orig_size != new_size:
                    print("Position interpolate from %dx%d to %dx%d" %
                        (orig_size, orig_size, new_size, new_size))
                    extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                    # only the position tokens are interpolated
                    pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                    # B, L, C -> BT, H, W, C -> BT, C, H, W
                    pos_tokens = pos_tokens.reshape(
                        -1, args.num_frames // base_model.patch_embed.tubelet_size,
                        orig_size, orig_size, embedding_size)
                    pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size,
                                                    embedding_size).permute(
                                                        0, 3, 1, 2)
                    pos_tokens = torch.nn.functional.interpolate(
                        pos_tokens,
                        size=(new_size, new_size),
                        mode='bicubic',
                        align_corners=False)
                    # BT, C, H, W -> BT, H, W, C ->  B, T, H, W, C
                    pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(
                        -1, args.num_frames // base_model.patch_embed.tubelet_size,
                        new_size, new_size, embedding_size)
                    pos_tokens = pos_tokens.flatten(1, 3)  # B, L, C
                    new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                    checkpoint_model['pos_embed'] = new_pos_embed
            model_dict = base_model.state_dict()
            
            load_state_dict(base_model,
                                checkpoint_model,
                                prefix=args.model_prefix)
    if args.model_name == 'ijepa':
        pretrained_dict = checkpoint['encoder']
        for k, v in pretrained_dict.items():
            video_model.state_dict()[k[len("module."):]].copy_(v)
        # pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}
        # pretrained_dict = {k.replace('backbone.', ''): v for k, v in pretrained_dict.items()}
        # video_model.load_state_dict(pretrained_dict)
        # video_model.load_state_dict(pretrained_dict)
        # logger.info(f'loaded pretrained encoder from epoch {epoch} with msg: {msg}')

        # -- loading predictor
        # pretrained_dict = checkpoint['predictor']
        # msg = predictor.load_state_dict(pretrained_dict)
        # logger.info(f'loaded pretrained encoder from epoch {epoch} with msg: {msg}')
        # print('target',checkpoint['encoder'].keys())
        # print('encoder',checkpoint['encoder'].keys())
        # pretrained_dict = checkpoint['target_encoder']
        # clean_state = OrderedDict()
        # for k, v in pretrained_dict.items():
        #     new_k = k.replace('module.', '') if k.startswith('module.') else k
        #     clean_state[new_k] = v
        # # print("\n", clean_state.keys())
        # checkpoint_model = clean_state
        # video_model.load_state_dict(checkpoint_model, strict=False)
        return video_model
    if args.model_name == "rope" or args.model_name == "mlcd_base":
        state_dict = torch.load(args.finetune, "cpu")
        state_dict = {
            k.replace("_orig_mod.", ""): v for k, v in state_dict.items()
        }
        video_model.load_state_dict(state_dict, strict=True)
        return video_model
    if args.model_name == "vjepa":
    #     def load_pretrained(
    # encoder,
    # pretrained,
    # checkpoint_key='target_encoder'
    # ):
        try:
            pretrained_dict = checkpoint['target_encoder']
        except Exception:
            pretrained_dict = checkpoint['encoder']

        pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}
        pretrained_dict = {k.replace('backbone.', ''): v for k, v in pretrained_dict.items()}
        for k, v in video_model.state_dict().items():
            if pretrained_dict[k].shape != v.shape:
                pretrained_dict[k] = v
        msg = video_model.load_state_dict(pretrained_dict, strict=False)
        del checkpoint
        return video_model
    checkpoint_model = None
    for model_key in args.model_key.split('|'):
        if model_key in checkpoint:
            checkpoint_model = checkpoint[model_key]
            print("Load state_dict by model_key = %s" % model_key)
            break
    if checkpoint_model is None:
        checkpoint_model = checkpoint
    
    if args.model_name == "mlcd_video":
        backbone_file = args.finetune
        backbone_state = torch.load(backbone_file, )
        unwrap_module(video_model).load_state_dict(backbone_state)
        checkpoint_model = video_model

    if args.model_name == 'videomae_v1' or args.model_name == 'videomae_v2':
        # videomae check
        for old_key in list(checkpoint_model.keys()):
            if old_key.startswith('_orig_mod.'):
                print("if old_key.startswith('_orig_mod.'):")
                new_key = old_key[10:]
                checkpoint_model[new_key] = checkpoint_model.pop(old_key)

    
    all_keys = list(checkpoint_model.keys())
    new_dict = OrderedDict()
    for key in all_keys:
        if key.startswith('backbone.'):
            new_dict[key[9:]] = checkpoint_model[key]
        elif key.startswith('encoder.'):
            new_dict[key[8:]] = checkpoint_model[key]
        else:
            new_dict[key] = checkpoint_model[key]
    checkpoint_model = new_dict

    if args.model_name == 'dino_v2':
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            if "mlp.w12" in k:
                new_k = k.replace("mlp.w3", "mlp.fc1")
            elif "mlp.w3" in k:
                new_k = k.replace("mlp.w12", "mlp.fc2")
            elif "ls1" in k or "ls2" in k:
                continue  # Drop LayerScale
            else:
                new_k = k

            # Optional: add blocks.0. prefix if needed
            if new_k.startswith("blocks."):
                parts = new_k.split(".")
                new_k = f"blocks.0.{'.'.join(parts[1:])}"
            
            new_state_dict[new_k] = v
        checkpoint_model = new_state_dict

    if args.model_name == 'viclip':
        all_keys = list(checkpoint_model.keys())
        vision_dict = OrderedDict()
        tex_dict = OrderedDict()
        for key in all_keys:
            if key.startswith('vision_encoder.'):
                vision_dict[key[15:]] = checkpoint_model[key]
            elif key.startswith('text_encoder.'):
                tex_dict[key[13:]] = checkpoint_model[key]
            else:
                continue
        checkpoint_model = vision_dict

    
    if 'pos_embed' in checkpoint_model:
        print("'pos_embed' in checkpoint_model")



    if args.model_name == 'viclip':
        def inflate_weight(weight_2d, time_dim, center=True):
            print('Init center: {center}')
            if weight_2d.ndim != 4:
                print(f"⚠️  跳过非4维权重：{weight_2d.shape}")
                return weight_2d  # 不处理
            if center:
                weight_3d = torch.zeros(*weight_2d.shape)
                weight_3d = weight_3d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
                middle_idx = time_dim // 2
                weight_3d[:, :, middle_idx, :, :] = weight_2d
            else:
                weight_3d = weight_2d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
                weight_3d = weight_3d / time_dim
            return weight_3d

        state_dict = checkpoint_model
        state_dict_3d = video_model.state_dict()
        for k in state_dict.keys():
            if k in state_dict_3d.keys() and state_dict[k].shape != state_dict_3d[k].shape:
                if len(state_dict_3d[k].shape) <= 2:
                    print('Ignore: {k}')
                    continue
                print('Inflate: {k}, {state_dict[k].shape} => {state_dict_3d[k].shape}')
                time_dim = state_dict_3d[k].shape[2]
                state_dict[k] = inflate_weight(state_dict[k], time_dim, center=True)

        pos_embed_checkpoint = state_dict['positional_embedding']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = (args.input_size // args.patch_size) ** 2
        orig_size = int((pos_embed_checkpoint.shape[-2] - 1) ** 0.5)
        new_size = int(num_patches ** 0.5)
        if orig_size != new_size:
            print('Pos_emb from {orig_size} to {new_size}')
            extra_tokens = pos_embed_checkpoint[:1]
            pos_tokens = pos_embed_checkpoint[1:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(0, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=0)
            state_dict['positional_embedding'] = new_pos_embed
        checkpoint_model = state_dict

    load_state_dict(video_model, checkpoint_model)
    
    return video_model


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total],
                         dtype=torch.float64,
                         device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def min(self):
        return min(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            min=self.min,
            value=self.value)


class MetricLogger(object):

    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

        self.step_time_start = 0
        self.init = False
        self.tic = 0

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None, 
                        world_size=None, batch_size=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f} ({min:.4f} -- {max:.4f})')
        data_time = SmoothedValue(fmt='{avg:.4f} ({min:.4f} -- {max:.4f})')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header, '[{0' + space_fmt + '}/{1}]', 
            'eta: {eta}', 
            '{meters}',
            'time: {time}', 
            'data: {data}',
            'max mem: {memory:.0f}']

        if (world_size is not None) and (batch_size is not None):
            # print("hhh")
            log_msg.append('video/s/gpu: {qps_v1}')
            log_msg.append('video/s: {qps_v2}')
        
        # print("???")
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        # print("?????")
        # import pdb; pdb.set_trace()
        for obj in iterable:
            # print("???????")
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                if self.init:
                    if (world_size is not None) and (batch_size is not None):
                        try:
                            speed = print_freq * batch_size / (time.time() - self.tic)
                            self.tic = time.time()
                            speed_total = speed * world_size
                        except ZeroDivisionError:
                            speed = float("inf")
                            speed_total = float("inf")

                    eta_seconds = iter_time.global_avg * (len(iterable) - i)
                    eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

                    if (world_size is not None) and (batch_size is not None):
                        speed = "{:.4f}".format(speed)
                        speed_total = "{:.4f}".format(speed_total)

                        print(log_msg.format(i, len(iterable), eta=eta_string, meters=str(self),
                            time=str(iter_time), data=str(data_time), memory=torch.cuda.max_memory_allocated() / MB,
                            qps_v1=str(speed), qps_v2=str(speed_total)))
                    else:
                        print(
                            log_msg.format(i, len(iterable), eta=eta_string, meters=str(self),
                                time=str(iter_time), data=str(data_time), memory=torch.cuda.max_memory_allocated() / MB))

                else:
                    self.init = True
                    self.tic = time.time()
                    self.step_time_start = time.time()
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        # print('{} Total time: {} ({:.4f} s / it)'.format(header, total_time_str, total_time / len(iterable)))