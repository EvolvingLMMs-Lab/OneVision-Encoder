import argparse, os,tqdm, json,random, numpy as np, torch
from pathlib import Path
import torch.nn.functional as F
from collections import OrderedDict
import torch.multiprocessing as mp
from torchvision import transforms
from decord import VideoReader, cpu  # GPU 解码: `gpu(0)`
import warnings
import threading
from queue import Queue
warnings.filterwarnings('ignore')
# NOTE: 下面这行用来注册自定义 ViT / InternVideo2 模型
import video_models  # noqa: F401 pylint: disable=unused-import
# _IMAGENET_MEAN = (0.485, 0.456, 0.406)
# _IMAGENET_STD  = (0.229, 0.224, 0.225)

# ---------- 1. Transform 与工具函数 ---------- #
# Constants
CLIP_LENGTH = 16  # Number of frames per clip

def to_normalized_float_tensor(vid: torch.Tensor) -> torch.Tensor:
    """(T, H, W, C[RGB]) uint8 -> (C, T, H, W) float32 in [0,1]."""
    vid = vid.permute(3, 0, 1, 2).to(torch.float32) / 255
    # if normalize:
    #     mean_t = torch.tensor(mean, device=vid.device).view(-1, 1, 1, 1)
    #     std_t  = torch.tensor(std,  device=vid.device).view(-1, 1, 1, 1)
    #     vid = (vid - mean_t) / std_t
    return vid
    
def resize(vid: torch.Tensor, size):
    """仅在空间维度做双线性 Resize."""
    scale = None
    if isinstance(size, int):
        scale = float(size) / min(vid.shape[-2:])
        size = None
    return torch.nn.functional.interpolate(
        vid, size=size, scale_factor=scale,
        
        mode='bilinear', align_corners=False
    )

class ToFloatTensorInZeroOne:  # 与官方脚本保持同名
    def __call__(self, vid): 
        return to_normalized_float_tensor(vid)

class Resize:
    def __init__(self, size): self.size = size
    def __call__(self, vid):  return resize(vid, self.size)

# ---------- 2. CLI ---------- #
def get_args():
    p = argparse.ArgumentParser("Extract TAD features with LLaVA-ViT")
    p.add_argument('--data_set',  default='charades',
                   choices=['thumos', 'fineaction', "charades", "hacs"])
    p.add_argument('--data_path', default="/video_vit/feilong/TAD-Dataset/charades/raw_data/video",)
    p.add_argument('--save_path', default="/video_vit/feilong/TAD-Dataset/charades/features/llavavitllavavit_test", )
    p.add_argument('--model_name',     default="llavavit",)
    p.add_argument('--model_type', default='llavavit')
    p.add_argument('--ckpt_path', default="/video_vit/xiangan/checkpoint_llava_vit/2025_11_22_new_l14_continue_128gpus_how_to_100m_448px_224px/00148000/backbone.pt",)
    # p.add_argument('--device',    default='cuda:0')
    p.add_argument('--world_size', default=8)

    p.add_argument("--anno_file", default="/video_vit/OpenTAD/data/data/charades/annotations/charades.json" ,type=str, help="path to annotation")
    # p.add_argument("--data_dir", default="/vlm/monash/feilong/OpenTAD-main/data/charades/raw_data/Charades_v1_480_30fps" , type=str, help="path to data folder")
    p.add_argument("--prefix", type=str, default="")
    p.add_argument('--model_key', default='model|module', type=str)
    p.add_argument("--suffix", type=str, default="")
    p.add_argument("--ext", type=str, default="npy")
    return p.parse_args()

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

# def load_finetune_checkpoint(args, video_model):
    if args.model_type == 'internvideo_v1':
        checkpoint = torch.load(args.ckpt_path, map_location='cpu')
        if 'state_dict' in checkpoint.keys():
            checkpoint = checkpoint['state_dict']

        print("\nLoad ckpt from %s" % args.ckpt_path)
        checkpoint_model = None
        for model_key in args.model_key.split('|'):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print("Load state_dict by model_key = %s" % model_key)
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint
        state_dict = video_model.state_dict()
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
            num_patches = video_model.patch_embed.num_patches  #
            num_extra_tokens = video_model.pos_embed.shape[-2] - num_patches  # 0/1

            # height (== width) for the checkpoint position embedding
            orig_size = int(
                ((pos_embed_checkpoint.shape[-2] - num_extra_tokens) //
                (args.num_frames // video_model.patch_embed.tubelet_size))**0.5)
            # height (== width) for the new position embedding
            new_size = int(
                (num_patches //
                (args.num_frames // video_model.patch_embed.tubelet_size))**0.5)
            # class_token and dist_token are kept unchanged
            if orig_size != new_size:
                print("Position interpolate from %dx%d to %dx%d" %
                    (orig_size, orig_size, new_size, new_size))
                extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                # only the position tokens are interpolated
                pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                # B, L, C -> BT, H, W, C -> BT, C, H, W
                pos_tokens = pos_tokens.reshape(
                    -1, args.num_frames // video_model.patch_embed.tubelet_size,
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
                    -1, args.num_frames // video_model.patch_embed.tubelet_size,
                    new_size, new_size, embedding_size)
                pos_tokens = pos_tokens.flatten(1, 3)  # B, L, C
                new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                checkpoint_model['pos_embed'] = new_pos_embed
        model_dict = video_model.state_dict()
        
        load_state_dict(video_model,
                            checkpoint_model)
    elif args.model_type == 'videomae':
        checkpoint_model = torch.load(args.ckpt_path, map_location='cpu')
        print("Load ckpt from %s" % args.ckpt_path)
        for key in ['state_dict', 'model', 'module']:      # 兼容各种保存方式
            if key in checkpoint_model:
                checkpoint_model = checkpoint_model[key]
                break
        # print(checkpoint_model.keys())
        checkpoint_model = {
        k.replace("module.", ""): v for k, v in checkpoint_model.items()
    } 
        load_state_dict(video_model, checkpoint_model)
        # video_model.load_state_dict(checkpoint_model, strict=True)
    
    return video_model
 
# ---------- 3. 数据集特定滑窗 ---------- #
def get_start_idx_range(name):
    if name.upper() == 'THUMOS':
        return lambda n_frames: range(0, n_frames - 15, 4)   # 16×4
    if name.upper() == 'FINEACTION':
        return lambda n_frames: range(0, n_frames - 15, 16)  # 16×16
    if name.upper() == 'HACS':
        return lambda n_frames: range(0, n_frames - 15, 16)  # 16×16    
    if name.upper() == 'CHARADES':                           # ★ 新增 ★
        # 30 fps, clip_len=16, stride=4, interval=1
        return lambda n_frames: range(0, n_frames - 15, 4)
    raise NotImplementedError(name)

def safe_get_batch(path, indices):
    try:
        vr = VideoReader(path, ctx=cpu(0), num_threads=1)  # 单线程
        return vr.get_batch(indices)  # (N, H, W, 3), uint8
    except Exception as e:
        # Fallback: OpenCV
        import cv2
        cap = cv2.VideoCapture(path)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Fallback failed on frame {idx} for {path}: {e}")
            frame = frame[:, :, ::-1]  # BGR->RGB
            frames.append(frame)
        cap.release()
        arr = np.stack(frames, axis=0)
        return torch.from_numpy(arr)

def build_model(weight_path: str, model_name: str, model_type: str):

    if model_type == 'dinov3':
        from transformers import AutoModel
        import torch
        processor = None
        model = AutoModel.from_pretrained(
            weight_path,
            trust_remote_code=True,
            local_files_only=True,   # 只用本地权重，避免联网
            torch_dtype=torch.float32
        )
    elif model_type == 'llavavit':
        from timm.models import create_model
        import model_factory
        # from model_factory.vit_preview_v0 import llava_vit_base_ln
        import torch
        model = create_model("llava_vit_base_ln", pretrained=False)
        # model = llava_vit_base_ln()
        state_dict = torch.load(weight_path, map_location="cpu")
        state_dict = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
        processor = None

    return model, processor

def generate_missing_txt(args):
    missing_list = []

    anno_database = json.load(open(args.anno_file))["database"]
    for video_name in tqdm.tqdm(list(anno_database.keys())):
        file_path = os.path.join(args.save_path, f"{args.prefix}{video_name}{args.suffix}.{args.ext}")

        if not os.path.exists(file_path):
            missing_list.append(video_name)

    saved_path = os.path.join(f"{args.save_path}", "missing_files.txt")
    with open(saved_path, "w") as f:
        f.write("\n".join(missing_list))

    print(
        f"Total {len(anno_database.keys())} videos/features in dataset, "
        f"missing {len(missing_list)} videos/features."
    )
    print(f"Missing file has been saved in {saved_path}")

def extract_by_dinov3(clip_batch, model, device):
    """逐帧前向，支持多种返回格式；最后对时间维做平均池化。"""
    stacked = torch.stack(clip_batch, dim=0).to(device, non_blocking=True)  # (B, C, T, H, W) with non-blocking
    outputs = []
    for frame_idx in range(stacked.shape[2]):
        frame = stacked[:, :, frame_idx, :, :]  # (B, C, H, W)
        with torch.no_grad():
            out = model(frame)
        # 兼容不同模型输出
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            feat = out.pooler_output
        elif hasattr(out, "last_hidden_state"):
            feat = out.last_hidden_state.mean(1)  # CLS 不一定有，取 token 平均
        elif isinstance(out, (tuple, list)) and len(out) > 0 and torch.is_tensor(out[0]):
            feat = out[0]
        else:
            raise RuntimeError("Unexpected DINOv3 model output; cannot extract features.")
        outputs.append(feat)
    return torch.stack(outputs, dim=1).mean(1)  # (B, D)

def interpolate_frame_indices(frame_indices: torch.Tensor, total_frames: torch.Tensor, target_frames: int = 64) -> torch.Tensor:
    """
    将帧索引从原始视频帧数插值到目标帧数

    Args:
        frame_indices: [B, seq_len] 原始帧索引
        total_frames: [B] 每个视频的总帧数
        target_frames: 目标帧数 (默认 64)

    Returns:
        interpolated_indices: [B, seq_len] 插值后的帧索引，范围在 [0, target_frames-1]
    """
    bs, seq_len = frame_indices.shape
    device = frame_indices.device

    # 将 total_frames 转换为浮点数以进行插值计算
    total_frames_float = total_frames.float().view(bs, 1)  # [B, 1]
    frame_indices_float = frame_indices.float()  # [B, seq_len]

    # 插值公式: new_idx = (old_idx / (total_frames - 1)) * (target_frames - 1)
    # 处理 total_frames = 1 的情况files.trimTrailingWhitespace: truefiles.trimTrailingWhitespace: tru
    total_frames_safe = torch.clamp(total_frames_float - 1, min=1.0)
    interpolated_indices = (frame_indices_float / total_frames_safe) * (target_frames - 1)

    # 四舍五入并转换为整数
    interpolated_indices = torch.round(interpolated_indices).long()

    # 确保索引在有效范围内
    interpolated_indices = torch.clamp(interpolated_indices, 0, target_frames - 1)

    return interpolated_indices

def extract_by_llavavit(
    videos,
    model,
    device,
    frame_indices=None,
    total_frames=None,
):
    """
    使用 llavavit 提取视频特征。
    输入:
        videos: (B, C, T, H, W)
        frame_indices: (B, seq_len)，可选
        total_frames: (B,)，可选
        args.frames_token_num: 每帧 token 数
        args.target_frames: 目标帧数（例如 64）
    输出:
        (B, D) 的视频级特征（来自 enc_out['head_output']）
    """
    # 使用 non_blocking 传输加速数据移动
    videos = torch.stack(videos, dim=0).to(device, non_blocking=True)
    bs, C, T, H, W = videos.shape
    # print(bs, C, T, H, W) # 20 3 16 224 224
    frame_tokens = 196
    target_frames = 64

    # # ========================
    # # 1. 构造 padded_videos 和 visible_index
    # # ========================
    # if frame_indices is not None and total_frames is not None:
    #     # ---- 有外部传入的 frame_indices 情况：用你原来的插值逻辑 ----
    #     interpolated_indices = interpolate_frame_indices(
    #         frame_indices,
    #         total_frames.view(-1),
    #         target_frames
    #     )  # [B, seq_len]

    #     padded_videos = torch.zeros(
    #         bs, C, target_frames, H, W,
    #         device=device,
    #         dtype=videos.dtype,
    #     )
    #     seq_len = frame_indices.shape[1]
    #     frame_idx_expanded = (
    #         interpolated_indices.view(bs, 1, seq_len, 1, 1)
    #         .expand(bs, C, seq_len, H, W)
    #     )
    #     padded_videos.scatter_(dim=2, index=frame_idx_expanded, src=videos)

    #     per = torch.arange(frame_tokens, device=device)
    #     visible_index = (
    #         interpolated_indices.unsqueeze(-1) * frame_tokens + per
    #     ).reshape(bs, -1)
    #     visible_index = visible_index.clamp_max(target_frames * frame_tokens - 1)

    # else:
    #     # ---- 没有 frame_indices / total_frames：在 T 维均匀取 target_frames 帧 ----
    #     # linspace 在 [0, T-1] 上均匀取 target_frames 个点，然后四舍五入到最近的帧索引
    #     # 即使 T < target_frames 也没关系，只是会有重复索引
    #     base = torch.linspace(
    #         0, max(T - 1, 0),
    #         steps=target_frames,
    #         device=device
    #     )  # (target_frames,)
    #     sampled_indices = base.round().long()  # (target_frames,)
    #     # 扩展到 batch 维度：(B, target_frames)
    #     sampled_indices = sampled_indices.unsqueeze(0).expand(bs, target_frames)

    #     # 构造 padded_videos
    #     padded_videos = torch.zeros(
    #         bs, C, target_frames, H, W,
    #         device=device,
    #         dtype=videos.dtype,
    #     )

    #     # scatter 用的 index：(B, C, target_frames, H, W)
    #     frame_idx_expanded = (
    #         sampled_indices.view(bs, 1, target_frames, 1, 1)
    #         .expand(bs, C, target_frames, H, W)
    #     )
    #     # 源视频只到 T，所以从原 videos 按 sampled_indices 取
    #     # 先 gather 出 (B, C, target_frames, H, W) 的对齐帧
    #     gathered = videos.gather(
    #         dim=2,
    #         index=frame_idx_expanded.clamp_max(T - 1)
    #     )
    #     # 直接赋值即可（不需要 scatter 一次）
    #     padded_videos = gathered

    #     # 计算 visible_index
    #     per = torch.arange(frame_tokens, device=device)  # (frame_tokens,)
    #     visible_index = (
    #         sampled_indices.unsqueeze(-1) * frame_tokens + per
    #     ).reshape(bs, -1)
    #     visible_index = visible_index.clamp_max(target_frames * frame_tokens - 1)

    # ========================
    # 2. 前向：用 head_output 而不是 visible_embeddings
    # ========================
    # with torch.no_grad():
    #     enc_out = model(padded_videos, visible_index, mask_ratio=None)

    with torch.no_grad():
        enc_out = model(videos, None, None)

    # 优先用 head_output
    if "head_output" in enc_out:
        outputs = enc_out["head_output"]
    else:
        # 兜底：如果没有 head_output，就退回原来的 visible_embeddings
        if "visible_embeddings" not in enc_out:
            raise RuntimeError(
                f"Unexpected llavavit enc_out keys: {list(enc_out.keys())}"
            )
        outputs = enc_out["visible_embeddings"]

    # ========================
    # 3. 池化成 (B, D)
    # ========================
    if outputs.dim() == 3:
        feats = outputs.mean(1)  # (B, D)
    else:
        feats = outputs  # 已经是 (B, D)

    return feats



import time

class ClipPrefetcher:
    """
    预取clip数据，在后台线程中加载和预处理数据
    实现数据加载与GPU计算的流水线并行
    """
    def __init__(self, vr, clip_starts, transform, prefetch_size=4):
        self.vr = vr
        self.clip_starts = clip_starts
        self.transform = transform
        self.queue = Queue(maxsize=prefetch_size)
        self.thread = None
        self.stop_flag = threading.Event()
        
    def _load_worker(self):
        """后台线程：持续加载和预处理clip"""
        try:
            for st in self.clip_starts:
                if self.stop_flag.is_set():
                    break
                # 读取和预处理
                frame_nd = self.vr.get_batch(np.arange(st, st + CLIP_LENGTH)).asnumpy()
                clip = self.transform(torch.from_numpy(frame_nd))
                # 放入队列，如果队列满则阻塞
                self.queue.put((st, clip))
        except Exception as e:
            self.queue.put(('ERROR', e))
        finally:
            self.queue.put(('DONE', None))
    
    def start(self):
        """启动预取线程"""
        self.thread = threading.Thread(target=self._load_worker, daemon=True)
        self.thread.start()
    
    def get(self):
        """获取下一个预处理好的clip"""
        return self.queue.get()
    
    def stop(self):
        """停止预取"""
        self.stop_flag.set()
        if self.thread:
            self.thread.join()

@torch.no_grad()
def extract_feature(rank, world_size, args):
    # 绑定 GPU
    local_rank = rank
    gpu_num = local_rank
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    # Enable TF32 for better performance on Ampere GPUs (A100)
    if torch.cuda.is_available() and torch.cuda.get_device_capability(device)[0] >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"[R{rank}] Enabled TF32 for faster matrix operations on Ampere GPU")

    model, processor = build_model(args.ckpt_path, args.model_name, args.model_type)
    Path(args.save_path).mkdir(parents=True, exist_ok=True)
    get_clip_range = get_start_idx_range(args.data_set)

    transform = transforms.Compose([
        ToFloatTensorInZeroOne(),
        Resize((224, 224)),
    ])

    model = model.to(device)
    model.eval()
    
    # Enable mixed precision for faster inference
    use_amp = torch.cuda.is_available() and torch.cuda.get_device_capability(device)[0] >= 7
    
    # Try to compile model for optimization (PyTorch 2.0+)
    if hasattr(torch, 'compile'):
        try:
            model = torch.compile(model, mode='max-autotune')
            print(f"[R{rank}] Model compiled with torch.compile for faster inference")
        except Exception as e:
            print(f"[R{rank}] torch.compile failed: {e}, continuing without compilation")

    # 所有视频按固定顺序分片给不同 rank
    vid_list = sorted(os.listdir(args.data_path))
    vid_list = vid_list[rank::world_size]

    # Increase batch size significantly for better GPU utilization
    BATCH_CLIPS = 64  # Increased from 20 to 64
    NUM_CPU = 64  # Increased to match available CPU threads

    # 全局统计（看整个进程级别的占比）
    total_read_time = 0.0  # 视频初始化时间
    total_infer_time = 0.0
    total_save_time = 0.0
    total_videos = 0

    for idx, vid_name in enumerate(vid_list):
        t_vid_start = time.perf_counter()

        out_npy = os.path.join(args.save_path, f'{Path(vid_name).stem}.npy')
        if os.path.isfile(out_npy):
            print(f"[R{rank}] ✔ 已有特征，跳过 {vid_name}")
            continue

        video_path = os.path.join(args.data_path, vid_name)
        print(f"[R{rank}] 开始处理视频: {video_path}")

        # 每个视频内部的统计
        read_time = 0.0  # 视频初始化时间
        infer_time = 0.0
        save_time = 0.0
        num_clips = 0

        try:
            # 打开视频计时 - 增加线程数以加速解码
            t0 = time.perf_counter()
            vr = VideoReader(video_path, ctx=cpu(0), num_threads=NUM_CPU)  # 使用更多线程
            clip_starts = list(get_clip_range(len(vr)))  # 只算一次
            t1 = time.perf_counter()
            read_time += (t1 - t0)

            feat_bank = []
            clip_batch = []
            
            # 使用预取器实现流水线并行
            # prefetch_size=8 表示最多预先准备8个clip在队列中
            prefetcher = ClipPrefetcher(vr, clip_starts, transform, prefetch_size=8)
            prefetcher.start()

            while True:
                # 从预取器获取clip - 时间统计已在预取器内部完成
                result = prefetcher.get()
                
                if result[0] == 'DONE':
                    break
                elif result[0] == 'ERROR':
                    raise result[1]
                
                st, clip = result
                # Note: read_time and transform_time are now overlapped with GPU computation
                # The timing here represents only the queue wait time
                
                clip_batch.append(clip)
                num_clips += 1

                # batch infer - 使用混合精度加速
                if len(clip_batch) == BATCH_CLIPS:
                    torch.cuda.synchronize(device)
                    t0 = time.perf_counter()
                    
                    if use_amp:
                        with torch.cuda.amp.autocast(dtype=torch.float16):
                            if args.model_type == 'llavavit':
                                out = extract_by_llavavit(clip_batch, model, device, None, None)
                            elif args.model_type == 'dinov3':
                                out = extract_by_dinov3(clip_batch, model, device)
                    else:
                        if args.model_type == 'llavavit':
                            out = extract_by_llavavit(clip_batch, model, device, None, None)
                        elif args.model_type == 'dinov3':
                            out = extract_by_dinov3(clip_batch, model, device)
                    
                    torch.cuda.synchronize(device)
                    t1 = time.perf_counter()
                    infer_time += (t1 - t0)
                    feat_bank.append(out.cpu().numpy())
                    clip_batch = []

            # 停止预取器
            prefetcher.stop()
            
            # 最后一批 infer - 使用混合精度加速
            if len(clip_batch) > 0:
                torch.cuda.synchronize(device)
                t0 = time.perf_counter()

                if use_amp:
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        if args.model_type == 'dinov3':
                            out = extract_by_dinov3(clip_batch, model, device)
                        elif args.model_type == 'llavavit':
                            out = extract_by_llavavit(clip_batch, model, device, None, None)
                else:
                    if args.model_type == 'dinov3':
                        out = extract_by_dinov3(clip_batch, model, device)
                    elif args.model_type == 'llavavit':
                        out = extract_by_llavavit(clip_batch, model, device, None, None)

                torch.cuda.synchronize(device)
                t1 = time.perf_counter()
                infer_time += (t1 - t0)

                feat_bank.append(out.cpu().numpy())
                clip_batch = []

            # ====== 你的保存+打印逻辑（加上计时） ======
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()

            features = np.vstack(feat_bank)
            np.save(out_npy, features)

            t1 = time.perf_counter()
            save_time += (t1 - t0)

            print(f"GPU:{gpu_num}:[{idx+1}/{len(vid_list)}] {vid_name} -> {out_npy}")

            t_vid_end = time.perf_counter()
            total_videos += 1

            # 全局累加
            total_read_time      += read_time
            total_infer_time     += infer_time
            total_save_time      += save_time

            # 打印当前视频耗时分布
            print(
                f"[R{rank}] {vid_name} 耗时 {t_vid_end - t_vid_start:.2f}s | "
                f"初始化 {read_time:.2f}s | "
                f"前向 {infer_time:.2f}s | 保存 {save_time:.2f}s | clips={num_clips}"
                f" (注: 数据加载和预处理已与GPU计算流水线并行)"
            )

        except Exception as e:
            print(f"[R{rank}] raise error: {e}, the error video is {vid_name}")
            continue

    # ====== 最后打印本 rank 的总统计 ======
    if total_videos > 0:
        print(
            f"[R{rank}] 总计处理 {total_videos} 个视频 | "
            f"初始化 {total_read_time:.2f}s | "
            f"前向 {total_infer_time:.2f}s | 保存 {total_save_time:.2f}s"
            f" (注: 数据加载和预处理时间已与GPU计算重叠)"
        )


if __name__ == '__main__':
    args = get_args()
    world_size = int(args.world_size)
    mp.spawn(extract_feature, nprocs=world_size, args=(world_size, args), join=True)
    generate_missing_txt(args)
