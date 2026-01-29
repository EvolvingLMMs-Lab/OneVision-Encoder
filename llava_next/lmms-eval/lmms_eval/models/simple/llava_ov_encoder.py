import torch

torch.backends.cuda.matmul.allow_tf32 = True


import copy
import warnings
from datetime import timedelta
from typing import List, Optional, Tuple, Union

from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from packaging import version
from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.utils import stop_sequences_criteria

warnings.filterwarnings("ignore")

from loguru import logger as eval_logger

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import (
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model
from PIL import Image


# inference implementation for attention, can be "sdpa", "eager", "flash_attention_2". Seems FA2 is not effective during inference: https://discuss.huggingface.co/t/flash-attention-has-no-effect-on-inference/73453/5
# if is_flash_attn_2_available:
#     best_fit_attn_implementation = "flash_attention_2" # flash_attn has a bug that says: ERROR Error query and key must have the same dtype in generating

if version.parse(torch.__version__) >= version.parse("2.1.2"):
    best_fit_attn_implementation = "sdpa"
else:
    best_fit_attn_implementation = "eager"

import os
from typing import List, Dict, Any
from pathlib import Path
import cv2

def build_patch_positions(num_frames: int = 8, total_frames: int = 64, h: int = 36, w: int = 36) -> torch.Tensor:
    """
    Build patch positions for video frames uniformly sampled from total_frames.
    
    Args:
        num_frames: Number of frames to sample (default: 8)
        total_frames: Total number of frames in the original video (default: 64)
        h: Height in patches (default: 36)
        w: Width in patches (default: 36)
    
    Returns:
        patch_positions: [1, seq_len, 3] tensor with [t, h, w] positions for each patch
                        where seq_len = num_frames * h * w
    """
    # Uniformly sample frame indices: e.g., for 8 frames from 64 -> [0, 9, 18, 27, 36, 45, 54, 63]
    frame_indices = torch.linspace(0, total_frames - 1, num_frames).long()
    
    # Build position indices for each patch
    # t_ids: frame index for each patch, repeated h*w times per frame
    t_ids = frame_indices.repeat_interleave(h * w)  # [seq_len]
    # h_ids: row index within each frame
    h_ids = torch.arange(h).repeat_interleave(w).repeat(num_frames)  # [seq_len]
    # w_ids: column index within each frame
    w_ids = torch.arange(w).repeat(h).repeat(num_frames)  # [seq_len]
    
    # Stack to [seq_len, 3] then add batch dimension -> [1, seq_len, 3]
    patch_positions = torch.stack([t_ids, h_ids, w_ids], dim=-1).unsqueeze(0)
    
    return patch_positions

def extract_frames_from_video(video_path, output_dir, num_frames=8, use_alternative_method=False):
    """
    Extract frames from video.
    Args:
        video_path (str): Path to video file
        output_dir (str): Output directory
        num_frames (int): Number of frames to extract
        use_alternative_method (bool): Whether to use alternative method for problematic videos
    Returns:
        list: List of saved frame paths
    """
    try:
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        # 获取视频文件名（不含扩展名）
        video_name = Path(video_path).stem
        saved_frames = []
        if use_alternative_method:
            return _extract_frames_robust_method(video_path, output_dir, video_name, num_frames)
        else:
            return _extract_frames_standard_method(video_path, output_dir, video_name, num_frames)
    except Exception as e:
        print(f"Error: Failed to process video {video_path}: {e}")
        # If standard method fails, try robust method
        if not use_alternative_method:
            print(f"Trying robust method for {video_path}")
            return extract_frames_from_video(video_path, output_dir, num_frames, use_alternative_method=True)
        return []
def _extract_frames_standard_method(video_path, output_dir, video_name, num_frames):
    """Standard video frame extraction method"""
    # Suppress OpenCV warnings
    cv2.setUseOptimized(False)  # Set to False to avoid NAL unit errors
    cv2.setNumThreads(1)  # Limit threads to avoid conflicts
    # Set environment variable for H264 issues
    os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return []
    # Try different decoding options
    cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_NONE)  # Disable HW acceleration
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'FFDS'))  # Try FFmpeg software decoding
    # Get total frame count
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        print(f"Error: Video file has no frames {video_path}")
        cap.release()
        return []
    # Calculate sampling interval
    interval = max(1, total_frames // num_frames)
    saved_frames = []
    consecutive_failures = 0
    max_consecutive_failures = 10  # Max consecutive failures
    # Extract frames
    
    for i in range(num_frames):
        #if i==7:
        #    import pdb;pdb.set_trace()
            
        frame_index = i * interval
        if frame_index >= total_frames:
            frame_index = total_frames - 1
        # Reset video position to avoid NAL unit error accumulation
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        # Try reading frame with retry mechanism
        ret, frame = cap.read()
        # If ret is None (extreme case), use decremented frame_index strategy until success
        if frame is None:
            decremented_index = frame_index
            temp_frame = None
            while decremented_index > 0:
                decremented_index -= 1
                cap.set(cv2.CAP_PROP_POS_FRAMES, decremented_index)
                ret_dec, temp_frame = cap.read()
                if ret_dec and temp_frame is not None:
                    # Successfully read, save it
                    output_filename = f"{video_name}_frame_{i:03d}.jpg"
                    output_path = os.path.join(output_dir, output_filename)
                    cv2.imwrite(output_path, temp_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    saved_frames.append(output_path)
                    print(f"Warning: Saved frame using decremented index: {output_path}, frame {decremented_index}")
                    break
            else:
                print(f"Error: Cannot read frame {frame_index} from {video_name} after decrementing")
            continue  # Skip to next frame


        retry_count = 0
        max_retries = 5
        while (not ret or frame is None) and retry_count < max_retries:
            retry_count += 1
            print(f"Warning: Failed to read frame {frame_index}, retry {retry_count}/{max_retries} from {video_name}")
            # Try different recovery strategies
            if retry_count == 1:
                # Reset video position
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            elif retry_count == 2:
                # Try going back a few frames
                new_index = max(0, frame_index - 5)
                cap.set(cv2.CAP_PROP_POS_FRAMES, new_index)
                # Skip a few frames
                for _ in range(min(5, frame_index - new_index)):
                    cap.grab()
            elif retry_count == 3:
                # Reopen video file
                cap.release()
                cap = cv2.VideoCapture(video_path)
                cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_NONE)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
        if ret and frame is not None:
            # Successfully read, reset failure count
            consecutive_failures = 0
            # Construct output filename
            output_filename = f"{video_name}_frame_{i:03d}.jpg"
            output_path = os.path.join(output_dir, output_filename)
            # Save frame with quality settings
            cv2.imwrite(output_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved_frames.append(output_path)
        else:
            
            #import pdb;pdb.set_trace()
            consecutive_failures += 1
            print(f"Error: Cannot read frame {frame_index} from {video_name} (consecutive failures: {consecutive_failures})")
            # If too many consecutive failures, switch to FFmpeg
            print("consecutive_failures >= max_consecutive_failures: ", consecutive_failures, max_consecutive_failures)
            if consecutive_failures >= max_consecutive_failures:
                print(f"Error: Too many consecutive failures, switching to FFmpeg for {video_name}")
                cap.release()
                return _extract_frames_robust_method(video_path, output_dir, video_name, num_frames)
    cap.release()
    return saved_frames
def _extract_frames_robust_method(video_path, output_dir, video_name, num_frames):
    """Robust video frame extraction method using ffmpeg"""
    try:
        import subprocess
        import tempfile
        print(f"Using ffmpeg robust method for {video_name}")
        # Create temp directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Use ffmpeg to extract frames with error handling options
            temp_pattern = os.path.join(temp_dir, f"{video_name}_frame_%03d.jpg")
            # Base command for handling NAL unit errors
            base_cmd = [
                'ffmpeg',
                '-v', 'error',  # 只显示错误信息
                '-err_detect', 'ignore_err',  # 忽略错误
                '-fflags', '+genpts+igndts',  # 生成时间戳，忽略DTS错误
                '-ignore_unknown',  # 忽略未知流
                '-i', video_path,
                '-avoid_negative_ts', 'make_zero',  # 避免负时间戳
                '-vf', f'select=not(mod(n\\,{max(1, 100//num_frames)})),scale=640:480',
                '-vsync', 'vfr',
                '-frames:v', str(num_frames),
                '-q:v', '3',  # 稍微降低质量以提高稳定性
                '-y',
                temp_pattern
            ]
            # Try base command
            result = subprocess.run(base_cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                # Success, move files to target directory
                saved_frames = []
                for i in range(num_frames):
                    # Check different naming patterns
                    possible_names = [
                        f"{video_name}_frame_{i:03d}.jpg",
                        f"{video_name}_frame_{i:03d:03d}.jpg",
                        f"frame_{i:03d}.jpg"
                    ]
                    for name in possible_names:
                        temp_file = os.path.join(temp_dir, name)
                        if os.path.exists(temp_file):
                            output_path = os.path.join(output_dir, f"{video_name}_frame_{i:03d}.jpg")
                            os.rename(temp_file, output_path)
                            saved_frames.append(output_path)
                            break
                return saved_frames
            else:
                print(f"FFmpeg basic command failed for {video_name}: {result.stderr}")
                # Try stronger error recovery options
                recovery_cmd = [
                    'ffmpeg',
                    '-v', 'error',
                    '-err_detect', 'ignore_err',
                    '-fflags', '+genpts+igndts+discardcorrupt',  # Discard corrupted packets
                    '-ignore_unknown',
                    '-max_muxing_queue_size', '1024',  # Increase buffer
                    '-thread_queue_size', '1024',
                    '-i', video_path,
                    '-c:v', 'libx264',  # Re-encode
                    '-preset', 'ultrafast',  # Fast preset
                    '-crf', '23',  # Reasonable quality
                    '-avoid_negative_ts', 'make_zero',
                    '-vf', f'select=not(mod(n\\,{max(1, 100//num_frames)})),scale=640:480',
                    '-vsync', 'vfr',
                    '-frames:v', str(num_frames),
                    '-y',
                    temp_pattern
                ]
                result = subprocess.run(recovery_cmd, capture_output=True, text=True, timeout=60)
                if result.returncode == 0:
                    # Success, move files
                    saved_frames = []
                    for i in range(num_frames):
                        for name in [f"{video_name}_frame_{i:03d}.jpg", f"frame_{i:03d}.jpg"]:
                            temp_file = os.path.join(temp_dir, name)
                            if os.path.exists(temp_file):
                                output_path = os.path.join(output_dir, f"{video_name}_frame_{i:03d}.jpg")
                                os.rename(temp_file, output_path)
                                saved_frames.append(output_path)
                                break
                    return saved_frames
                else:
                    print(f"FFmpeg recovery command failed for {video_name}: {result.stderr}")
    except subprocess.TimeoutExpired:
        print(f"Timeout processing {video_name} with ffmpeg")
    except Exception as e:
        print(f"FFmpeg method failed for {video_name}: {e}")
    # If FFmpeg also fails, try last resort: force extract keyframes
    try:
        print(f"Trying keyframe extraction for {video_name}")
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_pattern = os.path.join(temp_dir, f"keyframe_%03d.jpg")
            keyframe_cmd = [
                'ffmpeg',
                '-v', 'error',
                '-err_detect', 'ignore_err',
                '-fflags', '+genpts+igndts+discardcorrupt',
                '-skip_frame', 'nokey',  # 只处理关键帧
                '-i', video_path,
                '-vf', 'scale=640:480',
                '-frames:v', str(num_frames),
                '-q:v', '5',
                '-y',
                temp_pattern
            ]
            result = subprocess.run(keyframe_cmd, capture_output=True, text=True, timeout=20)
            if result.returncode == 0:
                saved_frames = []
                for i in range(num_frames):
                    temp_file = os.path.join(temp_dir, f"keyframe_{i:03d}.jpg")
                    if os.path.exists(temp_file):
                        output_path = os.path.join(output_dir, f"{video_name}_frame_{i:03d}.jpg")
                        os.rename(temp_file, output_path)
                        saved_frames.append(output_path)
                return saved_frames
    except Exception as e:
        print(f"Keyframe extraction failed for {video_name}: {e}")
    return []

def convert_inputs(inputs, image_root: str = '/workspace/tmp/vlm_imgs', num_frames: int = 8):
    image_inputs = []
    video_inputs = []
    
    # Extract visual info in format consistent with official implementation
    vision_infos = []
    if isinstance(inputs[0], dict):
        chats = [inputs]
    else:
        chats = inputs
    for chat in chats:
        for message in chat:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if (
                        "image" in ele or "image_url" in ele or "video" in ele
                        or ele.get("type", "") in ("image", "image_url", "video")
                    ):
                        vision_infos.append(ele)

    for vision_info in vision_infos:
        # Image
        if "image" in vision_info or "image_url" in vision_info:
            img_path = vision_info.get("image") or vision_info.get("image_url")
            if isinstance(img_path, str):
                image_inputs.append(Image.open(img_path))
            else:
                image_inputs.append(img_path)
        # Video, convert to PIL frame list
        elif "video" in vision_info:
            video_path = vision_info["video"]
            video_name = Path(video_path).stem
            img_dir = os.path.join(image_root, video_name)
            img_paths = extract_frames_from_video(video_path, img_dir, num_frames=num_frames)
            images = [Image.open(p) for p in img_paths]
            video_inputs.append(images)
        else:
            raise ValueError("image, image_url or video should in content.")

    if len(image_inputs) == 0:
        image_inputs = None
    if len(video_inputs) == 0:
        video_inputs = None
    # If video_inputs has fewer than num_frames, pad with last frame
    add = False
    if add:
        if video_inputs is not None:
            for idx, frames in enumerate(video_inputs):
                if len(frames) < num_frames:
                    if len(frames) > 0:
                        last_frame = frames[-1]
                        frames += [last_frame] * (num_frames - len(frames))
                        video_inputs[idx] = frames

    return image_inputs, video_inputs, vision_infos

@register_model("llava_ov_encoder")
class Llava_OV_Encoder(lmms):
    """
    Llava_OV_Encoder Model
    """

    def __init__(
        self,
        pretrained: str = "liuhaotian/llava-v1.5-7b",
        truncation: Optional[bool] = True,
        device: Optional[str] = "cuda:0",
        batch_size: Optional[Union[int, str]] = 1,
        model_name=None,
        attn_implementation=best_fit_attn_implementation,
        device_map="cuda:0",
        conv_template="vicuna_v1",
        use_cache=True,
        tie_weights: bool = True,
        truncate_context=False,  # whether to truncate the context in generation, set it False for LLaVA-1.6
        customized_config=None,  # ends in json
        is_prompt: bool=False,
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        self.accelerator = accelerator
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        llava_model_args = {
            "multimodal": True,
        }
        if customized_config is not None:
            llava_model_args["customized_config"] = customized_config
        if attn_implementation is not None:
            llava_model_args["attn_implementation"] = attn_implementation
        if "use_flash_attention_2" in kwargs:
            llava_model_args["use_flash_attention_2"] = kwargs["use_flash_attention_2"]
        model_name = model_name if model_name is not None else get_model_name_from_path(pretrained)
        try:
            # Try to load the model with the multimodal argument
            self._tokenizer, self._model, self._image_processor, self._max_length = load_pretrained_model(pretrained, None, model_name, device_map=self.device_map, **llava_model_args)
        except TypeError:
            # for older versions of LLaVA that don't have multimodal argument
            llava_model_args.pop("multimodal", None)
            self._tokenizer, self._model, self._image_processor, self._max_length = load_pretrained_model(pretrained, None, model_name, device_map=self.device_map, **llava_model_args)
        self._config = self._model.config
        self.model.eval()
        if tie_weights:
            self.model.tie_weights()
        self.is_prompt = is_prompt

        self.truncation = truncation
        self.batch_size_per_gpu = int(batch_size)
        self.conv_template = conv_template
        self.use_cache = use_cache
        self.truncate_context = truncate_context
        # assert self.batch_size_per_gpu == 1, "Llava currently does not support batched generation. See https://github.com/haotian-liu/LLaVA/issues/754. HF Llava also has this issue."
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            # If you want to use DistributedType.DEEPSPEED, you have to run accelerate config before using the model
            # Also, you have to select zero stage 0 (equivalent to DDP) in order to make the prepare model works
            # I tried to set different parameters in the kwargs to let default zero 2 stage works, but it didn't work.
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **kwargs)
                eval_logger.info("Detected that you are using DistributedType.DEEPSPEED. Make sure you run `accelerate config` and set zero stage to 0")

            if accelerator.distributed_type == DistributedType.FSDP or accelerator.distributed_type == DistributedType.DEEPSPEED:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        elif accelerator.num_processes == 1 and device_map == "auto":
            eval_logger.info(f"Using {accelerator.num_processes} devices with tensor parallelism")
            self._rank = 0
            self._world_size = 1
        else:
            eval_logger.info(f"Using single device: {self._device}")
            self.model.to(self._device)
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None) -> List[int]:
        """ """
        add_special_tokens = False if add_special_tokens is None else add_special_tokens
        encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def tok_decode(self, tokens):
        try:
            return self.tokenizer.decode(tokens)
        except:
            return self.tokenizer.decode([tokens])

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        # TODO
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, doc_to_target, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            # encode, pad, and truncate contexts for this batch
            if type(doc_to_target) == str:
                continuation = doc_to_target
            else:
                continuation = doc_to_target(self.task_dict[task][split][doc_id])
            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)
            image_sizes = [[visual.size[0], visual.size[1]] for visual in visuals]
            if visuals:
                image = process_images(visuals, self._image_processor, self._config)
                if type(image) is list:
                    image = [_image.to(dtype=torch.float16, device=self.device) for _image in image]
                else:
                    image = image.to(dtype=torch.float16, device=self.device)
            else:
                image = None

            prompts_input = contexts[0] if isinstance(contexts, list) else contexts

            if image is not None and len(image) != 0 and DEFAULT_IMAGE_TOKEN not in prompts_input:
                """
                Three senarios:
                1. No image, and there for, no image token should be added.
                2. image token is already specified in the context, so we don't need to add it.
                3. image token is not specified in the context and there is image inputs, so we need to add it. In this case, we add the image token at the beginning of the context and add a new line.
                """
                image_tokens = [DEFAULT_IMAGE_TOKEN] * len(visuals)
                image_tokens = " ".join(image_tokens)
                prompts_input = image_tokens + "\n" + (contexts[0] if isinstance(contexts, list) else contexts)

            # This is much safer for llama3, as we now have some object type in it
            if "llama_3" in self.conv_template:
                conv = copy.deepcopy(conv_templates[self.conv_template])
            else:
                conv = conv_templates[self.conv_template].copy()
            conv.append_message(conv.roles[0], prompts_input)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            contxt_id = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
            # Add the answer of the second role
            conv.messages[1][1] = continuation

            prompt = conv.get_prompt()
            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
            labels = input_ids.clone()
            # Context part no need to calculate for loss
            labels[0, : contxt_id.shape[1]] = -100
            with torch.inference_mode():
                outputs = self.model(input_ids=input_ids, labels=labels, images=image, use_cache=True, image_sizes=image_sizes)
            loss = outputs["loss"]
            # loss = torch.exp(loss)
            logits = outputs["logits"]
            greedy_tokens = logits.argmax(dim=-1)
            cont_toks = input_ids[:, contxt_id.shape[1] :]  # [1, seq]
            greedy_tokens = greedy_tokens[:, contxt_id.shape[1] : input_ids.shape[1]]  # [1, seq]
            max_equal = (greedy_tokens == cont_toks).all()
            res.append((float(loss.item()), bool(max_equal)))
            pbar.update(1)
        pbar.close()
        return res

    def flatten(self, input):
        if not input or any(i is None for i in input):
            return []
        new_list = []
        for i in input:
            if i:
                for j in i:
                    new_list.append(j)
        return new_list

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")
        for chunk in chunks:
            grid_thw = None
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            batched_visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]  # [B, N] 如果是视频的话，里面全是路径
            is_video = False
            if batched_visuals and isinstance(batched_visuals[0], list) and len(batched_visuals[0]) > 0 and isinstance(batched_visuals[0][0], str):
                is_video = True
                cur_video = batched_visuals[0][0]
                frames_paths = extract_frames_from_video(cur_video, '/workspace/tmp/vlm_imgs', num_frames=8)
                flattened_visuals = [Image.open(p) for p in frames_paths]
            
            else:
                flattened_visuals = self.flatten(batched_visuals)  # [B*N]
            # we assume all gen kwargs in the batch are the same
            # this is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]

            # Set default values for until and max_new_tokens
            until = [self.tok_decode(self.eot_token_id)]

            # Update values from gen_kwargs if present
            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str,list] but got {type(until)}")

            if "image_aspect_ratio" in gen_kwargs.keys() and "image_aspect_ratio" not in self._config.__dict__:
                # here we should pop it out of gen_kwargs so that it doesn't get passed to the model for next step of generation
                self._config.image_aspect_ratio = gen_kwargs.pop("image_aspect_ratio")
                eval_logger.info(f"Setting image aspect ratio: {self._config.image_aspect_ratio}")
            # encode, pad, and truncate contexts for this batch
            grid_thw = None
            if flattened_visuals:
                # assert 1==3, f'flattened_visuals len: {len(flattened_visuals)}\n shapes: {flattened_visuals.size}'
                image_tensor = process_images(flattened_visuals, self._image_processor, self._config)
                
                if type(image_tensor) is list:
                    image_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensor]
                elif type(image_tensor) is dict:
                    grid_thw = torch.tensor(image_tensor['grid_thw'])
                    image_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensor['image_patchs']]

                else:
                    image_tensor = image_tensor.to(dtype=torch.float16, device=self.device)
            else:
                image_tensor = None


            question_input = []

            for visual, context in zip(batched_visuals, contexts):
                if image_tensor is not None and len(image_tensor) != 0 and DEFAULT_IMAGE_TOKEN not in context:
                    """
                    Three senarios:
                    1. No image, and there for, no image token should be added.
                    2. image token is already specified in the context, so we don't need to add it.
                    3. image token is not specified in the context and there is image inputs, so we need to add it. In this case, we add the image token at the beginning of the context and add a new line.
                    """
                    image_tokens = [DEFAULT_IMAGE_TOKEN] * len(visual) if isinstance(visual, list) else [DEFAULT_IMAGE_TOKEN]
                    image_tokens = "\n".join(image_tokens)
                    question = image_tokens + "\n" + context
                else:
                    question = context
                # This is much safer for llama3, as we now have some object type in it
                if "llama_3" in self.conv_template:
                    conv = copy.deepcopy(conv_templates[self.conv_template])
                else:
                    conv = conv_templates[self.conv_template].copy()
                conv.append_message(conv.roles[0], question)
                conv.append_message(conv.roles[1], None)
                prompt_question = conv.get_prompt()
                question_input.append(prompt_question)

            # input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
            # preconfigure gen_kwargs with defaults
            gen_kwargs["image_sizes"] = [flattened_visuals[idx].size for idx in range(len(flattened_visuals))]
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1
            
            # gen_kwargs["visible_indices"] = 

            input_ids_list = [tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt") for prompt in question_input]
            pad_token_ids = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            input_ids = self.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_ids).to(self.device)
            attention_masks = input_ids.ne(pad_token_ids).to(self.device)
            # These steps are not in LLaVA's original code, but are necessary for generation to work
            # TODO: attention to this major generation step...
            if len(grid_thw) == 1:
                patch_positions = [build_patch_positions(num_frames=1, total_frames=1, h=item[1], w=item[2]).to(self.device) for item in grid_thw]
            else:
                patch_positions = [build_patch_positions(num_frames=8, total_frames=64, h=36, w=36).to(self.device)]
            try:
                cont = self.model.generate(
                    input_ids,
                    attention_mask=attention_masks,
                    pad_token_id=pad_token_ids,
                    images=image_tensor,
                    image_sizes=gen_kwargs["image_sizes"],
                    do_sample=True if gen_kwargs["temperature"] > 0 else False,
                    temperature=gen_kwargs["temperature"],
                    top_p=gen_kwargs["top_p"],
                    num_beams=gen_kwargs["num_beams"],
                    max_new_tokens=gen_kwargs["max_new_tokens"],
                    use_cache=self.use_cache,
                    grid_thw=grid_thw if grid_thw is not None else None,
                    patch_positions = patch_positions
                )
                text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
                if self.is_prompt:
                    text_outputs_filter = []
                    for out in text_outputs:
                        if out.startswith('assistant\n'):
                            out = out.replace('assistant\n', '')
                        text_outputs_filter.append(out)
                    text_outputs_filter2 = []
                    for out in text_outputs_filter:
                        if out.startswith('assistant\n\n'):
                            out = out.replace('assistant\n\n', '')
                        text_outputs_filter2.append(out)
                    text_outputs = text_outputs_filter2
            except Exception as e:
                raise e
                eval_logger.error(f"Error {e} in generating")
                cont = ""
                text_outputs = [""]

            # cont_toks_list = cont.tolist()
            # for cont_toks, context in zip(cont_toks_list, contexts):
            # discard context + left-padding toks if using causal decoder-only LMM
            # if self.truncate_context:
            #     cont_toks = cont_toks[input_ids.shape[1] :]
            # use secondary stop seqs to cut off should-have-been-stopped content post-hoc
            # if self.truncate_context:
            #     for term in until:
            #         if len(term) > 0:
            #             # ignore '' separator,
            #             # for seq2seq case where self.tok_decode(self.eot_token_id) = ''
            #             text_outputs = text_outputs.split(term)[0]
            res.extend(text_outputs)
            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), text_outputs)
            pbar.update(1)
            # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation for LLaVA")
