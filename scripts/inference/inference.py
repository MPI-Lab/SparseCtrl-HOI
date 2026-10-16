"""SparseCtrl-HOI inference.

Each line of ``--test_list`` describes one sample, four fields separated by ``#``:

    <reference image>#<audio .wav>#<keyframe directory>#[f0, f1, f2, f3]

The keyframe directory holds the K interaction keyframes as ``*_<idx>.png`` (sorted by
``<idx>``) and ``[f0, ...]`` are their designated frame indices in the generated clip.
Samples are split evenly across all visible GPUs, one process per GPU.
"""
import argparse
import json
import math
import multiprocessing
import os
import subprocess
import sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms as TT
from accelerate.utils import set_seed
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from sparsectrl_hoi import (SparseCtrlHOIPipeline, extract_motion_prior_features, load_qwen_vl, load_wan_dit,
                            load_wan_vae)
from sparsectrl_hoi.utils import add_silence_to_audio_ffmpeg, save_video

HEIGHT = 512
WIDTH = 512


def parse_args():
    parser = argparse.ArgumentParser(description="SparseCtrl-HOI inference.")
    # inputs / outputs
    parser.add_argument("--test_list", type=str, required=True, help="Text file listing the test samples (see module docstring).")
    parser.add_argument("--output_path", type=str, default="./output", help="Directory to save the generated videos.")
    # base models
    parser.add_argument("--dit_path", type=str, required=True, help="Wan2.1 DiT checkpoint; comma-separated shards for 14B.")
    parser.add_argument("--vae_path", type=str, required=True, help="Wan2.1 VAE checkpoint.")
    parser.add_argument("--wav2vec_path", type=str, default="checkpoints/wav2vec2-base-960h", help="wav2vec2-base-960h directory.")
    parser.add_argument("--pretrained_lora_path", type=str, required=True, help="OmniAvatar pytorch_model.pt (audio-driven LoRA).")
    parser.add_argument("--model_config_json", type=str, default=None,
                        help="JSON file with the DiT config overrides (key 'model_config'). Defaults to config.json next to --pretrained_lora_path.")
    # SparseCtrl-HOI weights
    parser.add_argument("--stage1_weights", type=str, required=True, help="Stage-1 checkpoint (LoRA + TiRoPE input adapter).")
    parser.add_argument("--stage2_weights", type=str, required=True, help="Stage-2 checkpoint (Motion Prior Injection Module).")
    parser.add_argument("--lora_rank", type=int, default=128)
    parser.add_argument("--lora_alpha", type=float, default=64)
    parser.add_argument("--lora_target_modules", type=str, default="q,k,v,o,ffn.0,ffn.2")
    # motion prior (MLLM)
    parser.add_argument("--mllm_version", type=str, default="7B", choices=["7B", "3B"], help="Qwen2.5-VL size.")
    parser.add_argument("--qwen_path", type=str, default=None,
                        help="Local Qwen2.5-VL directory; defaults to the HF hub id matching --mllm_version.")
    parser.add_argument("--num_qwen_feature_token", type=int, default=64, help="Number of motion prior tokens.")
    parser.add_argument("--num_layer_qwen_feature_proj", type=int, default=4, help="Number of Q-Former layers.")
    # text prompt (precomputed T5 embeddings)
    parser.add_argument("--caption_emb_path", type=str, required=True, help="Precomputed T5 embedding of the prompt (.pt).")
    parser.add_argument("--caption_nega_emb_path", type=str, required=True, help="Precomputed T5 embedding of the negative prompt (.pt).")
    # sampling
    parser.add_argument("--num_frames", type=int, default=121)
    parser.add_argument("--num_ref_multiviews", type=int, default=4, help="Number of interaction keyframes K.")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=4.5)
    parser.add_argument("--audio_cfg_scale", type=float, default=4.5)
    parser.add_argument("--seed", type=int, default=1028)
    parser.add_argument("--tile_size_height", type=int, default=64)
    parser.add_argument("--tile_size_width", type=int, default=64)
    parser.add_argument("--tile_stride_height", type=int, default=64)
    parser.add_argument("--tile_stride_width", type=int, default=64)
    parser.add_argument("--num_persistent_param_in_dit", type=int, default=6 * 10**9,
                        help="DiT parameters kept on the GPU; lower it to fit smaller GPUs.")
    return parser.parse_args()


def load_model_config(args):
    path = args.model_config_json or os.path.join(os.path.dirname(args.pretrained_lora_path), "config.json")
    with open(path, "r") as f:
        return json.load(f).get("model_config", None)


def compose_frame(ref_image, keyframes, generated):
    """Reference | keyframes | generated frame, side by side."""
    images = [ref_image] + list(keyframes) + [generated]
    height = ref_image.size[1]
    total_width = sum(im.size[0] for im in images)
    canvas = Image.new('RGB', (total_width, height))
    x = 0
    for im in images:
        canvas.paste(im, (x, 0))
        x += im.size[0]
    return canvas


def get_video_duration(video_file):
    cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', video_file]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    info = json.loads(result.stdout)
    return float(info['format']['duration'])


def mux_audio(video_file, audio_path, tmp_audio_path):
    """Replace ``video_file`` by a copy with the (silence-padded) audio track."""
    add_silence_to_audio_ffmpeg(audio_path, tmp_audio_path, silence_duration_s=1.0 / 25)
    video_duration = get_video_duration(video_file)
    final_file = video_file.replace(".mp4", "_with_audio.mp4")
    cmd = f'ffmpeg -y -v quiet -i {video_file} -i {tmp_audio_path} -c:v copy -c:a aac -strict experimental -t {video_duration} {final_file}'
    subprocess.run(cmd, shell=True, check=True)
    os.remove(video_file)
    os.rename(final_file, video_file)


def build_pipeline(args, device):
    dit = load_wan_dit(args.dit_path.split(","), model_config=load_model_config(args), torch_dtype=torch.bfloat16, device="cpu")
    vae = load_wan_vae(args.vae_path, torch_dtype=torch.bfloat16)
    pipe = SparseCtrlHOIPipeline(dit, vae, wav2vec_path=args.wav2vec_path, device=device, torch_dtype=torch.bfloat16)

    pipe.add_lora(
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_target_modules=args.lora_target_modules,
        pretrained_lora_path=args.pretrained_lora_path,
    )
    pipe.enable_motion_prior(
        num_qwen_feature_token=args.num_qwen_feature_token,
        mllm_version=args.mllm_version,
        num_layers=args.num_layer_qwen_feature_proj,
    )
    pipe.load_sparsectrl_weights(args.stage1_weights)
    pipe.load_sparsectrl_weights(args.stage2_weights)

    pipe.to(dtype=torch.bfloat16, device=device)
    pipe.requires_grad_(False)
    pipe.eval()
    pipe.enable_vram_management(num_persistent_param_in_dit=args.num_persistent_param_in_dit)
    return pipe


def gpu_worker(gpu_id, sample_list, args):
    device = f'cuda:{gpu_id}'
    print(f"Process on GPU {gpu_id} started with {len(sample_list)} samples.")
    set_seed(args.seed)

    pipe = build_pipeline(args, device)
    qwen_model, qwen_processor = load_qwen_vl(device, mllm_version=args.mllm_version, model_name_or_path=args.qwen_path)

    prompt_emb = torch.load(args.caption_emb_path, weights_only=True)
    negative_prompt_emb = torch.load(args.caption_nega_emb_path, weights_only=True)
    to_tensor = TT.Compose([TT.ToTensor()])

    for ref_image_path, audio_path, keyframe_dir, keyframe_positions in tqdm(sample_list, desc=f"GPU {gpu_id}", position=gpu_id):
        # reference image
        ref_image_pil = Image.fromarray(cv2.cvtColor(cv2.imread(ref_image_path), cv2.COLOR_BGR2RGB))
        ref_image_vis = torchvision.transforms.functional.resize(
            ref_image_pil, (HEIGHT, WIDTH), interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
        ref_image = to_tensor(ref_image_pil).unsqueeze(0).to(device=device)
        ref_image = ref_image * 2.0 - 1.0
        ref_image = ref_image[:, :, None]

        # keyframes, sorted by the trailing index in the file name
        keyframe_files = [f for f in os.listdir(keyframe_dir) if f.endswith('.png')]
        keyframe_files = sorted(keyframe_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        keyframe_pils = [Image.open(os.path.join(keyframe_dir, f)) for f in keyframe_files]
        keyframes = [np.array(im).astype(np.uint8) for im in keyframe_pils]

        # motion prior features from the frozen MLLM
        qwen_tokens = extract_motion_prior_features(qwen_model, qwen_processor, keyframe_pils, mllm_version=args.mllm_version)

        video = pipe.generate(
            prompt_emb=prompt_emb,
            negative_prompt_emb=negative_prompt_emb,
            audio_path=audio_path,
            ref_image=ref_image,
            keyframes=keyframes,
            keyframe_positions=keyframe_positions,
            num_keyframes=args.num_ref_multiviews,
            qwen_tokens=qwen_tokens,
            height=HEIGHT,
            width=WIDTH,
            num_frames=args.num_frames,
            cfg_scale=args.cfg_scale,
            audio_cfg_scale=args.audio_cfg_scale,
            num_inference_steps=args.num_inference_steps,
            tile_size=(args.tile_size_height, args.tile_size_width),
            tile_stride=(args.tile_stride_height, args.tile_stride_width),
        )

        # save: composite (reference | keyframes | generated) and the generated clip alone
        keyframe_rgb = [im.convert('RGB') if im.mode != 'RGB' else im for im in keyframe_pils]
        composite = [compose_frame(ref_image_vis, keyframe_rgb, frame) for frame in video]
        sample_name = os.path.basename(os.path.dirname(os.path.abspath(audio_path)))
        name = f'{sample_name}_{os.path.splitext(os.path.basename(audio_path))[0]}'
        save_file = os.path.join(args.output_path, f'video_{HEIGHT}_{name}.mp4')
        save_video(composite, save_file, fps=25, quality=10)
        os.makedirs(os.path.join(args.output_path, 'generated_single'), exist_ok=True)
        save_file_single = os.path.join(args.output_path, 'generated_single', f'{name}.mp4')
        save_video(video, save_file_single, fps=25, quality=10)

        tmp_audio_path = os.path.join(args.output_path, f'{name}_gpu{gpu_id}.wav')
        mux_audio(save_file, audio_path, tmp_audio_path)
        mux_audio(save_file_single, audio_path, tmp_audio_path)
        os.remove(tmp_audio_path)


def read_test_list(path):
    samples = []
    with open(path, 'r', encoding='utf-8') as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            parts = line.split('#')
            if len(parts) != 4:
                raise ValueError(f"Line {line_idx+1}: expected 4 fields separated by '#', got {len(parts)}: {line}")
            ref_image_path, audio_path, keyframe_dir, positions_str = [p.strip() for p in parts]
            keyframe_positions = [int(p.strip()) for p in positions_str.replace('[', '').replace(']', '').split(',')]
            for p in (ref_image_path, audio_path, keyframe_dir):
                if not os.path.exists(p):
                    raise FileNotFoundError(f"Line {line_idx+1}: {p} does not exist")
            samples.append([ref_image_path, audio_path, keyframe_dir, keyframe_positions])
    return samples


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_path, exist_ok=True)

    samples = read_test_list(args.test_list)
    print(f"Total test samples: {len(samples)}")

    num_gpus = torch.cuda.device_count()
    print(f"Detected {num_gpus} GPUs.")
    if num_gpus == 0:
        raise RuntimeError("No GPUs found.")

    chunk_size = math.ceil(len(samples) / num_gpus)
    chunks = [samples[i:i + chunk_size] for i in range(0, len(samples), chunk_size)]

    multiprocessing.set_start_method('spawn', force=True)
    processes = []
    for gpu_id in range(num_gpus):
        chunk = chunks[gpu_id] if gpu_id < len(chunks) else []
        p = multiprocessing.Process(target=gpu_worker, args=(gpu_id, chunk, args))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()

    failed = [p for p in processes if p.exitcode != 0]
    if failed:
        details = ", ".join(f"pid={p.pid}, exitcode={p.exitcode}" for p in failed)
        raise RuntimeError(f"Inference workers failed: {details}")

    print("All inference tasks completed.")
