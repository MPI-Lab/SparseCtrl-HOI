# Adapted from DiffSynth-Studio (https://github.com/modelscope/DiffSynth-Studio), Apache-2.0,
# and OmniAvatar (https://github.com/Omni-Avatar/OmniAvatar), Apache-2.0.
"""SparseCtrl-HOI inference pipeline.

Inputs: a reference image, an audio track, K sparse interaction keyframes with their
designated frame indices, a (precomputed) text embedding and the Qwen2.5-VL features
of the keyframes. Output: a list of PIL frames.
"""
import math
from typing import List, Optional

import librosa
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as TT
from einops import rearrange
from PIL import Image
from tqdm import tqdm

from ..models.loader import load_state_dict
from ..models.wan_dit import RMSNorm, WanModel, inject_motion_cross_attention
from ..models.wan_vae import CausalConv3d, RMS_norm, Upsample, WanVideoVAE
from ..schedulers.flow_match import FlowMatchScheduler
from ..vram_management import AutoWrappedLinear, AutoWrappedModule, enable_vram_management


class SparseCtrlHOIPipeline(torch.nn.Module):

    def __init__(self, dit: WanModel, vae: WanVideoVAE, wav2vec_path: str, device="cuda", torch_dtype=torch.bfloat16):
        super().__init__()
        self.device = device
        self.torch_dtype = torch_dtype
        self.height_division_factor = 16
        self.width_division_factor = 16
        self.cpu_offload = False
        self.model_names = ['dit', 'vae']

        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.dit = dit
        self.vae = vae
        self._init_wav2vec(wav2vec_path)

    # ------------------------------------------------------------------ setup

    def _init_wav2vec(self, wav2vec_path):
        """Wav2Vec2 feature extractor and encoder for the audio condition."""
        from transformers import Wav2Vec2FeatureExtractor

        from ..models.wav2vec import Wav2VecModel
        self.wav_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(wav2vec_path)
        self.audio_encoder = Wav2VecModel.from_pretrained(
            wav2vec_path,
            local_files_only=True,
            attn_implementation="eager",
        ).to(device=self.device, dtype=self.torch_dtype)
        self.audio_encoder.feature_extractor._freeze_parameters()

    def denoising_model(self):
        return self.dit

    def add_lora(self, lora_rank=128, lora_alpha=64, lora_target_modules="q,k,v,o,ffn.0,ffn.2", pretrained_lora_path: Optional[str] = None):
        """Add the LoRA branches to the DiT (peft layout) and optionally load the
        OmniAvatar audio-driven weights (LoRA + 33-channel patch embedding + audio projections)."""
        from peft import LoraConfig, inject_adapter_in_model

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            init_lora_weights=True,
            target_modules=lora_target_modules.split(","),
        )
        self.dit = inject_adapter_in_model(lora_config, self.dit)
        self.dit.enable_keyframe_input()
        if pretrained_lora_path is not None:
            state_dict = load_state_dict(pretrained_lora_path)
            missing_keys, unexpected_keys = self.dit.load_state_dict(state_dict, strict=False)
            all_keys = [i for i, _ in self.dit.named_parameters()]
            num_updated_keys = len(all_keys) - len(missing_keys)
            print(f"{num_updated_keys} parameters are loaded from {pretrained_lora_path}. {len(unexpected_keys)} parameters are unexpected.")

    def enable_motion_prior(self, num_qwen_feature_token=64, mllm_version="7B", num_layers=4):
        """Create the Motion Prior Injection Module: Q-Former + Motion Cross-Attention in every block."""
        self.dit.init_motion_prior_modules(num_qwen_feature_token=num_qwen_feature_token, mllm_version=mllm_version, num_layers=num_layers)
        inject_motion_cross_attention(self.dit, cond_dim=self.dit.dim, num_heads=self.dit.blocks[0].self_attn.num_heads)

    def load_sparsectrl_weights(self, path: str):
        """Load a SparseCtrl-HOI checkpoint (Stage-1 or Stage-2) into the DiT."""
        print(f"Loading SparseCtrl-HOI weights from {path}")
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        prefix = "pipe.dit."
        state_dict = {(k[len(prefix):] if k.startswith(prefix) else k): v for k, v in state_dict.items()}
        missing_keys, unexpected_keys = self.dit.load_state_dict(state_dict, strict=False)
        print(f"  missing keys: {len(missing_keys)}, unexpected keys: {len(unexpected_keys)}")

    def enable_cpu_offload(self):
        self.cpu_offload = True

    def enable_vram_management(self, num_persistent_param_in_dit=None):
        """Keep at most ``num_persistent_param_in_dit`` DiT parameters resident on the GPU;
        the rest (and the VAE) are streamed from CPU per module."""
        dtype = next(iter(self.dit.parameters())).dtype
        enable_vram_management(
            self.dit,
            module_map = {
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Conv3d: AutoWrappedModule,
                torch.nn.LayerNorm: AutoWrappedModule,
                RMSNorm: AutoWrappedModule,
            },
            module_config = dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device=self.device,
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
            max_num_param=num_persistent_param_in_dit,
            overflow_module_config = dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device="cpu",
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
        )
        dtype = next(iter(self.vae.parameters())).dtype
        enable_vram_management(
            self.vae,
            module_map = {
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Conv2d: AutoWrappedModule,
                RMS_norm: AutoWrappedModule,
                CausalConv3d: AutoWrappedModule,
                Upsample: AutoWrappedModule,
                torch.nn.SiLU: AutoWrappedModule,
                torch.nn.Dropout: AutoWrappedModule,
            },
            module_config = dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device=self.device,
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
        )
        self.enable_cpu_offload()

    def load_models_to_device(self, loadmodel_names=[]):
        # only load models to device if cpu_offload is enabled
        if not self.cpu_offload:
            return
        # offload the unneeded models to cpu
        for model_name in self.model_names:
            if model_name not in loadmodel_names:
                model = getattr(self, model_name)
                if model is not None:
                    if hasattr(model, "vram_management_enabled") and model.vram_management_enabled:
                        for module in model.modules():
                            if hasattr(module, "offload"):
                                module.offload()
                    else:
                        model.cpu()
        # load the needed models to device
        for model_name in loadmodel_names:
            model = getattr(self, model_name)
            if model is not None:
                if hasattr(model, "vram_management_enabled") and model.vram_management_enabled:
                    for module in model.modules():
                        if hasattr(module, "onload"):
                            module.onload()
                else:
                    model.to(self.device)
        # fresh the cuda cache
        torch.cuda.empty_cache()

    # ---------------------------------------------------------------- helpers

    def check_resize_height_width(self, height, width):
        if height % self.height_division_factor != 0:
            height = (height + self.height_division_factor - 1) // self.height_division_factor * self.height_division_factor
            print(f"The height cannot be evenly divided by {self.height_division_factor}. We round it up to {height}.")
        if width % self.width_division_factor != 0:
            width = (width + self.width_division_factor - 1) // self.width_division_factor * self.width_division_factor
            print(f"The width cannot be evenly divided by {self.width_division_factor}. We round it up to {width}.")
        return height, width

    def tensor2video(self, frames):
        frames = rearrange(frames, "C T H W -> T H W C")
        frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
        frames = [Image.fromarray(frame) for frame in frames]
        return frames

    def encode_video(self, input_video, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
        latents = self.vae.encode(input_video, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return latents

    def decode_video(self, latents, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
        frames = self.vae.decode(latents, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return frames

    def _transform_image(self, image_numpy, device):
        transform_image = TT.Compose([TT.ToTensor()])
        image = Image.fromarray(image_numpy)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = transform_image(image).unsqueeze(0).to(device)
        image = image * 2.0 - 1.0
        image = image[:, :, None]
        return image

    # --------------------------------------------------------------- generate

    @torch.no_grad()
    def generate(
        self,
        prompt_emb: dict,
        negative_prompt_emb: Optional[dict],
        audio_path: str,
        ref_image: torch.Tensor,
        keyframes: List[np.ndarray],
        keyframe_positions: List[int],
        num_keyframes: int,
        qwen_tokens: torch.Tensor,
        height=512,
        width=512,
        num_frames=121,
        cfg_scale=4.5,
        audio_cfg_scale=4.5,
        num_inference_steps=50,
        sigma_shift=5.0,
        tiled=True,
        tile_size=(64, 64),
        tile_stride=(64, 64),
        fps=25,
        val_mode=True,
        progress_bar_cmd=tqdm,
    ):
        """
        Args:
            prompt_emb: ``{"context": T5 embedding}`` of the text prompt, on ``self.device``.
            negative_prompt_emb: same layout for the negative prompt (``None`` when ``cfg_scale == 1``).
            audio_path: path of the driving audio.
            ref_image: reference image tensor in [-1, 1], (1, 3, 1, H, W).
            keyframes: K interaction keyframes as uint8 HxWx3 arrays.
            keyframe_positions: designated frame index of each keyframe.
            num_keyframes: K.
            qwen_tokens: Qwen2.5-VL hidden states of the keyframes, (1, n, d_mllm).
            val_mode: generate a single clip of ``num_frames`` frames (the released
                setting); when False, long audio is generated clip by clip.
        Returns:
            list of PIL frames.
        """
        height, width = self.check_resize_height_width(height, width)
        clip_num_frames = num_frames
        T = (clip_num_frames + 3) // 4  # latent frames
        first_fixed_frame = 1
        fixed_frame = 1

        # audio -> wav2vec embeddings
        speech_array, sampling_rate = librosa.load(audio_path, sr=16000)
        input_values = np.squeeze(
            self.wav_feature_extractor(speech_array, sampling_rate=sampling_rate).input_values
        )
        input_values = torch.from_numpy(input_values).float().to(device=self.device)
        ori_audio_len = audio_len = math.ceil(len(input_values) / sampling_rate * fps)
        input_values = input_values.unsqueeze(0)
        # padding audio
        if audio_len < clip_num_frames - first_fixed_frame:
            audio_len = audio_len + ((clip_num_frames - first_fixed_frame) - audio_len % (clip_num_frames - first_fixed_frame))
        elif (audio_len - (clip_num_frames - first_fixed_frame)) % (clip_num_frames - fixed_frame) != 0:
            audio_len = audio_len + ((clip_num_frames - fixed_frame) - (audio_len - (clip_num_frames - first_fixed_frame)) % (clip_num_frames - fixed_frame))
        input_values = F.pad(input_values, (0, audio_len * int(sampling_rate / fps) - input_values.shape[1]), mode='constant', value=0)
        input_values = input_values.to(dtype=self.torch_dtype)
        with torch.no_grad():
            hidden_states = self.audio_encoder(input_values, seq_len=audio_len, output_hidden_states=True)
            audio_embeddings = hidden_states.last_hidden_state
            for mid_hidden_states in hidden_states.hidden_states:
                audio_embeddings = torch.cat((audio_embeddings, mid_hidden_states), -1)
        seq_len = audio_len
        audio_embeddings = audio_embeddings.squeeze(0)
        audio_prefix = torch.zeros_like(audio_embeddings[:first_fixed_frame])

        # number of clips
        times = (seq_len - clip_num_frames + first_fixed_frame) // (clip_num_frames-fixed_frame) + 1
        if times * (clip_num_frames-fixed_frame) + fixed_frame < seq_len:
            times += 1

        tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}

        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=1.0, shift=sigma_shift)

        # text embeddings
        prompt_emb_posi = {"context": prompt_emb['context'].to(device=self.device, dtype=self.torch_dtype)}
        if cfg_scale != 1.0:
            prompt_emb_nega = {"context": negative_prompt_emb['context'].to(device=self.device, dtype=self.torch_dtype)}

        # reference image -> latent, repeated over the clip, with an all-ones mask channel
        image_emb = {}
        self.load_models_to_device(["vae"])
        img_lat = self.encode_video(ref_image.to(dtype=self.torch_dtype), **tiler_kwargs).to(self.device)
        msk = torch.zeros_like(img_lat.repeat(1, 1, T, 1, 1)[:,:1])
        image_cat = img_lat.repeat(1, 1, T, 1, 1)
        msk[:, :, 0:] = 1
        image_emb["y"] = torch.cat([image_cat, msk], dim=1)

        # keyframes -> latents, with an all-zeros mask channel
        object_frames_emb_list = []
        object_frames = [self._transform_image(frame, self.device) for frame in keyframes]
        for object_frame in object_frames:
            object_image_lat = self.encode_video(object_frame.to(dtype=self.torch_dtype), **tiler_kwargs).to(self.device)  # (1, 16, 1, h, w)
            msk_object = torch.zeros_like(object_image_lat[:,:1])
            object_frames_emb_list.append(torch.cat([object_image_lat, msk_object], dim=1))  # (1, 16+1, 1, h, w)
        image_emb["y_object"] = torch.cat(object_frames_emb_list, dim=2)  # (1, 16+1, K, h, w)

        dit_kwargs = dict(object_positions=keyframe_positions, num_hands_object_frames=num_keyframes, qwen_tokens=qwen_tokens)

        video = []
        for t in range(times):
            print(f"[{t+1}/{times}]")
            if val_mode:
                if t > 0:
                    break
            if t == 0:
                overlap = first_fixed_frame
            else:
                overlap = fixed_frame
            prefix_overlap = (3 + overlap) // 4
            if t == 0:
                audio_tensor = audio_embeddings[
                        :min(clip_num_frames - overlap, audio_embeddings.shape[0])
                    ]
            else:
                audio_start = clip_num_frames - first_fixed_frame + (t - 1) * (clip_num_frames - overlap)
                audio_tensor = audio_embeddings[
                    audio_start: min(audio_start + clip_num_frames - overlap, audio_embeddings.shape[0])
                ]
            audio_tensor = torch.cat([audio_prefix, audio_tensor], dim=0)
            audio_prefix = audio_tensor[-fixed_frame:]
            audio_tensor = audio_tensor.unsqueeze(0).to(device=self.device, dtype=self.torch_dtype)

            # Denoise
            self.load_models_to_device(["dit"])
            img_lat = torch.cat([img_lat, torch.zeros_like(img_lat[:, :, :1].repeat(1, 1, T - prefix_overlap, 1, 1))], dim=2)
            latents = torch.randn_like(img_lat)
            for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
                timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
                noise_pred_posi = self.dit(latents, timestep=timestep, **prompt_emb_posi, **image_emb, audio_emb=audio_tensor, **dit_kwargs)[:,:,num_keyframes:]

                if cfg_scale != 1.0:
                    if audio_cfg_scale == cfg_scale:
                        noise_pred_nega = self.dit(latents, timestep=timestep, **prompt_emb_nega, **image_emb, audio_emb=torch.zeros_like(audio_tensor), **dit_kwargs)[:,:,num_keyframes:]
                        noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
                    else:
                        audio_noise_pred_nega = self.dit(latents, timestep=timestep, **prompt_emb_posi, **image_emb, audio_emb=torch.zeros_like(audio_tensor), **dit_kwargs)[:,:,num_keyframes:]
                        text_noise_pred_nega = self.dit(latents, timestep=timestep, **prompt_emb_nega, **image_emb, audio_emb=audio_tensor, **dit_kwargs)[:,:,num_keyframes:]
                        noise_pred = text_noise_pred_nega + cfg_scale * (audio_noise_pred_nega - text_noise_pred_nega) + audio_cfg_scale * (noise_pred_posi - audio_noise_pred_nega)
                else:
                    noise_pred = noise_pred_posi

                latents = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], latents)

            # Decode
            self.load_models_to_device(['vae'])
            frames = self.decode_video(latents, **tiler_kwargs)
            self.load_models_to_device([])
            frames = self.tensor2video(frames[0])
            if t == 0:
                video.extend(frames)
            else:
                video.extend(frames[overlap:])
        if val_mode:
            return video
        video = video[:ori_audio_len+1]
        return video
