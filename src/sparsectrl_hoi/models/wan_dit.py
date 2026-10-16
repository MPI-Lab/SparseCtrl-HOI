# Adapted from DiffSynth-Studio (https://github.com/modelscope/DiffSynth-Studio), Apache-2.0,
# and OmniAvatar (https://github.com/Omni-Avatar/OmniAvatar), Apache-2.0.
"""Wan2.1 DiT backbone with SparseCtrl-HOI conditioning.

Additions on top of the Wan2.1 T2V DiT:
  * audio conditioning (``audio_proj`` / ``audio_cond_projs``, from OmniAvatar);
  * sparse interaction keyframes concatenated along the frame axis and anchored
    to their designated timestamps through TiRoPE (Time-Controlled RoPE);
  * a Motion Prior Injection Module: a Q-Former (``Qformer_qwen_feature_proj``)
    compressing Qwen2.5-VL features into motion prior tokens, injected into every
    DiT block by a Motion Cross-Attention layer (``cross_attn2`` / ``norm4``).

Attribute names are part of the released checkpoint format and must not change.
"""
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .audio_pack import AudioPack
from .loader import hash_state_dict_keys

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

try:
    from sageattention import sageattn
    SAGE_ATTN_AVAILABLE = True
except ModuleNotFoundError:
    SAGE_ATTN_AVAILABLE = False


def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, compatibility_mode=False):
    if compatibility_mode:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_3_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x, _ = flash_attn_interface.flash_attn_func(q, k, v)
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_2_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn.flash_attn_func(q, k, v)
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif SAGE_ATTN_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = sageattn(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    else:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    return x


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return (x * (1 + scale) + shift)


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(position.type(torch.float64), torch.pow(
        10000, -torch.arange(dim//2, dtype=torch.float64, device=position.device).div(dim//2)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    # 3d rope precompute
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    # 1d rope precompute
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                   [: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def rope_apply(x, freqs, num_heads):
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_out = torch.view_as_complex(x.to(torch.float64).reshape(
        x.shape[0], x.shape[1], x.shape[2], -1, 2))
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        dtype = x.dtype
        return self.norm(x.float()).to(dtype) * self.weight


class AttentionModule(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads

    def forward(self, q, k, v):
        x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads)
        return x


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)

        self.attn = AttentionModule(self.num_heads)

    def forward(self, x, freqs):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        q = rope_apply(q, freqs, self.num_heads)
        k = rope_apply(k, freqs, self.num_heads)
        x = self.attn(q, k, v)
        return self.o(x)


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6, has_image_input: bool = False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        self.has_image_input = has_image_input
        if has_image_input:
            self.k_img = nn.Linear(dim, dim)
            self.v_img = nn.Linear(dim, dim)
            self.norm_k_img = RMSNorm(dim, eps=eps)

        self.attn = AttentionModule(self.num_heads)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        if self.has_image_input:
            img = y[:, :257]
            ctx = y[:, 257:]
        else:
            ctx = y
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(ctx))
        v = self.v(ctx)
        x = self.attn(q, k, v)
        if self.has_image_input:
            k_img = self.norm_k_img(self.k_img(img))
            v_img = self.v_img(img)
            y = flash_attention(q, k_img, v_img, num_heads=self.num_heads)
            x = x + y
        return self.o(x)


class DiTBlock(nn.Module):
    def __init__(self, has_image_input: bool, dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.self_attn = SelfAttention(dim, num_heads, eps)
        self.cross_attn = CrossAttention(
            dim, num_heads, eps, has_image_input=has_image_input)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(
            approximate='tanh'), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(self, x, context, t_mod, freqs, extra_context=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=1)
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa * self.self_attn(input_x, freqs)
        x = x + self.cross_attn(self.norm3(x), context)
        # Motion Cross-Attention (present after inject_motion_cross_attention)
        if hasattr(self, 'cross_attn2') and extra_context is not None:
            x = x + self.cross_attn2(self.norm4(x), extra_context)
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp * self.ffn(input_x)
        return x


class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, t_mod):
        shift, scale = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(2, dim=1)
        x = (self.head(self.norm(x) * (1 + scale) + shift))
        return x


class WanModel(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        has_image_input: bool,
        audio_hidden_size: int = 32,
    ):
        super().__init__()
        self.dim = dim
        self.freq_dim = freq_dim
        self.has_image_input = has_image_input
        self.patch_size = patch_size

        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList([
            DiTBlock(has_image_input, dim, num_heads, ffn_dim, eps)
            for _ in range(num_layers)
        ])
        self.head = Head(dim, out_dim, patch_size, eps)
        head_dim = dim // num_heads
        self.freqs = precompute_freqs_cis_3d(head_dim)

        # audio conditioning (OmniAvatar)
        audio_input_dim = 10752
        audio_out_dim = dim
        self.audio_proj = AudioPack(audio_input_dim, [4, 1, 1], audio_hidden_size, layernorm=True)
        self.audio_cond_projs = nn.ModuleList()
        for d in range(num_layers // 2 - 1):
            l = nn.Linear(audio_hidden_size, audio_out_dim)
            self.audio_cond_projs.append(l)

    def enable_keyframe_input(self):
        """Switch the backbone to the image-conditioned input layout
        (reference latent + mask + keyframe latents concatenated to the noise)."""
        self.has_image_input = True

    def init_motion_prior_modules(self, num_qwen_feature_token: int = 2, mllm_version: str = '3B', num_layers: int = 2):
        """Create the Q-Former that compresses Qwen2.5-VL features into motion prior tokens."""
        from .motion_prior import QWEN_HIDDEN_DIM, TransformerProjector
        self.Qformer_qwen_feature_proj = TransformerProjector(
            d_input=QWEN_HIDDEN_DIM[mllm_version],
            d_output=self.dim,
            n_head=8,
            num_layers=num_layers,
            dropout=0.1,
            num_token=num_qwen_feature_token)

    def patchify(self, x: torch.Tensor):
        grid_size = x.shape[2:]
        x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()
        return x, grid_size  # x, grid_size: (f, h, w)

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
            f=grid_size[0], h=grid_size[1], w=grid_size[2],
            x=self.patch_size[0], y=self.patch_size[1], z=self.patch_size[2]
        )

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        y_object: Optional[torch.Tensor] = None,
        audio_emb: Optional[torch.Tensor] = None,
        object_positions=None,
        num_hands_object_frames: int = 3,
        qwen_tokens: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Args:
            x: noisy video latents, (b, c, f, h, w).
            timestep: diffusion timestep, (b,).
            context: T5 text embedding, (b, l, text_dim).
            y: reference-image latent repeated over f frames + mask channel, (b, c+1, f, h, w).
            y_object: keyframe latents + mask channel, (b, c+1, K, h, w).
            audio_emb: wav2vec features, (b, seq_len, 10752).
            object_positions: frame index (in pixel-frame units) of each keyframe.
            num_hands_object_frames: number of keyframes K.
            qwen_tokens: Qwen2.5-VL hidden states of the keyframes, (b, n, d_mllm).
        Returns:
            predicted flow for the extended sequence, (b, c, K + f, h, w); the first K
            frames correspond to the keyframe slots and are discarded by the pipeline.
        """
        t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
        context = self.text_embedding(context)
        # motion prior tokens
        extra_context = self.Qformer_qwen_feature_proj(qwen_tokens)

        if self.has_image_input:
            b, c, _, h, w = x.shape
            x = torch.cat([x, y], dim=1)  # (b, c_x + c_y, f, h, w)
            if y_object is not None:
                single_random_tensor = torch.randn(b, c, 1, h, w)
                x_ahead_object_noise = single_random_tensor.expand(-1, -1, num_hands_object_frames, -1, -1).to(dtype=x.dtype, device=x.device)
                y_object = torch.cat([x_ahead_object_noise, y_object], dim=1)  # (b, c_x + c_y, K, h, w)
                # keyframes are prepended along the frame axis
                x = torch.cat([y_object, x], dim=2)  # (b, c_x + c_y, K + f, h, w)

        lat_h, lat_w = x.shape[-2], x.shape[-1]

        if audio_emb is not None:
            audio_emb = audio_emb.permute(0, 2, 1)[:, :, :, None, None]  # (1, 10752, seq_len, 1, 1)
            audio_emb = torch.cat([audio_emb[:, :, :1].repeat(1, 1, 3, 1, 1), audio_emb], 2)  # (1, 10752, seq_len+3, 1, 1)
            audio_emb = self.audio_proj(audio_emb)  # (1, (seq_len+3)/4, 1, 1, audio_hidden_size)
            audio_emb = torch.concat([audio_cond_proj(audio_emb) for audio_cond_proj in self.audio_cond_projs], 0)  # (num_layers//2 - 1, (seq_len+3)/4, 1, 1, dim)

        x = self.patch_embedding(x)
        x, (f, h, w) = self.patchify(x)

        freqs = torch.cat([
            self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)

        # TiRoPE: anchor each keyframe slot to the temporal position of its designated timestamp
        if y_object is not None:
            total_frames = f
            position_mapping = torch.zeros(total_frames, dtype=torch.long)
            # regular frames (after the K keyframe slots) keep their natural positions 0..f-K-1
            position_mapping[num_hands_object_frames:] = torch.arange(total_frames - num_hands_object_frames)
            if object_positions is not None:
                object_positions = torch.tensor(object_positions, device=position_mapping.device)
                # 4x temporal downsampling of the VAE
                downsampled_object_positions = object_positions // 4
                downsampled_object_positions = torch.clamp(downsampled_object_positions, 0, total_frames - num_hands_object_frames - 1)
                position_mapping[:num_hands_object_frames] = downsampled_object_positions

            custom_freqs = torch.cat([
                self.freqs[0][position_mapping].view(total_frames, 1, 1, -1).expand(total_frames, h, w, -1),
                self.freqs[1][:h].view(1, h, 1, -1).expand(total_frames, h, w, -1),
                self.freqs[2][:w].view(1, 1, w, -1).expand(total_frames, h, w, -1)
            ], dim=-1).reshape(total_frames * h * w, 1, -1).to(x.device)
            freqs = custom_freqs

        if audio_emb is not None:
            audio_emb = audio_emb.reshape(x.shape[0], audio_emb.shape[0] // x.shape[0], -1, *audio_emb.shape[2:])

        for layer_i, block in enumerate(self.blocks):
            # audio features are added to the video tokens (not the keyframe slots)
            # in the first half of the blocks, one projection per block
            if audio_emb is not None:
                if (layer_i <= len(self.blocks) // 2 and layer_i > 1):
                    au_idx = layer_i - 2
                    audio_emb_tmp = audio_emb[:, au_idx].repeat(1, 1, lat_h // 2, lat_w // 2, 1)  # (1, seq_len, lat_h//2, lat_w//2, dim)
                    audio_cond_tmp = self.patchify(audio_emb_tmp.permute(0, 4, 1, 2, 3))[0]
                    x = torch.cat([x[:, :num_hands_object_frames*h*w, :], audio_cond_tmp + x[:, num_hands_object_frames*h*w:, :]], dim=1)
            x = block(x, context, t_mod, freqs, extra_context)

        x = self.head(x, t)
        x = self.unpatchify(x, (f, h, w))
        return x

    @staticmethod
    def state_dict_converter():
        return WanModelStateDictConverter()


class WanModelStateDictConverter:
    """Detects the Wan2.1 architecture from the official checkpoint keys."""

    def __init__(self):
        pass

    def from_civitai(self, state_dict, model_config: Optional[dict] = None):
        if hash_state_dict_keys(state_dict) == "9269f8db9040a9d860eaca435be61814":
            # Wan2.1-T2V-1.3B
            config = {
                "has_image_input": False,
                "patch_size": [1, 2, 2],
                "in_dim": 16,
                "dim": 1536,
                "ffn_dim": 8960,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 12,
                "num_layers": 30,
                "eps": 1e-6
            }
        elif hash_state_dict_keys(state_dict) == "aafcfd9672c3a2456dc46e1cb6e52c70":
            # Wan2.1-T2V-14B
            config = {
                "has_image_input": False,
                "patch_size": [1, 2, 2],
                "in_dim": 16,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6
            }
        elif hash_state_dict_keys(state_dict) == "6bfcfb3b342cb286ce886889d519a77e":
            # Wan2.1-I2V-14B
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 36,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6
            }
        else:
            raise ValueError("Unrecognized Wan DiT checkpoint; expected Wan2.1-T2V-1.3B / T2V-14B / I2V-14B.")
        if model_config is not None:
            config.update(model_config)
        return state_dict, config


def inject_motion_cross_attention(
    model: WanModel,
    block_indices=None,
    cond_dim: int = 1536,
    num_heads: int = 12,
    eps: float = 1e-6,
):
    """Add the Motion Cross-Attention layer (``cross_attn2`` + ``norm4``) to the given
    DiT blocks. ``DiTBlock.forward`` uses it whenever ``extra_context`` is provided."""
    if block_indices is None:
        block_indices = list(range(len(model.blocks)))

    for idx in block_indices:
        block = model.blocks[idx]
        block.cross_attn2 = CrossAttention(
            dim=cond_dim,
            num_heads=num_heads,
            eps=eps,
            has_image_input=False,
        )
        block.norm4 = nn.LayerNorm(cond_dim, eps=eps)
