"""Motion Prior Injection Module.

A frozen Qwen2.5-VL reads the sparse interaction keyframes and a task prompt; the
hidden states of an intermediate layer are compressed by a Q-Former
(``TransformerProjector``) into a fixed number of motion prior tokens, which the
DiT consumes through Motion Cross-Attention.
"""
from typing import List

import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

MOTION_PRIOR_PROMPT = """
    We are working on the hand-object interaction video generation task, where one person is introducing a product using hands in each generated video. To faciliate generation, we have provided four keyframes in the temporal order. We want you to infer the complete hand-object interaction process based on the four keyframes. In particular, we want you to infer the change of hand pose, object pose, the hand-object contact region in this process as priors for the video generation."""

# hidden layer whose states are used as motion prior features, per Qwen2.5-VL size
QWEN_SELECT_LAYER = {"7B": 28, "3B": 36}
QWEN_HIDDEN_DIM = {"7B": 3584, "3B": 2048}
QWEN_MODEL_NAME = {"7B": "Qwen/Qwen2.5-VL-7B-Instruct", "3B": "Qwen/Qwen2.5-VL-3B-Instruct"}


class TransformerProjector(nn.Module):
    """
    Transformer-based VLM projector that uses cross-attention to aggregate
    VLM token embeddings into a fixed number of condition tokens.
    """
    def __init__(self, d_input=2048, d_output=512, n_head=8, num_layers=2, dropout=0.1, num_token=2):
        super().__init__()

        assert d_output % n_head == 0, f"d_output ({d_output}) must be divisible by n_head ({n_head})"

        # Learnable query tokens that serve as information aggregation points
        self.query_token = nn.Parameter(torch.randn(1, num_token, d_output))

        # Project VLM embeddings to internal dimension
        self.input_projection = nn.Linear(d_input, d_output)
        self.input_norm = nn.LayerNorm(d_output)

        # Cross-attention transformer layers
        self.transformer_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=d_output,
                nhead=n_head,
                dim_feedforward=d_output * 2,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True
            ) for _ in range(num_layers)
        ])

        self.use_pos_encoding = False
        if self.use_pos_encoding:
            self.pos_encoding = nn.Parameter(torch.randn(1, 5000, d_output))

    def forward(self, vlm_embedding, vlm_mask=None):
        """
        Args:
            vlm_embedding: (BS, seq_len, d_input) - VLM token embeddings
            vlm_mask: (BS, seq_len) - 1 for real tokens, 0 for padding

        Returns:
            motion prior tokens: (BS, num_token, d_output)
        """
        bs, seq_len = vlm_embedding.shape[:2]

        vlm_projected = self.input_projection(vlm_embedding)  # (BS, seq_len, d_output)
        vlm_projected = self.input_norm(vlm_projected)

        if self.use_pos_encoding:
            vlm_projected = vlm_projected + self.pos_encoding[:, :seq_len, :]

        query = self.query_token.repeat(bs, 1, 1)  # (BS, num_token, d_output)

        if vlm_mask is not None:
            memory_key_padding_mask = ~vlm_mask.bool()  # (BS, seq_len)
        else:
            memory_key_padding_mask = None

        for layer in self.transformer_layers:
            query = layer(
                tgt=query,
                memory=vlm_projected,
                memory_key_padding_mask=memory_key_padding_mask
            )

        return query


def load_qwen_vl(device, mllm_version: str = "7B", model_name_or_path: str = None):
    """Load the frozen Qwen2.5-VL model and processor (flash-attention-2, bf16).

    ``model_name_or_path`` may be a local directory; by default the HF hub id for
    ``mllm_version`` is used."""
    if model_name_or_path is None:
        model_name_or_path = QWEN_MODEL_NAME[mllm_version]
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation='flash_attention_2',
        device_map=device)
    processor = AutoProcessor.from_pretrained(model_name_or_path, use_fast=True)
    model.eval()
    model.requires_grad_(False)
    return model, processor


def build_interleaved_messages(prompt, image_paths):
    """Interleave the keyframes with short captions, followed by the task prompt."""
    content = []
    for i, image_path in enumerate(image_paths):
        content.append({"type": "image", "image": f"file://{image_path}"})
        if i == 0:
            content.append({"type": "text", "text": f"This is start frame {i+1}。"})
        elif i == len(image_paths) - 1:
            content.append({"type": "text", "text": f"This is end frame {i+1}。"})
        else:
            content.append({"type": "text", "text": f"This is keyframe {i+1}。"})
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


@torch.no_grad()
def extract_motion_prior_features(model, processor, keyframes: List[Image.Image], mllm_version: str = "7B", prompt: str = MOTION_PRIOR_PROMPT):
    """Run Qwen2.5-VL on the keyframes and return the hidden states of the selected layer,
    shape (1, seq_len, d_mllm)."""
    fake_img_paths = [f'fake_path_{i}' for i in range(len(keyframes))]
    messages = build_interleaved_messages(prompt, fake_img_paths)
    text_inputs = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text_inputs],
                       images=keyframes, videos=None,
                       padding=True, return_tensors="pt").to(model.device)
    outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states[QWEN_SELECT_LAYER[mllm_version]]
    return hidden_states
