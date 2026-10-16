#!/usr/bin/env bash
# SparseCtrl-HOI 14B inference on the example clips.
set -euo pipefail

CKPT=checkpoints
WAN=$CKPT/Wan2.1-T2V-14B
DIT_PATH=$WAN/diffusion_pytorch_model-00001-of-00006.safetensors,$WAN/diffusion_pytorch_model-00002-of-00006.safetensors,$WAN/diffusion_pytorch_model-00003-of-00006.safetensors,$WAN/diffusion_pytorch_model-00004-of-00006.safetensors,$WAN/diffusion_pytorch_model-00005-of-00006.safetensors,$WAN/diffusion_pytorch_model-00006-of-00006.safetensors

python scripts/inference/inference.py \
  --test_list assets/examples/test_list.txt \
  --output_path outputs/examples \
  --dit_path "$DIT_PATH" \
  --vae_path $WAN/Wan2.1_VAE.pth \
  --pretrained_lora_path $CKPT/OmniAvatar-14B/pytorch_model.pt \
  --wav2vec_path $CKPT/wav2vec2-base-960h \
  --stage1_weights $CKPT/SparseCtrl-HOI-14B/stage1.ckpt \
  --stage2_weights $CKPT/SparseCtrl-HOI-14B/stage2.ckpt \
  --caption_emb_path $CKPT/SparseCtrl-HOI-14B/prompt_emb.pt \
  --caption_nega_emb_path $CKPT/SparseCtrl-HOI-14B/prompt_emb_negative.pt \
  --qwen_path $CKPT/Qwen2.5-VL-7B-Instruct \
  --num_persistent_param_in_dit 6000000000
