# Adapted from DiffSynth-Studio (https://github.com/modelscope/DiffSynth-Studio), Apache-2.0.
import hashlib
import os
from contextlib import contextmanager
from typing import List, Optional, Union

import torch
from safetensors import safe_open


@contextmanager
def init_weights_on_device(device=torch.device("meta"), include_buffers: bool = False):

    old_register_parameter = torch.nn.Module.register_parameter
    if include_buffers:
        old_register_buffer = torch.nn.Module.register_buffer

    def register_empty_parameter(module, name, param):
        old_register_parameter(module, name, param)
        if param is not None:
            param_cls = type(module._parameters[name])
            kwargs = module._parameters[name].__dict__
            kwargs["requires_grad"] = param.requires_grad
            module._parameters[name] = param_cls(module._parameters[name].to(device), **kwargs)

    def register_empty_buffer(module, name, buffer, persistent=True):
        old_register_buffer(module, name, buffer, persistent=persistent)
        if buffer is not None:
            module._buffers[name] = module._buffers[name].to(device)

    def patch_tensor_constructor(fn):
        def wrapper(*args, **kwargs):
            kwargs["device"] = device
            return fn(*args, **kwargs)

        return wrapper

    if include_buffers:
        tensor_constructors_to_patch = {
            torch_function_name: getattr(torch, torch_function_name)
            for torch_function_name in ["empty", "zeros", "ones", "full"]
        }
    else:
        tensor_constructors_to_patch = {}

    try:
        torch.nn.Module.register_parameter = register_empty_parameter
        if include_buffers:
            torch.nn.Module.register_buffer = register_empty_buffer
        for torch_function_name in tensor_constructors_to_patch.keys():
            setattr(torch, torch_function_name, patch_tensor_constructor(getattr(torch, torch_function_name)))
        yield
    finally:
        torch.nn.Module.register_parameter = old_register_parameter
        if include_buffers:
            torch.nn.Module.register_buffer = old_register_buffer
        for torch_function_name, old_torch_function in tensor_constructors_to_patch.items():
            setattr(torch, torch_function_name, old_torch_function)


def load_state_dict(file_path, torch_dtype=None):
    if file_path.endswith(".safetensors"):
        return load_state_dict_from_safetensors(file_path, torch_dtype=torch_dtype)
    else:
        return load_state_dict_from_bin(file_path, torch_dtype=torch_dtype)


def load_state_dict_from_safetensors(file_path, torch_dtype=None):
    state_dict = {}
    with safe_open(file_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            state_dict[k] = f.get_tensor(k)
            if torch_dtype is not None:
                state_dict[k] = state_dict[k].to(torch_dtype)
    return state_dict


def load_state_dict_from_bin(file_path, torch_dtype=None):
    state_dict = torch.load(file_path, map_location="cpu", weights_only=True)
    if torch_dtype is not None:
        for i in state_dict:
            if isinstance(state_dict[i], torch.Tensor):
                state_dict[i] = state_dict[i].to(torch_dtype)
    return state_dict


def convert_state_dict_keys_to_single_str(state_dict, with_shape=True):
    keys = []
    for key, value in state_dict.items():
        if isinstance(key, str):
            if isinstance(value, torch.Tensor):
                if with_shape:
                    shape = "_".join(map(str, list(value.shape)))
                    keys.append(key + ":" + shape)
                keys.append(key)
            elif isinstance(value, dict):
                keys.append(key + "|" + convert_state_dict_keys_to_single_str(value, with_shape=with_shape))
    keys.sort()
    keys_str = ",".join(keys)
    return keys_str


def hash_state_dict_keys(state_dict, with_shape=True):
    keys_str = convert_state_dict_keys_to_single_str(state_dict, with_shape=with_shape)
    keys_str = keys_str.encode(encoding="UTF-8")
    return hashlib.md5(keys_str).hexdigest()


def _load_weights_with_shape_padding(model, ckpt_state_dict):
    """Load weights whose shape may be smaller than the model's (e.g. the 16-channel
    Wan2.1 patch embedding into the 33-channel SparseCtrl-HOI patch embedding)."""
    model_state_dict = model.state_dict()
    new_state_dict = {}
    for name, param in model_state_dict.items():
        if name in ckpt_state_dict:
            ckpt_param = ckpt_state_dict[name]
            if param.shape == ckpt_param.shape:
                new_state_dict[name] = ckpt_param
            elif all(p >= c for p, c in zip(param.shape, ckpt_param.shape)):
                print(f"[Pad] {name}: ckpt {tuple(ckpt_param.shape)} -> model {tuple(param.shape)}")
                new_param = param.clone()
                slices = tuple(slice(0, s) for s in ckpt_param.shape)
                new_param[slices] = ckpt_param
                new_state_dict[name] = new_param
            else:
                print(f"[Skip] {name}: ckpt {tuple(ckpt_param.shape)} is larger than model {tuple(param.shape)}")
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, assign=True, strict=False)
    return model, missing_keys, unexpected_keys


def load_wan_dit(dit_path: Union[str, List[str]], model_config: Optional[dict] = None, torch_dtype=torch.bfloat16, device="cpu"):
    """Build the Wan2.1 DiT backbone from the official checkpoint(s).

    Args:
        dit_path: path of the DiT safetensors file, or a list of shard paths (14B).
        model_config: overrides applied on top of the architecture detected from the
            checkpoint, e.g. ``{"in_dim": 33}`` for the audio-conditioned backbone.
    """
    from .wan_dit import WanModel, WanModelStateDictConverter

    if isinstance(dit_path, str):
        dit_path = [dit_path]
    state_dict = {}
    for path in dit_path:
        state_dict.update(load_state_dict(path))
    state_dict, config = WanModelStateDictConverter().from_civitai(state_dict, model_config=model_config)
    print(f"Wan DiT config: {config}")
    with init_weights_on_device():
        model = WanModel(**config)
    model = model.eval().to_empty(device=device)
    model, missing_keys, unexpected_keys = _load_weights_with_shape_padding(model, state_dict)
    print(f"Wan DiT loaded: {len(missing_keys)} missing keys, {len(unexpected_keys)} unexpected keys.")
    return model.to(dtype=torch_dtype)


def load_wan_vae(vae_path: str, torch_dtype=torch.bfloat16):
    from .wan_vae import WanVideoVAE

    state_dict = load_state_dict(vae_path)
    state_dict = WanVideoVAE.state_dict_converter().from_civitai(state_dict)
    with init_weights_on_device():
        model = WanVideoVAE()
    model = model.eval()
    model.load_state_dict(state_dict, assign=True)
    return model.to(dtype=torch_dtype)
