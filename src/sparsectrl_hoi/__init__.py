"""SparseCtrl-HOI: Sparse Temporal Control for Human-Object Interaction Video Generation."""
from .models import (MOTION_PRIOR_PROMPT, WanModel, WanVideoVAE, extract_motion_prior_features, load_qwen_vl,
                     load_wan_dit, load_wan_vae)
from .pipelines import SparseCtrlHOIPipeline

__version__ = "0.1.0"

__all__ = [
    "MOTION_PRIOR_PROMPT", "SparseCtrlHOIPipeline", "WanModel", "WanVideoVAE",
    "extract_motion_prior_features", "load_qwen_vl", "load_wan_dit", "load_wan_vae",
]
