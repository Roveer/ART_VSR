from pathlib import Path

import torch
from safetensors import safe_open

from artvsr.ate import AdaptiveTimestepEstimator
from artvsr.ltr import LatentTrajectoryRectifier
from artvsr.wan.utils import init_weights_on_device
from artvsr.wan.wan_video_dit import WanModel
from artvsr.wan.wan_video_vae import WanVideoVAE


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "weights" / "artvsr_full.safetensors"


def select_prefix(state_dict, prefix):
    start = len(prefix)
    return {key[start:]: value for key, value in state_dict.items() if key.startswith(prefix)}


def load_safetensors(path, torch_dtype=None):
    tensors = {}
    with safe_open(str(path), framework="pt", device="cpu") as f:
        for key in f.keys():
            value = f.get_tensor(key)
            if torch_dtype is not None and value.is_floating_point():
                value = value.to(torch_dtype)
            tensors[key] = value
    return tensors


def build_wan_dit(dit_state_dict, torch_dtype, device):
    with init_weights_on_device():
        model = WanModel(
            dim=1536,
            in_dim=16,
            ffn_dim=8960,
            out_dim=16,
            text_dim=4096,
            freq_dim=256,
            eps=1e-6,
            patch_size=(1, 2, 2),
            num_heads=12,
            num_layers=30,
        )
    model.load_state_dict(dit_state_dict, assign=True)
    return model.eval().to(dtype=torch_dtype, device=device)


def build_wan_vae(vae_state_dict, torch_dtype, device):
    model = WanVideoVAE()
    model.load_state_dict(vae_state_dict, strict=False, assign=True)
    return model.eval().to(dtype=torch_dtype, device=device)


def load_artvsr_checkpoint(pipe, checkpoint_path):
    state_dict = load_safetensors(checkpoint_path, torch_dtype=pipe.torch_dtype)
    dit_state_dict = select_prefix(state_dict, "dit.")
    vae_state_dict = select_prefix(state_dict, "vae.")
    ate_state_dict = select_prefix(state_dict, "ate.")
    ltr_state_dict = select_prefix(state_dict, "ltr.")
    if not dit_state_dict:
        raise ValueError("Missing dit.* tensors in checkpoint.")
    if not vae_state_dict:
        raise ValueError("Missing vae.* tensors in checkpoint.")
    if not ate_state_dict:
        raise ValueError("Missing ate.* tensors in checkpoint.")
    if not ltr_state_dict:
        raise ValueError("Missing ltr.* tensors in checkpoint.")
    pipe.dit = build_wan_dit(dit_state_dict, pipe.torch_dtype, "cpu")
    pipe.vae = build_wan_vae(vae_state_dict, pipe.torch_dtype, "cpu")
    pipe.ate = AdaptiveTimestepEstimator().to(device=pipe.device, dtype=pipe.torch_dtype)
    pipe.ltr = LatentTrajectoryRectifier().to(device=pipe.device, dtype=pipe.torch_dtype)
    pipe.prompt_emb_read = state_dict.get("prompt_emb")
    if pipe.prompt_emb_read is None:
        raise ValueError("Missing prompt_emb tensor in checkpoint.")
    pipe.prompt_emb_read = pipe.prompt_emb_read.to(dtype=pipe.torch_dtype)
    pipe.ate.load_state_dict(ate_state_dict, strict=True)
    pipe.ltr.load_state_dict(ltr_state_dict, strict=True)
    pipe.height_division_factor = pipe.vae.upsampling_factor * 2
    pipe.width_division_factor = pipe.vae.upsampling_factor * 2
    return pipe
