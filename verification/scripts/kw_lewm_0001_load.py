"""Load a LeWM model from the official HF config.json + weights.pt (MIT).

Generic `_target_`-based instantiation (hydra-like) of the nested config,
then load_state_dict from weights.pt. Used by KW-LEWM-0001 (LeWM vs JEPA-WM
Push-T comparison).

Supports hydra's `_partial_` flag: when `_partial_: true`, the target is
returned as a `functools.partial` (a callable that still needs its remaining
args, e.g. `norm_fn(hidden_dim)` -> `BatchNorm1d(hidden_dim)`) rather than
being instantiated immediately.

Usage:
    python kw_lewm_0001_load.py            # load + verify, print param count
"""
import functools
import importlib
import json
import pathlib
import warnings

import torch

warnings.filterwarnings("ignore")

ROOT = pathlib.Path(__file__).resolve().parents[2]
CKPT_DIR = ROOT / "checkpoints" / "lewm"
CONFIG = CKPT_DIR / "config.json"
WEIGHTS = CKPT_DIR / "weights.pt"


def _instantiate(cfg):
    """Hydra-like recursive instantiation from {_target_: ..., **params} dicts."""
    if isinstance(cfg, list):
        return [_instantiate(c) for c in cfg]
    if not isinstance(cfg, dict):
        return cfg
    if "_target_" in cfg:
        target = cfg["_target_"]
        mod_name, _, cls_name = target.rpartition(".")
        mod = importlib.import_module(mod_name)
        cls = getattr(mod, cls_name)
        kwargs = {
            k: _instantiate(v)
            for k, v in cfg.items()
            if k not in ("_target_", "_partial_")
        }
        is_partial = bool(cfg.get("_partial_", False))
        if is_partial:
            # Return a callable that still needs its remaining args:
            # e.g. partial(BatchNorm1d) -> called as norm_fn(hidden_dim).
            return functools.partial(cls, **kwargs)
        return cls(**kwargs)
    return {k: _instantiate(v) for k, v in cfg.items()}


def load_lewm(device="cuda", checkpoint=None, config=None):
    """Construct LeWM from config.json and load weights.pt.

    Returns (model, config_dict). model is on `device`, in eval mode.
    """
    config = config or json.loads(CONFIG.read_text(encoding="utf-8"))
    ckpt = checkpoint or WEIGHTS

    model = _instantiate(config)
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(sd)
    model = model.to(device).eval()

    return model, config


if __name__ == "__main__":
    m, cfg = load_lewm(device="cuda")
    n_params = sum(p.numel() for p in m.parameters())
    n_train = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"LeWM loaded OK")
    print(f"  config keys: {list(cfg.keys())}")
    print(f"  total params: {n_params:,} ({n_params/1e6:.1f}M)")
    print(f"  trainable params: {n_train:,} ({n_train/1e6:.1f}M)")
    print(f"  device: {next(m.parameters()).device}")
    print(f"  dtype: {next(m.parameters()).dtype}")
