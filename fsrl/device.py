import os
from pathlib import Path

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
_requested_device = os.environ.get("FSRL_DEVICE")
if _requested_device is None:
    _requested_device = "cuda" if torch.cuda.is_available() else "cpu"
if _requested_device not in {"cpu", "cuda"}:
    raise ValueError("FSRL_DEVICE must be 'cpu' or 'cuda'")
if _requested_device == "cuda" and not torch.cuda.is_available():
    raise RuntimeError("CUDA was requested but is not available")
DEVICE = torch.device(_requested_device)


def log(message):
    print(message, flush=True)


def set_seed(seed):
    if seed < 0:
        log("[setup] No random seed.")
        return
    log(f"[setup] Setting random seed {seed}")
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
