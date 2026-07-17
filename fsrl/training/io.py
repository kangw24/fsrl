import hashlib
import json
from pathlib import Path

import torch

from fsrl.device import DEVICE, ROOT_DIR


def resolve_model_path(model_path=None):
    """解析模型权重路径（net.dat 或 net_active.dat）。"""
    if model_path:
        path = Path(model_path)
        return path if path.is_absolute() else ROOT_DIR / path
    for candidate in ("net.dat", "net_active.dat"):
        path = ROOT_DIR / candidate
        if path.exists():
            return path
    raise FileNotFoundError(
        "未找到模型文件。请在项目根目录放置 net.dat 或 net_active.dat，"
        "或通过 --model-path 指定。"
    )


def load_torch_payload(path: Path):
    try:
        return torch.load(path, map_location=DEVICE, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=DEVICE)


def load_model_state(path):
    manifest_path = Path(path).parent / "checkpoint_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = manifest.get("checkpoint_sha256", {}).get(Path(path).name)
        if expected is not None:
            digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
            if digest != expected:
                raise ValueError(
                    f"Checkpoint hash mismatch for {path}: expected {expected}, got {digest}"
                )
    try:
        return torch.load(path, map_location=DEVICE, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=DEVICE)
