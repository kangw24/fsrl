"""Strict checkpoint IO and provenance manifests for the source model."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from fsrl.miconi_kay.source.config import MiconiKaySourceConfig
from fsrl.miconi_kay.source.model import MiconiKayRetroModulRNN


SOURCE_STATE_KEYS = frozenset(
    {
        "w",
        "alpha",
        "etaet",
        "DAmult",
        "i2h.weight",
        "i2h.bias",
        "h2DA.weight",
        "h2DA.bias",
        "h2o.weight",
        "h2o.bias",
        "h2v.weight",
        "h2v.bias",
    }
)

PUBLIC_CHECKPOINT_SHA256 = {
    "net_active.dat": "aac06db04f457d98bc49b44cfbea23648721943fd6dc752c13703e188532940a",
    "net_passive.dat": "3726423c7eb3a4b7e994e4480dc28bbbd998f6ace39124c18449b5c1cb8336f4",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_tree_sha256() -> str:
    """Hash the isolated source package with stable relative-name framing."""

    directory = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(directory.glob("*.py"), key=lambda value: value.name):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _torch_load(path: Path, device: torch.device) -> Any:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def validate_source_state_dict(state_dict: dict[str, torch.Tensor]) -> None:
    actual = set(state_dict)
    missing = sorted(SOURCE_STATE_KEYS - actual)
    unexpected = sorted(actual - SOURCE_STATE_KEYS)
    if missing or unexpected:
        raise ValueError(
            "M&K source checkpoint key mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )


def load_source_checkpoint_strict(
    model: MiconiKayRetroModulRNN, path: str | Path
) -> str:
    checkpoint_path = Path(path)
    payload = _torch_load(checkpoint_path, model.w.device)
    if not isinstance(payload, dict) or not all(
        isinstance(value, torch.Tensor) for value in payload.values()
    ):
        raise ValueError("M&K source checkpoint must be a plain tensor state_dict")
    validate_source_state_dict(payload)
    model.load_state_dict(payload, strict=True)
    return sha256_file(checkpoint_path)


def verify_public_checkpoint_identity(path: str | Path, role: str) -> str:
    """Verify that a file is the authors' published active/passive checkpoint."""

    if role not in {"active", "passive"}:
        raise ValueError("role must be 'active' or 'passive'")
    checkpoint_path = Path(path)
    expected_name = f"net_{role}.dat"
    expected = PUBLIC_CHECKPOINT_SHA256[expected_name]
    actual = sha256_file(checkpoint_path)
    if actual != expected:
        raise ValueError(
            f"{role} checkpoint SHA-256 mismatch: expected={expected}, actual={actual}"
        )
    return actual


def save_source_checkpoint(
    model: MiconiKayRetroModulRNN,
    config: MiconiKaySourceConfig,
    output_dir: str | Path,
    *,
    episode_index: int,
    test_rewards: list[float],
) -> Path:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    state_dict = model.state_dict()
    validate_source_state_dict(state_dict)
    checkpoint_path = directory / "net.dat"
    seeded_path = directory / f"netAE{config.rng_seed}.dat"
    torch.save(state_dict, checkpoint_path)
    torch.save(state_dict, seeded_path)

    reward_path = directory / f"tAE{config.rng_seed}.txt"
    reward_path.write_text(
        "".join(f"{value}\n" for value in test_rewards[::10]),
        encoding="utf-8",
    )
    manifest = {
        "claim_status": "source reproduction checkpoint; active/passive identity unproven",
        "episode_index": episode_index,
        "config": config.to_manifest_dict(),
        "checkpoint_sha256": {
            checkpoint_path.name: sha256_file(checkpoint_path),
            seeded_path.name: sha256_file(seeded_path),
        },
        "source_tree_sha256": source_tree_sha256(),
        "reward_log": reward_path.name,
    }
    (directory / "checkpoint_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return checkpoint_path
