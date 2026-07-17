"""Fetch the small public Liu et al. behavior release with pinned hashes."""

from __future__ import annotations

import argparse
import hashlib
import os
import urllib.request
from pathlib import Path


FILES = {
    "preregistered_experiment_data.csv": (
        "https://osf.io/download/mjqpe/",
        "6dcae48511018a85765e3ab7ceed6f358f5185f5ce399ce47543dbc7aad0c227",
    ),
    "replication_experiment_data.csv": (
        "https://osf.io/download/698e858121336f1da3c72b17/",
        "c322cedd587d8e119442f873679d5fd5315e31736f4bf4266dbb89849c068249",
    ),
    "OSF_README.md": (
        "https://osf.io/download/yn6u2/",
        "b1b3b3ce77ef9ee4445e0f54effa6f03dc410ada83c86c8d9e29dad3f534aa7a",
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/liu2026_human_audit/raw"),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for name, (url, expected_hash) in FILES.items():
        destination = args.output_dir / name
        if destination.exists() and _sha256(destination) == expected_hash:
            print(f"verified existing: {destination}")
            continue
        if destination.exists() and not args.force:
            raise RuntimeError(
                f"refusing to overwrite hash-mismatched file without --force: {destination}"
            )
        temporary = destination.with_suffix(destination.suffix + ".download")
        try:
            urllib.request.urlretrieve(url, temporary)
            observed_hash = _sha256(temporary)
            if observed_hash != expected_hash:
                raise RuntimeError(
                    f"SHA-256 mismatch for {name}: {observed_hash} != {expected_hash}"
                )
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()
        print(f"downloaded and verified: {destination}")


if __name__ == "__main__":
    main()
