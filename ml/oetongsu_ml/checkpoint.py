from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from .alphazero_model import AlphaZeroNet


def create_version_id(number: int) -> str:
    return f"az_v{number:04d}"


def next_version_id(checkpoint_dir: str | Path) -> str:
    directory = Path(checkpoint_dir)
    existing = []
    if directory.exists():
        for path in directory.glob("az_v*.pt"):
            try:
                existing.append(int(path.stem.replace("az_v", "")))
            except ValueError:
                continue
    return create_version_id((max(existing) if existing else 0) + 1)


def save_checkpoint(
    model: AlphaZeroNet,
    checkpoint_dir: str | Path,
    metadata: dict[str, Any] | None = None,
    version: str | None = None,
) -> tuple[Path, Path]:
    directory = Path(checkpoint_dir)
    directory.mkdir(parents=True, exist_ok=True)
    version_id = version or next_version_id(directory)
    model_path = directory / f"{version_id}.pt"
    metadata_path = directory / f"{version_id}.json"
    payload = {
        "model_state": model.state_dict(),
        "channels": model.channels,
        "version": version_id,
    }
    torch.save(payload, model_path)
    full_metadata = {
        "version": version_id,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        **(metadata or {}),
    }
    metadata_path.write_text(json.dumps(full_metadata, indent=2), encoding="utf-8")
    return model_path, metadata_path


def load_checkpoint(model_path: str | Path, device: str | torch.device = "cpu") -> tuple[AlphaZeroNet, dict[str, Any]]:
    checkpoint = torch.load(model_path, map_location=device)
    model = AlphaZeroNet(channels=int(checkpoint.get("channels", 64))).to(device)
    model.load_state_dict(checkpoint["model_state"])
    metadata_path = Path(model_path).with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    return model, metadata
