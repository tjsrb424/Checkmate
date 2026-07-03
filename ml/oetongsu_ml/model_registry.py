from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

ModelStatus = Literal["candidate", "promoted", "rejected"]


def empty_registry() -> dict[str, Any]:
    return {"models": []}


def load_registry(path: str | Path) -> dict[str, Any]:
    registry_path = Path(path)
    if not registry_path.exists():
        return empty_registry()
    return json.loads(registry_path.read_text(encoding="utf-8"))


def save_registry(path: str | Path, registry: dict[str, Any]) -> None:
    registry_path = Path(path)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")


def register_candidate(
    registry: dict[str, Any],
    version: str,
    path: str,
    metadata_path: str | None = None,
    parent_version: str | None = None,
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    existing = find_entry(registry, version)
    if existing is not None:
        return existing
    entry = {
        "version": version,
        "path": path,
        "metadataPath": metadata_path,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "status": "candidate",
        "parentVersion": parent_version,
        "metrics": metrics or {},
        "arenaResults": [],
    }
    registry.setdefault("models", []).append(entry)
    return entry


def get_latest_promoted(registry: dict[str, Any]) -> dict[str, Any] | None:
    promoted = [entry for entry in registry.get("models", []) if entry.get("status") == "promoted"]
    if not promoted:
        return None
    return sorted(promoted, key=lambda entry: entry.get("createdAt", ""))[-1]


def promote_candidate(registry: dict[str, Any], version: str, arena_result: dict[str, Any] | None = None) -> dict[str, Any]:
    entry = require_entry(registry, version)
    entry["status"] = "promoted"
    if arena_result is not None:
        entry.setdefault("arenaResults", []).append(arena_result)
    return entry


def reject_candidate(registry: dict[str, Any], version: str, arena_result: dict[str, Any] | None = None) -> dict[str, Any]:
    entry = require_entry(registry, version)
    entry["status"] = "rejected"
    if arena_result is not None:
        entry.setdefault("arenaResults", []).append(arena_result)
    return entry


def find_entry(registry: dict[str, Any], version: str) -> dict[str, Any] | None:
    for entry in registry.get("models", []):
        if entry.get("version") == version:
            return entry
    return None


def require_entry(registry: dict[str, Any], version: str) -> dict[str, Any]:
    entry = find_entry(registry, version)
    if entry is None:
        raise KeyError(f"model version not found: {version}")
    return entry
