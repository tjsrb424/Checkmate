from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

ProgressStatus = Literal["idle", "running", "completed", "failed"]
ProgressPhase = Literal["selfplay", "train", "arena", "package", "completed", "failed"]

PHASE_WEIGHTS = {
    "selfplay": 0.45,
    "train": 0.30,
    "arena": 0.20,
    "package": 0.05,
    "completed": 1.0,
    "failed": 1.0,
}

STATUS_LABELS_KO = {
    "idle": "대기 중",
    "running": "진행 중",
    "completed": "완료",
    "failed": "실패",
}

PHASE_LABELS_KO = {
    "selfplay": "자기대국 생성",
    "train": "후보 학습",
    "arena": "승격 대국",
    "package": "결과 정리",
    "completed": "완료",
    "failed": "실패",
}


@dataclass
class ProgressSnapshot:
    job: str = "autotrain"
    runId: str | None = None
    status: ProgressStatus = "idle"
    statusLabelKo: str = STATUS_LABELS_KO["idle"]
    phaseKey: ProgressPhase = "selfplay"
    phaseLabelKo: str = PHASE_LABELS_KO["selfplay"]
    message: str = ""
    messageKo: str = ""
    overallPercent: float = 0.0
    phasePercent: float = 0.0
    progressAccuracy: str = "exact_phase_progress"
    startedAt: str | None = None
    updatedAt: str | None = None
    elapsedSeconds: int = 0
    elapsedText: str = "00:00:00"
    etaSeconds: int | None = None
    etaText: str | None = None
    iteration: dict[str, int] = field(default_factory=dict)
    selfPlay: dict[str, Any] = field(default_factory=dict)
    training: dict[str, Any] = field(default_factory=dict)
    arena: dict[str, Any] = field(default_factory=dict)
    models: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] = field(default_factory=dict)


class ProgressReporter:
    def __init__(
        self,
        progress_path: str | Path,
        events_path: str | Path,
        run_id: str,
        total_iterations: int,
    ) -> None:
        self.progress_path = Path(progress_path)
        self.events_path = Path(events_path)
        self.run_id = run_id
        self.total_iterations = max(1, int(total_iterations))
        self.started_at = utc_now()
        self.started_at_unix = datetime.now(timezone.utc).timestamp()
        self.phase_progress = {"selfplay": 0.0, "train": 0.0, "arena": 0.0, "package": 0.0}
        self.last_snapshot: ProgressSnapshot | None = None

    def reset_phase_progress(self) -> None:
        self.phase_progress = {"selfplay": 0.0, "train": 0.0, "arena": 0.0, "package": 0.0}

    def start_iteration(self) -> None:
        self.reset_phase_progress()

    def update(
        self,
        *,
        status: ProgressStatus = "running",
        phase: ProgressPhase,
        phase_percent: float,
        message: str,
        message_ko: str,
        current_iteration: int,
        completed_iterations: int,
        self_play: dict[str, Any] | None = None,
        training: dict[str, Any] | None = None,
        arena: dict[str, Any] | None = None,
        models: dict[str, Any] | None = None,
        result: dict[str, Any] | None = None,
        progress_accuracy: str = "exact_phase_progress",
    ) -> ProgressSnapshot:
        if phase in self.phase_progress:
            self.phase_progress[phase] = clamp_percent(phase_percent)
        if phase == "completed":
            for key in self.phase_progress:
                self.phase_progress[key] = 100.0
        if phase == "failed":
            self.phase_progress["package"] = max(self.phase_progress["package"], clamp_percent(phase_percent))

        overall = self.overall_percent(completed_iterations, phase)
        if status == "completed":
            overall = 100.0
        else:
            overall = min(overall, 99.9)
        phase_value = 100.0 if phase == "completed" else clamp_percent(phase_percent)
        now = utc_now()
        elapsed = max(0, int(datetime.now(timezone.utc).timestamp() - self.started_at_unix))
        eta = estimate_eta_seconds(elapsed, overall, status)
        snapshot = ProgressSnapshot(
            runId=self.run_id,
            status=status,
            statusLabelKo=STATUS_LABELS_KO[status],
            phaseKey=phase,
            phaseLabelKo=PHASE_LABELS_KO[phase],
            message=message,
            messageKo=message_ko,
            overallPercent=round(clamp_percent(overall), 1),
            phasePercent=round(phase_value, 1),
            progressAccuracy=progress_accuracy,
            startedAt=self.started_at,
            updatedAt=now,
            elapsedSeconds=elapsed,
            elapsedText=format_duration(elapsed),
            etaSeconds=eta,
            etaText=format_duration(eta) if eta is not None else None,
            iteration={
                "current": max(1, int(current_iteration)),
                "total": self.total_iterations,
                "completed": max(0, int(completed_iterations)),
            },
            selfPlay=self_play or {},
            training=training or {},
            arena=arena or {},
            models=models or {},
            result=result or {},
        )
        self.write_progress(snapshot)
        self.append_event(
            {
                "time": now,
                "status": status,
                "phaseKey": phase,
                "overallPercent": snapshot.overallPercent,
                "phasePercent": snapshot.phasePercent,
                "message": message,
                "messageKo": message_ko,
            }
        )
        self.last_snapshot = snapshot
        return snapshot

    def mark_failed(self, error: BaseException, current_iteration: int, completed_iterations: int) -> ProgressSnapshot:
        return self.update(
            status="failed",
            phase="failed",
            phase_percent=100,
            message=f"AutoTrain failed: {error}",
            message_ko=f"AutoTrain이 실패했습니다: {error}",
            current_iteration=current_iteration,
            completed_iterations=completed_iterations,
            progress_accuracy="terminal_status",
            result={"status": "failed", "error": str(error)},
        )

    def overall_percent(self, completed_iterations: int, phase: ProgressPhase) -> float:
        if phase == "completed":
            return 100.0
        current = sum(self.phase_progress[key] * PHASE_WEIGHTS[key] for key in self.phase_progress)
        return ((completed_iterations + current / 100) / self.total_iterations) * 100

    def write_progress(self, snapshot: ProgressSnapshot) -> None:
        write_progress(snapshot, self.progress_path)

    def append_event(self, event: dict[str, Any]) -> None:
        append_event(event, self.events_path)


def write_progress(snapshot: ProgressSnapshot, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(asdict(snapshot), ensure_ascii=False, indent=2)
    tmp = target.with_name(f"{target.name}.tmp")
    tmp.write_text(payload, encoding="utf-8")
    os.replace(tmp, target)


def append_event(event: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as file:
        file.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")


def clamp_percent(value: float | int | None) -> float:
    if value is None:
        return 0.0
    return max(0.0, min(100.0, float(value)))


def format_duration(seconds: int | float | None) -> str:
    if seconds is None:
        return "-"
    total = max(0, int(seconds))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def estimate_eta_seconds(elapsed_seconds: int, overall_percent: float, status: ProgressStatus) -> int | None:
    if status in {"completed", "failed"} or overall_percent <= 0:
        return None
    remaining_ratio = (100.0 - clamp_percent(overall_percent)) / max(overall_percent, 0.1)
    return max(0, int(elapsed_seconds * remaining_ratio))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
