from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .model_registry import get_latest_promoted, load_registry

SERVER_VERSION = "0.1.0"
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
]


class StartAutoTrainRequest(BaseModel):
    quick: bool = False
    iterations: Optional[int] = None
    gamesPerIteration: Optional[int] = None
    simulations: Optional[int] = None
    maxPlies: Optional[int] = None
    trainEpochs: Optional[int] = None
    batchSize: Optional[int] = None
    promotionGames: Optional[int] = None
    threshold: Optional[float] = None
    ruleset: Optional[Literal["kakao-like", "oetongsu-basic", "kja-like"]] = None
    selfplayWorkers: Optional[int] = None
    parallelSelfPlay: bool = False


class TrainingServerController:
    def __init__(self, ml_dir: Path | None = None) -> None:
        self.ml_dir = ml_dir or Path(__file__).resolve().parents[1]
        self.data_dir = self.ml_dir.parent / "data"
        self.training_dir = self.data_dir / "training"
        self.models_dir = self.data_dir / "models"
        self.arena_dir = self.models_dir / "arena"
        self.process: subprocess.Popen | None = None
        self.started_at: str | None = None
        self.ended_at: str | None = None
        self.command: list[str] | None = None
        self.last_error: str | None = None

    def health(self) -> dict[str, Any]:
        return {
            "ok": True,
            "server": "oetongsu-training-server",
            "version": SERVER_VERSION,
            "cwd": str(self.ml_dir),
            "python": sys.executable,
        }

    def status(self) -> dict[str, Any]:
        self.refresh_process()
        registry = load_registry(self.registry_path())
        latest_champion = get_latest_promoted(registry)
        warnings = []
        if latest_champion is None:
            warnings.append("No promoted champion is registered yet.")
        return {
            "serverStatus": self.server_status(),
            "pid": self.process.pid if self.process and self.process.poll() is None else None,
            "startedAt": self.started_at,
            "endedAt": self.ended_at,
            "command": self.command,
            "lastError": self.last_error,
            "autotrainState": read_json_or_none(self.training_dir / "autotrain_state.json"),
            "autotrainSummary": read_json_or_none(self.training_dir / "autotrain_summary.json"),
            "latestChampion": latest_champion,
            "registryModelCount": len(registry.get("models", [])),
            "warnings": warnings,
        }

    def registry_response(self) -> dict[str, Any]:
        registry = load_registry(self.registry_path())
        models = registry.get("models", [])
        return {
            "registry": registry,
            "latestPromoted": get_latest_promoted(registry),
            "promotedCount": sum(1 for entry in models if entry.get("status") == "promoted"),
            "rejectedCount": sum(1 for entry in models if entry.get("status") == "rejected"),
            "candidateCount": sum(1 for entry in models if entry.get("status") == "candidate"),
        }

    def start_autotrain(self, request: StartAutoTrainRequest) -> dict[str, Any]:
        self.refresh_process()
        if self.process and self.process.poll() is None:
            raise HTTPException(status_code=409, detail="AutoTrain is already running")

        self.training_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = self.training_dir / "server_last_stdout.log"
        stderr_path = self.training_dir / "server_last_stderr.log"
        command = self.build_autotrain_command(request)
        stdout = stdout_path.open("w", encoding="utf-8")
        stderr = stderr_path.open("w", encoding="utf-8")
        self.process = subprocess.Popen(command, cwd=self.ml_dir, stdout=stdout, stderr=stderr)
        self.started_at = utc_now()
        self.ended_at = None
        self.command = command
        self.last_error = None
        return {
            "started": True,
            "pid": self.process.pid,
            "command": command,
            "startedAt": self.started_at,
        }

    def stop_autotrain(self) -> dict[str, Any]:
        self.refresh_process()
        if not self.process or self.process.poll() is not None:
            return {"requestedStop": False, "message": "No AutoTrain process is running."}
        self.process.terminate()
        self.ended_at = utc_now()
        return {"requestedStop": True, "pid": self.process.pid}

    def log_tail(self, limit: int) -> dict[str, Any]:
        path = self.training_dir / "autotrain_log.jsonl"
        if not path.exists():
            return {"entries": []}
        lines = path.read_text(encoding="utf-8").splitlines()
        entries = []
        for line in lines[-max(0, limit) :]:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                entries.append({"raw": line})
        return {"entries": entries}

    def arena_results(self) -> dict[str, Any]:
        results = []
        if self.arena_dir.exists():
            for path in sorted(self.arena_dir.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
                payload = read_json_or_none(path) or {}
                results.append(
                    {
                        "file": path.name,
                        "path": str(path),
                        "candidateScoreRate": payload.get("candidateScoreRate"),
                        "championScoreRate": payload.get("championScoreRate"),
                        "promoted": payload.get("promoted"),
                        "illegalMoves": payload.get("illegalMoves"),
                        "forfeits": payload.get("forfeits"),
                        "averagePlies": payload.get("averagePlies"),
                        "modifiedAt": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
                    }
                )
        return {"results": results}

    def build_autotrain_command(self, request: StartAutoTrainRequest) -> list[str]:
        command = [sys.executable, "-m", "oetongsu_ml.autotrain"]
        if request.quick:
            command.append("--quick")
        option_map: list[tuple[str, Any]] = [
            ("--iterations", request.iterations),
            ("--gamesPerIteration", request.gamesPerIteration),
            ("--simulations", request.simulations),
            ("--maxPlies", request.maxPlies),
            ("--trainEpochs", request.trainEpochs),
            ("--batchSize", request.batchSize),
            ("--promotionGames", request.promotionGames),
            ("--threshold", request.threshold),
            ("--ruleset", request.ruleset),
            ("--selfplayWorkers", selfplay_workers_for(request)),
        ]
        for option, value in option_map:
            if value is not None:
                command.extend([option, str(value)])
        if request.parallelSelfPlay:
            command.append("--parallelSelfPlay")
        return command

    def server_status(self) -> str:
        if self.process and self.process.poll() is None:
            return "running"
        if self.last_error:
            return "failed"
        return "idle"

    def refresh_process(self) -> None:
        if not self.process:
            return
        code = self.process.poll()
        if code is None:
            return
        if self.ended_at is None:
            self.ended_at = utc_now()
        if code != 0:
            self.last_error = f"AutoTrain exited with code {code}"

    def registry_path(self) -> Path:
        return self.models_dir / "registry.json"


def create_app(controller: TrainingServerController | None = None) -> FastAPI:
    app = FastAPI(title="Oetongsu Training Server", version=SERVER_VERSION)
    app.state.controller = controller or TrainingServerController()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return app.state.controller.health()

    @app.get("/api/training/status")
    def training_status() -> dict[str, Any]:
        return app.state.controller.status()

    @app.post("/api/training/autotrain/start")
    def start_autotrain(request: StartAutoTrainRequest) -> dict[str, Any]:
        return app.state.controller.start_autotrain(request)

    @app.post("/api/training/autotrain/stop")
    def stop_autotrain() -> dict[str, Any]:
        return app.state.controller.stop_autotrain()

    @app.get("/api/models/registry")
    def model_registry() -> dict[str, Any]:
        return app.state.controller.registry_response()

    @app.get("/api/training/logs")
    def training_logs(limit: int = 50) -> dict[str, Any]:
        return app.state.controller.log_tail(limit)

    @app.get("/api/training/summary")
    def training_summary() -> dict[str, Any]:
        return {"summary": read_json_or_none(app.state.controller.training_dir / "autotrain_summary.json")}

    @app.get("/api/arena/results")
    def arena_results() -> dict[str, Any]:
        return app.state.controller.arena_results()

    return app


def read_json_or_none(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def selfplay_workers_for(request: StartAutoTrainRequest) -> int | None:
    if request.selfplayWorkers is not None:
        return max(1, request.selfplayWorkers)
    if request.quick and request.parallelSelfPlay:
        return 2
    return None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local Oetongsu training API server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--reload", action="store_true")
    return parser.parse_args(argv)


app = create_app()


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.host != "127.0.0.1":
        raise SystemExit("Training server must bind to 127.0.0.1 for local MVP use.")
    import uvicorn

    uvicorn.run("oetongsu_ml.training_server:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
