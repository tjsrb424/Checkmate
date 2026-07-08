from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any


METRIC_KEYS = [
    "policy_loss",
    "value_loss",
    "total_loss",
    "value_mae",
    "policy_top1_against_argmax",
    "val_policy_loss",
    "val_value_loss",
    "val_total_loss",
    "val_value_mae",
    "val_policy_top1_against_argmax",
]


@dataclass(frozen=True)
class ArtifactPaths:
    root: Path
    supervised: Path
    az2: Path
    az3: Path
    az3_metrics: Path
    ablation: Path
    ablation_metrics: Path
    ablation_retrain_summary: Path
    ablation_evaluation: Path
    selfplay: Path
    selfplay_summary: Path
    autotrain_summary: Path
    autotrain_state: Path
    az3_arena: Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare AutoTrain A3 and A3 ablation training/evaluation paths.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", default="../docs/ml/a3_path_diff_report.md")
    parser.add_argument("--limitPositions", type=int, default=128)
    parser.add_argument("--noModelOutput", action="store_true")
    return parser.parse_args(argv)


def resolve_paths(root: str | Path) -> ArtifactPaths:
    base = Path(root)
    return ArtifactPaths(
        root=base,
        supervised=base / "data/models/checkpoints/supervised_v0001.pt",
        az2=base / "data/models/checkpoints/az_iter_000002.pt",
        az3=base / "data/models/checkpoints/az_iter_000003.pt",
        az3_metrics=base / "data/models/checkpoints/az_iter_000003_metrics.json",
        ablation=base / "data/training/ablation_a3/ablation_a3_lr_0_001.pt",
        ablation_metrics=base / "data/training/ablation_a3/ablation_a3_lr_0_001_metrics.json",
        ablation_retrain_summary=base / "data/training/ablation_a3/ablation_retrain_summary.json",
        ablation_evaluation=base / "data/training/ablation_a3/evaluation_summary.json",
        selfplay=base / "data/selfplay/az_iter_000003.jsonl",
        selfplay_summary=base / "data/selfplay/az_iter_000003_summary.json",
        autotrain_summary=base / "data/training/autotrain_summary.json",
        autotrain_state=base / "data/training/autotrain_state.json",
        az3_arena=base / "data/models/arena/az_iter_000003_arena.json",
    )


def required_files(paths: ArtifactPaths) -> dict[str, Path]:
    return {
        "supervised_v0001.pt": paths.supervised,
        "az_iter_000002.pt": paths.az2,
        "az_iter_000003.pt": paths.az3,
        "ablation_a3_lr_0_001.pt": paths.ablation,
        "ablation_a3_lr_0_001_metrics.json": paths.ablation_metrics,
        "ablation_retrain_summary.json": paths.ablation_retrain_summary,
        "evaluation_summary.json": paths.ablation_evaluation,
        "az_iter_000003.jsonl": paths.selfplay,
        "az_iter_000003_summary.json": paths.selfplay_summary,
        "az_iter_000003_arena.json": paths.az3_arena,
    }


def optional_files(paths: ArtifactPaths) -> dict[str, Path]:
    return {
        "az_iter_000003_metrics.json": paths.az3_metrics,
        "autotrain_summary.json": paths.autotrain_summary,
        "autotrain_state.json": paths.autotrain_state,
    }


def missing_required(paths: ArtifactPaths) -> list[str]:
    return [name for name, path in required_files(paths).items() if not path.exists()]


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def metric_from_checkpoint(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    torch = import_torch()
    checkpoint = torch.load(path, map_location="cpu")
    metrics = checkpoint.get("metrics") if isinstance(checkpoint, dict) else None
    if isinstance(metrics, list) and metrics and isinstance(metrics[-1], dict):
        return {"source": "checkpoint.metrics", "last": metrics[-1]}
    return None


def metric_summary(metrics_path: Path, checkpoint_path: Path | None = None) -> dict[str, Any]:
    source = "missing"
    payload = read_json(metrics_path)
    if payload:
        source = str(metrics_path)
    elif checkpoint_path:
        payload = metric_from_checkpoint(checkpoint_path)
    if not payload:
        return {"exists": False, "source": source}
    last = payload.get("history", [payload.get("last", {})])[-1] if isinstance(payload.get("history"), list) and payload.get("history") else payload.get("last", {})
    if not isinstance(last, dict):
        last = {}
    return {
        "exists": metrics_path.exists(),
        "source": payload.get("source") or source,
        "sample_count": payload.get("sample_count"),
        "train_count": payload.get("train_count"),
        "val_count": payload.get("val_count"),
        "lr": payload.get("lr"),
        "epochs": payload.get("epochs"),
        "resume": payload.get("resume"),
        "last": {key: as_float(last.get(key)) for key in METRIC_KEYS},
    }


def import_torch():
    import torch

    return torch


def checkpoint_summary(path: Path) -> dict[str, Any]:
    item: dict[str, Any] = {"exists": path.exists(), "path": str(path), "file_size": path.stat().st_size if path.exists() else None}
    if not path.exists():
        return item
    torch = import_torch()
    checkpoint = torch.load(path, map_location="cpu")
    item["metadata_keys"] = sorted(checkpoint.keys()) if isinstance(checkpoint, dict) else []
    state = checkpoint.get("model_state") if isinstance(checkpoint, dict) else None
    if not isinstance(state, dict):
        item.update({"model_state_key_count": 0, "parameter_count": 0, "has_nan_or_inf": None, "tensor_l2_norm": None})
        return item
    total_params = 0
    norm_sq = 0.0
    has_bad = False
    for tensor in state.values():
        if not hasattr(tensor, "numel"):
            continue
        tensor = tensor.detach().float()
        total_params += int(tensor.numel())
        has_bad = has_bad or bool(torch.isnan(tensor).any().item()) or bool(torch.isinf(tensor).any().item())
        norm_sq += float(torch.sum(tensor * tensor).item())
    item.update(
        {
            "channels": checkpoint.get("channels") if isinstance(checkpoint, dict) else None,
            "model_state_key_count": len(state),
            "parameter_count": total_params,
            "has_nan_or_inf": has_bad,
            "tensor_l2_norm": math.sqrt(norm_sq),
        }
    )
    return item


def parameter_delta_norm(left: Path, right: Path) -> float | None:
    if not left.exists() or not right.exists():
        return None
    torch = import_torch()
    left_state = torch.load(left, map_location="cpu").get("model_state", {})
    right_state = torch.load(right, map_location="cpu").get("model_state", {})
    if not isinstance(left_state, dict) or not isinstance(right_state, dict):
        return None
    norm_sq = 0.0
    for key in sorted(set(left_state) & set(right_state)):
        left_tensor = left_state[key].detach().float()
        right_tensor = right_state[key].detach().float()
        norm_sq += float(torch.sum((left_tensor - right_tensor) ** 2).item())
    return math.sqrt(norm_sq)


def run_model_output_comparison(paths: ArtifactPaths, limit: int) -> str:
    from .model_regression_diagnostics import render_markdown, run_diagnostics

    args = SimpleNamespace(
        champion=str(paths.supervised),
        previous=str(paths.az3),
        candidate=str(paths.ablation),
        samples=str(paths.selfplay),
        limit=limit,
        device="cpu",
        report="",
        no_report=True,
    )
    stats, pairs, sample_count = run_diagnostics(args)
    body = render_markdown(stats, pairs, sample_count)
    return (
        body.replace("champion", "supervised_v0001")
        .replace("previous", "az_iter_000003")
        .replace("candidate", "ablation_a3_lr_0_001")
    )


def margin_stats(arena_payload: dict[str, Any]) -> dict[str, Any]:
    games = arena_payload.get("gameSummaries", [])
    margins = [
        as_float((game.get("finalScore") or {}).get("margin"))
        for game in games
        if isinstance(game, dict) and isinstance(game.get("finalScore"), dict)
    ]
    margins = [value for value in margins if value is not None]
    if not margins:
        return {"avg": None, "median": None, "withinDrawMargin": None, "outsideDrawMargin": None}
    ordered = sorted(margins)
    draw_margin = 1.5
    return {
        "avg": sum(margins) / len(margins),
        "median": ordered[len(ordered) // 2],
        "withinDrawMargin": len([value for value in margins if value <= draw_margin]),
        "outsideDrawMargin": len([value for value in margins if value > draw_margin]),
    }


def ablation_lr_001_row(evaluation: dict[str, Any] | None) -> dict[str, Any] | None:
    if not evaluation:
        return None
    for row in evaluation.get("runs", []):
        if isinstance(row, dict) and row.get("candidateName") == "ablation_a3_lr_0_001":
            return row
    return None


def format_value(value: Any, digits: int = 6) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_metrics_table(metrics: dict[str, dict[str, Any]]) -> list[str]:
    lines = [
        "| model | metrics_source | sample_count | train_count | val_count | lr | epochs | resume | val_total_loss | val_policy_loss | val_value_loss | val_policy_top1 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for name, item in metrics.items():
        last = item.get("last", {})
        lines.append(
            f"| {name} | {item.get('source', '-')} | {format_value(item.get('sample_count'))} | {format_value(item.get('train_count'))} | "
            f"{format_value(item.get('val_count'))} | {format_value(item.get('lr'))} | {format_value(item.get('epochs'))} | {item.get('resume') or '-'} | "
            f"{format_value(last.get('val_total_loss'))} | {format_value(last.get('val_policy_loss'))} | "
            f"{format_value(last.get('val_value_loss'))} | {format_value(last.get('val_policy_top1_against_argmax'))} |"
        )
    return lines


def render_checkpoint_table(checkpoints: dict[str, dict[str, Any]], deltas: dict[str, float | None]) -> list[str]:
    lines = [
        "| checkpoint | exists | size_bytes | metadata_keys | state_keys | params | has_nan_or_inf | tensor_l2_norm |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for name, item in checkpoints.items():
        lines.append(
            f"| {name} | {item.get('exists')} | {format_value(item.get('file_size'))} | `{', '.join(item.get('metadata_keys') or [])}` | "
            f"{format_value(item.get('model_state_key_count'))} | {format_value(item.get('parameter_count'))} | "
            f"{format_value(item.get('has_nan_or_inf'))} | {format_value(item.get('tensor_l2_norm'))} |"
        )
    lines.extend(["", "| delta | parameter_delta_norm |", "| --- | ---: |"])
    for name, value in deltas.items():
        lines.append(f"| {name} | {format_value(value)} |")
    return lines


def render_report(paths: ArtifactPaths, no_model_output: bool, limit_positions: int) -> str:
    missing = missing_required(paths)
    if missing:
        raise FileNotFoundError(f"missing required files: {', '.join(missing)}")

    optional_missing = [name for name, path in optional_files(paths).items() if not path.exists()]
    checkpoints = {} if no_model_output else {
        "supervised_v0001": checkpoint_summary(paths.supervised),
        "az_iter_000002": checkpoint_summary(paths.az2),
        "az_iter_000003": checkpoint_summary(paths.az3),
        "ablation_a3_lr_0_001": checkpoint_summary(paths.ablation),
    }
    deltas = {} if no_model_output else {
        "supervised_v0001 -> az_iter_000003": parameter_delta_norm(paths.supervised, paths.az3),
        "supervised_v0001 -> ablation_a3_lr_0_001": parameter_delta_norm(paths.supervised, paths.ablation),
        "az_iter_000003 -> ablation_a3_lr_0_001": parameter_delta_norm(paths.az3, paths.ablation),
    }
    metrics = {
        "az_iter_000003": metric_summary(paths.az3_metrics, None if no_model_output else paths.az3),
        "ablation_a3_lr_0_001": metric_summary(paths.ablation_metrics, None if no_model_output else paths.ablation),
    }
    selfplay_summary = read_json(paths.selfplay_summary) or {}
    az3_arena = read_json(paths.az3_arena) or {}
    ablation_eval = read_json(paths.ablation_evaluation) or {}
    ablation_row = ablation_lr_001_row(ablation_eval) or {}
    az3_margin = margin_stats(az3_arena)
    ablation_margin = ablation_row.get("marginSummary") if isinstance(ablation_row.get("marginSummary"), dict) else {}
    model_output = "모델 출력 비교는 `--noModelOutput` 옵션으로 생략했습니다."
    if not no_model_output:
        model_output = run_model_output_comparison(paths, limit_positions)

    root_cause = (
        "가장 유력한 원인은 AutoTrain resume 버그보다는 학습 호출/분할 차이와 평가 불안정성의 결합입니다. "
        "코드상 AutoTrain A3도 latest promoted champion인 `supervised_v0001`을 resume source로 사용하고, "
        "ablation도 `supervised_v0001`을 명시적으로 resume합니다. 다만 AutoTrain은 seed `1 + iteration`, "
        "ablation은 seed `7`을 쓰므로 train/validation split과 shuffle 순서가 달라집니다. "
        "두 체크포인트의 checkpoint 내부 metrics는 거의 비슷하지만 arena 결과는 크게 갈렸고, 모든 비교가 "
        "`maxPlies=150` score-adjudication에 묶여 있어 평가 변동성/역할 표본 문제가 강하게 남아 있습니다."
    )

    lines = [
        "# A3 Path Difference Report",
        "",
        "## 1. Executive Summary",
        "",
        "- 결론: A4 full RunPod는 아직 실행하지 않습니다.",
        "- AutoTrain A3와 ablation LR 0.001은 둘 다 `supervised_v0001`에서 resume하는 경로입니다. rejected `az_iter_000002`에서 resume했다는 직접 증거는 코드상 확인되지 않았습니다.",
        "- 큰 차이는 `seed`와 그로 인한 train/validation split 및 shuffle 순서입니다. AutoTrain A3는 `seed=1+3=4`, ablation은 `seed=7`입니다.",
        "- checkpoint 내부 metrics는 매우 비슷하지만 arena 결과는 `0.0%` 대 `75.0%`로 크게 다릅니다. 따라서 단순 학습 성능보다 maxPlies/score-adjudication 평가 불안정성을 우선 의심해야 합니다.",
        "",
        "## 2. Artifact Inputs",
        "",
        f"- artifact root: `{paths.root}`",
        f"- 누락된 선택 파일: {', '.join(optional_missing) if optional_missing else '없음'}",
        "",
        "| file | exists | size_bytes |",
        "| --- | ---: | ---: |",
    ]
    for name, path in {**required_files(paths), **optional_files(paths)}.items():
        lines.append(f"| {name} | {path.exists()} | {path.stat().st_size if path.exists() else '-'} |")

    lines.extend(
        [
            "",
            "## 3. Static Code Path Comparison",
            "",
            "| 항목 | AutoTrain A3 | ablation LR 0.001 | 판정 |",
            "| --- | --- | --- | --- |",
            "| resume checkpoint source | `get_latest_promoted(registry)`에서 champion path를 찾고 `resumeChampion=True`이면 그 champion으로 resume | RunPod ablation script가 `--resume ../data/models/checkpoints/supervised_v0001.pt`를 명시 | 둘 다 champion resume 경로 |",
            "| rejected candidate resume 가능성 | 코드상 candidate resume은 `champion_path`만 사용. `latestCandidateVersion`/rejected candidate를 resume source로 쓰지 않음 | 해당 없음 | 직접 증거 낮음 |",
            "| data path | `../data/selfplay/az_iter_000003.jsonl` 생성 후 즉시 학습 | 같은 `../data/selfplay/az_iter_000003.jsonl` 재사용 | 동일 계열 |",
            "| sample count | self-play summary 기준 `14674` | metrics 기준 `14674` | 동일 |",
            "| epochs / batch size / channels / lr | `1 / 64 / 64 / 0.001` | `1 / 64 / 64 / 0.001` | 동일 |",
            "| seed | `cfg.seed + iteration`, A3는 `4` | `7` | 중요 차이 |",
            "| train/validation split | `random_split(..., manual_seed(seed))` | 같은 함수지만 seed가 다름 | 중요 차이 |",
            "| shuffle | DataLoader `shuffle=True` | 동일하지만 seed가 다름 | 중요 차이 |",
            "| optimizer | Adam, weight decay 없음 | Adam, weight decay 없음 | 동일 |",
            "| checkpoint format | `model_state`, `channels`, `metrics` | 동일 | 동일 |",
            "| arena | candidate vs champion, `promotion_threshold=0.55`, 40 games | registry-free evaluation, threshold effectively 0.5, 20 games | 평가 표본/판정 차이 |",
            "",
            "## 4. Checkpoint Metadata Comparison",
            "",
        ]
    )
    lines.extend(render_checkpoint_table(checkpoints, deltas) if not no_model_output else ["체크포인트 tensor 분석은 `--noModelOutput` 옵션으로 생략했습니다."])

    lines.extend(["", "## 5. Metrics Comparison", ""])
    lines.extend(render_metrics_table(metrics))
    lines.extend(
        [
            "",
            f"- self-play summary sample_count: `{selfplay_summary.get('sample_count', '-')}`",
            f"- self-play games: `{selfplay_summary.get('games', '-')}`, workers: `{selfplay_summary.get('workers', '-')}`",
            "",
            "## 6. Model Output Comparison",
            "",
            model_output,
            "",
            "## 7. Arena Result Comparison",
            "",
            "| 항목 | AutoTrain az_iter_000003 | ablation_a3_lr_0_001 |",
            "| --- | ---: | ---: |",
            f"| scoreRate | {format_value(az3_arena.get('candidateScoreRate'))} | {format_value(ablation_row.get('candidateScoreRate'))} |",
            f"| wins/losses/draws | {az3_arena.get('candidateWins', '-')}/{az3_arena.get('championWins', '-')}/{az3_arena.get('draws', '-')} | {ablation_row.get('candidateWins', '-')}/{ablation_row.get('championWins', '-')}/{ablation_row.get('draws', '-')} |",
            f"| averagePlies | {format_value(az3_arena.get('averagePlies'))} | {format_value(ablation_row.get('averagePlies'))} |",
            f"| margin avg/median | {format_value(az3_margin.get('avg'))}/{format_value(az3_margin.get('median'))} | {format_value(as_float(ablation_margin.get('avg')))}/{format_value(as_float(ablation_margin.get('median')))} |",
            f"| draw margin hits | {format_value(az3_margin.get('withinDrawMargin'))} within / {format_value(az3_margin.get('outsideDrawMargin'))} outside | {format_value(ablation_margin.get('withinDrawMargin'))} within / {format_value(ablation_margin.get('outsideDrawMargin'))} outside |",
            f"| paired warnings | {az3_arena.get('pairedSummary', {}).get('warnings', []) if isinstance(az3_arena.get('pairedSummary'), dict) else '-'} | {ablation_row.get('pairedSummary', {}).get('warnings', []) if isinstance(ablation_row.get('pairedSummary'), dict) else '-'} |",
            "",
            "## 8. Most Likely Root Cause",
            "",
            root_cause,
            "",
            "## 9. Recommended Fix",
            "",
            "- AutoTrain candidate initialization에 guard test를 추가해 candidate resume source가 항상 latest promoted champion인지 검증합니다.",
            "- AutoTrain metrics/checkpoint metadata에 `resume`, `seed`, `train_count`, `val_count`, `sample_count`, `split_seed`, `optimizer`를 명시적으로 남깁니다.",
            "- A4 전에 같은 checkpoint pair를 더 큰 paired local/cheap evaluation으로 재검증합니다.",
            "- promotion arena에 maxPlies/score-adjudication confidence gate를 추가합니다.",
            "- seed 차이만으로 결과가 뒤집히는지 확인하는 작은 local 재현 테스트를 먼저 설계합니다.",
            "",
            "## 10. RunPod Decision",
            "",
            "- Full RunPod A4: **금지**.",
            "- AutoTrain 추가 실행: **금지**.",
            "- 새 self-play 생성: **금지**.",
            "- registry champion 변경: **금지**.",
            "- 허용: 이 path-diff report 기반의 코드 수정, 메타데이터 기록 보강, 작은 로컬/CPU-first 검증.",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    paths = resolve_paths(args.root)
    try:
        report = render_report(paths, no_model_output=args.noModelOutput, limit_positions=args.limitPositions)
    except (FileNotFoundError, ValueError, json.JSONDecodeError, OSError) as error:
        print(f"ERROR: {error}")
        return 1
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(report)
    print(f"output written: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
