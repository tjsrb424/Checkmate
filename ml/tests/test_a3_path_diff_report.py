import json

from oetongsu_ml import a3_path_diff_report


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_file(path, text="x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def create_fixture_root(tmp_path):
    root = tmp_path / "artifact"
    for relative in [
        "data/models/checkpoints/supervised_v0001.pt",
        "data/models/checkpoints/az_iter_000002.pt",
        "data/models/checkpoints/az_iter_000003.pt",
        "data/training/ablation_a3/ablation_a3_lr_0_001.pt",
        "data/selfplay/az_iter_000003.jsonl",
    ]:
        write_file(root / relative)
    write_json(
        root / "data/training/ablation_a3/ablation_a3_lr_0_001_metrics.json",
        {
            "sample_count": 10,
            "train_count": 8,
            "val_count": 2,
            "lr": 0.001,
            "epochs": 1,
            "resume": "../data/models/checkpoints/supervised_v0001.pt",
            "history": [{"val_total_loss": 1.0, "val_policy_loss": 0.8, "val_value_loss": 0.2, "val_policy_top1_against_argmax": 0.5}],
        },
    )
    write_json(root / "data/training/ablation_a3/ablation_retrain_summary.json", {"runs": []})
    write_json(
        root / "data/training/ablation_a3/evaluation_summary.json",
        {
            "runs": [
                {
                    "candidateName": "ablation_a3_lr_0_001",
                    "candidateWins": 1,
                    "championWins": 0,
                    "draws": 1,
                    "candidateScoreRate": 0.75,
                    "averagePlies": 150,
                    "marginSummary": {"avg": 3.5, "median": 3.5, "withinDrawMargin": 1, "outsideDrawMargin": 1},
                    "pairedSummary": {"warnings": []},
                }
            ]
        },
    )
    write_json(root / "data/selfplay/az_iter_000003_summary.json", {"sample_count": 10, "games": 2, "workers": 1})
    write_json(
        root / "data/models/arena/az_iter_000003_arena.json",
        {
            "candidateWins": 0,
            "championWins": 2,
            "draws": 0,
            "candidateScoreRate": 0.0,
            "averagePlies": 150,
            "gameSummaries": [{"finalScore": {"margin": 12.0}}, {"finalScore": {"margin": 8.0}}],
            "pairedSummary": {"warnings": []},
        },
    )
    return root


def test_resolve_paths_finds_required_files(tmp_path):
    root = create_fixture_root(tmp_path)
    paths = a3_path_diff_report.resolve_paths(root)

    assert a3_path_diff_report.missing_required(paths) == []
    assert paths.ablation.name == "ablation_a3_lr_0_001.pt"


def test_missing_required_file_returns_error(tmp_path):
    output = tmp_path / "report.md"

    status = a3_path_diff_report.main(["--root", str(tmp_path / "missing"), "--output", str(output), "--noModelOutput"])

    assert status == 1
    assert not output.exists()


def test_no_model_output_generates_metrics_and_root_cause_report(tmp_path):
    root = create_fixture_root(tmp_path)
    output = tmp_path / "report.md"

    status = a3_path_diff_report.main(["--root", str(root), "--output", str(output), "--noModelOutput"])

    assert status == 0
    body = output.read_text(encoding="utf-8")
    assert "## 5. Metrics Comparison" in body
    assert "가장 유력한 원인은" in body
    assert "체크포인트 tensor 분석은 `--noModelOutput` 옵션으로 생략했습니다." in body
