import json

from oetongsu_ml import a3_local_validation_report


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def create_root(tmp_path, cheap_same=False):
    root = tmp_path / "artifact"
    write_json(
        root / "data/training/seed_probe/seed_sensitivity_summary.json",
        {
            "runs": [
                {"seed": 4, "checkpoint": "seed_4.pt", "last": {"val_total_loss": 1.0, "val_policy_loss": 0.8, "val_value_loss": 0.2, "val_policy_top1_against_argmax": 0.5}},
                {"seed": 7, "checkpoint": "seed_7.pt", "last": {"val_total_loss": 1.03, "val_policy_loss": 0.82, "val_value_loss": 0.21, "val_policy_top1_against_argmax": 0.49}},
            ]
        },
    )
    write_json(root / "data/training/cheap_validation_az_iter_000003.json", {"status": "pass" if cheap_same else "fail", "candidateScoreRate": 0.0 if not cheap_same else 0.5, "games": 4, "simulations": 8, "maxPlies": 80, "candidateWins": 0, "championWins": 4, "draws": 0, "warnings": []})
    write_json(root / "data/training/cheap_validation_ablation_lr_0_001.json", {"status": "pass", "candidateScoreRate": 0.5, "games": 4, "simulations": 8, "maxPlies": 80, "candidateWins": 2, "championWins": 2, "draws": 0, "warnings": []})
    write_json(root / "data/models/arena/az_iter_000003_arena.json", {"candidateScoreRate": 0.0, "candidateWins": 0, "championWins": 40, "draws": 0, "averagePlies": 150})
    write_json(root / "data/training/ablation_a3/evaluation_summary.json", {"runs": [{"candidateName": "ablation_a3_lr_0_001", "candidateScoreRate": 0.75, "candidateWins": 10, "championWins": 0, "draws": 10, "averagePlies": 150}]})
    return root


def test_report_reads_seed_and_cheap_validation_results(tmp_path):
    root = create_root(tmp_path)
    output = tmp_path / "report.md"

    status = a3_local_validation_report.main(["--root", str(root), "--output", str(output)])

    assert status == 0
    body = output.read_text(encoding="utf-8")
    assert "Seed verdict" in body
    assert "Cheap/full consistency: `consistent`" in body
    assert "RunPod full A4 remains blocked" in body


def test_report_marks_inconsistent_cheap_gate(tmp_path):
    root = create_root(tmp_path, cheap_same=True)

    body = a3_local_validation_report.render_report(root)

    assert "Cheap/full consistency: `inconsistent`" in body
    assert "redesign cheap gate" in body


def test_missing_file_returns_error(tmp_path):
    output = tmp_path / "report.md"

    status = a3_local_validation_report.main(["--root", str(tmp_path / "missing"), "--output", str(output)])

    assert status == 1
    assert not output.exists()
