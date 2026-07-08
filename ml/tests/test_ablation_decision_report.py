import json

from oetongsu_ml import ablation_decision_report


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def evaluation_payload():
    return {
        "runs": [
            {
                "candidate": "../data/training/ablation_a3/ablation_a3_lr_0_0001.pt",
                "candidateName": "ablation_a3_lr_0_0001",
                "candidateWins": 10,
                "championWins": 0,
                "draws": 10,
                "candidateScoreRate": 0.75,
                "averagePlies": 150.0,
                "marginSummary": {"avg": 4.0, "median": 4.0, "withinDrawMargin": 10, "outsideDrawMargin": 10},
                "pairedSummary": {"warnings": []},
            },
            {
                "candidate": "../data/training/ablation_a3/ablation_a3_lr_0_001.pt",
                "candidateName": "ablation_a3_lr_0_001",
                "candidateWins": 10,
                "championWins": 0,
                "draws": 10,
                "candidateScoreRate": 0.75,
                "averagePlies": 150.0,
                "marginSummary": {"avg": 3.5, "median": 3.5, "withinDrawMargin": 10, "outsideDrawMargin": 10},
                "pairedSummary": {"warnings": []},
            },
        ]
    }


def retrain_payload():
    return {
        "runs": [
            {
                "lr": 0.0001,
                "last": {
                    "val_total_loss": 5.65,
                    "val_policy_loss": 4.71,
                    "val_value_loss": 0.94,
                    "val_policy_top1_against_argmax": 0.27,
                },
            },
            {
                "lr": 0.001,
                "last": {
                    "val_total_loss": 3.26,
                    "val_policy_loss": 2.78,
                    "val_value_loss": 0.48,
                    "val_policy_top1_against_argmax": 0.50,
                },
            },
        ]
    }


def a3_payload():
    return {
        "games": 40,
        "candidateWins": 0,
        "championWins": 40,
        "draws": 0,
        "candidateScoreRate": 0.0,
        "averagePlies": 150.0,
        "gameSummaries": [
            {"finalScore": {"margin": 15.5}},
            {"finalScore": {"margin": 8.5}},
        ],
        "pairedSummary": {"championDominatedPairs": 20},
    }


def test_best_candidate_breaks_score_rate_tie_by_margin_and_validation_loss():
    candidates = ablation_decision_report.build_candidates(evaluation_payload(), retrain_payload())
    best = ablation_decision_report.best_candidate(candidates)

    assert best is not None
    assert best.name == "ablation_a3_lr_0_001"
    assert best.lr == 0.001


def test_report_output_mentions_baseline_improvement_and_stop_gate(tmp_path):
    root = tmp_path / "artifact"
    summary = root / "data/training/ablation_a3/evaluation_summary.json"
    retrain = root / "data/training/ablation_a3/ablation_retrain_summary.json"
    a3 = root / "data/models/arena/az_iter_000003_arena.json"
    output = tmp_path / "a3_ablation_decision_report.md"
    write_json(summary, evaluation_payload())
    write_json(retrain, retrain_payload())
    write_json(a3, a3_payload())

    status = ablation_decision_report.main(["--root", str(root), "--output", str(output)])

    assert status == 0
    body = output.read_text(encoding="utf-8")
    assert "A3 baseline improvement: yes" in body
    assert "Full RunPod A4: **STOP**" in body
    assert "LR `0.0010`" in body


def test_missing_summary_returns_error(tmp_path):
    output = tmp_path / "report.md"

    status = ablation_decision_report.main(["--root", str(tmp_path / "missing"), "--output", str(output)])

    assert status == 1
    assert not output.exists()
