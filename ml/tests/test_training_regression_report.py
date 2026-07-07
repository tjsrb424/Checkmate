import json

from oetongsu_ml.training_regression_report import load_metrics, render_markdown


def metrics(path, total_loss, sample_count=4):
    payload = {
        "sample_count": sample_count,
        "train_count": 3,
        "val_count": 1,
        "lr": 0.001,
        "epochs": 1,
        "resume": "../data/models/checkpoints/supervised_v0001.pt",
        "history": [
            {
                "policy_loss": total_loss - 0.1,
                "value_loss": 0.1,
                "total_loss": total_loss,
                "value_mae": 0.2,
                "policy_top1_against_argmax": 0.5,
                "val_policy_loss": total_loss - 0.05,
                "val_value_loss": 0.2,
                "val_total_loss": total_loss + 0.1,
                "val_value_mae": 0.3,
                "val_policy_top1_against_argmax": 0.4,
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_metrics_comparison_report_generation(tmp_path):
    first = tmp_path / "az_iter_000002_metrics.json"
    second = tmp_path / "az_iter_000003_metrics.json"
    metrics(first, 1.0)
    metrics(second, 1.2)

    snapshots = [load_metrics(first), load_metrics(second)]
    body, causes, actions = render_markdown(snapshots)

    assert "az_iter_000002" in body
    assert "az_iter_000003" in body
    assert any("worsened" in cause for cause in causes)
    assert actions
