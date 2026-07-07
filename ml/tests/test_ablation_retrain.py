from oetongsu_ml import ablation_retrain


def test_ablation_retrain_creates_checkpoint_and_metrics_paths(tmp_path, monkeypatch):
    data = tmp_path / "data.jsonl"
    resume = tmp_path / "resume.pt"
    output_dir = tmp_path / "ablation"
    data.write_text("x")
    resume.write_text("x")

    def fake_train_alphazero(**kwargs):
        output = kwargs["output"]
        output.write_text("checkpoint")
        output.with_name(f"{output.stem}_metrics.json").write_text("{}")
        return {
            "history": [
                {
                    "total_loss": 1.0,
                    "val_total_loss": 1.1,
                    "policy_loss": 0.5,
                    "val_policy_loss": 0.6,
                    "value_loss": 0.5,
                    "val_value_loss": 0.5,
                }
            ]
        }

    monkeypatch.setattr(ablation_retrain, "train_alphazero", fake_train_alphazero)
    status = ablation_retrain.main(
        [
            "--data",
            str(data),
            "--resume",
            str(resume),
            "--outputDir",
            str(output_dir),
            "--lrs",
            "0.001",
            "--no-report",
        ]
    )

    assert status == 0
    assert (output_dir / "ablation_a3_lr_0_001.pt").exists()
    assert (output_dir / "ablation_retrain_summary.json").exists()
