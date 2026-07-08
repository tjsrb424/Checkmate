import json

from oetongsu_ml import seed_sensitivity_probe


def test_seed_sensitivity_probe_creates_seed_outputs(tmp_path, monkeypatch):
    data = tmp_path / "selfplay.jsonl"
    resume = tmp_path / "champion.pt"
    output_dir = tmp_path / "seed_probe"
    data.write_text("{}", encoding="utf-8")
    resume.write_text("checkpoint", encoding="utf-8")

    def fake_train_alphazero(**kwargs):
        output = kwargs["output"]
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("checkpoint", encoding="utf-8")
        output.with_name(f"{output.stem}_metrics.json").write_text("{}", encoding="utf-8")
        return {
            "history": [
                {
                    "val_total_loss": float(kwargs["seed"]),
                    "val_policy_loss": 1.0,
                    "val_value_loss": 0.5,
                    "val_policy_top1_against_argmax": 0.25,
                }
            ]
        }

    monkeypatch.setattr(seed_sensitivity_probe, "train_alphazero", fake_train_alphazero)

    status = seed_sensitivity_probe.main(
        [
            "--data",
            str(data),
            "--resume",
            str(resume),
            "--outputDir",
            str(output_dir),
            "--seeds",
            "4",
            "7",
            "--limit",
            "2",
            "--noArena",
        ]
    )

    assert status == 0
    summary = json.loads((output_dir / "seed_sensitivity_summary.json").read_text(encoding="utf-8"))
    assert [row["seed"] for row in summary["runs"]] == [4, 7]
    assert (output_dir / "seed_4.pt").exists()
    assert (output_dir / "seed_7.pt").exists()
