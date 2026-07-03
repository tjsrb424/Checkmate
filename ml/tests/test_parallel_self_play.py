import json
from pathlib import Path

import pytest

from oetongsu_ml.parallel_self_play import ParallelSelfPlayConfig, run_parallel_self_play, split_games


def test_split_games_balances_work():
    counts = split_games(10, 3)

    assert counts == [4, 3, 3]
    assert sum(counts) == 10


def test_split_games_limits_workers_to_games():
    counts = split_games(2, 8)

    assert counts == [1, 1]
    assert sum(counts) == 2


def test_split_games_rejects_zero_games():
    with pytest.raises(ValueError):
        split_games(0, 2)


def test_parallel_self_play_random_model_writes_shards_and_merge(tmp_path):
    output = tmp_path / "selfplay" / "parallel_latest.jsonl"
    summary = tmp_path / "selfplay" / "parallel_latest_summary.json"
    shard_dir = tmp_path / "selfplay" / "shards"

    result = run_parallel_self_play(
        ParallelSelfPlayConfig(
            games=2,
            workers=2,
            simulations=1,
            max_plies=2,
            seed=31,
            output=str(output),
            summary=str(summary),
            shard_dir=str(shard_dir),
            random_model=True,
        )
    )

    assert result.status == "completed"
    assert result.games == 2
    assert result.workers == 2
    assert result.sample_count > 0
    assert output.exists()
    assert summary.exists()
    assert len(result.shards) == 2
    assert all(Path(path).exists() for path in result.shards)

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert len(rows) == result.sample_count
    assert payload["sample_count"] == result.sample_count
    assert payload["workers"] == 2
    assert payload["partial"] is False
    assert payload["total_ms"] >= 0
    assert payload["samples_per_sec"] >= 0
    assert payload["games_per_sec"] >= 0
    assert payload["worker_summaries"][0]["performance"]["samples_per_sec"] >= 0
