import json

from oetongsu_ml.progress_reporter import ProgressReporter, clamp_percent, format_duration


def test_percent_clamp_and_duration_format():
    assert clamp_percent(-10) == 0
    assert clamp_percent(120) == 100
    assert format_duration(3661) == "01:01:01"


def test_progress_reporter_writes_atomic_snapshot_and_events(tmp_path):
    progress_path = tmp_path / "training" / "progress.json"
    events_path = tmp_path / "training" / "progress_events.jsonl"
    reporter = ProgressReporter(progress_path, events_path, run_id="autotrain-test", total_iterations=2)

    reporter.update(
        phase="selfplay",
        phase_percent=50,
        message="Generating self-play games",
        message_ko="자기대국을 생성하는 중입니다.",
        current_iteration=1,
        completed_iterations=0,
        self_play={"currentGames": 5, "totalGames": 10},
    )

    payload = json.loads(progress_path.read_text(encoding="utf-8"))
    assert payload["runId"] == "autotrain-test"
    assert payload["status"] == "running"
    assert payload["phaseKey"] == "selfplay"
    assert payload["phaseLabelKo"] == "자기대국 생성"
    assert payload["messageKo"] == "자기대국을 생성하는 중입니다."
    assert payload["phasePercent"] == 50
    assert 0 < payload["overallPercent"] < 100

    events = events_path.read_text(encoding="utf-8").splitlines()
    assert len(events) == 1
    assert json.loads(events[0])["messageKo"] == "자기대국을 생성하는 중입니다."


def test_progress_reporter_records_failed_terminal_state(tmp_path):
    reporter = ProgressReporter(tmp_path / "progress.json", tmp_path / "events.jsonl", "run", 1)

    reporter.mark_failed(RuntimeError("boom"), current_iteration=1, completed_iterations=0)

    payload = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["phaseKey"] == "failed"
    assert payload["result"]["error"] == "boom"
