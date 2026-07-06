from fastapi.testclient import TestClient

from oetongsu_ml.training_server import StartAutoTrainRequest, TrainingServerController, create_app


class FakeRunningProcess:
    pid = 12345

    def poll(self):
        return None

    def terminate(self):
        return None


def client(tmp_path):
    ml_dir = tmp_path / "ml"
    ml_dir.mkdir()
    controller = TrainingServerController(ml_dir=ml_dir)
    return TestClient(create_app(controller)), controller


def test_health_endpoint_returns_ok(tmp_path):
    test_client, _ = client(tmp_path)

    response = test_client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert response.json()["server"] == "oetongsu-training-server"


def test_status_endpoint_works_without_files(tmp_path):
    test_client, _ = client(tmp_path)

    response = test_client.get("/api/training/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["serverStatus"] == "idle"
    assert payload["autotrainState"] is None
    assert payload["registryModelCount"] == 0


def test_registry_logs_and_arena_work_without_files(tmp_path):
    test_client, _ = client(tmp_path)

    assert test_client.get("/api/models/registry").json()["promotedCount"] == 0
    assert test_client.get("/api/training/logs").json()["entries"] == []
    assert test_client.get("/api/arena/results").json()["results"] == []


def test_progress_endpoint_reports_missing_file(tmp_path):
    test_client, _ = client(tmp_path)

    response = test_client.get("/api/training/progress")

    assert response.status_code == 200
    assert response.json() == {"progress": None, "exists": False}


def test_progress_endpoint_reads_progress_json(tmp_path):
    test_client, controller = client(tmp_path)
    controller.training_dir.mkdir(parents=True)
    (controller.training_dir / "progress.json").write_text('{"status":"running"}', encoding="utf-8")

    payload = test_client.get("/api/training/progress").json()

    assert payload["exists"] is True
    assert payload["progress"]["status"] == "running"
    assert payload["source"] == "progress.json"
    assert "updatedAt" in payload


def test_progress_endpoint_handles_invalid_json(tmp_path):
    test_client, controller = client(tmp_path)
    controller.training_dir.mkdir(parents=True)
    (controller.training_dir / "progress.json").write_text("{", encoding="utf-8")

    payload = test_client.get("/api/training/progress").json()

    assert payload["exists"] is True
    assert payload["progress"] is None
    assert "Invalid progress JSON" in payload["error"]


def test_start_endpoint_rejects_when_already_running(tmp_path):
    test_client, controller = client(tmp_path)
    controller.process = FakeRunningProcess()

    response = test_client.post("/api/training/autotrain/start", json={"quick": True})

    assert response.status_code == 409


def test_build_autotrain_command_includes_parallel_selfplay_options(tmp_path):
    _, controller = client(tmp_path)

    command = controller.build_autotrain_command(
        StartAutoTrainRequest(quick=True, selfplayWorkers=3, parallelSelfPlay=True)
    )

    assert "--selfplayWorkers" in command
    assert command[command.index("--selfplayWorkers") + 1] == "3"
    assert "--parallelSelfPlay" in command
