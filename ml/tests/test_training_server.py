from fastapi.testclient import TestClient

from oetongsu_ml.training_server import TrainingServerController, create_app


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


def test_start_endpoint_rejects_when_already_running(tmp_path):
    test_client, controller = client(tmp_path)
    controller.process = FakeRunningProcess()

    response = test_client.post("/api/training/autotrain/start", json={"quick": True})

    assert response.status_code == 409
