from oetongsu_ml.alphazero_model import AlphaZeroNet
from oetongsu_ml.checkpoint import create_version_id, load_checkpoint, next_version_id, save_checkpoint


def test_checkpoint_save_and_load(tmp_path):
    model = AlphaZeroNet(channels=4)
    model_path, metadata_path = save_checkpoint(model, tmp_path, {"epochs": 1}, version="az_v0007")

    restored, metadata = load_checkpoint(model_path)

    assert model_path.exists()
    assert metadata_path.exists()
    assert restored.channels == 4
    assert metadata["version"] == "az_v0007"
    assert metadata["epochs"] == 1
    assert create_version_id(8) == "az_v0008"
    assert next_version_id(tmp_path) == "az_v0008"
