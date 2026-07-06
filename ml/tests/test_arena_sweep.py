from oetongsu_ml.arena_sweep import main


def test_arena_sweep_quick_runs(capsys):
    assert main(["--quick", "--versions", "--maxPlies", "0", "--games", "2", "--simulations", "1"]) == 0

    output = capsys.readouterr().out
    assert "maxPlies | pair | score_adj" in output
    assert "random vs random" in output
