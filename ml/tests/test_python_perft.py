from oetongsu_ml.perft import perft, perft_divide
from oetongsu_ml.python_rules import create_initial_position


def test_initial_inner_inner_perft_depth_1():
    assert perft(create_initial_position("inner-elephant", "inner-elephant"), 1) == 31


def test_initial_inner_inner_perft_depth_2():
    assert perft(create_initial_position("inner-elephant", "inner-elephant"), 2) == 949


def test_initial_inner_inner_perft_depth_3():
    assert perft(create_initial_position("inner-elephant", "inner-elephant"), 3) == 29697


def test_perft_divide_sums_to_perft():
    position = create_initial_position("inner-elephant", "inner-elephant")
    rows = perft_divide(position, 2)

    assert len(rows) == 31
    assert sum(row["nodes"] for row in rows) == 949
    assert all("move" in row and "move_index" in row and "nodes" in row for row in rows)
