"""Oetongsu machine-learning data utilities."""

from .constants import BOARD_HEIGHT, BOARD_WIDTH, ENCODER_CHANNELS, POLICY_SIZE
from .encoder import decode_piece_planes, encode_position, side_to_move_planes
from .move_index import index_to_move, is_valid_policy_index, legal_moves_to_mask, move_to_index
from .schema import (
    Move,
    Piece,
    PolicyTrainingSample,
    Position,
    SelfPlaySample,
    TrainingPosition,
    ValueTrainingSample,
)

_LAZY_EXPORTS = {
    "PolicyNet": (".model", "PolicyNet"),
    "ValueNet": (".model", "ValueNet"),
    "count_parameters": (".model", "count_parameters"),
    "PolicyValueModel": (".inference", "PolicyValueModel"),
    "RandomPolicyValueModel": (".inference", "RandomPolicyValueModel"),
    "TorchPolicyValueModel": (".inference", "TorchPolicyValueModel"),
    "SelfPlayConfig": (".self_play", "SelfPlayConfig"),
    "SelfPlayGameResult": (".self_play", "SelfPlayGameResult"),
    "play_self_play_game": (".self_play", "play_self_play_game"),
    "AlphaZeroNet": (".alphazero_model", "AlphaZeroNet"),
    "ModelArenaConfig": (".model_arena", "ModelArenaConfig"),
    "ModelArenaResult": (".model_arena", "ModelArenaResult"),
    "RandomModelPlayer": (".model_arena", "RandomModelPlayer"),
    "run_model_arena": (".model_arena", "run_model_arena"),
}


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    module_name, attribute_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value

__all__ = [
    "BOARD_HEIGHT",
    "BOARD_WIDTH",
    "ENCODER_CHANNELS",
    "POLICY_SIZE",
    "Piece",
    "Position",
    "Move",
    "TrainingPosition",
    "PolicyTrainingSample",
    "ValueTrainingSample",
    "SelfPlaySample",
    "encode_position",
    "decode_piece_planes",
    "side_to_move_planes",
    "move_to_index",
    "index_to_move",
    "is_valid_policy_index",
    "legal_moves_to_mask",
    "PolicyNet",
    "ValueNet",
    "count_parameters",
    "PolicyValueModel",
    "RandomPolicyValueModel",
    "TorchPolicyValueModel",
    "SelfPlayConfig",
    "SelfPlayGameResult",
    "play_self_play_game",
    "AlphaZeroNet",
    "ModelArenaConfig",
    "ModelArenaResult",
    "RandomModelPlayer",
    "run_model_arena",
]
