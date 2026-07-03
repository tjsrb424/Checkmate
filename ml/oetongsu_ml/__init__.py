"""Oetongsu machine-learning data utilities."""

from .constants import BOARD_HEIGHT, BOARD_WIDTH, ENCODER_CHANNELS, POLICY_SIZE
from .encoder import decode_piece_planes, encode_position, side_to_move_planes
from .model import PolicyNet, ValueNet, count_parameters
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
from .inference import PolicyValueModel, RandomPolicyValueModel, TorchPolicyValueModel
from .self_play import SelfPlayConfig, SelfPlayGameResult, play_self_play_game
from .alphazero_model import AlphaZeroNet
from .model_arena import ModelArenaConfig, ModelArenaResult, RandomModelPlayer, run_model_arena

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
