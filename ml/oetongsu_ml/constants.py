from __future__ import annotations

BOARD_WIDTH = 9
BOARD_HEIGHT = 10

SIDES = ("CHO", "HAN")
PIECE_KINDS = ("GENERAL", "GUARD", "ELEPHANT", "HORSE", "CHARIOT", "CANNON", "SOLDIER")

PIECE_CHANNELS = len(SIDES) * len(PIECE_KINDS)
SIDE_TO_MOVE_CHANNELS = len(SIDES)
ENCODER_CHANNELS = PIECE_CHANNELS + SIDE_TO_MOVE_CHANNELS

POLICY_SIZE = BOARD_WIDTH * BOARD_HEIGHT * BOARD_WIDTH * BOARD_HEIGHT

PIECE_TO_CHANNEL = {
    (side, kind): side_index * len(PIECE_KINDS) + kind_index
    for side_index, side in enumerate(SIDES)
    for kind_index, kind in enumerate(PIECE_KINDS)
}

CHANNEL_TO_PIECE = {channel: piece for piece, channel in PIECE_TO_CHANNEL.items()}

