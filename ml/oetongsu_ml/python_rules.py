from __future__ import annotations

from .schema import Move, Piece, Position, Side, TrainingPosition

Formation = str

FORMATION_HOME_PIECES: dict[Formation, dict[int, str]] = {
    "inner-elephant": {1: "HORSE", 2: "ELEPHANT", 6: "ELEPHANT", 7: "HORSE"},
    "outer-elephant": {1: "ELEPHANT", 2: "HORSE", 6: "HORSE", 7: "ELEPHANT"},
    "left-elephant": {1: "ELEPHANT", 2: "HORSE", 6: "ELEPHANT", 7: "HORSE"},
    "right-elephant": {1: "HORSE", 2: "ELEPHANT", 6: "HORSE", 7: "ELEPHANT"},
}

ORTHOGONAL_DIRECTIONS = (
    Position(1, 0),
    Position(-1, 0),
    Position(0, 1),
    Position(0, -1),
)

HORSE_STEPS = (
    (2, 1, Position(1, 0)),
    (2, -1, Position(1, 0)),
    (-2, 1, Position(-1, 0)),
    (-2, -1, Position(-1, 0)),
    (1, 2, Position(0, 1)),
    (-1, 2, Position(0, 1)),
    (1, -2, Position(0, -1)),
    (-1, -2, Position(0, -1)),
)

ELEPHANT_STEPS = (
    (3, 2, (Position(1, 0), Position(2, 1))),
    (3, -2, (Position(1, 0), Position(2, -1))),
    (-3, 2, (Position(-1, 0), Position(-2, 1))),
    (-3, -2, (Position(-1, 0), Position(-2, -1))),
    (2, 3, (Position(0, 1), Position(1, 2))),
    (-2, 3, (Position(0, 1), Position(-1, 2))),
    (2, -3, (Position(0, -1), Position(1, -2))),
    (-2, -3, (Position(0, -1), Position(-1, -2))),
)

PALACE_LINES = (
    (Position(3, 0), Position(4, 1), Position(5, 2)),
    (Position(5, 0), Position(4, 1), Position(3, 2)),
    (Position(3, 7), Position(4, 8), Position(5, 9)),
    (Position(5, 7), Position(4, 8), Position(3, 9)),
)


def other_side(side: Side) -> Side:
    return "HAN" if side == "CHO" else "CHO"


def create_initial_position(
    cho_formation: Formation = "inner-elephant",
    han_formation: Formation = "inner-elephant",
    turn: Side = "CHO",
) -> TrainingPosition:
    board: list[list[Piece | None]] = [[None for _ in range(9)] for _ in range(10)]
    place_side(board, "HAN", han_formation)
    place_side(board, "CHO", cho_formation)
    return TrainingPosition(
        board=board,
        turn=turn,
        metadata={"choFormation": cho_formation, "hanFormation": han_formation},
    )


def generate_legal_moves(position: TrainingPosition | dict, side: Side | None = None) -> list[Move]:
    parsed = TrainingPosition.from_raw(position)
    moving_side = side or parsed.turn
    moves: list[Move] = []
    for move in generate_pseudo_moves(parsed, moving_side):
        next_position = apply_move(parsed, move, append_history=False)
        if not is_in_check(next_position, moving_side):
            moves.append(move)
    return moves


def generate_pseudo_moves(position: TrainingPosition | dict, side: Side) -> list[Move]:
    parsed = TrainingPosition.from_raw(position)
    moves: list[Move] = []
    for y, row in enumerate(parsed.board):
        for x, piece in enumerate(row):
            if piece is None or piece.side != side:
                continue
            moves.extend(piece_moves(parsed.board, Position(x, y), piece))
    return moves


def apply_move(position: TrainingPosition | dict, move: Move | dict, append_history: bool = True) -> TrainingPosition:
    parsed = TrainingPosition.from_raw(position)
    parsed_move = Move.from_raw(move)
    board = clone_board(parsed.board)
    moving_piece = get_piece(board, parsed_move.from_)
    board[parsed_move.from_.y][parsed_move.from_.x] = None
    board[parsed_move.to.y][parsed_move.to.x] = moving_piece
    next_turn = other_side(parsed.turn)
    history = [*parsed.history, parsed_move] if append_history else list(parsed.history)
    winner = parsed.turn if append_history and is_checkmate_board(board, next_turn) else None
    return TrainingPosition(board=board, turn=next_turn, history=history, winner=winner, metadata=dict(parsed.metadata))


def is_checkmate(position: TrainingPosition | dict, side: Side) -> bool:
    parsed = TrainingPosition.from_raw(position)
    return is_in_check(parsed, side) and len(generate_legal_moves(parsed, side)) == 0


def is_in_check(position: TrainingPosition | dict, side: Side) -> bool:
    parsed = TrainingPosition.from_raw(position)
    general = find_general(parsed.board, side)
    if general is None:
        return True
    if is_facing_enemy_general(parsed.board, side, general):
        return True
    return is_square_attacked(parsed.board, general, other_side(side))


def piece_moves(board: list[list[Piece | None]], from_pos: Position, piece: Piece) -> list[Move]:
    if piece.kind in ("GENERAL", "GUARD"):
        return palace_step_moves(board, from_pos, piece)
    if piece.kind == "CHARIOT":
        return [*ray_moves(board, from_pos, piece, ORTHOGONAL_DIRECTIONS), *palace_ray_moves(board, from_pos, piece, False)]
    if piece.kind == "HORSE":
        return horse_moves(board, from_pos, piece)
    if piece.kind == "ELEPHANT":
        return elephant_moves(board, from_pos, piece)
    if piece.kind == "CANNON":
        return [*cannon_ray_moves(board, from_pos, piece, ORTHOGONAL_DIRECTIONS), *palace_ray_moves(board, from_pos, piece, True)]
    if piece.kind == "SOLDIER":
        return soldier_moves(board, from_pos, piece)
    return []


def ray_moves(board: list[list[Piece | None]], from_pos: Position, piece: Piece, directions: tuple[Position, ...]) -> list[Move]:
    moves: list[Move] = []
    for direction in directions:
        to = Position(from_pos.x + direction.x, from_pos.y + direction.y)
        while in_bounds(to):
            target = get_piece(board, to)
            if target is None:
                moves.append(Move(from_pos, to))
            else:
                if target.side != piece.side:
                    moves.append(Move(from_pos, to))
                break
            to = Position(to.x + direction.x, to.y + direction.y)
    return moves


def horse_moves(board: list[list[Piece | None]], from_pos: Position, piece: Piece) -> list[Move]:
    moves: list[Move] = []
    for dx, dy, block in HORSE_STEPS:
        if get_piece(board, Position(from_pos.x + block.x, from_pos.y + block.y)) is not None:
            continue
        push_if_available(moves, board, from_pos, Position(from_pos.x + dx, from_pos.y + dy), piece)
    return moves


def elephant_moves(board: list[list[Piece | None]], from_pos: Position, piece: Piece) -> list[Move]:
    moves: list[Move] = []
    for dx, dy, blocks in ELEPHANT_STEPS:
        if any(get_piece(board, Position(from_pos.x + block.x, from_pos.y + block.y)) is not None for block in blocks):
            continue
        push_if_available(moves, board, from_pos, Position(from_pos.x + dx, from_pos.y + dy), piece)
    return moves


def cannon_ray_moves(board: list[list[Piece | None]], from_pos: Position, piece: Piece, directions: tuple[Position, ...]) -> list[Move]:
    moves: list[Move] = []
    for direction in directions:
        screen: Piece | None = None
        to = Position(from_pos.x + direction.x, from_pos.y + direction.y)
        while in_bounds(to):
            target = get_piece(board, to)
            if screen is None:
                if target is not None:
                    if target.kind == "CANNON":
                        break
                    screen = target
            elif target is None:
                moves.append(Move(from_pos, to))
            else:
                if target.side != piece.side and target.kind != "CANNON":
                    moves.append(Move(from_pos, to))
                break
            to = Position(to.x + direction.x, to.y + direction.y)
    return moves


def palace_step_moves(board: list[list[Piece | None]], from_pos: Position, piece: Piece) -> list[Move]:
    moves: list[Move] = []
    for direction in ORTHOGONAL_DIRECTIONS:
        to = Position(from_pos.x + direction.x, from_pos.y + direction.y)
        if is_in_palace(to, piece.side):
            push_if_available(moves, board, from_pos, to, piece)
    for to in adjacent_palace_diagonal_positions(from_pos):
        if is_in_palace(to, piece.side):
            push_if_available(moves, board, from_pos, to, piece)
    return moves


def soldier_moves(board: list[list[Piece | None]], from_pos: Position, piece: Piece) -> list[Move]:
    moves: list[Move] = []
    forward = -1 if piece.side == "CHO" else 1
    for direction in (Position(0, forward), Position(1, 0), Position(-1, 0)):
        push_if_available(moves, board, from_pos, Position(from_pos.x + direction.x, from_pos.y + direction.y), piece)
    for to in adjacent_palace_diagonal_positions(from_pos):
        if to.y - from_pos.y == forward and is_in_palace(to, other_side(piece.side)):
            push_if_available(moves, board, from_pos, to, piece)
    return moves


def palace_ray_moves(board: list[list[Piece | None]], from_pos: Position, piece: Piece, cannon: bool) -> list[Move]:
    moves: list[Move] = []
    for line in palace_lines_containing(from_pos):
        index = line.index(from_pos)
        for direction in (-1, 1):
            screen: Piece | None = None
            i = index + direction
            while 0 <= i < len(line):
                to = line[i]
                target = get_piece(board, to)
                if not cannon:
                    if target is None:
                        moves.append(Move(from_pos, to))
                    else:
                        if target.side != piece.side:
                            moves.append(Move(from_pos, to))
                        break
                else:
                    if screen is None:
                        if target is not None:
                            if target.kind == "CANNON":
                                break
                            screen = target
                    elif target is None:
                        moves.append(Move(from_pos, to))
                    else:
                        if target.side != piece.side and target.kind != "CANNON":
                            moves.append(Move(from_pos, to))
                        break
                i += direction
    return moves


def push_if_available(moves: list[Move], board: list[list[Piece | None]], from_pos: Position, to: Position, piece: Piece) -> None:
    if not in_bounds(to):
        return
    target = get_piece(board, to)
    if target is None or target.side != piece.side:
        moves.append(Move(from_pos, to))


def is_square_attacked(board: list[list[Piece | None]], target: Position, by_side: Side) -> bool:
    for y, row in enumerate(board):
        for x, piece in enumerate(row):
            if piece is None or piece.side != by_side:
                continue
            if any(move.to == target for move in piece_moves(board, Position(x, y), piece)):
                return True
    return False


def find_general(board: list[list[Piece | None]], side: Side) -> Position | None:
    for y, row in enumerate(board):
        for x, piece in enumerate(row):
            if piece is not None and piece.side == side and piece.kind == "GENERAL":
                return Position(x, y)
    return None


def is_facing_enemy_general(board: list[list[Piece | None]], side: Side, general: Position) -> bool:
    enemy = find_general(board, other_side(side))
    if enemy is None or enemy.x != general.x:
        return False
    step = 1 if enemy.y > general.y else -1
    for y in range(general.y + step, enemy.y, step):
        if board[y][general.x] is not None:
            return False
    return True


def is_checkmate_board(board: list[list[Piece | None]], side: Side) -> bool:
    return is_checkmate(TrainingPosition(board=board, turn=side), side)


def palace_lines_containing(pos: Position) -> list[tuple[Position, ...]]:
    return [line for line in PALACE_LINES if pos in line]


def adjacent_palace_diagonal_positions(pos: Position) -> list[Position]:
    result: list[Position] = []
    for line in palace_lines_containing(pos):
        index = line.index(pos)
        for next_index in (index - 1, index + 1):
            if 0 <= next_index < len(line):
                result.append(line[next_index])
    return result


def is_in_palace(pos: Position, side: Side) -> bool:
    min_y = 0 if side == "HAN" else 7
    max_y = 2 if side == "HAN" else 9
    return 3 <= pos.x <= 5 and min_y <= pos.y <= max_y


def get_piece(board: list[list[Piece | None]], pos: Position) -> Piece | None:
    if not in_bounds(pos):
        return None
    return board[pos.y][pos.x]


def in_bounds(pos: Position) -> bool:
    return 0 <= pos.x < 9 and 0 <= pos.y < 10


def clone_board(board: list[list[Piece | None]]) -> list[list[Piece | None]]:
    return [[Piece(piece.side, piece.kind) if piece is not None else None for piece in row] for row in board]


def place_side(board: list[list[Piece | None]], side: Side, formation: Formation) -> None:
    if formation not in FORMATION_HOME_PIECES:
        raise ValueError(f"unknown formation: {formation}")
    home_y = 0 if side == "HAN" else 9
    general_y = 1 if side == "HAN" else 8
    cannon_y = 2 if side == "HAN" else 7
    soldier_y = 3 if side == "HAN" else 6

    board[home_y][0] = Piece(side, "CHARIOT")
    board[home_y][8] = Piece(side, "CHARIOT")
    board[home_y][3] = Piece(side, "GUARD")
    board[home_y][5] = Piece(side, "GUARD")
    board[general_y][4] = Piece(side, "GENERAL")
    board[cannon_y][1] = Piece(side, "CANNON")
    board[cannon_y][7] = Piece(side, "CANNON")

    for x in (0, 2, 4, 6, 8):
        board[soldier_y][x] = Piece(side, "SOLDIER")

    for x, kind in FORMATION_HOME_PIECES[formation].items():
        board[home_y][x] = Piece(side, kind)
