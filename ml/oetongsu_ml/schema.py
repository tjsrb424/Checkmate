from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Literal, TypeAlias

Side: TypeAlias = Literal["CHO", "HAN"]
PieceKind: TypeAlias = Literal["GENERAL", "GUARD", "ELEPHANT", "HORSE", "CHARIOT", "CANNON", "SOLDIER"]


@dataclass(frozen=True)
class Piece:
    side: Side
    kind: PieceKind

    @classmethod
    def from_raw(cls, raw: "Piece | dict[str, Any] | None") -> "Piece | None":
        if raw is None:
            return None
        if isinstance(raw, Piece):
            return raw
        return cls(side=raw["side"], kind=raw["kind"])


@dataclass(frozen=True)
class Position:
    x: int
    y: int

    @classmethod
    def from_raw(cls, raw: "Position | dict[str, Any]") -> "Position":
        if isinstance(raw, Position):
            return raw
        return cls(x=int(raw["x"]), y=int(raw["y"]))


@dataclass(frozen=True)
class Move:
    from_: Position
    to: Position

    @classmethod
    def from_raw(cls, raw: "Move | dict[str, Any]") -> "Move":
        if isinstance(raw, Move):
            return raw
        from_raw = raw.get("from", raw.get("from_"))
        return cls(from_=Position.from_raw(from_raw), to=Position.from_raw(raw["to"]))

    def to_json(self) -> dict[str, Any]:
        return {"from": asdict(self.from_), "to": asdict(self.to)}


@dataclass
class TrainingPosition:
    board: list[list[Piece | None]]
    turn: Side
    history: list[Move] = field(default_factory=list)
    position_history: list[str] = field(default_factory=list)
    winner: Side | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_raw(cls, raw: "TrainingPosition | dict[str, Any]") -> "TrainingPosition":
        if isinstance(raw, TrainingPosition):
            return raw
        board = [[Piece.from_raw(cell) for cell in row] for row in raw["board"]]
        history = [Move.from_raw(move) for move in raw.get("history", [])]
        position_history = [str(key) for key in raw.get("position_history", raw.get("positionHistory", []))]
        return cls(
            board=board,
            turn=raw["turn"],
            history=history,
            position_history=position_history,
            winner=raw.get("winner"),
            metadata=dict(raw.get("metadata", {})),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "board": [[asdict(cell) if cell else None for cell in row] for row in self.board],
            "turn": self.turn,
            "history": [move.to_json() for move in self.history],
            "position_history": self.position_history,
            "winner": self.winner,
            "metadata": self.metadata,
        }


@dataclass
class PolicyTrainingSample:
    position: TrainingPosition
    move: Move
    move_index: int
    result: float | int | str | None = None
    source: str | None = None

    @classmethod
    def from_raw(cls, raw: "PolicyTrainingSample | dict[str, Any]") -> "PolicyTrainingSample":
        if isinstance(raw, PolicyTrainingSample):
            return raw
        return cls(
            position=TrainingPosition.from_raw(raw["position"]),
            move=Move.from_raw(raw["move"]),
            move_index=int(raw["move_index"]),
            result=raw.get("result"),
            source=raw.get("source"),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "position": self.position.to_json(),
            "move": self.move.to_json(),
            "move_index": self.move_index,
            "result": self.result,
            "source": self.source,
        }


@dataclass
class ValueTrainingSample:
    position: TrainingPosition
    value: float
    result: float | int | str | None = None
    source: str | None = None

    @classmethod
    def from_raw(cls, raw: "ValueTrainingSample | dict[str, Any]") -> "ValueTrainingSample":
        if isinstance(raw, ValueTrainingSample):
            return raw
        return cls(
            position=TrainingPosition.from_raw(raw["position"]),
            value=float(raw["value"]),
            result=raw.get("result"),
            source=raw.get("source"),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "position": self.position.to_json(),
            "value": self.value,
            "result": self.result,
            "source": self.source,
        }


@dataclass
class SelfPlaySample:
    position: TrainingPosition
    policy_target: list[float]
    value_target: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_raw(cls, raw: "SelfPlaySample | dict[str, Any]") -> "SelfPlaySample":
        if isinstance(raw, SelfPlaySample):
            return raw
        return cls(
            position=TrainingPosition.from_raw(raw["position"]),
            policy_target=[float(value) for value in raw["policy_target"]],
            value_target=float(raw["value_target"]),
            metadata=dict(raw.get("metadata", {})),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "position": self.position.to_json(),
            "policy_target": self.policy_target,
            "value_target": self.value_target,
            "metadata": self.metadata,
        }


def to_jsonable(value: Any) -> Any:
    if hasattr(value, "to_json"):
        return value.to_json()
    if is_dataclass(value):
        return asdict(value)
    return value
