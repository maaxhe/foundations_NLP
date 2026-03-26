"""
Core NPC data model used by the command line generator and chat loop.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any
from uuid import uuid4


@dataclass
class NPC:
    npc_id: str = field(default_factory=lambda: uuid4().hex[:8])
    name: str = "Unknown Wanderer"
    race: str = "Human"
    primary_class: str = "Fighter"
    subclass: str = "Veteran"
    background: str = "Outlander"
    alignment: str = "True Neutral"
    emotional_state: str = "curious"
    weapon: str = "Longsword"
    level: int = 1
    HP: int = 10
    AC: int = 10
    Str: int = 10
    Dex: int = 10
    Con: int = 10
    Int: int = 10
    Wis: int = 10
    Cha: int = 10
    story: str = ""
    goal: str = ""
    quirk: str = ""
    secret: str = ""
    notes: list[str] = field(default_factory=list)
    source_prompt: str = ""
    history: list[dict[str, str]] = field(default_factory=list, repr=False)

    @property
    def stat_block(self) -> str:
        return (
            f"STR {self.Str:2d}  DEX {self.Dex:2d}  CON {self.Con:2d}  "
            f"INT {self.Int:2d}  WIS {self.Wis:2d}  CHA {self.Cha:2d}"
        )

    @property
    def short_label(self) -> str:
        return f"{self.name} [{self.npc_id}]"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "NPC":
        allowed_fields = {field.name for field in cls.__dataclass_fields__.values()}
        cleaned = {key: value for key, value in payload.items() if key in allowed_fields}
        if "npc_id" not in cleaned or not cleaned["npc_id"]:
            cleaned["npc_id"] = uuid4().hex[:8]
        return cls(**cleaned)

    def record_turn(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})
        if len(self.history) > 24:
            self.history = self.history[-24:]

    def append_note(self, note: str) -> None:
        note = note.strip()
        if note:
            self.notes.append(note)
        if len(self.notes) > 12:
            self.notes = self.notes[-12:]
