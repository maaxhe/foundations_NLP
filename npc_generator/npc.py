"""
npc.py
NPC dataclass that bundles character stats, story, quest, and dialogue.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class NPC:
    name: str
    race: str
    primary_class: str
    background: str
    alignment: str
    level: int
    HP: int
    AC: int
    Str: int
    Dex: int
    Con: int
    Int: int
    Wis: int
    Cha: int
    story: str = ""
    quest: str = ""
    persona_prompt: str = ""
    # Dialogue history (DialoGPT token ids tensor, or None)
    _history: Optional[object] = field(default=None, repr=False)

    @property
    def stat_block(self) -> str:
        return (
            f"STR {self.Str:2d}  DEX {self.Dex:2d}  CON {self.Con:2d}  "
            f"INT {self.Int:2d}  WIS {self.Wis:2d}  CHA {self.Cha:2d}"
        )
