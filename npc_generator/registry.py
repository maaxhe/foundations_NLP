"""
Persistent storage for generated NPCs.
"""

from __future__ import annotations

import json
from pathlib import Path

from npc_generator.npc import NPC

DEFAULT_STORAGE_PATH = Path(__file__).resolve().parent.parent / "data_sets" / "generated_npcs.json"


class NpcRegistry:
    def __init__(self, storage_path: Path | str = DEFAULT_STORAGE_PATH):
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._npcs: dict[str, NPC] = {}
        self._load()

    def _load(self) -> None:
        if not self.storage_path.exists():
            return
        try:
            payload = json.loads(self.storage_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = []
        if isinstance(payload, dict):
            payload = payload.get("npcs", [])
        for item in payload:
            npc = NPC.from_dict(item)
            self._npcs[npc.npc_id] = npc

    def save(self) -> None:
        data = [npc.to_dict() for npc in self._npcs.values()]
        self.storage_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def upsert(self, npc: NPC) -> None:
        self._npcs[npc.npc_id] = npc
        self.save()

    def remove(self, npc: NPC) -> None:
        del self._npcs[npc.npc_id]
        self.save()

    def all(self) -> list[NPC]:
        return list(self._npcs.values())

    def count(self) -> int:
        return len(self._npcs)

    def resolve(self, reference: str | None) -> NPC | None:
        if not reference:
            return None
        ref = reference.strip()
        if ref.isdigit():
            index = int(ref) - 1
            items = self.all()
            if 0 <= index < len(items):
                return items[index]
            return None
        lower = ref.lower()
        if ref in self._npcs:
            return self._npcs[ref]
        for npc in self._npcs.values():
            if npc.name.lower() == lower or npc.npc_id.lower() == lower:
                return npc
        return None
