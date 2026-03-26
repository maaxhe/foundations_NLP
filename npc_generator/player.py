"""
player.py
Player state: level, XP, gold, quest log.
"""

from dataclasses import dataclass, field

XP_PER_LEVEL = 100  # XP needed to level up


@dataclass
class QuestEntry:
    title: str
    description: str
    giver: str
    reward_xp: int
    reward_gold: int
    status: str = "active"  # active | completed


@dataclass
class Player:
    name: str = "Adventurer"
    level: int = 1
    xp: int = 0
    gold: int = 10
    active_quests: list = field(default_factory=list)
    completed_quests: list = field(default_factory=list)

    def xp_to_next(self) -> int:
        return XP_PER_LEVEL * self.level - self.xp

    def add_quest(self, quest: QuestEntry):
        self.active_quests.append(quest)

    def complete_quest(self, index: int) -> QuestEntry | None:
        if not 0 <= index < len(self.active_quests):
            return None
        quest = self.active_quests.pop(index)
        quest.status = "completed"
        self.completed_quests.append(quest)
        self.gold += quest.reward_gold
        self.xp += quest.reward_xp
        leveled_up = False
        while self.xp >= XP_PER_LEVEL * self.level:
            self.xp -= XP_PER_LEVEL * self.level
            self.level += 1
            leveled_up = True
        return quest, leveled_up
