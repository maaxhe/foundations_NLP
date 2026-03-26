"""
Heuristic parser for converting free-form text into NPC overrides.
"""

from __future__ import annotations

import re
from typing import Iterable

ALIGNMENTS = [
    "Lawful Good",
    "Neutral Good",
    "Chaotic Good",
    "Lawful Neutral",
    "True Neutral",
    "Chaotic Neutral",
    "Lawful Evil",
    "Neutral Evil",
    "Chaotic Evil",
]

EMOTIONAL_STATES = [
    "friendly",
    "hostile",
    "cynical",
    "trustful",
    "suspicious",
    "curious",
    "hopeful",
    "grieving",
    "anxious",
    "calm",
    "proud",
]

STAT_KEYS = {
    "str": "Str",
    "strength": "Str",
    "dex": "Dex",
    "dexterity": "Dex",
    "con": "Con",
    "constitution": "Con",
    "int": "Int",
    "intelligence": "Int",
    "wis": "Wis",
    "wisdom": "Wis",
    "cha": "Cha",
    "charisma": "Cha",
}


def _find_number(text: str, patterns: list[str]) -> int | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def _match_known_value(text: str, options: Iterable[str]) -> str | None:
    lowered = text.lower()
    for option in sorted(set(options), key=len, reverse=True):
        escaped = re.escape(option.lower())
        if re.search(rf"(?<!\w){escaped}(?!\w)", lowered):
            return option
    return None


def _extract_named_value(text: str, labels: list[str], stop_tokens: list[str] | None = None) -> str | None:
    stop_tokens = stop_tokens or [
        "level", "lvl", "hp", "hit", "ac", "armor", "weapon", "subclass",
        "archetype", "background", "alignment", "race", "class", "goal",
        "quirk", "secret", "str", "dex", "con", "int", "wis", "cha",
        "lawful", "neutral", "chaotic", "friendly", "hostile", "cynical",
        "trustful", "suspicious", "curious", "hopeful", "grieving", "anxious",
        "calm", "proud",
    ]
    stop_pattern = "|".join(re.escape(token) for token in stop_tokens)
    for label in labels:
        pattern = rf"\b{label}\b\s*(?:is|=|:)?\s*([A-Za-z][A-Za-z' -]{{1,40}}?)(?=\s+(?:{stop_pattern})\b|$)"
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip(" .,:;")
    return None


def parse_character_specs(text: str, sampler) -> dict:
    raw = text.strip()
    if not raw:
        return {}

    overrides: dict[str, object] = {}

    name = _extract_named_value(raw, ["name", "named", "called"])
    if name:
        overrides["name"] = name.title()

    level = _find_number(raw, [r"\blevel\b\s*(?:is|=|:)?\s*(\d+)", r"\blvl\b\s*(?:is|=|:)?\s*(\d+)"])
    hp = _find_number(raw, [r"\bhp\b\s*(?:is|=|:)?\s*(\d+)", r"\bhit points?\b\s*(?:is|=|:)?\s*(\d+)"])
    ac = _find_number(raw, [r"\bac\b\s*(?:is|=|:)?\s*(\d+)", r"\barmor class\b\s*(?:is|=|:)?\s*(\d+)"])
    if level is not None:
        overrides["level"] = level
    if hp is not None:
        overrides["HP"] = hp
    if ac is not None:
        overrides["AC"] = ac

    for stat_name, field_name in STAT_KEYS.items():
        value = _find_number(raw, [rf"\b{stat_name}\b\s*(?:is|=|:)?\s*(\d+)"])
        if value is not None:
            overrides[field_name] = value

    alignment = _match_known_value(raw, ALIGNMENTS)
    if alignment:
        overrides["alignment"] = alignment

    emotional_state = _match_known_value(raw, EMOTIONAL_STATES)
    if emotional_state:
        overrides["emotional_state"] = emotional_state

    race = _match_known_value(raw, sampler.known_races)
    if race:
        overrides["race"] = race

    primary_class = _match_known_value(raw, sampler.known_classes)
    if primary_class:
        overrides["primary_class"] = primary_class

    background = _match_known_value(raw, sampler.known_backgrounds)
    if background:
        overrides["background"] = background

    subclass = _extract_named_value(raw, ["subclass", "archetype"])
    if subclass:
        overrides["subclass"] = subclass.title()

    weapon = _extract_named_value(raw, ["weapon"])
    if not weapon:
        wield_match = re.search(r"\bwields?\s+(?:an?\s+)?([A-Za-z][A-Za-z' -]{1,40})", raw, flags=re.IGNORECASE)
        if wield_match:
            weapon = wield_match.group(1).strip(" .,:;")
    if weapon:
        overrides["weapon"] = weapon.title()

    goal = _extract_named_value(raw, ["goal", "wants", "motivation"])
    if goal:
        overrides["goal"] = goal

    quirk = _extract_named_value(raw, ["quirk"])
    if quirk:
        overrides["quirk"] = quirk

    secret = _extract_named_value(raw, ["secret"])
    if secret:
        overrides["secret"] = secret

    return overrides


def apply_update_instruction(npc, instruction: str, sampler) -> dict[str, object]:
    updates = parse_character_specs(instruction, sampler)
    if updates:
        for key, value in updates.items():
            setattr(npc, key, value)
    npc.append_note(instruction)
    return updates
