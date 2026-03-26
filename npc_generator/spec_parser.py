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


def _extract_labeled_pairs(text: str) -> dict[str, str]:
    pairs: dict[str, str] = {}
    pattern = r"([A-Za-z][A-Za-z _-]{1,30})\s*[:=]\s*(.+?)(?=\s+[A-Za-z][A-Za-z _-]{1,30}\s*[:=]|[,;\n]|$)"
    for key, value in re.findall(pattern, text):
        cleaned_key = key.strip().lower().replace(" ", "_")
        cleaned_value = value.strip(" .,:;")
        if cleaned_value:
            pairs[cleaned_key] = cleaned_value
    return pairs


def _normalize_override_value(field: str, value: str, sampler):
    if field in {"level", "HP", "AC", "Str", "Dex", "Con", "Int", "Wis", "Cha"}:
        try:
            return int(value)
        except ValueError:
            return value
    if field == "race":
        return _match_known_value(value, sampler.known_races) or value.title()
    if field == "primary_class":
        return _match_known_value(value, sampler.known_classes) or value.title()
    if field == "background":
        return _match_known_value(value, sampler.known_backgrounds) or value.title()
    if field == "subclass":
        return _match_known_value(value, getattr(sampler, "known_subclasses", [])) or value.title()
    if field == "weapon":
        return _match_known_value(value, getattr(sampler, "known_weapons", [])) or value.title()
    if field == "alignment":
        return _match_known_value(value, ALIGNMENTS) or value.title()
    if field == "emotional_state":
        return _match_known_value(value, EMOTIONAL_STATES) or value.lower()
    return value


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
    consumed_fields: set[str] = set()
    labeled_pairs = _extract_labeled_pairs(raw)

    name = _extract_named_value(raw, ["name", "named", "called"])
    if name:
        overrides["name"] = name.title()
        consumed_fields.add("name")

    level = _find_number(raw, [r"\blevel\b\s*(?:is|=|:)?\s*(\d+)", r"\blvl\b\s*(?:is|=|:)?\s*(\d+)"])
    hp = _find_number(raw, [r"\bhp\b\s*(?:is|=|:)?\s*(\d+)", r"\bhit points?\b\s*(?:is|=|:)?\s*(\d+)"])
    ac = _find_number(raw, [r"\bac\b\s*(?:is|=|:)?\s*(\d+)", r"\barmor class\b\s*(?:is|=|:)?\s*(\d+)"])
    if level is not None:
        overrides["level"] = level
        consumed_fields.add("level")
    if hp is not None:
        overrides["HP"] = hp
        consumed_fields.add("hp")
    if ac is not None:
        overrides["AC"] = ac
        consumed_fields.add("ac")

    for stat_name, field_name in STAT_KEYS.items():
        value = _find_number(raw, [rf"\b{stat_name}\b\s*(?:is|=|:)?\s*(\d+)"])
        if value is not None:
            overrides[field_name] = value
            consumed_fields.add(stat_name)

    alignment = _match_known_value(raw, ALIGNMENTS)
    if alignment:
        overrides["alignment"] = alignment
        consumed_fields.add("alignment")

    emotional_state = _match_known_value(raw, EMOTIONAL_STATES)
    if emotional_state:
        overrides["emotional_state"] = emotional_state
        consumed_fields.add("emotional_state")

    race = _match_known_value(raw, sampler.known_races)
    if race:
        overrides["race"] = race
        consumed_fields.add("race")

    primary_class = _match_known_value(raw, sampler.known_classes)
    if primary_class:
        overrides["primary_class"] = primary_class
        consumed_fields.add("class")

    background = _match_known_value(raw, sampler.known_backgrounds)
    if background:
        overrides["background"] = background
        consumed_fields.add("background")

    subclass = _match_known_value(raw, getattr(sampler, "known_subclasses", []))
    if not subclass:
        subclass = _extract_named_value(raw, ["subclass", "archetype"])
        if subclass:
            subclass = subclass.title()
    if subclass:
        overrides["subclass"] = subclass
        consumed_fields.add("subclass")
        if "primary_class" not in overrides:
            inferred_class = getattr(sampler, "subclass_to_class", {}).get(subclass.lower())
            if inferred_class:
                overrides["primary_class"] = inferred_class
                consumed_fields.add("class")

    weapon = _match_known_value(raw, getattr(sampler, "known_weapons", []))
    if not weapon:
        weapon = _extract_named_value(raw, ["weapon"])
    if not weapon:
        wield_match = re.search(r"\bwields?\s+(?:an?\s+)?([A-Za-z][A-Za-z' -]{1,40})", raw, flags=re.IGNORECASE)
        if wield_match:
            weapon = wield_match.group(1).strip(" .,:;")
    if weapon:
        overrides["weapon"] = weapon.title()
        consumed_fields.add("weapon")

    goal = _extract_named_value(raw, ["goal", "wants", "motivation"])
    if goal:
        overrides["goal"] = goal
        consumed_fields.add("goal")

    quirk = _extract_named_value(raw, ["quirk"])
    if quirk:
        overrides["quirk"] = quirk
        consumed_fields.add("quirk")

    secret = _extract_named_value(raw, ["secret"])
    if secret:
        overrides["secret"] = secret
        consumed_fields.add("secret")

    alias_map = {
        "hp": "HP",
        "hit_points": "HP",
        "ac": "AC",
        "armor_class": "AC",
        "class": "primary_class",
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
        "emotion": "emotional_state",
        "emotion_state": "emotional_state",
        "mood": "emotional_state",
    }
    extra_traits: dict[str, str] = {}
    for key, value in labeled_pairs.items():
        normalized_key = alias_map.get(key, key)
        if normalized_key in {"name", "race", "primary_class", "subclass", "background", "alignment", "emotional_state", "weapon", "goal", "quirk", "secret", "HP", "AC", "level", "Str", "Dex", "Con", "Int", "Wis", "Cha"}:
            if normalized_key not in overrides:
                normalized_value = _normalize_override_value(normalized_key, value, sampler)
                overrides[normalized_key] = normalized_value
                if normalized_key == "subclass" and "primary_class" not in overrides:
                    inferred_class = getattr(sampler, "subclass_to_class", {}).get(str(normalized_value).lower())
                    if inferred_class:
                        overrides["primary_class"] = inferred_class
            continue
        if key not in consumed_fields:
            extra_traits[normalized_key] = value

    if extra_traits:
        overrides["extra_traits"] = extra_traits

    return overrides


def apply_update_instruction(npc, instruction: str, sampler) -> dict[str, object]:
    updates = parse_character_specs(instruction, sampler)
    if updates:
        for key, value in updates.items():
            setattr(npc, key, value)
    npc.append_note(instruction)
    return updates
