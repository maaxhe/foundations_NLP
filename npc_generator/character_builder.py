"""
Helpers for regex-based and language-model-based NPC creation.
"""

from __future__ import annotations

import difflib
import json
import re

from npc_generator.dialogue_engine import DialogueEngine
from npc_generator.npc import NPC
from npc_generator.spec_parser import ALIGNMENTS, EMOTIONAL_STATES, parse_character_specs
from npc_generator.story_generator import StoryGenerator

CANONICAL_FIELDS = {
    "name",
    "race",
    "primary_class",
    "subclass",
    "background",
    "alignment",
    "emotional_state",
    "weapon",
    "level",
    "HP",
    "AC",
    "Str",
    "Dex",
    "Con",
    "Int",
    "Wis",
    "Cha",
    "goal",
    "quirk",
    "secret",
    "story",
    "extra_traits",
}

NUMERIC_FIELDS = {"level", "HP", "AC", "Str", "Dex", "Con", "Int", "Wis", "Cha"}

FIELD_ALIASES = {
    "hp": "HP",
    "ac": "AC",
    "str": "Str",
    "dex": "Dex",
    "con": "Con",
    "int": "Int",
    "wis": "Wis",
    "cha": "Cha",
    "class": "primary_class",
    "archetype": "subclass",
    "mood": "emotional_state",
    "emotion": "emotional_state",
    "hit_points": "HP",
    "armor_class": "AC",
    "strength": "Str",
    "dexterity": "Dex",
    "constitution": "Con",
    "intelligence": "Int",
    "wisdom": "Wis",
    "charisma": "Cha",
    "backstory": "story",
}

MODEL_DETAIL_FIELDS = ("goal", "quirk", "secret")

NAME_HINT_PATTERNS = [
    r"\b(?:named|called|name)\b\s+([A-Za-z][A-Za-z' -]{0,60})",
    r"\b(?:namens|heisst)\b\s+([A-Za-z][A-Za-z' -]{0,60})",
]

NAME_STOP_WORDS = (
    "with",
    "who",
    "from",
    "wields",
    "wielding",
    "uses",
    "using",
    "wears",
    "wearing",
    "carries",
    "carrying",
    "level",
    "hp",
    "ac",
    "background",
    "alignment",
    "weapon",
    "class",
    "race",
    "goal",
    "quirk",
    "secret",
)

FUZZY_FIELD_CONFIGS = (
    ("alignment", ALIGNMENTS, 0.84),
    ("emotional_state", EMOTIONAL_STATES, 0.84),
)

GOAL_FALLBACKS = {
    "Barbarian": "make sure their people are never cornered again",
    "Bard": "turn one buried truth into a story nobody can ignore",
    "Cleric": "find where their faith is needed before it arrives too late",
    "Druid": "keep an old place from being stripped of what makes it sacred",
    "Fighter": "finish a duty others quietly abandoned",
    "Monk": "master the impulse that still threatens their discipline",
    "Paladin": "prove their oath still means something in a compromised world",
    "Ranger": "keep a dangerous route out of the wrong hands",
    "Rogue": "get ahead of the danger closing in around them",
    "Sorcerer": "understand the force inside them before it answers for them",
    "Warlock": "untangle a bargain before its cost reaches someone else",
    "Wizard": "decode a piece of magic no one else has understood",
}

QUIRK_FALLBACKS = {
    "Barbarian": "tests the weight of anything they might need as a weapon",
    "Bard": "quietly rewrites conversations into better dialogue later",
    "Cleric": "touches a charm or symbol before answering difficult questions",
    "Druid": "notices weather shifts before anyone else comments on them",
    "Fighter": "checks the balance of objects in one hand while thinking",
    "Monk": "slows their breathing on purpose whenever tension rises",
    "Paladin": "straightens small signs of disorder without noticing",
    "Ranger": "marks routes and landmarks on any scrap of paper nearby",
    "Rogue": "maps exits before letting their guard down",
    "Sorcerer": "lets small traces of magic leak through strong emotions",
    "Warlock": "pauses a second too long before naming old promises",
    "Wizard": "murmurs fragments of theory under their breath while thinking",
}

SECRET_FALLBACKS = {
    "Barbarian": "they left someone behind when retreat was the only way out",
    "Bard": "one of their most famous stories is closer to confession than performance",
    "Cleric": "they are still waiting for an answer they once swore they already received",
    "Druid": "an ancient place still recognizes them for something they regret",
    "Fighter": "the battle that made their reputation also left a truth buried",
    "Monk": "their calm was built around a mistake they never made peace with",
    "Paladin": "their oath began with a compromise no one else knows about",
    "Ranger": "they are tracking someone for reasons they refuse to explain",
    "Rogue": "they kept proof from a job that should have disappeared years ago",
    "Sorcerer": "their power first surfaced during a moment they never describe",
    "Warlock": "the price of their pact is already closer than they admit",
    "Wizard": "they learned one forbidden thing and have never truly set it down",
}


def build_npc_from_prompt_regex(prompt: str, sampler, story_gen: StoryGenerator) -> NPC:
    overrides = parse_character_specs(prompt, sampler)
    overrides["source_prompt"] = prompt
    char = sampler.sample_character(overrides)
    char["story"] = story_gen.generate_story(char)
    return NPC.from_dict(char)


def can_build_with_language_model(engine: DialogueEngine) -> bool:
    return engine.model is not None and engine.tokenizer is not None


def build_npc_from_prompt_lm(
    prompt: str,
    sampler,
    story_gen: StoryGenerator,
    engine: DialogueEngine,
) -> NPC:
    if not can_build_with_language_model(engine):
        raise RuntimeError("Language-model-based character creation requires a loaded Qwen model.")

    regex_overrides = _normalize_payload(parse_character_specs(prompt, sampler), sampler)
    if "name" not in regex_overrides:
        name_hint = _extract_name_hint(prompt)
        if name_hint:
            regex_overrides["name"] = name_hint

    fuzzy_overrides = _infer_fuzzy_overrides(prompt, sampler, regex_overrides)
    base_overrides = _merge_overrides(fuzzy_overrides, regex_overrides)
    base_overrides["source_prompt"] = prompt

    seed_character = sampler.sample_character(base_overrides, fill_personality_details=False)
    lm_overrides = _complete_character_with_model(prompt, seed_character, base_overrides, engine, sampler)
    merged_overrides = _merge_overrides(lm_overrides, base_overrides)
    merged_overrides["source_prompt"] = prompt

    char = sampler.sample_character(merged_overrides, fill_personality_details=False)
    for field in MODEL_DETAIL_FIELDS:
        char[field] = merged_overrides.get(field) or _fallback_detail(field, char)
    char["story"] = merged_overrides.get("story") or story_gen.generate_story(char)
    return NPC.from_dict(char)


def _complete_character_with_model(
    prompt: str,
    character: dict,
    fixed_overrides: dict,
    engine: DialogueEngine,
    sampler,
) -> dict:
    fixed_values = {
        key: value
        for key, value in fixed_overrides.items()
        if key in CANONICAL_FIELDS and value not in (None, "", {}, [])
    }
    messages = [
        {
            "role": "system",
            "content": (
                "Finalize fantasy NPC creation data and return JSON only. "
                "Allowed keys: name, race, primary_class, subclass, background, alignment, emotional_state, "
                "weapon, level, HP, AC, Str, Dex, Con, Int, Wis, Cha, goal, quirk, secret, story, extra_traits. "
                "Treat fixed_values as locked and do not change them. "
                "If you output a name, output only the name itself, never descriptors, titles, or trailing clauses, and keep it to 1 to 3 words. "
                "Correct obvious typos in fantasy terms when the intent is clear. "
                "For goal, quirk, and secret, create fresh phrasing that fits the character and do not reuse stock phrases verbatim. "
                "Keep story to 2 or 3 sentences."
            ),
        },
        {
            "role": "user",
            "content": (
                f"User prompt:\n{prompt}\n\n"
                "fixed_values:\n"
                f"{json.dumps(fixed_values, ensure_ascii=True, indent=2)}\n\n"
                "current_character_draft:\n"
                f"{_format_character_draft(character)}\n\n"
                "If a field is already fixed, omit it from your JSON."
            ),
        },
    ]
    response = engine.generate_text(
        messages,
        temperature=0.45,
        top_p=0.85,
        max_new_tokens=210,
        repetition_penalty=1.02,
    )
    payload = _load_json_object(response)
    return _normalize_payload(payload, sampler)


def _infer_fuzzy_overrides(prompt: str, sampler, existing_overrides: dict) -> dict:
    candidates = _build_phrase_candidates(prompt)
    inferred: dict[str, object] = {}

    dynamic_fields = [
        ("race", getattr(sampler, "known_races", []), 0.84),
        ("primary_class", getattr(sampler, "known_classes", []), 0.82),
        ("subclass", getattr(sampler, "known_subclasses", []), 0.84),
        ("weapon", getattr(sampler, "known_weapons", []), 0.85),
    ]

    for field, options, cutoff in (*FUZZY_FIELD_CONFIGS, *dynamic_fields):
        if field in existing_overrides:
            continue
        match = _best_fuzzy_option(candidates, options, cutoff)
        if match:
            inferred[field] = match

    if "subclass" in inferred and "primary_class" not in existing_overrides and "primary_class" not in inferred:
        inferred_class = getattr(sampler, "subclass_to_class", {}).get(str(inferred["subclass"]).lower())
        if inferred_class:
            inferred["primary_class"] = inferred_class

    return inferred


def _best_fuzzy_option(candidates: set[str], options: list[str], cutoff: float) -> str | None:
    best_option = None
    best_score = cutoff
    normalized_options = [(_normalize_phrase(option), option) for option in options]

    for candidate in candidates:
        if len(candidate) < 4:
            continue
        for normalized_option, original_option in normalized_options:
            if candidate == normalized_option:
                return original_option
            score = difflib.SequenceMatcher(None, candidate, normalized_option).ratio()
            if score > best_score:
                best_score = score
                best_option = original_option
    return best_option


def _build_phrase_candidates(text: str, max_words: int = 4) -> set[str]:
    tokens = re.findall(r"[a-z0-9']+", text.lower())
    candidates = {" ".join(tokens).strip()}
    for start in range(len(tokens)):
        for width in range(1, max_words + 1):
            chunk = tokens[start:start + width]
            if not chunk:
                continue
            candidates.add(" ".join(chunk))
    return {candidate for candidate in candidates if candidate}


def _merge_overrides(model_overrides: dict, regex_overrides: dict) -> dict:
    merged = dict(model_overrides)
    merged.update(regex_overrides)

    extra_traits: dict[str, str] = {}
    for source in (model_overrides, regex_overrides):
        for key, value in dict(source.get("extra_traits", {})).items():
            cleaned_key = _normalize_extra_key(key)
            cleaned_value = _stringify(value)
            if cleaned_key and cleaned_value:
                extra_traits[cleaned_key] = cleaned_value
    if extra_traits:
        merged["extra_traits"] = extra_traits
    return merged


def _format_character_draft(character: dict) -> str:
    ordered_fields = [
        "name",
        "race",
        "primary_class",
        "subclass",
        "background",
        "alignment",
        "emotional_state",
        "weapon",
        "level",
        "HP",
        "AC",
        "Str",
        "Dex",
        "Con",
        "Int",
        "Wis",
        "Cha",
    ]
    lines = [f"{field}: {character[field]}" for field in ordered_fields if character.get(field) not in (None, "")]
    extra_traits = character.get("extra_traits", {})
    if extra_traits:
        lines.append(f"extra_traits: {json.dumps(extra_traits, ensure_ascii=True, sort_keys=True)}")
    return "\n".join(lines)


def _normalize_payload(payload: dict | None, sampler) -> dict:
    if not isinstance(payload, dict):
        return {}

    normalized: dict[str, object] = {}
    extra_traits: dict[str, str] = {}

    for raw_key, raw_value in payload.items():
        if raw_value in (None, "", [], {}):
            continue
        raw_slug = _normalize_extra_key(raw_key)
        normalized_key = FIELD_ALIASES.get(raw_slug, raw_slug)
        if normalized_key == "extra_traits" and isinstance(raw_value, dict):
            for extra_key, extra_value in raw_value.items():
                cleaned_key = _normalize_extra_key(extra_key)
                cleaned_value = _stringify(extra_value)
                if cleaned_key and cleaned_value:
                    extra_traits[cleaned_key] = cleaned_value
            continue
        if normalized_key not in CANONICAL_FIELDS:
            cleaned_key = _normalize_extra_key(normalized_key)
            cleaned_value = _stringify(raw_value)
            if cleaned_key and cleaned_value:
                extra_traits[cleaned_key] = cleaned_value
            continue

        cleaned_value = _normalize_field_value(normalized_key, raw_value, sampler)
        if cleaned_value not in (None, ""):
            normalized[normalized_key] = cleaned_value

    if extra_traits:
        merged_extra = dict(normalized.get("extra_traits", {}))
        merged_extra.update(extra_traits)
        normalized["extra_traits"] = merged_extra

    return normalized


def _normalize_field_value(field: str, value, sampler):
    if field in NUMERIC_FIELDS:
        try:
            return int(str(value).strip())
        except ValueError:
            return None

    text_value = _stringify(value)
    if not text_value:
        return None

    if field == "name":
        return _clean_name(text_value)
    if field == "alignment":
        return _match_known_value(text_value, ALIGNMENTS) or text_value.title()
    if field == "emotional_state":
        return _match_known_value(text_value, EMOTIONAL_STATES) or text_value.lower()
    if field == "race":
        return _match_known_value(text_value, getattr(sampler, "known_races", [])) or text_value.title()
    if field == "primary_class":
        return _match_known_value(text_value, getattr(sampler, "known_classes", [])) or text_value.title()
    if field == "background":
        return _match_known_value(text_value, getattr(sampler, "known_backgrounds", [])) or text_value.title()
    if field == "subclass":
        return _match_known_value(text_value, getattr(sampler, "known_subclasses", [])) or text_value.title()
    if field == "weapon":
        return _match_known_value(text_value, getattr(sampler, "known_weapons", [])) or text_value.title()
    return text_value


def _match_known_value(value: str, options: list[str]) -> str | None:
    if not value:
        return None
    lowered = _normalize_phrase(value)
    exact_map = {_normalize_phrase(option): option for option in options}
    if lowered in exact_map:
        return exact_map[lowered]

    best_option = None
    best_score = 0.0
    for option in sorted(set(options), key=len, reverse=True):
        option_lower = _normalize_phrase(option)
        if lowered in option_lower or option_lower in lowered:
            return option
        score = difflib.SequenceMatcher(None, lowered, option_lower).ratio()
        if score > best_score:
            best_score = score
            best_option = option
    return best_option if best_score >= 0.82 else None


def _extract_name_hint(prompt: str) -> str | None:
    for pattern in NAME_HINT_PATTERNS:
        match = re.search(pattern, prompt, flags=re.IGNORECASE)
        if match:
            cleaned = _clean_name(match.group(1))
            if cleaned:
                return cleaned
    return None


def _clean_name(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = _stringify(value).strip(" '\".,;:-")
    cleaned = re.sub(r"^(?:named|called|name|namens|heisst)\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.split(r"[,.;:]", cleaned, maxsplit=1)[0]
    stop_pattern = "|".join(NAME_STOP_WORDS)
    cleaned = re.split(rf"\b(?:{stop_pattern})\b", cleaned, maxsplit=1, flags=re.IGNORECASE)[0]
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", cleaned)
    if not words:
        return None
    limited = words[:3]
    return " ".join(word.title() for word in limited)


def _fallback_detail(field: str, character: dict) -> str:
    primary_class = character.get("primary_class", "")
    emotional_state = character.get("emotional_state", "curious")

    if field == "goal":
        base = GOAL_FALLBACKS.get(primary_class, "settle a piece of unfinished business before it catches up with them")
        if emotional_state in {"anxious", "suspicious", "grieving"}:
            return f"quietly {base}"
        return base

    if field == "quirk":
        base = QUIRK_FALLBACKS.get(primary_class, "goes noticeably still before making important decisions")
        if emotional_state in {"proud", "friendly"}:
            return base.replace("quietly ", "")
        return base

    if field == "secret":
        return SECRET_FALLBACKS.get(primary_class, "someone from their past would recognize them for the wrong reason")

    return ""


def _normalize_phrase(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9']+", value.lower()))


def _normalize_extra_key(value) -> str:
    return str(value).strip().lower().replace(" ", "_")


def _stringify(value) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        items = [_stringify(item) for item in value]
        return ", ".join(item for item in items if item)
    return str(value).strip()


def _load_json_object(text: str | None) -> dict | None:
    if not text:
        return None
    json_text = _extract_json_object(text)
    if not json_text:
        return None
    try:
        payload = json.loads(json_text)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _extract_json_object(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:index + 1]
    return None
