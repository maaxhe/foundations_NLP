from npc_generator.character_builder import (
    build_npc_from_prompt_lm,
    build_npc_from_prompt_regex,
)
from npc_generator.character_sampler import CharacterSampler
from npc_generator.story_generator import StoryGenerator


class FakeLmEngine:
    def __init__(self, *responses: str):
        self.responses = list(responses)
        self.model = object()
        self.tokenizer = object()

    def generate_text(self, messages, **kwargs):
        assert messages
        assert kwargs
        if not self.responses:
            raise AssertionError("No fake LM response left.")
        return self.responses.pop(0)


def test_regex_builder_stays_compatible():
    sampler = CharacterSampler()
    story_gen = StoryGenerator()

    npc = build_npc_from_prompt_regex(
        "elf rogue named Mira level 7 hp 42 weapon rapier chaotic good suspicious",
        sampler,
        story_gen,
    )

    assert npc.race == "Elf"
    assert npc.primary_class == "Rogue"
    assert npc.name == "Mira"
    assert npc.level == 7
    assert npc.HP == 42
    assert npc.weapon == "Rapier"
    assert npc.alignment == "Chaotic Good"
    assert npc.emotional_state == "suspicious"
    assert npc.story


def test_lm_builder_uses_model_analysis_and_generated_details():
    sampler = CharacterSampler()
    story_gen = StoryGenerator()
    engine = FakeLmEngine(
        """{
          "goal": "chart forgotten moon paths",
          "quirk": "folds every map into perfect squares",
          "secret": "is secretly tracking a missing sibling",
          "story": "Liora learned to read the wild by moonlight and old trail markers. As an Outlander ranger, she trusts maps almost as much as instinct. She now travels with a longbow, chasing clues that might lead her back to family.",
          "background": "Outlander",
          "weapon": "Longbow",
          "extra_traits": {
            "hometown": "Waterdeep",
            "eye_color": "silver"
          }
        }""",
    )

    npc = build_npc_from_prompt_lm(
        "An elf ranger named Liora with silver eyes from Waterdeep.",
        sampler,
        story_gen,
        engine,
    )

    assert npc.name == "Liora"
    assert npc.race == "Elf"
    assert npc.primary_class == "Ranger"
    assert npc.background == "Outlander"
    assert npc.weapon == "Longbow"
    assert npc.goal == "chart forgotten moon paths"
    assert npc.quirk == "folds every map into perfect squares"
    assert npc.secret == "is secretly tracking a missing sibling"
    assert "moonlight" in npc.story
    assert npc.extra_traits["hometown"] == "Waterdeep"
    assert npc.extra_traits["eye_color"] == "silver"


def test_lm_builder_preserves_explicit_regex_fields_over_model_guess():
    sampler = CharacterSampler()
    story_gen = StoryGenerator()
    engine = FakeLmEngine(
        """{
          "primary_class": "Wizard",
          "weapon": "Quarterstaff",
          "background": "Criminal",
          "goal": "stay ahead of old enemies",
          "quirk": "counts exits before sitting down",
          "secret": "keeps one stolen letter hidden in a boot",
          "story": "Mira learned early that trust is expensive. Her rogue training sharpened that lesson into instinct. Even when she seems calm, she is measuring the room and the people in it. The rapier at her side reminds her to move first when danger closes in."
        }""",
    )

    npc = build_npc_from_prompt_lm(
        "human rogue named Mira weapon rapier",
        sampler,
        story_gen,
        engine,
    )

    assert npc.primary_class == "Rogue"
    assert npc.weapon == "Rapier"
    assert npc.background == "Criminal"
    assert npc.goal == "stay ahead of old enemies"


def test_lm_builder_cleans_name_without_trailing_description():
    sampler = CharacterSampler()
    story_gen = StoryGenerator()
    engine = FakeLmEngine(
        """{
          "name": "Liora from Waterdeep with silver eyes",
          "primary_class": "Ranger",
          "goal": "trace a forgotten trail before rivals reach it",
          "quirk": "keeps route sketches folded into careful squares",
          "secret": "still reads letters from someone presumed gone",
          "story": "Liora learned to trust moonlight, wind, and silence over crowded rooms. The road keeps pulling her toward clues other people overlook. She follows them with a patience that looks calmer than it feels."
        }""",
    )

    npc = build_npc_from_prompt_lm(
        "An elf scout from Waterdeep with silver eyes.",
        sampler,
        story_gen,
        engine,
    )

    assert npc.name == "Liora"


def test_lm_builder_handles_small_typos_with_fuzzy_matching():
    sampler = CharacterSampler()
    story_gen = StoryGenerator()
    engine = FakeLmEngine(
        """{
          "goal": "stay one step ahead of the people looking for her",
          "quirk": "checks every doorway before she settles in",
          "secret": "keeps one incriminating letter hidden in her boot",
          "story": "Mira survives by noticing what others miss first. Even in quiet moments, she reads the room like it might turn on her. That habit has kept her alive more than once."
        }""",
    )

    npc = build_npc_from_prompt_lm(
        "humen rogeu named Mira weilding rapiar",
        sampler,
        story_gen,
        engine,
    )

    assert npc.race == "Human"
    assert npc.primary_class == "Rogue"
    assert npc.weapon == "Rapier"
