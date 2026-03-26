"""
quick integration test - just check the whole pipeline runs without crashing
"""
from npc_generator.character_sampler import CharacterSampler
from npc_generator.story_generator import StoryGenerator
from npc_generator.quest_generator import generate_quest
from npc_generator.dialogue_engine import DialogueEngine, build_persona_prompt
from npc_generator.npc import NPC


def test_full_pipeline():
    sampler = CharacterSampler()
    story_gen = StoryGenerator(model_path="models/__nonexistent__")
    engine = DialogueEngine()

    char = sampler.sample_character()
    story = story_gen.generate_story(char)
    quest = generate_quest(char)
    persona = build_persona_prompt(char, story)

    npc = NPC(
        name=char["name"], race=char["race"], primary_class=char["primary_class"],
        background=char["background"], alignment=char["alignment"],
        level=char["level"], HP=char["HP"], AC=char["AC"],
        Str=char["Str"], Dex=char["Dex"], Con=char["Con"],
        Int=char["Int"], Wis=char["Wis"], Cha=char["Cha"],
        story=story, quest=quest, persona_prompt=persona,
    )

    assert npc.name
    assert npc.story
    assert npc.quest

    response, _ = engine.chat(persona, "hello", character=npc.__dict__)
    assert isinstance(response, str) and len(response) > 0
