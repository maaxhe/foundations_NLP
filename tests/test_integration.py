"""
quick integration test - just check the whole pipeline runs without crashing
"""
from npc_generator.character_sampler import CharacterSampler
from npc_generator.story_generator import StoryGenerator
from npc_generator.dialogue_engine import DialogueEngine
from npc_generator.npc import NPC


def test_full_pipeline():
    sampler = CharacterSampler()
    story_gen = StoryGenerator()
    engine = DialogueEngine()

    char = sampler.sample_character()
    story = story_gen.generate_story(char)
    char["story"] = story

    npc = NPC.from_dict(char)

    assert npc.name
    assert npc.story
    assert npc.goal
    assert npc.weapon

    response, _ = engine.chat(npc, "hello")
    assert isinstance(response, str) and len(response) > 0
