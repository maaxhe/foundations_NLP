from npc_generator.character_sampler import CharacterSampler
from npc_generator.npc import NPC
from npc_generator.spec_parser import apply_update_instruction, parse_character_specs


sampler = CharacterSampler()


def test_parse_keeps_explicit_fields():
    overrides = parse_character_specs(
        "elf rogue named Mira level 7 hp 42 weapon rapier chaotic good suspicious",
        sampler,
    )
    assert overrides["race"] == "Elf"
    assert overrides["primary_class"] == "Rogue"
    assert overrides["name"] == "Mira"
    assert overrides["level"] == 7
    assert overrides["HP"] == 42
    assert overrides["weapon"] == "Rapier"
    assert overrides["alignment"] == "Chaotic Good"
    assert overrides["emotional_state"] == "suspicious"


def test_update_instruction_changes_alignment():
    npc = NPC(name="Test")
    changes = apply_update_instruction(npc, "you are now lawful evil from now on", sampler)
    assert changes["alignment"] == "Lawful Evil"
    assert npc.alignment == "Lawful Evil"
    assert npc.notes
