from npc_generator.character_sampler import CharacterSampler

# takes a second to load but only happens once
sampler = CharacterSampler()


def test_loads():
    assert len(sampler.df) > 0


def test_sample_has_keys():
    char = sampler.sample_character()
    for key in ["name", "race", "primary_class", "subclass", "weapon", "emotional_state", "goal", "level", "HP", "AC", "Str"]:
        assert key in char


def test_stats_in_range():
    char = sampler.sample_character()
    assert 1 <= char["Str"] <= 20
    assert 1 <= char["level"] <= 20


def test_alignment_valid():
    char = sampler.sample_character()
    valid = ["Lawful Good", "Neutral Good", "Chaotic Good",
             "Lawful Neutral", "True Neutral", "Chaotic Neutral",
             "Lawful Evil", "Neutral Evil", "Chaotic Evil"]
    assert char["alignment"] in valid
