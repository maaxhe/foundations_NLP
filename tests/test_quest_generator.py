from npc_generator.quest_generator import generate_quest


def test_returns_string():
    char = {"primary_class": "Fighter", "name": "Aldric"}
    result = generate_quest(char)
    assert isinstance(result, str)


def test_name_in_quest():
    char = {"primary_class": "Fighter", "name": "Mira"}
    result = generate_quest(char)
    assert "Mira" in result


def test_unknown_class():
    # should not crash for unknown class
    char = {"primary_class": "Artificer", "name": "Hero"}
    result = generate_quest(char)
    assert "Hero" in result


def test_empty_dict():
    # just make sure it doesnt crash
    result = generate_quest({})
    assert isinstance(result, str)
