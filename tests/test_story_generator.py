from npc_generator.story_generator import StoryGenerator


def get_char():
    return {
        "name": "Aldric Stonebrew",
        "race": "Human",
        "primary_class": "Fighter",
        "background": "Soldier",
        "alignment": "Neutral Good",
    }


# use a fake model path so it falls back to templates
gen = StoryGenerator(model_path="models/__nonexistent__")


def test_story_is_string():
    result = gen.generate_story(get_char())
    assert isinstance(result, str)


def test_story_contains_name():
    result = gen.generate_story(get_char())
    assert "Aldric Stonebrew" in result


def test_story_not_empty():
    result = gen.generate_story(get_char())
    assert len(result) > 20


def test_no_model_loaded():
    assert gen.model is None
