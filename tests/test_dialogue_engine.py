from npc_generator.dialogue_engine import DialogueEngine, build_persona_prompt


engine = DialogueEngine()

char = {
    "name": "Aldric Stonebrew",
    "race": "Human",
    "primary_class": "Fighter",
    "alignment": "Neutral Good",
    "background": "Soldier",
}


def test_persona_prompt():
    prompt = build_persona_prompt(char, "A brave warrior.")
    assert "Aldric Stonebrew" in prompt
    assert "Fighter" in prompt


def test_chat_returns_string():
    response, _ = engine.chat("persona", "hello", character=char)
    assert isinstance(response, str)
    assert len(response) > 0


def test_greeting_has_name():
    response, _ = engine.chat("persona", "hello", character=char)
    assert "Aldric Stonebrew" in response


def test_who_are_you():
    response, _ = engine.chat("persona", "who are you?", character=char)
    assert isinstance(response, str)


def test_farewell():
    response, _ = engine.chat("persona", "goodbye", character=char)
    assert isinstance(response, str)


def test_no_model():
    # we disabled DialoGPT, should always be None
    assert engine.model is None


# TODO: test more alignments
# TODO: add tests for chaotic/evil responses
