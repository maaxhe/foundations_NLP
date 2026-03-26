from npc_generator.dialogue_engine import DialogueEngine, build_persona_prompt
from npc_generator.npc import NPC


engine = DialogueEngine()

char = NPC(
    name="Aldric Stonebrew",
    race="Human",
    primary_class="Fighter",
    subclass="Champion",
    alignment="Neutral Good",
    emotional_state="friendly",
    background="Soldier",
    weapon="Longsword",
    goal="protect the village",
    quirk="writes down every promise they hear",
)


def test_persona_prompt():
    prompt = build_persona_prompt(char.to_dict(), "A brave warrior.")
    assert "Aldric Stonebrew" in prompt
    assert "Fighter" in prompt


def test_chat_returns_string():
    response, _ = engine.chat(char, "hello")
    assert isinstance(response, str)
    assert len(response) > 0


def test_greeting_has_name():
    response, _ = engine.chat(char, "hello")
    assert "Aldric Stonebrew" in response


def test_who_are_you():
    response, _ = engine.chat(char, "who are you?")
    assert isinstance(response, str)


def test_farewell():
    response, _ = engine.chat(char, "goodbye")
    assert isinstance(response, str)


def test_no_model():
    assert engine.model is None


def test_suggest_replies_returns_four_options():
    options = engine.suggest_replies(char)
    assert len(options) == 4
