from chat import apply_hyper_command
from npc_generator.dialogue_engine import DialogueEngine


def test_hyper_command_updates_generation_config():
    engine = DialogueEngine()
    apply_hyper_command(engine, "temperature 0.7 top_p 0.8 max_new_tokens 120 repetition_penalty 1.05")

    assert engine.generation_config.temperature == 0.7
    assert engine.generation_config.top_p == 0.8
    assert engine.generation_config.max_new_tokens == 120
    assert engine.generation_config.repetition_penalty == 1.05


def test_hyper_command_reset_restores_defaults():
    engine = DialogueEngine()
    apply_hyper_command(engine, "temperature 0.7")
    apply_hyper_command(engine, "reset")

    assert engine.generation_config.temperature == 0.9
    assert engine.generation_config.top_p == 0.9
    assert engine.generation_config.max_new_tokens == 96
    assert engine.generation_config.repetition_penalty == 1.15
