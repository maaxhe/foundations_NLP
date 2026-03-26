from npc_generator.npc import NPC


def make_npc():
    return NPC(
        name="Test Hero", race="Human", primary_class="Fighter",
        subclass="Champion", background="Soldier", alignment="Neutral Good",
        emotional_state="friendly", weapon="Longsword",
        level=5, HP=50, AC=15,
        Str=16, Dex=12, Con=14, Int=10, Wis=11, Cha=9,
    )


def test_npc_creation():
    npc = make_npc()
    assert npc.name == "Test Hero"
    assert npc.race == "Human"


def test_stat_block():
    npc = make_npc()
    sb = npc.stat_block
    assert "STR" in sb
    assert "16" in sb


def test_optional_fields():
    npc = make_npc()
    assert npc.story == ""
    assert npc.goal == ""
    assert npc.notes == []
