"""
quest_generator.py
Generates quests that the NPC gives TO the player.
"""

import random

CLASS_QUESTS = {
    "Fighter": [
        "A gang of mercenaries has blockaded the road to the next village. "
        "Drive them off and make sure they don't come back. {name} will pay well.",
        "Three soldiers deserted their post last week and robbed a farmer on the way out. "
        "{name} wants them found and brought to justice — alive if possible.",
        "Someone broke into the town armory last night. {name} needs you to track the thief "
        "before the stolen weapons reach the wrong hands.",
        "A warlord's scout was spotted near the bridge. {name} wants you to follow them "
        "back to camp and report what you find.",
    ],
    "Wizard": [
        "A page has gone missing from {name}'s spellbook — torn out, not lost. "
        "Find who took it and why before someone uses it.",
        "Strange runes have appeared overnight on the old mill. {name} needs a rubbing of them "
        "brought back without disturbing the site.",
        "A crate of alchemical reagents was stolen from the market. "
        "{name} suspects a rival — retrieve the crate and leave no evidence you were there.",
        "An apprentice wandered into the forbidden archive three days ago and hasn't returned. "
        "{name} is too old to go in after them. You're not.",
    ],
    "Rogue": [
        "A letter needs to reach a contact on the other side of the city — tonight, before dawn. "
        "{name} can't be seen doing it themselves.",
        "Someone is skimming coin from the guild's collections. {name} wants proof, not assumptions. "
        "Watch the collector on their rounds and report back.",
        "A fence in the lower quarter has been selling stolen goods linked to a murder. "
        "{name} needs their ledger. Don't let them know you were there.",
        "A noble's bodyguard has been asking questions about {name}. "
        "Find out who sent them and what they already know.",
    ],
    "Cleric": [
        "A dying man keeps muttering about a debt he owes to someone in the next town. "
        "{name} asks you to deliver his final message before it's too late.",
        "The old cemetery has been disturbed — graves dug up, bones rearranged. "
        "{name} believes it's ritual work. Find the culprit and end it.",
        "A village has been refusing to send their tithe and no messenger has returned. "
        "{name} fears something is very wrong. Go and find out.",
        "A relic belonging to the temple was sold at auction by mistake. "
        "{name} needs it recovered quietly — without causing a scene.",
    ],
    "Ranger": [
        "Something has been killing livestock at the edge of the forest — cleanly, surgically. "
        "Not a wolf. {name} wants you to find out what it is.",
        "A trapper went missing in the eastern woods four days ago. "
        "{name} tracked them to a ravine but won't go further alone.",
        "A hunting party from the city killed more than their quota and left the rest to rot. "
        "{name} wants them reported to the warden — with witnesses.",
        "Strange tracks have appeared near the water source. {name} can't identify them. "
        "Follow the trail and mark on this map where it leads.",
    ],
    "Paladin": [
        "A merchant has been selling fake holy relics to grieving families. "
        "{name} wants hard proof of the fraud brought before the magistrate.",
        "A knight errant was stripped of their title under suspicious circumstances. "
        "{name} believes they were framed. Find the real evidence.",
        "Children in the lower district have gone missing over the past month. "
        "The guard has done nothing. {name} is done waiting.",
        "A confession was coerced from an innocent prisoner. {name} needs a witness "
        "who was there that night to come forward.",
    ],
    "Barbarian": [
        "A member of {name}'s clan was taken as a slave and sold north. "
        "Find them. Bring them home. Don't worry about being subtle.",
        "A merchant caravan was ambushed and the survivors are stranded in the hills. "
        "{name} asks you to reach them before the weather does.",
        "Something lives in the mine shaft east of town and the miners won't go back in. "
        "{name} needs it dealt with. No questions, no ceremony.",
        "A rival clan has stolen a sacred totem. {name} wants it back — "
        "and wants the thieves to remember who took it from them.",
    ],
    "Bard": [
        "A song has been circulating in taverns that reveals secrets {name} would rather keep buried. "
        "Find the composer and make them stop — creatively.",
        "A noble is hosting a private gathering where a dangerous deal is being negotiated. "
        "{name} needs someone inside. You'd blend in better than most.",
        "An old journal belonging to a famous hero was auctioned off last week. "
        "{name} is certain it contains something that shouldn't be public. Acquire it.",
        "A traveling performer has been using {name}'s name and reputation without permission. "
        "Track them down and sort it out.",
    ],
    "Druid": [
        "Poachers have been setting traps in the protected forest. "
        "{name} needs the traps removed and the poachers identified — not harmed.",
        "A spring that feeds three villages has dried up overnight. "
        "{name} suspects someone upstream. Go and find out what changed.",
        "A rare plant blooms only once a decade and only in a specific cave. "
        "{name} needs three cuttings brought back alive.",
        "Loggers have started cutting near an old standing stone. "
        "{name} needs someone to delay them long enough to get a court order.",
    ],
    "Sorcerer": [
        "An object in {name}'s possession has started reacting to something nearby. "
        "Follow the pull and find the source before it gets stronger.",
        "A child in the village has begun manifesting uncontrolled magic. "
        "{name} needs you to bring them here safely — before someone gets hurt or afraid.",
        "A ritual circle was found carved into the floor of an abandoned house. "
        "{name} needs a sketch of it and any objects left inside. Don't step in the circle.",
        "Someone has been selling counterfeit magical charms that actually work — badly. "
        "Find the supplier and shut them down.",
    ],
    "Warlock": [
        "A binding contract {name} made years ago has resurfaced. "
        "Find the broker who sold it and recover the document. By any means necessary.",
        "Something has been following {name} for three nights. "
        "They need you to stay awake, watch, and describe exactly what you see.",
        "A rival has stolen an object tied to {name}'s patron. "
        "Retrieve it. Do not open it. Do not ask what's inside.",
        "A door has appeared in a building where no door was before. "
        "{name} won't go near it. You're being paid to go through it.",
    ],
    "Monk": [
        "A stolen manuscript from the monastery must be recovered before it is copied. "
        "{name} gives you one day.",
        "A former student has fallen in with a dangerous faction. "
        "{name} asks you to reach them before they do something they can't undo.",
        "A debt collector has been extorting the monastery under a false legal claim. "
        "{name} needs proof the claim is forged.",
        "A pilgrim carrying a sacred object was robbed on the mountain road. "
        "{name} asks you to retrieve it and escort the pilgrim safely back.",
    ],
}

CHAT_QUESTS = [
    "I need to know if I can trust you. Answer me honestly: "
    "if you found a bag of gold with no owner in sight, what would you do with it? "
    "{name} watches your face carefully as you answer.",
    "There's a man in this tavern — don't look now — who I believe is lying about his name. "
    "Talk to him. Find out who he really is. Come back and tell me.",
    "I have three suspects and one stolen key. I'll describe what I know — "
    "you tell me which one did it. {name} slides a folded note across the table.",
    "You want the quest? First convince me you're worth hiring. "
    "{name} crosses their arms. 'Tell me about the worst thing you've ever done.'",
    "I need a message passed to someone in this room without anyone noticing. "
    "The message is: 'The river runs north at midnight.' Figure out who it's for.",
]

FALLBACK_QUESTS = [
    "A crate went missing from the docks three nights ago. {name} won't say what's in it — "
    "only that it needs to come back before someone opens it.",
    "A name keeps coming up in all the wrong places. {name} wants to know who they are "
    "and what they want.",
    "Strange lights were seen in the old watchtower last night. "
    "{name} needs someone to go up there and make sure it's nothing.",
    "A child claims to have seen something in the cellar of the inn. "
    "{name} thinks they're telling the truth.",
    "An old map was found in a dead man's pocket. {name} thinks it leads somewhere real. "
    "They're too cautious to go alone.",
]


QUEST_REWARDS = {
    "Fighter":   (40, 25),
    "Wizard":    (50, 30),
    "Rogue":     (35, 40),
    "Cleric":    (45, 20),
    "Ranger":    (40, 25),
    "Paladin":   (50, 20),
    "Barbarian": (35, 30),
    "Bard":      (40, 35),
    "Druid":     (45, 20),
    "Sorcerer":  (50, 30),
    "Warlock":   (55, 25),
    "Monk":      (45, 20),
}


def generate_quest(character: dict) -> str:
    cls = character.get("primary_class", "Fighter")
    name = character.get("name", "the NPC")
    if random.random() < 0.25:
        return random.choice(CHAT_QUESTS).format(name=name)
    templates = CLASS_QUESTS.get(cls, FALLBACK_QUESTS)
    return random.choice(templates).format(name=name)


def quest_rewards(character: dict) -> tuple[int, int]:
    """Returns (xp, gold) reward for a quest from this NPC."""
    cls = character.get("primary_class", "Fighter")
    base_xp, base_gold = QUEST_REWARDS.get(cls, (40, 25))
    # slight randomness
    xp = base_xp + random.randint(-10, 20)
    gold = base_gold + random.randint(-5, 15)
    return max(10, xp), max(5, gold)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from npc_generator.character_sampler import CharacterSampler
    char = CharacterSampler().sample_character()
    print(generate_quest(char))
