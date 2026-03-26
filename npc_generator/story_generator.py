"""
Template-based backstory generation for NPCs.
"""

import random

STORY_TEMPLATES = [
    "{name} was born into a {background} life among the {race} people. "
    "As a {primary_class}, they spent years honing their craft before leaving home. "
    "{alignment_sentence} Now they wander the land, shaped by both hardship and wonder.",

    "Few knew the early struggles of {name}, a {race} {primary_class} raised in {background_lower} circumstances. "
    "{alignment_sentence} Their journey has taken them far from where they began, "
    "and every scar tells a story.",

    "{name} never expected to become a {primary_class}. "
    "Born of {race} heritage, their {background_lower} background gave them skills others overlooked. "
    "{alignment_sentence} Life on the road has taught them that nothing comes without a cost.",

    "The story of {name} is one of quiet determination. "
    "This {race} {primary_class} emerged from a {background_lower} past with little more than talent and grit. "
    "{alignment_sentence} People who underestimate them rarely make that mistake twice.",
]

ALIGNMENT_SENTENCES = {
    "Lawful Good":    "They believe in order and justice above all else.",
    "Neutral Good":   "They try to do right by the people they meet.",
    "Chaotic Good":   "Rules matter less to them than doing what feels right.",
    "Lawful Neutral": "They follow a strict personal code, though not always kindly.",
    "True Neutral":   "They keep to themselves and let the world spin as it will.",
    "Chaotic Neutral":"They live by no one's rules — not even their own.",
    "Lawful Evil":    "They pursue their goals with cold, calculated precision.",
    "Neutral Evil":   "Self-interest guides every decision they make.",
    "Chaotic Evil":   "They leave chaos in their wake, and prefer it that way.",
}


class StoryGenerator:
    def __init__(self, model_path: str | None = None):
        self.model_path = model_path
        self.model = None

    def generate_story(self, character: dict) -> str:
        """Return a 3-4 sentence background story for the given character dict."""
        alignment_sentence = ALIGNMENT_SENTENCES.get(
            character.get("alignment", "True Neutral"), ""
        )
        background_lower = character.get("background", "humble").lower()
        emotional_state = character.get("emotional_state", "curious")
        goal = character.get("goal", "find direction")
        quirk = character.get("quirk", "keeps careful notes")
        secret = character.get("secret", "is carrying old regret")
        extra_traits = character.get("extra_traits", {})

        template = random.choice(STORY_TEMPLATES)
        opening = template.format(
            name=character["name"],
            race=character["race"],
            primary_class=character["primary_class"],
            background=character.get("background", "humble"),
            background_lower=background_lower,
            alignment_sentence=alignment_sentence,
        )
        addon = (
            f" They currently seem {emotional_state} and are focused on {goal}. "
            f"A noticeable quirk: {quirk}. One thing they keep hidden is that they {secret}."
        )
        if extra_traits:
            extra_summary = "; ".join(f"{key}: {value}" for key, value in extra_traits.items())
            addon += f" Additional details others notice: {extra_summary}."
        return opening + addon


if __name__ == "__main__":
    from character_sampler import CharacterSampler
    char = CharacterSampler().sample_character()
    gen = StoryGenerator()
    print(gen.generate_story(char))
