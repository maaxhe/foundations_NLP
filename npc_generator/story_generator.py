"""
story_generator.py
Generates NPC background stories using a fine-tuned GPT-2 model.
If no fine-tuned model is found, falls back to the base GPT-2.
"""

import os
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
    """
    Generates NPC background stories.
    First tries to load a fine-tuned GPT-2 from `model_path`.
    Falls back to template-based generation if no model is found.
    """

    def __init__(self, model_path: str = "models/gpt2_npc"):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self._try_load_model()

    def _try_load_model(self):
        abs_path = os.path.join(os.path.dirname(__file__), "..", self.model_path)
        if not os.path.isdir(abs_path):
            return  # No fine-tuned model yet — use templates
        try:
            from transformers import GPT2LMHeadModel, GPT2Tokenizer
            self.tokenizer = GPT2Tokenizer.from_pretrained(abs_path)
            self.model = GPT2LMHeadModel.from_pretrained(abs_path)
            self.model.eval()
        except Exception:
            self.model = None
            self.tokenizer = None

    def _generate_with_model(self, prompt: str, max_new_tokens: int = 80) -> str:
        import torch
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.85,
                top_p=0.92,
                repetition_penalty=1.3,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = self.tokenizer.decode(output[0], skip_special_tokens=True)
        # Return only the newly generated part after the prompt
        story = generated[len(prompt):].strip()
        # Cut at sentence boundary
        for punct in [". ", "! ", "? "]:
            idx = story.rfind(punct)
            if idx > 40:
                story = story[: idx + 1]
                break
        return story

    def generate_story(self, character: dict) -> str:
        """Return a 3-4 sentence background story for the given character dict."""
        alignment_sentence = ALIGNMENT_SENTENCES.get(
            character.get("alignment", "True Neutral"), ""
        )
        background_lower = character.get("background", "humble").lower()

        # Template fallback (GPT-2 fine-tune disabled: model reproduces training tags)
        template = random.choice(STORY_TEMPLATES)
        return template.format(
            name=character["name"],
            race=character["race"],
            primary_class=character["primary_class"],
            background=character.get("background", "humble"),
            background_lower=background_lower,
            alignment_sentence=alignment_sentence,
        )


if __name__ == "__main__":
    from character_sampler import CharacterSampler
    char = CharacterSampler().sample_character()
    gen = StoryGenerator()
    print(gen.generate_story(char))
