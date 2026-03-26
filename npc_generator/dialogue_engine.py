"""
dialogue_engine.py
NPC dialogue system using DialoGPT conditioned on a persona prefix.

The model is fine-tuned (or prompted) with the NPC's persona so responses
stay in character. Uses the PersonaChat dataset for additional fine-tuning
context if available.
"""

import os
from typing import List, Optional

# Persona tone modifiers per alignment
ALIGNMENT_TONE = {
    "Lawful Good":    "You are honorable, direct, and kind. You uphold the law.",
    "Neutral Good":   "You are warm, helpful, and genuine.",
    "Chaotic Good":   "You are free-spirited, bold, and care deeply about doing right.",
    "Lawful Neutral": "You are formal, disciplined, and impartial.",
    "True Neutral":   "You are reserved, observant, and speak carefully.",
    "Chaotic Neutral":"You are unpredictable, witty, and march to your own drum.",
    "Lawful Evil":    "You are cold, calculating, and choose your words for maximum effect.",
    "Neutral Evil":   "You are self-serving and guarded. Every word is a transaction.",
    "Chaotic Evil":   "You are menacing, erratic, and enjoy unsettling others.",
}

CLASS_MANNERISMS = {
    "Fighter":   "You speak plainly and value action over words.",
    "Wizard":    "You tend to use precise, sometimes overly technical language.",
    "Rogue":     "You choose your words carefully and rarely reveal more than needed.",
    "Cleric":    "You often reference your faith and speak with quiet conviction.",
    "Ranger":    "You are brief, observant, and comfortable with silence.",
    "Paladin":   "You are earnest and formal, with a strong moral compass.",
    "Barbarian": "You speak bluntly, sometimes too loud, always honest.",
    "Bard":      "You are charming, verbose, and love a good story.",
    "Druid":     "You speak with reverence for nature and think in long timescales.",
    "Sorcerer":  "You are intense and prone to cryptic remarks about fate.",
    "Warlock":   "You are guarded, enigmatic, and occasionally ominous.",
    "Monk":      "You are calm, measured, and choose every word with intention.",
}


def build_persona_prompt(character: dict, story: str) -> str:
    """Build a system/persona string that anchors DialoGPT responses."""
    name = character["name"]
    race = character["race"]
    cls = character["primary_class"]
    alignment = character.get("alignment", "True Neutral")
    tone = ALIGNMENT_TONE.get(alignment, "You speak plainly.")
    mannerism = CLASS_MANNERISMS.get(cls, "You speak naturally.")

    return (
        f"You are {name}, a {race} {cls}. "
        f"{story} "
        f"{tone} {mannerism} "
        f"Always respond as {name} in the first person. Keep answers under 3 sentences."
    )


class DialogueEngine:
    """
    NPC dialogue engine — tries Claude API first, falls back to rule-based.
    """

    CLAUDE_MODEL = "claude-haiku-4-5"

    def __init__(self, model_name: str = "microsoft/DialoGPT-small"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self._claude = None
        self._load_model()

    GPT2_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "gpt2_dialogue")

    def _load_model(self):
        try:
            from transformers import GPT2LMHeadModel, GPT2Tokenizer
            import torch
            path = os.path.abspath(self.GPT2_MODEL_PATH)
            if not os.path.isdir(path):
                print("[INFO] No fine-tuned GPT-2 found, using rule-based responses.")
                return
            self.tokenizer = GPT2Tokenizer.from_pretrained(path)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = GPT2LMHeadModel.from_pretrained(path)
            self.model.eval()
            self._torch = torch
            print("[INFO] Loaded fine-tuned GPT-2 NPC model.")
        except Exception as e:
            print(f"[WARNING] Could not load GPT-2: {e}")

    def _gpt2_chat(self, character: dict, user_input: str, history: list) -> tuple[str, list]:
        torch = self._torch
        name = character.get("name", "NPC")
        race = character.get("race", "")
        cls = character.get("primary_class", "")
        alignment = character.get("alignment", "True Neutral")
        tone = ALIGNMENT_TONE.get(alignment, "")

        # build prompt in the same format the model was trained on
        context = ""
        for turn in history[-6:]:
            context += turn
        context += f"Player: {user_input}\nNPC:"

        inputs = self.tokenizer(context, return_tensors="pt", truncation=True, max_length=200)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=60,
                do_sample=True,
                temperature=0.85,
                top_p=0.92,
                repetition_penalty=1.4,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = self.tokenizer.decode(output[0], skip_special_tokens=True)
        after_name = generated.split("NPC:")[-1].strip()
        # cut at first newline or sentence end
        for stop in ["\n", "Player:", "<NPC>"]:
            if stop in after_name:
                after_name = after_name.split(stop)[0].strip()
        # cut at sentence boundary if too long
        if len(after_name) > 120:
            for p in [". ", "! ", "? "]:
                idx = after_name.find(p)
                if 10 < idx < 120:
                    after_name = after_name[:idx + 1]
                    break

        if not after_name or len(after_name) < 3:
            return self._fallback_response(user_input, character), history

        new_history = history + [f"Player: {user_input}\nNPC: {after_name}\n"]
        return after_name, new_history

    def chat(
        self,
        persona_prompt: str,
        user_input: str,
        history_ids: Optional[object] = None,
        character: Optional[dict] = None,
    ):
        """
        Generate a single NPC response.
        Returns (response_text, new_history).
        """
        if self.model is not None:
            try:
                history = history_ids if isinstance(history_ids, list) else []
                return self._gpt2_chat(character or {}, user_input, history)
            except Exception as e:
                print(f"[WARNING] GPT-2 error: {e}")

        return self._fallback_response(user_input, character), None

        tok = self.tokenizer
        torch = self._torch

        # Encode: persona prefix + user input + EOS
        persona_enc = tok.encode(persona_prompt + tok.eos_token, return_tensors="pt")
        user_enc = tok.encode(user_input + tok.eos_token, return_tensors="pt")

        if history_ids is not None:
            bot_input = torch.cat([history_ids, user_enc], dim=-1)
        else:
            bot_input = torch.cat([persona_enc, user_enc], dim=-1)

        # Limit context length to avoid OOM
        if bot_input.shape[-1] > 900:
            bot_input = torch.cat([persona_enc, user_enc], dim=-1)

        attention_mask = (bot_input != tok.eos_token_id).long()
        with torch.no_grad():
            output = self.model.generate(
                bot_input,
                attention_mask=attention_mask,
                max_new_tokens=80,
                do_sample=True,
                temperature=0.75,
                top_p=0.90,
                repetition_penalty=1.3,
                pad_token_id=tok.eos_token_id,
            )

        response = tok.decode(
            output[:, bot_input.shape[-1]:][0], skip_special_tokens=True
        ).strip()

        # Fallback if empty
        if not response:
            response = self._fallback_response(user_input, character)

        return response, output

    def _fallback_response(self, user_input: str, character: Optional[dict]) -> str:
        """Persona-aware rule-based response engine."""
        import random

        lower = user_input.lower()
        name = character["name"] if character else "traveler"
        cls = character.get("primary_class", "wanderer") if character else "wanderer"
        race = character.get("race", "") if character else ""
        alignment = character.get("alignment", "True Neutral") if character else "True Neutral"
        bg = character.get("background", "humble") if character else "humble"
        tone = ALIGNMENT_TONE.get(alignment, "You speak plainly.")
        mannerism = CLASS_MANNERISMS.get(cls, "")

        # ── keyword buckets ───────────────────────────────────────────────────
        greetings  = {"hello", "hi", "hey", "greetings", "good day", "well met",
                      "darkness", "friend", "salute", "howdy"}
        who_kw     = {"who are you", "your name", "who", "name", "introduce"}
        quest_kw   = {"quest", "job", "task", "work", "mission", "help", "need"}
        fight_kw   = {"fight", "battle", "combat", "weapon", "war", "enemy", "attack", "danger"}
        trade_kw   = {"buy", "sell", "trade", "gold", "coin", "price", "shop", "merchant"}
        info_kw    = {"where", "town", "road", "map", "direction", "know", "tell me", "news"}
        farewell_kw= {"bye", "farewell", "goodbye", "leave", "go now", "see you"}
        threat_kw  = {"die", "kill", "threat", "hurt", "blood", "destroy"}
        compliment_kw = {"good", "great", "brave", "strong", "wise", "well done", "impressive"}

        # ── alignment-flavored openers ────────────────────────────────────────
        evil_warn  = alignment in ("Lawful Evil", "Neutral Evil", "Chaotic Evil")
        chaotic    = "Chaotic" in alignment
        lawful     = "Lawful" in alignment

        if any(g in lower for g in greetings):
            options = [
                f"Well met, stranger. I am {name}.",
                f"*nods* {name}. And you are?",
                f"Greetings. I don't often get visitors — what brings you to me?",
            ]
            if evil_warn:
                options = [
                    f"...*eyes you carefully*... {name}. State your purpose.",
                    f"You address me directly. Bold. I am {name}.",
                ]
            elif chaotic:
                options = [
                    f"Ha! A face I haven't seen before. {name}'s the name.",
                    f"*spins around* Oh! Didn't hear you coming. {name}, at your service — maybe.",
                ]
            return random.choice(options)

        if any(w in lower for w in who_kw):
            return (
                f"I am {name}, a {race} {cls} of {bg} background. "
                f"{ALIGNMENT_TONE.get(alignment, '')} What more do you need to know?"
            )

        if any(q in lower for q in quest_kw):
            options = [
                f"Aye, I do have something that needs doing. Are you willing to listen?",
                f"Since you ask — there is a matter I cannot handle alone. Interested?",
                f"*lowers voice* There is work, if you have the stomach for it.",
            ]
            return random.choice(options)

        if any(f in lower for f in fight_kw):
            if cls in ("Fighter", "Barbarian", "Paladin", "Ranger"):
                return f"Combat? *{name} grips their weapon.* I've seen enough battles to know when one is coming."
            elif cls in ("Wizard", "Sorcerer", "Warlock"):
                return f"Brute force is a last resort. I have other ways of dealing with enemies."
            else:
                return f"I try to avoid unnecessary conflict. But I am no easy target."

        if any(t in lower for t in trade_kw):
            options = [
                f"I'm not a merchant. But I may know someone who deals in such things.",
                f"Gold, is it? *shrugs* I have little interest in coin beyond what keeps me moving.",
            ]
            if evil_warn:
                options = [f"Everything has a price. What exactly are you offering?"]
            return random.choice(options)

        if any(i in lower for i in info_kw):
            options = [
                f"I've traveled these roads. What do you want to know?",
                f"*thinks for a moment* The last I heard... things have changed out there.",
                f"Information costs nothing, I suppose. Ask your question.",
            ]
            return random.choice(options)

        if any(f in lower for f in farewell_kw):
            options = [
                f"Safe travels, adventurer. Watch the roads.",
                f"*nods* Until next time.",
                f"Go well. This world has enough graves.",
            ]
            return random.choice(options)

        if any(t in lower for t in threat_kw):
            if evil_warn:
                return f"*smiles coldly* Is that a threat? Interesting choice."
            elif cls in ("Barbarian", "Fighter"):
                return f"*stands up slowly* Say that again. I dare you."
            else:
                return f"I would choose your next words carefully."

        if any(c in lower for c in compliment_kw):
            options = [
                f"*raises an eyebrow* ...Thank you. I suppose.",
                f"Flattery. I've learned not to trust it — but I appreciate the thought.",
                f"Kind words. Don't expect me to go soft because of them.",
            ]
            return random.choice(options)

        # ── generic fallbacks ─────────────────────────────────────────────────
        generic = [
            f"*{name} considers your words.* Go on.",
            f"Hmm. I'm not sure what to make of that.",
            f"*shifts weight* You're an odd one. I'll give you that.",
            f"I've heard stranger things. What's your point?",
            f"...Keep talking. I'm listening.",
        ]
        if chaotic:
            generic.append(f"*laughs unexpectedly* Sorry — you just reminded me of something. What were you saying?")
        if lawful:
            generic.append(f"Speak plainly. I have little patience for riddles.")
        return random.choice(generic)


def download_persona_dataset(save_dir: str = "data_sets/persona_chat"):
    """
    Download a small persona-grounded dialogue dataset from HuggingFace.
    Uses 'bavard/personachat_truecased' — ~5MB, persona-conditioned dialogues.
    """
    try:
        from datasets import load_dataset
        os.makedirs(save_dir, exist_ok=True)
        print("Downloading PersonaChat dataset (~5MB)...")
        ds = load_dataset("bavard/personachat_truecased", split="train")
        # Save as jsonl
        out_path = os.path.join(save_dir, "personachat_train.jsonl")
        ds.to_json(out_path)
        print(f"Saved {len(ds)} examples → {out_path}")
        return out_path
    except Exception as e:
        print(f"[WARNING] Could not download PersonaChat: {e}")
        return None


if __name__ == "__main__":
    download_persona_dataset()
