"""
NPC dialogue system with optional Qwen inference and a deterministic fallback.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

from npc_generator.npc import NPC

ALIGNMENT_TONE = {
    "Lawful Good": "You are honorable, direct, and kind. You uphold the law.",
    "Neutral Good": "You are warm, helpful, and genuine.",
    "Chaotic Good": "You are free-spirited, bold, and care deeply about doing right.",
    "Lawful Neutral": "You are formal, disciplined, and impartial.",
    "True Neutral": "You are reserved, observant, and speak carefully.",
    "Chaotic Neutral": "You are unpredictable, witty, and march to your own drum.",
    "Lawful Evil": "You are cold, calculating, and choose your words for maximum effect.",
    "Neutral Evil": "You are self-serving and guarded. Every word is a transaction.",
    "Chaotic Evil": "You are menacing, erratic, and enjoy unsettling others.",
}

CLASS_MANNERISMS = {
    "Fighter": "You speak plainly and value action over words.",
    "Wizard": "You tend to use precise, sometimes overly technical language.",
    "Rogue": "You choose your words carefully and rarely reveal more than needed.",
    "Cleric": "You often reference your faith and speak with quiet conviction.",
    "Ranger": "You are brief, observant, and comfortable with silence.",
    "Paladin": "You are earnest and formal, with a strong moral compass.",
    "Barbarian": "You speak bluntly, sometimes too loud, always honest.",
    "Bard": "You are charming, verbose, and love a good story.",
    "Druid": "You speak with reverence for nature and think in long timescales.",
    "Sorcerer": "You are intense and prone to cryptic remarks about fate.",
    "Warlock": "You are guarded, enigmatic, and occasionally ominous.",
    "Monk": "You are calm, measured, and choose every word with intention.",
}

DEFAULT_QWEN_MODEL = os.getenv("QWEN_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")


def build_persona_prompt(character: dict, story: str) -> str:
    """Build a system/persona string that anchors Qwen or fallback responses."""
    name = character["name"]
    race = character["race"]
    primary_class = character["primary_class"]
    subclass = character.get("subclass", "Wanderer")
    alignment = character.get("alignment", "True Neutral")
    tone = ALIGNMENT_TONE.get(alignment, "You speak plainly.")
    mannerism = CLASS_MANNERISMS.get(primary_class, "You speak naturally.")
    emotional_state = character.get("emotional_state", "curious")
    weapon = character.get("weapon", "an adventurer's weapon")
    goal = character.get("goal", "stay alive")
    quirk = character.get("quirk", "watch people carefully")
    notes = character.get("notes", [])
    recent_notes = "; ".join(notes[-3:]) if notes else "No recent updates."
    extra_traits = character.get("extra_traits", {})
    extra_text = "; ".join(f"{key}: {value}" for key, value in extra_traits.items()) if extra_traits else "No extra traits."

    return (
        f"You are {name}, a {race} {primary_class} with the subclass or specialty '{subclass}'. "
        f"You wield {weapon}. Your alignment is {alignment} and your current emotional state is {emotional_state}. "
        f"Your background story: {story} "
        f"Your current goal is to {goal}. Your quirk is that you {quirk}. "
        f"Extra described traits: {extra_text}. "
        f"Recent world updates affecting you: {recent_notes}. "
        f"{tone} {mannerism} "
        f"Always respond as {name} in the first person. Stay in character. Keep answers under 3 sentences."
    )


@dataclass
class GenerationConfig:
    temperature: float = 0.9
    top_p: float = 0.9
    max_new_tokens: int = 96
    repetition_penalty: float = 1.15


class DialogueEngine:
    """Qwen-backed NPC dialogue engine with a deterministic fallback mode."""

    def __init__(self, model_name: str = DEFAULT_QWEN_MODEL, load_model: bool = False):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.generation_config = GenerationConfig()
        self.load_error: str | None = None
        self._torch = None
        self._device = None
        if load_model:
            self._load_model()

    def _load_model(self) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._torch = torch
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if self._device == "cuda" else torch.float32
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
            )
            self.model.to(self._device)
            self.model.eval()
            self.load_error = None
            print(f"[INFO] Loaded Qwen model: {self.model_name}")
        except Exception as exc:
            self.model = None
            self.tokenizer = None
            self.load_error = str(exc)
            print(f"[WARNING] Could not load Qwen model '{self.model_name}': {exc}")

    def describe_backend(self) -> str:
        if self.model is not None:
            return f"Qwen ({self.model_name})"
        if self.load_error:
            return f"rule-based fallback (Qwen unavailable: {self.load_error})"
        return "rule-based fallback"

    def set_generation_config(
        self,
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        max_new_tokens: int | None = None,
        repetition_penalty: float | None = None,
    ) -> None:
        if temperature is not None:
            self.generation_config.temperature = temperature
        if top_p is not None:
            self.generation_config.top_p = top_p
        if max_new_tokens is not None:
            self.generation_config.max_new_tokens = max_new_tokens
        if repetition_penalty is not None:
            self.generation_config.repetition_penalty = repetition_penalty

    def _qwen_chat(self, npc: NPC, user_input: str) -> str:
        torch = self._torch
        messages = [
            {"role": "system", "content": build_persona_prompt(npc.to_dict(), npc.story)},
        ]
        for turn in npc.history[-10:]:
            role = "assistant" if turn["role"] == "assistant" else "user"
            messages.append({"role": role, "content": turn["content"]})
        messages.append({"role": "user", "content": user_input})

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self._device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.generation_config.max_new_tokens,
                do_sample=True,
                temperature=self.generation_config.temperature,
                top_p=self.generation_config.top_p,
                repetition_penalty=self.generation_config.repetition_penalty,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated_ids = output[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return response or self._fallback_response(user_input, npc.to_dict())

    def chat(self, npc: NPC, user_input: str):
        """
        Generate a single NPC response.
        Returns (response_text, history_placeholder).
        """
        if self.model is not None:
            try:
                return self._qwen_chat(npc, user_input), None
            except Exception as exc:
                print(f"[WARNING] Qwen generation failed, using fallback: {exc}")

        return self._fallback_response(user_input, npc.to_dict()), None

    def suggest_replies(self, npc: NPC) -> list[str]:
        base = [
            "Tell me about yourself.",
            "How are you feeling right now?",
            "What do you want most at the moment?",
            f"What should I know about your {npc.weapon.lower()}?",
            "What part of your past still defines you?",
            "What do you think of me so far?",
        ]
        if npc.emotional_state in {"hostile", "suspicious", "cynical"}:
            base.append("Why are you so guarded?")
        if npc.notes:
            base.append("How did the recent events change you?")
        random.shuffle(base)
        return base[:4]

    def _fallback_response(self, user_input: str, character: dict) -> str:
        """Persona-aware rule-based response engine."""
        lower = user_input.lower()
        name = character["name"]
        cls = character.get("primary_class", "wanderer")
        race = character.get("race", "")
        alignment = character.get("alignment", "True Neutral")
        bg = character.get("background", "humble")
        emotion = character.get("emotional_state", "curious")
        goal = character.get("goal", "keep going")
        quirk = character.get("quirk", "watch everything")

        greetings = {"hello", "hi", "hey", "greetings", "good day", "well met", "howdy"}
        who_kw = {"who are you", "your name", "who", "name", "introduce"}
        quest_kw = {"quest", "job", "task", "work", "mission", "help", "need"}
        fight_kw = {"fight", "battle", "combat", "weapon", "war", "enemy", "attack", "danger"}
        trade_kw = {"buy", "sell", "trade", "gold", "coin", "price", "shop", "merchant"}
        info_kw = {"where", "town", "road", "map", "direction", "know", "tell me", "news"}
        farewell_kw = {"bye", "farewell", "goodbye", "leave", "go now", "see you"}
        threat_kw = {"die", "kill", "threat", "hurt", "blood", "destroy"}
        compliment_kw = {"good", "great", "brave", "strong", "wise", "well done", "impressive"}

        evil_warn = alignment in ("Lawful Evil", "Neutral Evil", "Chaotic Evil")
        chaotic = "Chaotic" in alignment
        lawful = "Lawful" in alignment

        if any(g in lower for g in greetings):
            options = [
                f"Well met, stranger. I am {name}.",
                f"*nods* {name}. I am feeling {emotion} today. And you are?",
                f"Greetings. I do not often get visitors. What brings you to me?",
            ]
            if evil_warn:
                options = [
                    f"...*eyes you carefully*... {name}. State your purpose.",
                    f"You address me directly. Bold. I am {name}.",
                ]
            elif chaotic:
                options = [
                    f"Ha! A face I have not seen before. {name}'s the name.",
                    f"*spins around* Oh! Did not hear you coming. {name}, at your service. Maybe.",
                ]
            return random.choice(options)

        if any(w in lower for w in who_kw):
            return (
                f"I am {name}, a {race} {cls} from a {bg} background. "
                f"I have been focused on {goal}, and I am known for someone who {quirk}. "
                f"What more do you need to know?"
            )

        if any(q in lower for q in quest_kw):
            options = [
                f"I do not hand out quests here, but I can tell you what I am dealing with if you insist.",
                f"I am focused on {goal}. If you want to help, ask the right questions.",
                f"*lowers voice* I have my own concerns, not a task board.",
            ]
            return random.choice(options)

        if any(f in lower for f in fight_kw):
            if cls in ("Fighter", "Barbarian", "Paladin", "Ranger"):
                return f"Combat? *{name} grips their weapon.* I know exactly what {bg.lower()} steel feels like in a bad situation."
            if cls in ("Wizard", "Sorcerer", "Warlock"):
                return f"Brute force is a last resort. I have other ways of dealing with enemies."
            return "I try to avoid unnecessary conflict. But I am no easy target."

        if any(t in lower for t in trade_kw):
            options = [
                "I am not a merchant. But I may know someone who deals in such things.",
                "Gold has its uses, but I care more about leverage than coin.",
            ]
            if evil_warn:
                options = ["Everything has a price. What exactly are you offering?"]
            return random.choice(options)

        if any(i in lower for i in info_kw):
            options = [
                "I have traveled these roads. What do you want to know?",
                "The last I heard, things changed faster than people admit.",
                "Information costs nothing, I suppose. Ask your question.",
            ]
            return random.choice(options)

        if any(f in lower for f in farewell_kw):
            options = [
                "Safe travels. Watch the roads.",
                "*nods* Until next time.",
                "Go well. This world has enough graves already.",
            ]
            return random.choice(options)

        if any(t in lower for t in threat_kw):
            if evil_warn:
                return "*smiles coldly* Is that a threat? Interesting choice."
            if cls in ("Barbarian", "Fighter"):
                return "*stands up slowly* Say that again. I dare you."
            return "I would choose your next words carefully."

        if any(c in lower for c in compliment_kw):
            options = [
                "*raises an eyebrow* ...Thank you. I suppose.",
                "Flattery is a dangerous tool, but I appreciate the attempt.",
                "Kind words are rare enough that I remember them.",
            ]
            return random.choice(options)

        generic = [
            f"*{name} considers your words.* Go on.",
            "Hmm. I am not sure what to make of that.",
            "You are an odd one. I will give you that.",
            "I have heard stranger things. What is your point?",
            "Keep talking. I am listening.",
            f"I am still focused on {goal}. If you want my trust, be direct.",
        ]
        if chaotic:
            generic.append("*laughs unexpectedly* Sorry, you reminded me of something. What were you saying?")
        if lawful:
            generic.append("Speak plainly. I have little patience for riddles.")
        return random.choice(generic)
