"""
state_extractor.py -- Dialog-state extraction via the local fine-tuned model.

Responsibility:
    Given the accumulated dialog history and current NPC context, call the
    local model (via Ollama or direct HuggingFace inference) and return a
    validated DialogState object.

This module is the bridge between the raw dialog and the NPC's internal state.
It is called once per player turn, AFTER the NPC dialog response has been
generated (to keep latency low for the user-facing response).

Architecture note:
    Two extraction backends are supported, switchable via configs/inference_config.yaml:
        backend: "ollama"      -- Use Ollama's REST API (simplest, no GPU required)
        backend: "transformers"-- Use HuggingFace Transformers directly (requires GPU)

    The Ollama backend calls the fine-tuned GGUF model served by Ollama.
    The Transformers backend uses the LoRA adapter loaded by lora_trainer.py.

Dependencies:
    pip install ollama fastapi pydantic
    (for transformers backend) pip install transformers peft torch
"""

from __future__ import annotations

import json
import logging
from typing import Literal

# TODO: import ollama                               # pip install ollama
# TODO: from data_pipeline.schemas import DialogState
# TODO: from data_pipeline.dataset_builder import build_prompt
# TODO: from training.constrained_decoder import RepairFallbackDecoder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

class StateExtractor:
    """
    Extracts a DialogState from the accumulated dialog history.

    Args:
        backend:        "ollama" or "transformers".
        ollama_model:   Ollama model tag for the extraction model.
                        Example: "llama3:3b-instruct-q4_K_M"
                        After fine-tuning, replace with the custom model name:
                        "dnd-state-extractor:latest"
        hf_model_path:  Path to the HuggingFace model or LoRA adapter.
                        Used only when backend="transformers".
        temperature:    Sampling temperature.  Use 0.0 for deterministic
                        extraction (recommended; we want consistency, not creativity).

    TODO: Load backend and model names from configs/inference_config.yaml
          instead of constructor arguments.
    """

    def __init__(
        self,
        backend:        Literal["ollama", "transformers"] = "ollama",
        ollama_model:   str = "llama3:3b-instruct-q4_K_M",
        hf_model_path:  str = "models/lora_adapter",
        temperature:    float = 0.0,
    ) -> None:
        self.backend       = backend
        self.ollama_model  = ollama_model
        self.hf_model_path = hf_model_path
        self.temperature   = temperature
        self._hf_model     = None   # lazy-loaded for transformers backend
        self._tokenizer    = None   # lazy-loaded for transformers backend

    # ------------------------------------------------------------------
    # Prompt assembly
    # ------------------------------------------------------------------

    def _format_dialog_history(self, history: list[dict]) -> str:
        """
        Convert a list of turn dicts into a single dialog string for the prompt.

        Args:
            history: List of {"role": "player"|"npc", "content": str} dicts.
                     Ordered from oldest to newest.

        Returns:
            Formatted string, e.g.:
                Player: I need to find the eastern pass.
                NPC: Why should I trust you?
                Player: I have a letter from the guild.

        TODO: Implement:
            lines = []
            for turn in history:
                label = "Player" if turn["role"] == "player" else "NPC"
                lines.append(f"{label}: {turn['content']}")
            return "\\n".join(lines)

        TODO: Truncate history to the last N turns (configurable) to stay within
              the model's context window.  Track token count with the tokenizer.
        """
        raise NotImplementedError

    def _build_extraction_prompt(
        self,
        dialog_str: str,
        npc_name: str,
        turn_index: int,
    ) -> str:
        """
        Build the full extraction prompt including NPC name and turn context.

        Args:
            dialog_str:  Formatted dialog history string.
            npc_name:    Current NPC name (injected into the system prompt).
            turn_index:  Current turn index.

        Returns:
            Full prompt string ready for model inference.

        TODO: Use build_prompt from data_pipeline.dataset_builder for the
              base structure (so training and inference prompts are identical).
              Then prepend or append NPC-specific context as a system message.

        IMPORTANT: The extraction prompt format at inference MUST exactly match
                   the format used during fine-tuning (PROMPT_TEMPLATE in
                   dataset_builder.py).  Any mismatch will degrade accuracy
                   because the model has never seen the inference format during
                   training.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Backend implementations
    # ------------------------------------------------------------------

    def _extract_via_ollama(self, prompt: str, npc_name: str) -> "DialogState":
        """
        Call the Ollama local server to run extraction.

        Args:
            prompt:    Full extraction prompt.
            npc_name:  Used by the fallback decoder if parsing fails.

        Returns:
            Validated DialogState instance.

        TODO: Implement using RepairFallbackDecoder:
            decoder = RepairFallbackDecoder(ollama_model_name=self.ollama_model)
            return decoder.decode(prompt, default_npc_name=npc_name)

        Alternative (direct Ollama call without the repair wrapper):
            response = ollama.generate(
                model=self.ollama_model,
                prompt=prompt,
                format="json",   # Ollama's built-in JSON mode (less strict than outlines)
                options={"temperature": self.temperature},
            )
            raw = response["response"]
            return DialogState(**json.loads(raw))
        """
        raise NotImplementedError

    def _load_hf_model(self) -> None:
        """
        Lazily load the HuggingFace model and tokenizer for the transformers backend.

        TODO: Implement:
            from transformers import AutoTokenizer
            from training.lora_trainer import LoRADialogStateTrainer
            self._tokenizer = AutoTokenizer.from_pretrained(self.hf_model_path)
            self._hf_model  = LoRADialogStateTrainer.load_adapter(
                base_model_id=<read from inference_config.yaml>,
                adapter_path=self.hf_model_path,
            )
        """
        raise NotImplementedError

    def _extract_via_transformers(self, prompt: str, npc_name: str) -> "DialogState":
        """
        Run extraction using the HuggingFace transformers + PEFT stack.

        Args:
            prompt:   Full extraction prompt.
            npc_name: Used by the fallback decoder if parsing fails.

        Returns:
            Validated DialogState instance.

        TODO: Implement using OutlinesConstrainedDecoder for guaranteed JSON:
            from training.constrained_decoder import OutlinesConstrainedDecoder
            decoder = OutlinesConstrainedDecoder(
                model_name_or_path=self.hf_model_path
            )
            return decoder.decode(prompt)

        Or use standard generation + repair if outlines is not available:
            if self._hf_model is None:
                self._load_hf_model()
            inputs = self._tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                output = self._hf_model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                )
            raw = self._tokenizer.decode(output[0], skip_special_tokens=True)
            # Strip the prompt prefix from the output
            raw = raw[len(prompt):]
            return DialogState(**json.loads(raw))
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def extract(
        self,
        dialog_history: list[dict],
        npc_name: str,
        turn_index: int,
    ) -> "DialogState":
        """
        Main extraction method called by NpcEngine after each player turn.

        Args:
            dialog_history: List of {"role": "player"|"npc", "content": str} dicts.
            npc_name:        Current NPC name.
            turn_index:      Zero-based turn index.

        Returns:
            A validated DialogState reflecting the current conversation state.

        TODO: Implement:
            dialog_str = self._format_dialog_history(dialog_history)
            prompt     = self._build_extraction_prompt(dialog_str, npc_name, turn_index)
            if self.backend == "ollama":
                state = self._extract_via_ollama(prompt, npc_name)
            else:
                state = self._extract_via_transformers(prompt, npc_name)
            logger.debug("Extracted state: %s", state.model_dump_json(indent=2))
            return state

        TODO: Run extraction in a background asyncio task (asyncio.create_task)
              so the NPC dialog response can be sent to the frontend immediately
              without blocking on the extraction latency.
              Update the session state asynchronously after the response is sent.
        """
        raise NotImplementedError
