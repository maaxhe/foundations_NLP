"""
constrained_decoder.py -- Grammar-constrained JSON decoding (Phase 2/3).

Responsibility:
    Wrap the model's generation call so that the output is GUARANTEED to be
    valid JSON conforming to the DialogState schema.  This eliminates the
    class of failures where the model produces malformed JSON (missing braces,
    wrong field names, etc.) that would otherwise crash the app.

Two complementary strategies are implemented here (as skeletons):

    Strategy A -- Outlines / LMQL (token-level grammar constraints):
        At each decoding step, mask logits for tokens that would violate
        the JSON grammar.  The model never "sees" invalid continuations.
        Library: `outlines` (pip install outlines)

    Strategy B -- Post-hoc repair (simpler fallback):
        Let the model generate freely, then attempt to repair malformed JSON
        with a lightweight fixer before falling back to a default DialogState.

Use Strategy A for production (lower error rate); Strategy B as a fallback.

Dependencies:
    pip install outlines        # Strategy A
    pip install json-repair     # Strategy B fallback
"""

from __future__ import annotations

import json
import logging
from typing import Optional

# TODO: import outlines                              # pip install outlines
# TODO: import outlines.models as outlines_models
# TODO: import outlines.generate as outlines_generate
# TODO: from json_repair import repair_json          # pip install json-repair
# TODO: from data_pipeline.schemas import DialogState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Strategy A -- Outlines grammar-constrained generation
# ---------------------------------------------------------------------------

class OutlinesConstrainedDecoder:
    """
    Uses the `outlines` library to constrain token sampling to the DialogState
    JSON schema at every generation step.

    The outlines library accepts a Pydantic model and automatically builds
    an efficient finite-state machine (FSM) over the token vocabulary that
    only allows token sequences forming valid JSON for that schema.

    Args:
        model_name_or_path: HuggingFace model ID or local path.
                            Must be the same base/fine-tuned model used for training.
        max_new_tokens:     Maximum generation length.

    TODO: Evaluate whether outlines is compatible with PEFT-loaded models.
          If not, use the merged (adapter-baked) model instead.
          See: lora_trainer.LoRADialogStateTrainer.save (merge_and_unload).

    Reference: https://github.com/outlines-dev/outlines
    """

    def __init__(
        self,
        model_name_or_path: str,
        max_new_tokens: int = 512,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.max_new_tokens     = max_new_tokens
        self._model             = None   # lazy-loaded
        self._generator         = None   # lazy-loaded

    def _load(self) -> None:
        """
        Load the model through outlines and build the JSON generator.

        TODO: Implement:
            self._model = outlines_models.transformers(
                self.model_name_or_path,
                device="cuda",   # or "cpu" / "mps"
            )
            self._generator = outlines_generate.json(
                self._model,
                DialogState,         # Pydantic model drives the FSM
                sampler=outlines.samplers.greedy(),
            )

        TODO: Wrap in a try/except to fall back to OutlinesFallbackDecoder
              if the outlines package is not installed.
        """
        raise NotImplementedError

    def decode(self, prompt: str) -> "DialogState":  # noqa: F821
        """
        Generate a DialogState by running the constrained decoder.

        Args:
            prompt: The full formatted prompt string
                    (output of data_pipeline.dataset_builder.build_prompt).

        Returns:
            A validated DialogState instance.  Guaranteed to be schema-valid
            because outlines enforces the grammar token-by-token.

        TODO: Implement:
            if self._generator is None:
                self._load()
            return self._generator(prompt, max_tokens=self.max_new_tokens)
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Strategy B -- Post-hoc repair fallback
# ---------------------------------------------------------------------------

class RepairFallbackDecoder:
    """
    Runs unconstrained model generation and attempts to fix malformed JSON
    before validation.

    Use this as a fallback when outlines is unavailable or incompatible with
    the deployed model format (e.g. GGUF via Ollama).

    Args:
        ollama_model_name: The name of the model loaded into Ollama
                           (e.g. "llama3:3b-instruct-q4_K_M").
                           Used when calling the Ollama Python client.

    TODO: Decide at startup (in app/state_extractor.py) which decoder to use
          based on configs/inference_config.yaml field `decoder_strategy`.
    """

    def __init__(self, ollama_model_name: str) -> None:
        self.ollama_model_name = ollama_model_name

    def _call_ollama(self, prompt: str) -> str:
        """
        Call the local Ollama server to generate a raw response string.

        Args:
            prompt: Full formatted extraction prompt.

        Returns:
            Raw response string from Ollama (may be malformed JSON).

        TODO: Implement using the `ollama` Python library:
            import ollama
            response = ollama.generate(
                model=self.ollama_model_name,
                prompt=prompt,
                options={
                    "temperature": 0.0,     # deterministic for extraction
                    "num_predict": 512,
                    "stop": ["}"],          # stop after JSON object closes
                },
            )
            return response["response"]

        TODO: Handle `ollama.ResponseError` and connection errors gracefully.
              If Ollama is not running, raise a descriptive RuntimeError.
        """
        raise NotImplementedError

    def _repair(self, raw: str) -> Optional[str]:
        """
        Attempt to fix common JSON formatting errors in `raw`.

        Args:
            raw: Potentially malformed JSON string.

        Returns:
            Repaired JSON string, or None if repair fails.

        TODO: Implement:
            try:
                repaired = repair_json(raw, return_objects=False)
                # Verify it now parses cleanly
                json.loads(repaired)
                return repaired
            except Exception as e:
                logger.warning("JSON repair failed: %s", e)
                return None
        """
        raise NotImplementedError

    def decode(self, prompt: str, default_npc_name: str) -> "DialogState":  # noqa: F821
        """
        Generate, repair, and validate a DialogState from the Ollama model.

        Args:
            prompt:           Full formatted extraction prompt.
            default_npc_name: NPC name used to construct the fallback
                              DialogState if all repair attempts fail.

        Returns:
            A validated DialogState.  Falls back to a neutral default state
            if both generation and repair fail, to keep the app running.

        TODO: Implement:
            raw = self._call_ollama(prompt)
            repaired = self._repair(raw)
            if repaired:
                try:
                    return DialogState(**json.loads(repaired))
                except Exception:
                    pass
            logger.error("Falling back to default DialogState for NPC: %s",
                         default_npc_name)
            return DialogState(npc_name=default_npc_name)

        TODO: Emit a metric counter for fallback events so the dashboard
              (or a log aggregator) can alert if fallback rate exceeds a threshold.
        """
        raise NotImplementedError
