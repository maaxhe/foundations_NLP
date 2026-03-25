"""
generator.py — Synthetic dataset generator (Phase 1).

Responsibility:
    Call an external LLM API (e.g. OpenAI GPT-4o or Anthropic Claude) with
    carefully crafted prompts to produce (dialog_excerpt, DialogState JSON)
    training pairs.  The generated JSON must conform to `schemas.DialogState`.

High-level flow:
    1. Load a prompt template from configs/data_config.yaml.
    2. For each sample, sample random NPC metadata (name, race, quest hook)
       to create diverse scenarios.
    3. Call the external API and extract the JSON block from the response.
    4. Yield raw (text, json_string) tuples; validation happens in validator.py.

Dependencies:
    pip install openai pyyaml tenacity
    (or `pip install anthropic` if using Claude as the synthesis API)
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterator

# TODO: import openai                    # pip install openai
# TODO: import yaml                      # pip install pyyaml
# TODO: from tenacity import (           # pip install tenacity
#           retry, stop_after_attempt, wait_exponential
#       )
# TODO: from data_pipeline.schemas import DialogState


# ---------------------------------------------------------------------------
# Constants — replace with yaml config loading
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_TEMPLATE = """
You are a D&D world-building assistant.  Given a short dialog excerpt between
a player and an NPC, output ONLY a valid JSON object that matches the
following schema exactly (no markdown, no extra keys):

{schema_json}

The NPC's name is: {npc_name}
Current turn index: {turn_index}
"""

NPC_NAME_POOL = [
    "Grimtooth", "Mira the Fence", "Brother Aldous",
    "Captain Yeva", "Zeph the Alchemist",
]

QUEST_HOOK_POOL = [
    "Retrieve the Crystal Shard from the northern ruins.",
    "Find the missing merchant caravan on the old road.",
    "Silence the bandit leader before the harvest festival.",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class SyntheticDataGenerator:
    """
    Generates (dialog_text, dialog_state_json) pairs by prompting an external
    LLM API.

    Args:
        api_key:       API key for the synthesis model provider.
        model_name:    Provider model ID, e.g. "gpt-4o" or "claude-3-5-sonnet".
        n_samples:     Total number of training pairs to generate.
        output_path:   Directory where raw JSONL files are written.
        temperature:   Sampling temperature for the synthesis model.
                       Lower values produce more deterministic JSON; higher values
                       produce more diverse dialog scenarios.  Recommended: 0.8-1.1.

    TODO: Accept a config dict/dataclass instead of individual kwargs so that
          all hyperparameters flow from configs/data_config.yaml.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "gpt-4o",
        n_samples: int = 1000,
        output_path: str = "data/raw",
        temperature: float = 0.9,
    ) -> None:
        self.api_key     = api_key
        self.model_name  = model_name
        self.n_samples   = n_samples
        self.output_path = Path(output_path)
        self.temperature = temperature

        # TODO: Initialise the API client here, e.g.:
        #   self.client = openai.OpenAI(api_key=api_key)
        self.client = None  # placeholder

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_system_prompt(self, npc_name: str, turn_index: int) -> str:
        """
        Render the system prompt template with the given NPC parameters.

        Args:
            npc_name:   Name of the NPC for this scenario.
            turn_index: Turn number to embed in the prompt for temporal grounding.

        Returns:
            Fully rendered system prompt string ready to send to the API.

        TODO: Load the template from configs/data_config.yaml instead of the
              module-level constant so prompts are version-controlled separately
              from code.  Use `yaml.safe_load` + `str.format_map`.

        TODO: Inject the JSON Schema of DialogState so the model knows the
              exact field names and types:
                  schema_json = json.dumps(
                      DialogState.model_json_schema(), indent=2
                  )
        """
        # TODO: return SYSTEM_PROMPT_TEMPLATE.format(
        #           schema_json=json.dumps(
        #               DialogState.model_json_schema(), indent=2
        #           ),
        #           npc_name=npc_name,
        #           turn_index=turn_index,
        #       )
        raise NotImplementedError

    def _build_user_message(self, dialog_excerpt: str) -> str:
        """
        Wrap the dialog excerpt in a user-role message for the API call.

        Args:
            dialog_excerpt: Raw multi-turn dialog string, e.g.:
                "Player: I need to find the eastern pass.
NPC: Why should I trust you?"

        Returns:
            Formatted user message string.

        TODO: Consider prepending a brief chain-of-thought instruction
              ("Think step by step then output JSON") and using a
              two-shot example of (dialog, JSON) to improve reliability.
              Strip the CoT reasoning from the response before saving.
        """
        raise NotImplementedError

    @staticmethod
    def _sample_scenario() -> dict:
        """
        Randomly sample NPC metadata to create diverse training scenarios.

        Returns:
            dict with keys: npc_name, quest_hook, turn_index, initial_inventory.

        TODO: Replace random.choice with a weighted sampler that ensures
              balanced distribution across aggression levels and quest types.
              Use `numpy.random.Generator` seeded from configs for reproducibility.
        """
        # TODO: return {
        #     "npc_name":          random.choice(NPC_NAME_POOL),
        #     "quest_hook":        random.choice(QUEST_HOOK_POOL),
        #     "turn_index":        random.randint(0, 10),
        #     "initial_inventory": [],  # TODO: sample from an item pool
        # }
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Core generation loop
    # ------------------------------------------------------------------

    # TODO: Decorate with @retry(stop=stop_after_attempt(3),
    #                            wait=wait_exponential(min=1, max=10))
    def _call_api(self, system_prompt: str, user_message: str) -> str:
        """
        Send a chat completion request to the synthesis API and return the
        raw response string.

        Args:
            system_prompt: Rendered system prompt with schema + NPC context.
            user_message:  The dialog excerpt to annotate.

        Returns:
            Raw text content of the assistant's response.  May contain
            markdown fences if the model ignores the "no markdown" instruction;
            caller must strip them.

        Expected API call (openai SDK):
            response = self.client.chat.completions.create(
                model=self.model_name,
                temperature=self.temperature,
                response_format={"type": "json_object"},  # forces JSON mode
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_message},
                ],
            )
            return response.choices[0].message.content

        TODO: Add token-usage tracking (prompt_tokens, completion_tokens) and
              log cumulative cost so you can estimate total synthesis cost
              before launching a full run.
        TODO: Handle `openai.RateLimitError` and `openai.APIStatusError`
              explicitly in addition to the tenacity retry decorator.
        """
        raise NotImplementedError

    def generate(self) -> Iterator[dict]:
        """
        Main generation loop.  Yields one dict per sample:
            {
                "dialog":     <str>  -- the raw dialog excerpt,
                "state_json": <str>  -- the model's JSON string (unvalidated),
                "npc_name":   <str>,
                "turn_index": <int>,
            }

        The caller (usually DatasetBuilder.build) passes each yielded dict
        through Validator.validate before writing to disk.

        TODO: Implement the loop body:
            1. Call `_sample_scenario()` to get NPC metadata.
            2. Synthesise a plausible dialog excerpt (either via a second API
               call or from a template bank).
            3. Call `_build_system_prompt` + `_build_user_message`.
            4. Call `_call_api` and strip markdown fences from the response.
            5. Yield the resulting dict.
            6. Write a checkpoint JSONL file every 100 samples so a crashed
               run can be resumed without re-generating from scratch.

        TODO: Parallelise with `concurrent.futures.ThreadPoolExecutor` to
              saturate API rate limits.  Use a `tqdm` progress bar for UX.
        """
        raise NotImplementedError
