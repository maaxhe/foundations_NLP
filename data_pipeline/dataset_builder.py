"""
dataset_builder.py -- Assemble validated samples into a HuggingFace Dataset (Phase 1).

Responsibility:
    1. Orchestrate the Generator -> Validator pipeline.
    2. Convert accepted samples into prompt/completion format for causal LM fine-tuning.
    3. Persist the dataset to disk in Arrow format (datasets.Dataset.save_to_disk)
       and optionally push to HuggingFace Hub.
    4. Produce a held-out validation split (default 10%).

Prompt format (instruction-tuning / Alpaca style):
    ### Instruction:
    Extract the dialog state from the following D&D NPC conversation as JSON.

    ### Dialog:
    {dialog_text}

    ### Response:
    {state_json}

Dependencies:
    pip install datasets transformers
"""

from __future__ import annotations

import logging
from pathlib import Path

# TODO: from datasets import Dataset, DatasetDict
# TODO: from data_pipeline.generator import SyntheticDataGenerator
# TODO: from data_pipeline.validator import DataValidator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

INSTRUCTION = (
    "Extract the dialog state from the following D&D NPC conversation "
    "and return a valid JSON object matching the schema exactly."
)

PROMPT_TEMPLATE = (
    "### Instruction:\n{instruction}\n\n"
    "### Dialog:\n{dialog}\n\n"
    "### Response:\n"
)


def build_prompt(dialog_text: str) -> str:
    """
    Render the input portion of the fine-tuning prompt (without the response).

    Args:
        dialog_text: Raw multi-turn dialog string.

    Returns:
        Formatted prompt string.  The response (JSON) is concatenated by
        format_sample so the tokeniser can mask prompt tokens during loss
        computation using a DataCollatorForSeq2Seq or custom collator.

    TODO: Experiment with few-shot examples embedded in the prompt.
          Load 2-3 canonical (dialog, JSON) pairs from a curated file
          (e.g. data/few_shot_examples.jsonl) and prepend them.
          Measure whether few-shot prompting improves exact-match accuracy
          on the validation set compared to zero-shot prompting.
    """
    raise NotImplementedError


def format_sample(dialog_text: str, state_json: str) -> dict:
    """
    Produce a single training record with prompt and completion keys.

    Args:
        dialog_text: Input dialog excerpt.
        state_json:  Target JSON string (already validated by DataValidator).

    Returns:
        {
            "prompt":     <str>  -- the full instruction + dialog prompt,
            "completion": <str>  -- the target JSON string,
            "text":       <str>  -- prompt + completion concatenated (used by
                                    trainers that expect a single text field),
        }

    TODO: Append the tokeniser's EOS token to completion so the model learns
          to stop after the JSON closes.  Load the token with:
              AutoTokenizer.from_pretrained(model_id).eos_token
          The exact token depends on the base model (Llama-3: "</s>").
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# DatasetBuilder
# ---------------------------------------------------------------------------

class DatasetBuilder:
    """
    Orchestrates data generation, validation, and dataset assembly.

    Args:
        generator:    A configured SyntheticDataGenerator instance.
        validator:    A configured DataValidator instance.
        output_dir:   Path where the final DatasetDict is saved.
        val_fraction: Fraction of samples reserved for validation (default 0.1).
        seed:         Random seed for the train/val split.

    TODO: Accept a configs/data_config.yaml path and construct the
          Generator and Validator internally so this class can be invoked
          directly from the CLI:
              python -m data_pipeline.dataset_builder --config configs/data_config.yaml
    """

    def __init__(
        self,
        generator:    object,
        validator:    object,
        output_dir:   str   = "data/processed",
        val_fraction: float = 0.1,
        seed:         int   = 42,
    ) -> None:
        self.generator    = generator
        self.validator    = validator
        self.output_dir   = Path(output_dir)
        self.val_fraction = val_fraction
        self.seed         = seed

    def _generate_and_validate(self) -> list[dict]:
        """
        Run the generator and filter samples through the validator.

        Returns:
            List of format_sample dicts that passed all validation layers.

        TODO: Implement:
            accepted, rejected = [], []
            for raw in self.generator.generate():
                result = self.validator.validate(
                    raw["dialog"], raw["state_json"]
                )
                if result.is_valid:
                    accepted.append(format_sample(raw["dialog"], raw["state_json"]))
                else:
                    rejected.append(raw)
                    logger.warning("Rejected sample: %s", result.errors)
            acceptance_rate = len(accepted) / (len(accepted) + len(rejected) + 1e-9)
            logger.info("Accepted %d samples (%.1f%%)", len(accepted), 100*acceptance_rate)
            return accepted

        TODO: Persist rejected samples to data/rejected.jsonl for analysis.
              High rejection rates likely indicate a prompt engineering issue in
              generator.py and should trigger a prompt revision loop.
        """
        raise NotImplementedError

    def build(self) -> object:   # TODO: return type -> DatasetDict
        """
        Full pipeline: generate -> validate -> format -> split -> save.

        Returns:
            A datasets.DatasetDict with keys "train" and "validation".

        Steps to implement:
            1. Call _generate_and_validate() to get the accepted list.
            2. Create datasets.Dataset.from_list(accepted).
            3. Call .train_test_split(test_size=self.val_fraction,
                                       seed=self.seed).
            4. Call .save_to_disk(str(self.output_dir)).
            5. Log dataset statistics: num rows, avg prompt length in tokens,
               aggression-level distribution as a bar chart via matplotlib.
            6. Return the DatasetDict.

        TODO: Optionally push to HuggingFace Hub with
              dataset_dict.push_to_hub(repo_id).  Read hub_repo_id from
              configs/data_config.yaml; skip the push if the field is empty.
        """
        raise NotImplementedError

    def load(self) -> object:    # TODO: return type -> DatasetDict
        """
        Load a previously built dataset from disk.

        Returns:
            datasets.DatasetDict loaded from self.output_dir.

        TODO: Use datasets.load_from_disk(str(self.output_dir)).
              Raise a descriptive FileNotFoundError if the directory is absent
              (the user may have forgotten to run the generation step).
        """
        raise NotImplementedError
