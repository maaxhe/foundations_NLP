"""
evaluator.py -- Evaluation metrics for the fine-tuned dialog-state extraction model.

Responsibility:
    Given a trained model and a held-out validation dataset, compute:
        1. Schema Validity Rate (SVR)   -- % of outputs parseable by DialogState.
        2. Exact Match (EM)             -- % of outputs identical to gold JSON
                                           after normalisation.
        3. Field-level F1               -- token overlap per field (like SQuAD F1)
                                           to catch partial correctness.
        4. Aggression Accuracy          -- classification accuracy on the
                                           aggression enum field specifically.
        5. Trust Score MAE              -- mean absolute error on trust_score float.

These metrics are computed over the validation split and written to a report file.

Dependencies:
    pip install transformers peft datasets evaluate scikit-learn
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# TODO: import torch
# TODO: from transformers import AutoTokenizer, pipeline
# TODO: from peft import PeftModel
# TODO: from datasets import Dataset
# TODO: from sklearn.metrics import accuracy_score, classification_report
# TODO: from data_pipeline.schemas import DialogState, AggressionLevel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metric result containers
# ---------------------------------------------------------------------------

@dataclass
class EvaluationReport:
    """
    Aggregated evaluation metrics for one model checkpoint.

    Attributes:
        schema_validity_rate: Fraction of outputs that parse into DialogState.
        exact_match:          Fraction of outputs identical to gold JSON
                              (after key-sorting and whitespace normalisation).
        aggression_accuracy:  Classification accuracy on the aggression field.
        trust_mae:            Mean absolute error on the trust_score field.
        field_f1:             Dict mapping each top-level field name to its
                              average token-level F1 score.
        num_samples:          Total number of samples evaluated.
        num_invalid:          Number of outputs that failed schema validation.

    TODO: Add `per_aggression_class_report: dict` (from sklearn
          classification_report) for per-class precision/recall/F1.
    """
    schema_validity_rate: float             = 0.0
    exact_match:          float             = 0.0
    aggression_accuracy:  float             = 0.0
    trust_mae:            float             = 0.0
    field_f1:             dict[str, float]  = field(default_factory=dict)
    num_samples:          int               = 0
    num_invalid:          int               = 0


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class ModelEvaluator:
    """
    Runs inference on a validation Dataset and computes evaluation metrics.

    Args:
        model:       A loaded PeftModel (or plain AutoModelForCausalLM).
        tokenizer:   Matching tokenizer.
        dataset:     A datasets.Dataset split (the "validation" split).
        batch_size:  Inference batch size.
        max_new_tokens: Maximum tokens to generate per sample.
                        Should be >= the longest gold JSON in the dataset.
                        Start with 512; adjust based on actual gold lengths.

    TODO: Add a `device` argument to support explicit device placement when
          `device_map="auto"` is not available (e.g. on Apple Silicon MPS).
    """

    def __init__(
        self,
        model:          object,
        tokenizer:      object,
        dataset:        object,
        batch_size:     int = 8,
        max_new_tokens: int = 512,
    ) -> None:
        self.model          = model
        self.tokenizer      = tokenizer
        self.dataset        = dataset
        self.batch_size     = batch_size
        self.max_new_tokens = max_new_tokens

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _run_inference(self) -> list[tuple[str, str]]:
        """
        Run the model on every sample in self.dataset and return
        (predicted_json_string, gold_json_string) pairs.

        Returns:
            List of (prediction, gold) string tuples.

        TODO: Implement batched inference:
            results = []
            for i in range(0, len(self.dataset), self.batch_size):
                batch = self.dataset[i:i+self.batch_size]
                inputs = self.tokenizer(
                    batch["prompt"],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).to(self.model.device)
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,        # greedy for deterministic eval
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
                decoded = self.tokenizer.batch_decode(
                    outputs[:, inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                )
                for pred, gold in zip(decoded, batch["completion"]):
                    results.append((pred.strip(), gold.strip()))
            return results

        TODO: Use constrained_decoder.py's grammar sampler here instead of
              greedy decoding to enforce JSON validity during generation.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Metric helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_json(s: str) -> str:
        """
        Normalise a JSON string for comparison:
            1. Parse with json.loads (raises ValueError if invalid).
            2. Re-serialise with sorted keys and no extra whitespace.

        Args:
            s: Raw JSON string (possibly with formatting variation).

        Returns:
            Canonical JSON string, or the original string if parsing fails.

        TODO: Use json.dumps(json.loads(s), sort_keys=True, separators=(",", ":"))
        """
        raise NotImplementedError

    @staticmethod
    def _token_f1(prediction: str, gold: str) -> float:
        """
        Compute token-level F1 between two strings (SQuAD-style).

        Args:
            prediction: Predicted string for a single field value.
            gold:       Gold standard string for the same field.

        Returns:
            F1 score in [0, 1].

        TODO: Implement:
            pred_tokens = prediction.lower().split()
            gold_tokens = gold.lower().split()
            common = Counter(pred_tokens) & Counter(gold_tokens)
            num_common = sum(common.values())
            if num_common == 0: return 0.0
            precision = num_common / len(pred_tokens)
            recall    = num_common / len(gold_tokens)
            return 2 * precision * recall / (precision + recall)
        """
        raise NotImplementedError

    def _compute_field_f1(
        self,
        pred_state: object,   # TODO: type as DialogState
        gold_state: object,   # TODO: type as DialogState
    ) -> dict[str, float]:
        """
        Compute token-level F1 for each top-level string/enum field.

        Returns:
            Dict mapping field name to F1 score, e.g.
            {"player_intent": 0.8, "npc_mood_note": 0.5, "aggression": 1.0}

        TODO: Iterate over DialogState.model_fields and for each field that
              holds a string or enum value, call _token_f1(str(pred), str(gold)).
              Skip complex nested fields (inventory, known_facts) for now.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def evaluate(self) -> EvaluationReport:
        """
        Full evaluation loop.

        Returns:
            A populated EvaluationReport.

        TODO: Implement:
            pairs = self._run_inference()
            report = EvaluationReport(num_samples=len(pairs))
            valid, preds, golds = [], [], []
            for pred_str, gold_str in pairs:
                try:
                    pred_state = DialogState(**json.loads(pred_str))
                    gold_state = DialogState(**json.loads(gold_str))
                    valid.append((pred_state, gold_state))
                except Exception:
                    report.num_invalid += 1
            report.schema_validity_rate = len(valid) / len(pairs)
            report.exact_match = sum(
                _normalise_json(p) == _normalise_json(g)
                for p, g in zip(pred_str, gold_str)  # use raw strings for EM
            ) / len(pairs)
            report.aggression_accuracy = accuracy_score(
                [s.aggression for _, s in valid],
                [p.aggression for p, _ in valid],
            )
            report.trust_mae = float(np.mean([
                abs(p.trust_score - g.trust_score) for p, g in valid
            ]))
            # Field-level F1 (average over all valid pairs)
            field_scores = defaultdict(list)
            for p, g in valid:
                for k, v in _compute_field_f1(p, g).items():
                    field_scores[k].append(v)
            report.field_f1 = {k: float(np.mean(v)) for k, v in field_scores.items()}
            return report

        TODO: Write the report to configs/../results/eval_report.json.
        """
        raise NotImplementedError

    def save_report(self, report: EvaluationReport, path: str) -> None:
        """
        Serialise an EvaluationReport to a JSON file.

        Args:
            report: The populated EvaluationReport.
            path:   File path for the output JSON.

        TODO: Use dataclasses.asdict(report) then json.dump with indent=2.
        """
        raise NotImplementedError
