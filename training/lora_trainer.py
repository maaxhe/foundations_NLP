"""
lora_trainer.py -- LoRA fine-tuning for dialog-state extraction (Phase 2).

Responsibility:
    Fine-tune a local Llama-3B model with LoRA (Low-Rank Adaptation) so that
    it reliably outputs valid DialogState JSON given a dialog excerpt as input.
    Uses the PEFT library from HuggingFace and the SFTTrainer from TRL.

Key design decisions:
    - We fine-tune ONLY the extraction model (not the chat/dialog model).
    - LoRA targets: query and value projection layers (q_proj, v_proj).
    - Training objective: causal LM loss over the completion tokens only
      (prompt tokens are masked with label=-100).
    - Mixed precision: bfloat16 on CUDA; float32 fallback on CPU/MPS.

Dependencies:
    pip install transformers>=4.40 peft>=0.10 trl>=0.8 bitsandbytes accelerate datasets
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# TODO: import torch
# TODO: from transformers import (
#           AutoModelForCausalLM,
#           AutoTokenizer,
#           BitsAndBytesConfig,
#           TrainingArguments,
#       )
# TODO: from peft import (
#           LoraConfig,
#           get_peft_model,
#           TaskType,
#           PeftModel,
#       )
# TODO: from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
# TODO: from datasets import DatasetDict


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class LoraTrainingConfig:
    """
    All hyperparameters for the LoRA fine-tuning run.

    Attributes:
        base_model_id:    HuggingFace model ID or local path to the Llama-3B
                          base model.  Example: "meta-llama/Llama-3.2-3B".
        dataset_dir:      Path to the DatasetDict saved by dataset_builder.py.
        output_dir:       Where to save the LoRA adapter weights.
        lora_r:           LoRA rank.  Higher rank = more capacity but more VRAM.
                          Recommended starting point: 16.
        lora_alpha:       LoRA scaling factor (usually 2 * lora_r).
        lora_dropout:     Dropout on LoRA layers.  Helps prevent overfitting on
                          small datasets.
        target_modules:   Which linear layers to adapt.  For Llama-3:
                          ["q_proj", "v_proj"] is a safe minimum; add
                          "k_proj", "o_proj" for more capacity.
        num_epochs:       Number of training epochs.
        batch_size:       Per-device training batch size.
        grad_accum_steps: Gradient accumulation steps (effective batch =
                          batch_size * grad_accum_steps).
        learning_rate:    Peak AdamW learning rate.
        max_seq_length:   Maximum token length; longer samples are truncated.
        load_in_4bit:     If True, load the base model in 4-bit NF4 quantisation
                          via bitsandbytes to reduce VRAM usage from ~6 GB to ~2 GB.
        response_template: String that marks the beginning of the completion in
                           the prompt template, used to mask prompt tokens.
                           Must match PROMPT_TEMPLATE in dataset_builder.py.
                           Example: "### Response:\\n"

    TODO: Load all of these from configs/training_config.yaml with:
              import yaml
              cfg = LoraTrainingConfig(**yaml.safe_load(open(path))["lora"])
    """
    base_model_id:    str        = "meta-llama/Llama-3.2-3B"
    dataset_dir:      str        = "data/processed"
    output_dir:       str        = "models/lora_adapter"
    lora_r:           int        = 16
    lora_alpha:       int        = 32
    lora_dropout:     float      = 0.05
    target_modules:   list[str]  = None   # TODO: default_factory=["q_proj","v_proj"]
    num_epochs:       int        = 3
    batch_size:       int        = 4
    grad_accum_steps: int        = 4
    learning_rate:    float      = 2e-4
    max_seq_length:   int        = 1024
    load_in_4bit:     bool       = True
    response_template: str       = "### Response:\n"


# ---------------------------------------------------------------------------
# Trainer class
# ---------------------------------------------------------------------------

class LoRADialogStateTrainer:
    """
    Wraps HuggingFace PEFT + TRL SFTTrainer to fine-tune Llama-3B with LoRA.

    Args:
        config: A LoraTrainingConfig instance.

    Usage:
        trainer = LoRADialogStateTrainer(config)
        trainer.train()
        trainer.save()
    """

    def __init__(self, config: LoraTrainingConfig) -> None:
        self.config    = config
        self.model     = None   # populated in _load_model
        self.tokenizer = None   # populated in _load_tokenizer
        self.trainer   = None   # populated in _build_trainer

    # ------------------------------------------------------------------
    # Private setup methods
    # ------------------------------------------------------------------

    def _load_tokenizer(self) -> None:
        """
        Load the tokenizer for the base model and configure padding.

        TODO: Implement:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.base_model_id,
                trust_remote_code=True,
            )
            # Llama-3 does not have a pad token by default.
            # Set pad_token = eos_token as a common workaround.
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            # Pad on the right for causal LM training.
            self.tokenizer.padding_side = "right"
        """
        raise NotImplementedError

    def _load_model(self) -> None:
        """
        Load the base Llama-3B model, optionally in 4-bit quantisation.

        TODO: Implement:
            bnb_config = None
            if self.config.load_in_4bit:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,   # QLoRA trick
                )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model_id,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
            self.model.config.use_cache = False   # required for gradient checkpointing

        TODO: After loading, call `model.print_trainable_parameters()` and log
              the ratio of trainable / total parameters to confirm LoRA is applied
              correctly (expected ~0.5-1% for r=16 on Llama-3B).
        """
        raise NotImplementedError

    def _apply_lora(self) -> None:
        """
        Wrap the loaded model with LoRA adapters using PEFT.

        TODO: Implement:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=self.config.target_modules or ["q_proj", "v_proj"],
                bias="none",
            )
            self.model = get_peft_model(self.model, lora_config)

        TODO: Call `self.model.print_trainable_parameters()` here to verify.
        """
        raise NotImplementedError

    def _build_trainer(self, dataset: object) -> None:
        """
        Construct the TRL SFTTrainer with a completion-only data collator.

        Args:
            dataset: A datasets.DatasetDict with "train" and "validation" splits.

        The DataCollatorForCompletionOnlyLM masks all tokens before the
        response_template, ensuring the loss is computed ONLY on the JSON
        completion tokens.  This is critical: without masking, the model
        also optimises the prompt tokens, wasting capacity.

        TODO: Implement:
            training_args = TrainingArguments(
                output_dir=self.config.output_dir,
                num_train_epochs=self.config.num_epochs,
                per_device_train_batch_size=self.config.batch_size,
                gradient_accumulation_steps=self.config.grad_accum_steps,
                learning_rate=self.config.learning_rate,
                bf16=torch.cuda.is_bf16_supported(),
                fp16=not torch.cuda.is_bf16_supported(),
                logging_steps=10,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                report_to="none",   # TODO: switch to "wandb" for experiment tracking
            )
            collator = DataCollatorForCompletionOnlyLM(
                response_template=self.config.response_template,
                tokenizer=self.tokenizer,
            )
            self.trainer = SFTTrainer(
                model=self.model,
                args=training_args,
                train_dataset=dataset["train"],
                eval_dataset=dataset["validation"],
                data_collator=collator,
                dataset_text_field="text",
                max_seq_length=self.config.max_seq_length,
                tokenizer=self.tokenizer,
            )

        TODO: Integrate Weights & Biases (wandb) for experiment tracking.
              Set report_to="wandb" and add a run name that includes
              lora_r, num_epochs, and learning_rate for easy comparison.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def train(self) -> None:
        """
        Full training pipeline: load -> apply LoRA -> build trainer -> train.

        TODO: Implement:
            self._load_tokenizer()
            self._load_model()
            self._apply_lora()
            from data_pipeline.dataset_builder import DatasetBuilder
            dataset = DatasetBuilder(...).load()
            self._build_trainer(dataset)
            self.trainer.train()

        TODO: Log the final train/eval loss to a results JSON file at
              self.config.output_dir / "training_results.json".
        """
        raise NotImplementedError

    def save(self) -> None:
        """
        Save only the LoRA adapter weights (not the full model) to disk.

        The adapter weights are typically ~10-50 MB vs ~6 GB for the full model,
        making them easy to version-control and share.

        TODO: Implement:
            self.model.save_pretrained(self.config.output_dir)
            self.tokenizer.save_pretrained(self.config.output_dir)
            # This saves only the adapter_model.bin and adapter_config.json files.

        TODO: Also export a merged (LoRA-baked) model for use with Ollama:
            merged = self.model.merge_and_unload()
            merged.save_pretrained(self.config.output_dir + "_merged")
            Then convert to GGUF format with llama.cpp for Ollama compatibility.
        """
        raise NotImplementedError

    @staticmethod
    def load_adapter(base_model_id: str, adapter_path: str) -> object:
        """
        Load a previously saved LoRA adapter onto the base model at inference time.

        Args:
            base_model_id: HuggingFace model ID of the same base model used for training.
            adapter_path:  Path to the saved adapter weights directory.

        Returns:
            A PeftModel (base model + LoRA adapter merged in inference mode).

        TODO: Implement:
            base = AutoModelForCausalLM.from_pretrained(
                base_model_id, device_map="auto"
            )
            model = PeftModel.from_pretrained(base, adapter_path)
            model.eval()
            return model

        TODO: Consider calling model.merge_and_unload() here for faster
              inference (no adapter overhead), but note this is irreversible.
        """
        raise NotImplementedError
