"""
run_pipeline.py -- Top-level CLI entry point for the three project phases.

Usage:
    python run_pipeline.py --phase data    --config configs/data_config.yaml
    python run_pipeline.py --phase train   --config configs/training_config.yaml
    python run_pipeline.py --phase serve   --config configs/inference_config.yaml
    python run_pipeline.py --phase eval    --config configs/training_config.yaml

Each phase delegates to the corresponding module:
    data  -> data_pipeline.dataset_builder.DatasetBuilder
    train -> training.lora_trainer.LoRADialogStateTrainer
    eval  -> training.evaluator.ModelEvaluator
    serve -> uvicorn app.main:app  (subprocess)

Dependencies:
    pip install pyyaml click
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# TODO: import yaml
# TODO: from data_pipeline.generator import SyntheticDataGenerator
# TODO: from data_pipeline.validator import DataValidator
# TODO: from data_pipeline.dataset_builder import DatasetBuilder
# TODO: from training.lora_trainer import LoRADialogStateTrainer, LoraTrainingConfig
# TODO: from training.evaluator import ModelEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Phase runners
# ---------------------------------------------------------------------------

def run_data_phase(config_path: str) -> None:
    """
    Execute Phase 1: generate, validate, and save the training dataset.

    Args:
        config_path: Path to configs/data_config.yaml.

    TODO: Implement:
        config = yaml.safe_load(open(config_path))
        api_key = os.environ[config["synthesis_api"]["api_key_env_var"]]
        generator = SyntheticDataGenerator(
            api_key=api_key,
            model_name=config["synthesis_api"]["model"],
            n_samples=config["generation"]["n_samples"],
            output_path=config["generation"]["output_dir"],
            temperature=config["synthesis_api"]["temperature"],
        )
        validator = DataValidator(
            run_coherence_check=config["validation"]["run_coherence_check"],
        )
        builder = DatasetBuilder(
            generator=generator,
            validator=validator,
            output_dir=config["dataset"]["output_dir"],
            val_fraction=config["dataset"]["val_fraction"],
            seed=config["dataset"]["seed"],
        )
        dataset = builder.build()
        logger.info("Dataset saved to %s", config["dataset"]["output_dir"])
        logger.info("Train: %d, Val: %d", len(dataset["train"]), len(dataset["validation"]))
    """
    logger.info("Starting data generation phase with config: %s", config_path)
    # TODO: implement (see docstring above)
    raise NotImplementedError("Data phase not yet implemented.")


def run_train_phase(config_path: str) -> None:
    """
    Execute Phase 2: LoRA fine-tuning on the generated dataset.

    Args:
        config_path: Path to configs/training_config.yaml.

    TODO: Implement:
        config = yaml.safe_load(open(config_path))
        train_cfg = LoraTrainingConfig(**config["lora"])
        trainer = LoRADialogStateTrainer(train_cfg)
        trainer.train()
        trainer.save()
        logger.info("LoRA adapter saved to %s", train_cfg.output_dir)
    """
    logger.info("Starting training phase with config: %s", config_path)
    # TODO: implement (see docstring above)
    raise NotImplementedError("Training phase not yet implemented.")


def run_eval_phase(config_path: str) -> None:
    """
    Execute evaluation of the fine-tuned model on the validation split.

    Args:
        config_path: Path to configs/training_config.yaml.

    TODO: Implement:
        config = yaml.safe_load(open(config_path))
        from datasets import load_from_disk
        dataset = load_from_disk(config["lora"]["dataset_dir"])
        model, tokenizer = load_trained_model(config["lora"]["output_dir"])
        evaluator = ModelEvaluator(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset["validation"],
        )
        report = evaluator.evaluate()
        evaluator.save_report(report, config["evaluation"]["results_output"])
        logger.info("Eval report: %s", report)
    """
    logger.info("Starting evaluation phase with config: %s", config_path)
    # TODO: implement (see docstring above)
    raise NotImplementedError("Eval phase not yet implemented.")


def run_serve_phase(config_path: str) -> None:
    """
    Launch the FastAPI server with uvicorn.

    Args:
        config_path: Path to configs/inference_config.yaml.

    TODO: Implement:
        import subprocess, os
        config = yaml.safe_load(open(config_path))
        os.environ["INFERENCE_CONFIG"] = config_path
        subprocess.run([
            sys.executable, "-m", "uvicorn",
            "app.main:app",
            "--host", config["server"]["host"],
            "--port", str(config["server"]["port"]),
            "--reload" if config["server"]["reload"] else "",
        ], check=True)
    """
    logger.info("Starting FastAPI server with config: %s", config_path)
    # TODO: implement (see docstring above)
    raise NotImplementedError("Serve phase not yet implemented.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Parse CLI arguments and dispatch to the appropriate phase runner.

    TODO: Replace argparse with `click` for subcommand support and
          auto-generated --help text.  Example:
              @click.group()
              @click.option("--config", required=True)
              def cli(config): ...

              @cli.command()
              @click.pass_context
              def data(ctx): run_data_phase(ctx.obj["config"])
    """
    parser = argparse.ArgumentParser(
        description="Run a phase of the D&D NPC NLP pipeline."
    )
    parser.add_argument(
        "--phase",
        required=True,
        choices=["data", "train", "eval", "serve"],
        help="Which pipeline phase to execute.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML config file for the chosen phase.",
    )
    args = parser.parse_args()

    phase_runners = {
        "data":  run_data_phase,
        "train": run_train_phase,
        "eval":  run_eval_phase,
        "serve": run_serve_phase,
    }

    runner = phase_runners[args.phase]
    try:
        runner(args.config)
    except NotImplementedError as e:
        logger.error("Phase '%s' is not yet implemented: %s", args.phase, e)
        sys.exit(1)


if __name__ == "__main__":
    main()
