"""
train_gpt2.py
Fine-tunes GPT-2 small on NPC-style background stories
built from the D&D character dataset.

Usage:
    python train_gpt2.py
"""

import os
import sys

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


class TextFileDataset(Dataset):
    def __init__(self, tokenizer, file_path: str, block_size: int = 128):
        with open(file_path, encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        pad_id = tokenizer.pad_token_id
        self.examples = []
        for line in lines:
            ids = tokenizer.encode(line, truncation=True, max_length=block_size)
            # pad to block_size so all tensors are the same length
            ids = ids + [pad_id] * (block_size - len(ids))
            self.examples.append(torch.tensor(ids, dtype=torch.long))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return {"input_ids": self.examples[idx], "labels": self.examples[idx]}

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "data_sets", "dnd_character_database", "dnd_chars_all.csv")
TRAIN_FILE = os.path.join(BASE_DIR, "data_sets", "gpt2_train.txt")
MODEL_OUT = os.path.join(BASE_DIR, "models", "gpt2_npc")

ALIGNMENT_MAP = {
    "LG": "Lawful Good", "NG": "Neutral Good", "CG": "Chaotic Good",
    "LN": "Lawful Neutral", "N": "True Neutral", "CN": "Chaotic Neutral",
    "LE": "Lawful Evil", "NE": "Neutral Evil", "CE": "Chaotic Evil",
}

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


def build_training_data(csv_path: str, out_file: str):
    """Convert D&D characters into short story prompts for GPT-2 training."""
    df = pd.read_csv(csv_path, sep=";", low_memory=False)
    df = df.dropna(subset=["race", "justClass"])
    df["primary_class"] = df["justClass"].str.split("|").str[0].str.strip()
    df["race_clean"] = df["processedRace"].fillna(df["race"])

    stories = []
    for _, row in df.iterrows():
        race = str(row.get("race_clean", "unknown"))
        cls = str(row.get("primary_class", "adventurer"))
        bg = str(row.get("background", "humble")).lower()
        align_code = str(row.get("processedAlignment", "NG"))
        alignment = ALIGNMENT_MAP.get(align_code, "True Neutral")
        align_sent = ALIGNMENT_SENTENCES.get(alignment, "")

        text = (
            f"<NPC> A {race} {cls} with a {bg} background. "
            f"{align_sent} "
            f"They travel the roads seeking purpose and coin. </NPC>"
        )
        stories.append(text)

    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(stories))
    print(f"Wrote {len(stories)} training examples → {out_file}")


DEFAULT_HYPERPARAMS = {
    "epochs":     3,
    "batch_size": 8,
    "lr":         5e-5,
    "block_size": 128,
    "warmup_steps": 100,
}


def train(train_file: str, model_out: str, hp: dict | None = None):
    if hp is None:
        hp = DEFAULT_HYPERPARAMS.copy()

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained("gpt2")

    dataset = TextFileDataset(
        tokenizer=tokenizer,
        file_path=train_file,
        block_size=int(hp["block_size"]),
    )
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    os.makedirs(model_out, exist_ok=True)

    args = TrainingArguments(
        output_dir=model_out,
        num_train_epochs=int(hp["epochs"]),
        per_device_train_batch_size=int(hp["batch_size"]),
        learning_rate=float(hp["lr"]),
        warmup_steps=int(hp["warmup_steps"]),
        save_steps=500,
        save_total_limit=1,
        logging_steps=100,
        prediction_loss_only=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        data_collator=collator,
        train_dataset=dataset,
    )

    print("Starting GPT-2 fine-tuning...")
    trainer.train()
    trainer.save_model(model_out)
    tokenizer.save_pretrained(model_out)
    print(f"Model saved → {model_out}")


if __name__ == "__main__":
    os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
    build_training_data(CSV_PATH, TRAIN_FILE)
    train(TRAIN_FILE, MODEL_OUT)
