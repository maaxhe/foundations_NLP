"""
train_dialogue.py
Fine-tunes GPT-2 on the Synthetic Persona Chat dataset
so it can generate in-character NPC dialogue.

Usage:
    python train_dialogue.py
"""

import os
import torch
from torch.utils.data import Dataset
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_CSV = os.path.join(BASE_DIR, "data_sets", "Synthetic_Persona_Chat", "Synthetic-Persona-Chat_train.csv")
TRAIN_FILE = os.path.join(BASE_DIR, "data_sets", "dialogue_train.txt")
MODEL_OUT = os.path.join(BASE_DIR, "models", "gpt2_dialogue")


def build_training_data(csv_path: str, out_file: str):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["Best Generated Conversation"])

    dialogues = []
    for _, row in df.iterrows():
        convo = str(row["Best Generated Conversation"]).strip()
        # rename generic User 1/2 to NPC/Player for our use case
        convo = convo.replace("User 1:", "NPC:").replace("User 2:", "Player:")
        dialogues.append(convo)

    # 3000 reichen für ein erstes Modell, geht viel schneller
    dialogues = dialogues[:3000]

    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n\n".join(dialogues))
    print(f"Wrote {len(dialogues)} dialogues → {out_file}")


class DialogueDataset(Dataset):
    def __init__(self, tokenizer, file_path: str, block_size: int = 256):
        with open(file_path, encoding="utf-8") as f:
            dialogues = [d.strip() for d in f.read().split("\n\n") if d.strip()]
        pad = tokenizer.eos_token_id
        self.examples = []
        for dialogue in dialogues:
            ids = tokenizer.encode(dialogue, truncation=True, max_length=block_size)
            ids += [pad] * (block_size - len(ids))
            self.examples.append(torch.tensor(ids, dtype=torch.long))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return {"input_ids": self.examples[idx], "labels": self.examples[idx]}


def train():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained("gpt2")

    dataset = DialogueDataset(tokenizer, TRAIN_FILE, block_size=256)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    os.makedirs(MODEL_OUT, exist_ok=True)

    args = TrainingArguments(
        output_dir=MODEL_OUT,
        num_train_epochs=1,
        per_device_train_batch_size=16,
        learning_rate=5e-5,
        warmup_steps=100,
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

    print("Starting GPT-2 dialogue fine-tuning...")
    trainer.train()
    trainer.save_model(MODEL_OUT)
    tokenizer.save_pretrained(MODEL_OUT)
    print(f"Model saved → {MODEL_OUT}")


if __name__ == "__main__":
    os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
    build_training_data(TRAIN_CSV, TRAIN_FILE)
    train()
