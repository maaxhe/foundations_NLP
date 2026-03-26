import torch
from transformers import GPT2Tokenizer
from train_gpt2 import DEFAULT_HYPERPARAMS, TextFileDataset


def get_tokenizer():
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    return tok


def test_default_hyperparams():
    assert "epochs" in DEFAULT_HYPERPARAMS
    assert "lr" in DEFAULT_HYPERPARAMS
    assert DEFAULT_HYPERPARAMS["epochs"] > 0


def test_dataset(tmp_path):
    tok = get_tokenizer()
    f = tmp_path / "train.txt"
    f.write_text("A human fighter with a soldier background.\nAn elf wizard from the north.\n")

    ds = TextFileDataset(tok, str(f), block_size=32)
    assert len(ds) == 2

    item = ds[0]
    assert "input_ids" in item
    assert item["input_ids"].shape[0] == 32


def test_dataset_labels_equal_input_ids(tmp_path):
    tok = get_tokenizer()
    f = tmp_path / "t.txt"
    f.write_text("Some NPC text here.\n")
    ds = TextFileDataset(tok, str(f), block_size=32)
    assert torch.equal(ds[0]["input_ids"], ds[0]["labels"])
