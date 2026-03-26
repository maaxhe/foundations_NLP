# Foundations of NLP

A repository for exploring foundational NLP concepts — built around a D&D NPC Generator with GPT-2 fine-tuning and a playable chat interface.

## Contributors

Ole, Maluna, Max

---

## Installation

**Requirements:** Python 3.11+, conda or venv recommended.

```bash
git clone https://github.com/maaxhe/foundations_NLP.git
cd foundations_NLP
pip install -r requirements.txt
pip install 'transformers[torch]' accelerate rich
```

---

## Training the Dialogue Model

Before chatting you need to fine-tune GPT-2 on the dialogue dataset:

```bash
python train_dialogue.py
```

This trains GPT-2 on the Synthetic Persona Chat dataset (~10 min on Apple Silicon).
The model is saved to `models/gpt2_dialogue/` and loaded automatically on the next run.

To also train the NPC story model:

```bash
python train_gpt2.py
```

---

## Starting the Chat

```bash
python chat.py
```

Enter your name, then start talking to randomly generated D&D NPCs.

---

## Chat Commands

| Command | What it does |
|---|---|
| `status` | Show your level, XP, gold |
| `quests` | Show active and completed quests |
| `complete` | Complete your first active quest and collect rewards |
| `complete 2` | Complete quest number 2 |
| `new` | Generate a new NPC |
| `hyper` | Adjust GPT-2 hyperparameters and retrain |
| `quit` | Exit |

**Quick replies:** Type a number (1–4) to use a preset dialogue option. Options rotate after each use. Quest-related options appear automatically when talking about quests.

---

## Gameplay

- Talk to NPCs and accept quests with **"I'll take the quest."**
- Complete quests by typing `complete` — you earn **XP** and **gold**
- Collect enough XP to **level up** (100 XP × current level)
- Use `new` to find a new NPC and take on more quests

---

## Project Structure

```
foundations_NLP/
├── chat.py                  # main entry point
├── train_gpt2.py            # fine-tunes GPT-2 on NPC stories
├── train_dialogue.py        # fine-tunes GPT-2 on dialogue data
├── npc_generator/
│   ├── character_sampler.py # samples NPCs from D&D dataset
│   ├── story_generator.py   # generates NPC backstories
│   ├── quest_generator.py   # generates quests for the player
│   ├── dialogue_engine.py   # GPT-2 powered NPC responses
│   ├── player.py            # player state (level, XP, quest log)
│   └── npc.py               # NPC dataclass
├── data_sets/
│   ├── dnd_character_database/
│   └── Synthetic_Persona_Chat/
├── models/
│   ├── gpt2_npc/            # story model
│   └── gpt2_dialogue/       # dialogue model
├── notebooks/
│   └── 04_hyperparameter_report.ipynb
└── tests/
```

---

## Git Workflow

```bash
git add .
git commit -m "beschreibung der änderung"
git push
```

Neueste Änderungen holen:
```bash
git pull
```
