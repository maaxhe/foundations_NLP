# Foundations of NLP

Command-line NPC generator for D&D-style characters with slash commands, persistent storage, and a Qwen-based chat backend.

## Contributors

Ole, Maluna, Max

## Installation

Python 3.11+ is recommended.

```bash
pip install -r requirements.txt
```

If you want real model-based chat instead of the built-in fallback responses, make sure a Qwen model is available locally or downloadable via Hugging Face. The default model name is `Qwen/Qwen2.5-1.5B-Instruct`.

## Start The App

```bash
python chat.py
```

If you want to skip model loading and use the fallback dialogue logic directly:

```bash
python chat.py --no-model
```

## Main Commands

All commands start with `/`.

| Command | What it does |
|---|---|
| `/create <text>` | Generate a new NPC from a free-text description |
| `/list` | Show all saved NPCs |
| `/list <ref>` | Load an NPC by list number, id, or name |
| `/chat [ref]` | Chat with the current or selected NPC |
| `/status` | Show the current NPC sheet |
| `/edit <field> <value>` | Edit a specific NPC field directly |
| `/update <text>` | Apply a natural-language state update |
| `/hyper` | Adjust Qwen generation settings |
| `/quit` | Exit |

## Example

```text
/create elf rogue named Mira level 7 HP 42 weapon rapier chaotic good suspicious
/chat
/update you are now lawful evil from now on
/edit emotional_state friendly
/list
```

## Behavior

- Unspecified fields are generated automatically from the D&D dataset.
- Provided details such as `HP`, `level`, `race`, `weapon`, `subclass`, or alignment are kept.
- NPC emotional state is stored explicitly and can be generated, edited, or updated later.
- Generated NPCs are saved to `data_sets/generated_npcs.json`.
- There are no player levels and no quest system in the current command app.

## Project Structure

```text
foundations_NLP/
|-- chat.py
|-- npc_generator/
|   |-- character_sampler.py
|   |-- dialogue_engine.py
|   |-- npc.py
|   |-- registry.py
|   |-- spec_parser.py
|   `-- story_generator.py
|-- data_sets/
|-- tests/
`-- requirements.txt
```
