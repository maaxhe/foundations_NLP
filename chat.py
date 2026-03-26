"""
CLI entry point for the NPC generator and command-based chat application.
"""

from __future__ import annotations

import argparse
import random

try:
    from rich.console import Console
    from rich.panel import Panel

    RICH = True
    console = Console()
except ImportError:
    RICH = False
    console = None

from npc_generator.character_sampler import CharacterSampler
from npc_generator.dialogue_engine import DEFAULT_QWEN_MODEL, DialogueEngine
from npc_generator.npc import NPC
from npc_generator.registry import NpcRegistry
from npc_generator.spec_parser import apply_update_instruction, parse_character_specs
from npc_generator.story_generator import StoryGenerator


def print_line(message: str = "") -> None:
    if RICH:
        console.print(message)
    else:
        print(message)


def print_help() -> None:
    help_text = """
/create <text>   Generate a new NPC from free text
/list            List saved NPCs
/list <ref>      Select an NPC by number, id, or exact name
/chat [ref]      Chat with the current or selected NPC
/status          Show the current NPC sheet
/edit <field> <value>
/update <text>   Apply a narrative update, e.g. "/update you are now lawful evil"
/hyper           Adjust Qwen generation settings
/help            Show commands
/quit            Exit
"""
    print_line(help_text.strip())


def render_npc_sheet(npc: NPC) -> None:
    body = (
        f"{npc.name} [{npc.npc_id}]\n"
        f"{npc.race} {npc.primary_class} | {npc.subclass}\n"
        f"Level {npc.level} | HP {npc.HP} | AC {npc.AC}\n"
        f"Alignment: {npc.alignment}\n"
        f"Emotional state: {npc.emotional_state}\n"
        f"Weapon: {npc.weapon}\n"
        f"Goal: {npc.goal}\n"
        f"Quirk: {npc.quirk}\n"
        f"Secret: {npc.secret}\n"
        f"{npc.stat_block}\n\n"
        f"Story:\n{npc.story}"
    )
    if npc.notes:
        body += "\n\nUpdates:\n- " + "\n- ".join(npc.notes[-5:])
    if npc.extra_traits:
        extras = "\n".join(f"- {key}: {value}" for key, value in npc.extra_traits.items())
        body += "\n\nExtra traits:\n" + extras

    if RICH:
        console.print(Panel(body, title="NPC Status", border_style="cyan"))
    else:
        print_line(body)


def list_npcs(registry: NpcRegistry, current_npc_id: str | None) -> None:
    npcs = registry.all()
    if not npcs:
        print_line("No NPCs saved yet. Use /create to generate one.")
        return
    print_line("Saved NPCs:")
    for index, npc in enumerate(npcs, start=1):
        marker = "*" if npc.npc_id == current_npc_id else " "
        summary = f"{index:>2}. {marker} {npc.name} [{npc.npc_id}] | {npc.race} {npc.primary_class} | {npc.emotional_state}"
        print_line(summary)


def build_npc_from_prompt(prompt: str, sampler: CharacterSampler, story_gen: StoryGenerator) -> NPC:
    overrides = parse_character_specs(prompt, sampler)
    overrides["source_prompt"] = prompt
    char = sampler.sample_character(overrides)
    char["story"] = story_gen.generate_story(char)
    return NPC.from_dict(char)


def edit_current_npc(npc: NPC, args: str) -> str:
    parts = args.strip().split(None, 1)
    if len(parts) != 2:
        return "Use /edit <field> <value>."
    field_name, value = parts
    if field_name not in npc.__dataclass_fields__:
        return f"Unknown field '{field_name}'."

    current_value = getattr(npc, field_name)
    if isinstance(current_value, int):
        try:
            value = int(value)
        except ValueError:
            return f"Field '{field_name}' expects a number."
    elif isinstance(current_value, list):
        value = [item.strip() for item in value.split(",") if item.strip()]
    setattr(npc, field_name, value)
    return f"Updated {field_name} for {npc.name}."


def run_hyper_wizard(engine: DialogueEngine) -> None:
    cfg = engine.generation_config
    print_line("Qwen generation settings. Press Enter to keep the current value.")
    try:
        temperature = input(f"temperature [{cfg.temperature}]: ").strip()
        top_p = input(f"top_p [{cfg.top_p}]: ").strip()
        max_new_tokens = input(f"max_new_tokens [{cfg.max_new_tokens}]: ").strip()
        repetition_penalty = input(f"repetition_penalty [{cfg.repetition_penalty}]: ").strip()
    except (KeyboardInterrupt, EOFError):
        print_line("\nNo changes applied.")
        return

    try:
        engine.set_generation_config(
            temperature=float(temperature) if temperature else None,
            top_p=float(top_p) if top_p else None,
            max_new_tokens=int(max_new_tokens) if max_new_tokens else None,
            repetition_penalty=float(repetition_penalty) if repetition_penalty else None,
        )
        print_line("Generation settings updated.")
    except ValueError:
        print_line("Invalid numeric value. Settings unchanged.")


def chat_with_npc(npc: NPC, engine: DialogueEngine, registry: NpcRegistry, sampler: CharacterSampler) -> bool:
    print_line(f"Chatting with {npc.name}. Use 1-4 for suggestions, free text for your own line, /back to leave chat.")

    while True:
        options = engine.suggest_replies(npc)
        print_line("")
        for index, option in enumerate(options, start=1):
            print_line(f"{index}. {option}")
        print_line("Free text is also allowed.")

        try:
            raw = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print_line("\nLeaving chat.")
            return False

        if not raw:
            continue

        if raw.startswith("/"):
            command, _, args = raw.partition(" ")
            command = command.lower()
            if command == "/back":
                registry.upsert(npc)
                return False
            if command == "/status":
                render_npc_sheet(npc)
                continue
            if command == "/hyper":
                run_hyper_wizard(engine)
                continue
            if command == "/edit":
                message = edit_current_npc(npc, args)
                registry.upsert(npc)
                print_line(message)
                continue
            if command == "/update":
                changes = apply_update_instruction(npc, args, sampler)
                registry.upsert(npc)
                if changes:
                    print_line(f"Updated: {', '.join(sorted(changes.keys()))}")
                else:
                    print_line("Stored the update as a note.")
                continue
            if command == "/quit":
                registry.upsert(npc)
                return True
            print_line("Only /back, /status, /edit, /update, /hyper, and /quit are available inside chat.")
            continue

        user_input = raw
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            user_input = options[int(raw) - 1]
            print_line(f"> {user_input}")

        response, _ = engine.chat(npc, user_input)
        npc.record_turn("user", user_input)
        npc.record_turn("assistant", response)
        registry.upsert(npc)
        print_line(f"{npc.name}: {response}")


def main() -> None:
    parser = argparse.ArgumentParser(description="NPC Generator Command App")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--model", default=DEFAULT_QWEN_MODEL, help="Qwen model name or local path")
    parser.add_argument("--no-model", action="store_true", help="Skip Qwen loading and use fallback responses")
    args = parser.parse_args()

    if args.seed is not None:
        import numpy as np

        random.seed(args.seed)
        np.random.seed(args.seed)

    sampler = CharacterSampler()
    story_gen = StoryGenerator()
    engine = DialogueEngine(model_name=args.model, load_model=not args.no_model)
    registry = NpcRegistry()
    current_npc_id: str | None = None

    print_line("NPC Generator Command App")
    print_line(f"Chat backend: {engine.describe_backend()}")
    print_help()

    while True:
        prompt = f"{current_npc_id or 'npc'}> "
        try:
            raw = input(prompt).strip()
        except (KeyboardInterrupt, EOFError):
            print_line("\nBye.")
            break

        if not raw:
            continue
        if not raw.startswith("/"):
            print_line("Commands start with /. Use /create, /chat, or /help.")
            continue

        command, _, args = raw.partition(" ")
        command = command.lower()
        args = args.strip()

        if command in {"/quit", "/exit"}:
            print_line("Bye.")
            break

        if command == "/help":
            print_help()
            continue

        if command in {"/create", "/generate", "/new"}:
            if not args:
                try:
                    args = input("Describe the NPC: ").strip()
                except (KeyboardInterrupt, EOFError):
                    print_line("\nCancelled.")
                    continue
            npc = build_npc_from_prompt(args, sampler, story_gen)
            registry.upsert(npc)
            current_npc_id = npc.npc_id
            render_npc_sheet(npc)
            continue

        if command == "/list":
            if not args:
                list_npcs(registry, current_npc_id)
                continue
            npc = registry.resolve(args)
            if npc is None:
                print_line(f"No NPC found for '{args}'.")
                continue
            current_npc_id = npc.npc_id
            render_npc_sheet(npc)
            continue

        current_npc = registry.resolve(current_npc_id) if current_npc_id else None

        if command == "/status":
            if current_npc is None:
                print_line(f"{registry.count()} NPC(s) saved. No active NPC selected.")
            else:
                render_npc_sheet(current_npc)
            continue

        if command == "/chat":
            npc = registry.resolve(args) if args else current_npc
            if npc is None:
                print_line("Select an NPC first with /list or create one with /create.")
                continue
            current_npc_id = npc.npc_id
            should_quit = chat_with_npc(npc, engine, registry, sampler)
            if should_quit:
                print_line("Bye.")
                break
            continue

        if command == "/edit":
            if current_npc is None:
                print_line("Select an NPC first.")
                continue
            message = edit_current_npc(current_npc, args)
            registry.upsert(current_npc)
            print_line(message)
            continue

        if command == "/update":
            if current_npc is None:
                print_line("Select an NPC first.")
                continue
            changes = apply_update_instruction(current_npc, args, sampler)
            registry.upsert(current_npc)
            if changes:
                print_line(f"Updated: {', '.join(sorted(changes.keys()))}")
            else:
                print_line("Stored the update as a note.")
            continue

        if command == "/hyper":
            run_hyper_wizard(engine)
            continue

        print_line(f"Unknown command: {command}. Use /help.")


if __name__ == "__main__":
    main()
