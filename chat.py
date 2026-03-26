"""
chat.py — D&D NPC Generator & Chat CLI
=======================================
Usage:
    python chat.py          # generate random NPC and start chat
    python chat.py --seed 42
"""

import argparse
import os
import sys

# ── Try rich for pretty output, fall back to plain print ──────────────────────
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    from rich import print as rprint
    RICH = True
    console = Console()
except ImportError:
    RICH = False
    console = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_gpt2 import DEFAULT_HYPERPARAMS, build_training_data, train, CSV_PATH, TRAIN_FILE, MODEL_OUT
from npc_generator.character_sampler import CharacterSampler
from npc_generator.story_generator import StoryGenerator
from npc_generator.quest_generator import generate_quest, quest_rewards
from npc_generator.dialogue_engine import DialogueEngine, build_persona_prompt
from npc_generator.npc import NPC
from npc_generator.player import Player, QuestEntry


def run_hyper_wizard():
    """Interactive /hyper command: edit hyperparams and retrain GPT-2."""
    hp = DEFAULT_HYPERPARAMS.copy()

    if RICH:
        console.print("\n[bold yellow]⚙  Hyperparameter Tuning[/bold yellow]")
        console.print("[dim]Press Enter to keep current value.[/dim]\n")
    else:
        print("\n=== Hyperparameter Tuning (Enter = keep current) ===")

    fields = [
        ("epochs",       "Training epochs",          int),
        ("batch_size",   "Batch size",                int),
        ("lr",           "Learning rate",             float),
        ("block_size",   "Token block size",          int),
        ("warmup_steps", "Warmup steps",              int),
    ]

    for key, label, cast in fields:
        current = hp[key]
        try:
            raw = input(f"  {label} [{current}]: ").strip()
            if raw:
                hp[key] = cast(raw)
        except (ValueError, KeyboardInterrupt):
            pass  # keep default

    if RICH:
        console.print("\n[bold]Settings:[/bold]")
        for k, v in hp.items():
            console.print(f"  [cyan]{k}[/cyan] = [yellow]{v}[/yellow]")
        confirm = input("\nStart training? [y/N]: ").strip().lower()
    else:
        print("\nSettings:", hp)
        confirm = input("Start training? [y/N]: ").strip().lower()

    if confirm != "y":
        if RICH:
            console.print("[dim]Training cancelled.[/dim]")
        else:
            print("Cancelled.")
        return

    if RICH:
        console.print("\n[bold cyan]Building training data...[/bold cyan]")
    build_training_data(CSV_PATH, TRAIN_FILE)

    if RICH:
        console.print("[bold cyan]Training GPT-2 — this runs in the background.[/bold cyan]")
        console.print("[dim]You can keep chatting; training output appears in this terminal.[/dim]\n")
    else:
        print("Training started...")

    import threading
    def _train():
        train(TRAIN_FILE, MODEL_OUT, hp)
        msg = f"\n[bold green]✓ Training complete! Model saved → {MODEL_OUT}[/bold green]"
        if RICH:
            console.print(msg)
        else:
            print(f"\nTraining complete! Model saved → {MODEL_OUT}")

    threading.Thread(target=_train, daemon=True).start()


def print_player_status(player: Player):
    xp_bar_filled = int((player.xp / (100 * player.level)) * 20)
    xp_bar = "█" * xp_bar_filled + "░" * (20 - xp_bar_filled)
    if RICH:
        console.print(Panel(
            f"[bold]{player.name}[/bold]  •  Level [yellow]{player.level}[/yellow]\n"
            f"XP  [{xp_bar}] {player.xp}/{100 * player.level}\n"
            f"Gold [yellow]{player.gold}[/yellow]  •  "
            f"Active quests [cyan]{len(player.active_quests)}[/cyan]  •  "
            f"Completed [green]{len(player.completed_quests)}[/green]",
            title="[bold]Player Status[/bold]", border_style="yellow"
        ))
    else:
        print(f"\n=== {player.name} | Lvl {player.level} ===")
        print(f"XP: {player.xp}/{100 * player.level}  Gold: {player.gold}")
        print(f"Active quests: {len(player.active_quests)}  Completed: {len(player.completed_quests)}\n")


def print_quest_log(player: Player):
    if RICH:
        active = "\n".join(
            f"  [yellow]{i+1}.[/yellow] [bold]{q.title}[/bold] — from {q.giver}\n"
            f"     {q.description[:80]}...\n"
            f"     Reward: [green]{q.reward_xp} XP[/green]  [yellow]{q.reward_gold} gold[/yellow]"
            for i, q in enumerate(player.active_quests)
        ) or "  [dim]No active quests.[/dim]"
        completed = "\n".join(
            f"  [dim]✓ {q.title} — from {q.giver}[/dim]"
            for q in player.completed_quests
        ) or "  [dim]None yet.[/dim]"
        console.print(Panel(
            f"[bold cyan]Active[/bold cyan]\n{active}\n\n"
            f"[bold green]Completed[/bold green]\n{completed}",
            title="[bold]Quest Log[/bold]", border_style="cyan"
        ))
    else:
        print("\n=== Active Quests ===")
        for i, q in enumerate(player.active_quests):
            print(f"  {i+1}. {q.title} (from {q.giver}) — {q.reward_xp}xp / {q.reward_gold}g")
        print("\n=== Completed ===")
        for q in player.completed_quests:
            print(f"  ✓ {q.title}")
        print()


def print_divider():
    if RICH:
        console.rule(style="bold cyan")
    else:
        print("─" * 60)


def print_npc_sheet(npc: NPC):
    if RICH:
        content = (
            f"[bold]{npc.name}[/bold]\n"
            f"[cyan]{npc.race} {npc.primary_class}[/cyan]  •  "
            f"Level [yellow]{npc.level}[/yellow]  •  "
            f"[magenta]{npc.alignment}[/magenta]\n"
            f"[dim]{npc.background}[/dim]\n\n"
            f"HP [red]{npc.HP}[/red]   AC [blue]{npc.AC}[/blue]\n"
            f"{npc.stat_block}\n\n"
            f"[bold green]Story:[/bold green]\n{npc.story}\n\n"
            f"[bold yellow]Quest:[/bold yellow]\n{npc.quest}"
        )
        console.print(Panel(content, title="[bold]NPC Sheet[/bold]", border_style="cyan"))
    else:
        print_divider()
        print(f"  {npc.name}")
        print(f"  {npc.race} {npc.primary_class}  |  Level {npc.level}  |  {npc.alignment}")
        print(f"  Background: {npc.background}")
        print(f"  HP: {npc.HP}   AC: {npc.AC}")
        print(f"  {npc.stat_block}")
        print_divider()
        print(f"  Story:\n  {npc.story}")
        print_divider()
        print(f"  Quest:\n  {npc.quest}")
        print_divider()


def chat_loop(npc: NPC, engine: DialogueEngine, player: Player, char: dict):
    if RICH:
        console.print(
            f"\n[bold cyan]You approach {npc.name}...[/bold cyan]\n"
            "[dim]Type your message. Commands: 'quit' / 'new' / 'quest' / 'hyper'[/dim]\n"
        )
    else:
        print(f"\nYou approach {npc.name}...")
        print("Commands: 'quit' / 'new' / 'quest' / 'hyper'\n")

    history = None

    import random

    ALL_OPTIONS = [
        "Tell me about yourself.",
        "What do you need help with?",
        "What can you tell me about this area?",
        "Do you have a quest for me?",
        "What's your story?",
        "What do you think about magic?",
        "Have you heard any rumors lately?",
        "What do you know about the roads ahead?",
        "Are you a fighter or a talker?",
        "What brings you to this place?",
        "Do you trust strangers?",
        "What do you fear most?",
    ]
    QUEST_OPTIONS = [
        "I'll take the quest.",
        "Tell me more about the reward.",
        "How dangerous is it?",
        "Who else knows about this?",
        "How long will it take?",
        "Can I bring companions?",
    ]
    FAREWELL = "I need to go. Farewell."

    state = {"quest_mode": False}

    def fresh_options(exclude=None):
        if state["quest_mode"]:
            pool = [o for o in QUEST_OPTIONS if o != exclude]
            return random.sample(pool, min(3, len(pool))) + [FAREWELL]
        pool = [o for o in ALL_OPTIONS if o != exclude]
        return random.sample(pool, 3) + [FAREWELL]

    current_options = fresh_options()

    def show_options():
        if RICH:
            console.print("\n[dim]── Quick replies ──[/dim]")
            for i, opt in enumerate(current_options, 1):
                console.print(f"  [yellow]{i}[/yellow]  {opt}")
            console.print("[dim]  or type anything freely[/dim]\n")
        else:
            print("\n── Quick replies ──")
            for i, opt in enumerate(current_options, 1):
                print(f"  {i}  {opt}")
            print("  or type anything freely\n")

    show_options()

    while True:
        try:
            raw = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nFarewell, traveler.")
            break

        if not raw:
            continue

        # number shortcut → expand to full option text
        if raw.isdigit() and 1 <= int(raw) <= len(current_options):
            user_input = current_options[int(raw) - 1]
            current_options[:] = fresh_options(exclude=user_input)
            if RICH:
                console.print(f"[dim]> {user_input}[/dim]")
            else:
                print(f"> {user_input}")
        else:
            user_input = raw

        cmd = user_input.lower().strip()

        if cmd in ("quit", "exit", "q", FAREWELL.lower()):
            print(f"\n{npc.name}: Safe travels, adventurer.")
            break
        if cmd == "new":
            return "new"
        if cmd == "status":
            print_player_status(player)
            continue
        if cmd == "quests":
            print_quest_log(player)
            continue
        if cmd.startswith("complete"):
            # complete [number] or just "complete" → complete first quest
            parts = cmd.split()
            idx = int(parts[1]) - 1 if len(parts) > 1 and parts[1].isdigit() else 0
            result = player.complete_quest(idx)
            if result is None:
                print("No quest at that number.")
            else:
                quest, leveled_up = result
                if RICH:
                    console.print(f"\n[green]✓ Quest completed: {quest.title}[/green]")
                    console.print(f"  +{quest.reward_xp} XP  +{quest.reward_gold} gold")
                    if leveled_up:
                        console.print(f"\n[bold yellow]⬆  LEVEL UP! You are now level {player.level}![/bold yellow]")
                else:
                    print(f"\n✓ Quest completed: {quest.title}")
                    print(f"  +{quest.reward_xp} XP  +{quest.reward_gold} gold")
                    if leveled_up:
                        print(f"\n*** LEVEL UP! You are now level {player.level}! ***")
            continue
        if cmd == "hyper":
            run_hyper_wizard()
            continue

        # quest accept
        if cmd in ("quest", "tell me about your quest.") or "quest" in cmd:
            user_input = "Tell me about your quest." if cmd == "quest" else user_input
            state["quest_mode"] = True
        elif cmd in ("i'll take the quest.", "i'll take the quest"):
            xp, gold = quest_rewards(char)
            entry = QuestEntry(
                title=npc.quest[:50].rstrip(".") + ".",
                description=npc.quest,
                giver=npc.name,
                reward_xp=xp,
                reward_gold=gold,
            )
            player.add_quest(entry)
            state["quest_mode"] = False
            if RICH:
                console.print(f"\n[green]Quest accepted! Reward: {xp} XP + {gold} gold[/green]")
                console.print(f"[dim]Type 'complete' when you're done.[/dim]\n")
            else:
                print(f"\nQuest accepted! Reward: {xp} XP + {gold} gold")

        response, history = engine.chat(
            persona_prompt=npc.persona_prompt,
            user_input=user_input,
            history_ids=history,
            character=npc.__dict__,
        )

        if RICH:
            console.print(f"\n[bold cyan]{npc.name}:[/bold cyan] {response}\n")
        else:
            print(f"\n{npc.name}: {response}\n")

        show_options()

    return "done"


def generate_npc(sampler: CharacterSampler, story_gen: StoryGenerator):
    char = sampler.sample_character()
    story = story_gen.generate_story(char)
    quest = generate_quest(char)
    persona = build_persona_prompt(char, story)

    npc = NPC(
        name=char["name"],
        race=char["race"],
        primary_class=char["primary_class"],
        background=char["background"],
        alignment=char["alignment"],
        level=char["level"],
        HP=char["HP"],
        AC=char["AC"],
        Str=char["Str"],
        Dex=char["Dex"],
        Con=char["Con"],
        Int=char["Int"],
        Wis=char["Wis"],
        Cha=char["Cha"],
        story=story,
        quest=quest,
        persona_prompt=persona,
    )
    return npc, char


def main():
    parser = argparse.ArgumentParser(description="D&D NPC Generator")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--no-chat", action="store_true", help="Only generate NPC, no chat")
    args = parser.parse_args()

    if args.seed is not None:
        import random, numpy as np
        random.seed(args.seed)
        np.random.seed(args.seed)

    if RICH:
        console.print("[bold magenta]\n⚔  D&D NPC Generator  ⚔[/bold magenta]\n")
    else:
        print("\n=== D&D NPC Generator ===\n")

    if RICH:
        with console.status("[cyan]Loading models...[/cyan]"):
            sampler = CharacterSampler()
            story_gen = StoryGenerator()
            engine = DialogueEngine() if not args.no_chat else None
    else:
        print("Loading models...")
        sampler = CharacterSampler()
        story_gen = StoryGenerator()
        engine = DialogueEngine() if not args.no_chat else None

    # ask for player name
    try:
        player_name = input("Enter your name, adventurer: ").strip() or "Adventurer"
    except (KeyboardInterrupt, EOFError):
        player_name = "Adventurer"
    player = Player(name=player_name)

    if RICH:
        console.print(f"\n[dim]Commands: 'status' · 'quests' · 'complete [n]' · 'new' · 'quit' · 'hyper'[/dim]\n")

    while True:
        if RICH:
            with console.status("[cyan]Generating NPC...[/cyan]"):
                npc, char = generate_npc(sampler, story_gen)
        else:
            print("\nGenerating NPC...")
            npc, char = generate_npc(sampler, story_gen)

        print_npc_sheet(npc)

        if args.no_chat:
            break

        result = chat_loop(npc, engine, player, char)
        if result != "new":
            break

        if RICH:
            console.print("\n[dim]Generating a new NPC...[/dim]\n")
        else:
            print("\nGenerating a new NPC...\n")


if __name__ == "__main__":
    main()
