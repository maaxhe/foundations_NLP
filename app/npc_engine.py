"""
npc_engine.py -- NPC state manager and dialog generator (Phase 3).

Responsibility:
    1. Maintain per-session NPC state (a DialogState object) in memory.
    2. Generate the NPC's dialog response given the current state and the
       player's message, using the chat/dialog Llama-3B model via Ollama.
    3. After generating the response, call StateExtractor to update the state.
    4. Expose a clean interface to the FastAPI routes in main.py.

Design note on model usage:
    - Dialog model (Ollama, base Llama-3B-Instruct):
        Used for generating the NPC's in-character speech.
        This model is NOT fine-tuned; its creativity and fluency are used as-is.
        The DialogState is injected into its system prompt to guide tone and content.
    - Extraction model (fine-tuned via LoRA):
        Used silently in the background by StateExtractor to parse the dialog
        into the structured DialogState.

Dependencies:
    pip install ollama fastapi pydantic
"""

from __future__ import annotations

import logging
import uuid
from typing import Optional

# TODO: import ollama
# TODO: from data_pipeline.schemas import DialogState, AggressionLevel
# TODO: from app.state_extractor import StateExtractor
# TODO: from app.models import NPCResponse, StateSnapshot

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# NPC system prompt templates
# ---------------------------------------------------------------------------

NPC_SYSTEM_PROMPT = """
You are {npc_name}, a D&D NPC.  Stay in character at all times.

Your current state:
- Aggression: {aggression}
- Trust in player: {trust_score:.0%}
- Active quest: {active_quest}
- Inventory: {inventory_summary}
- What you know about the player: {player_intent}
{mood_note}

Guidelines:
- Respond in 1-3 sentences maximum.
- Mirror your aggression level in your tone.
- If aggression is hostile, refuse to help and threaten the player.
- If aggression is friendly, be warm and offer assistance.
- Do NOT break character or mention JSON, AI, or game mechanics.
"""


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

class SessionState:
    """
    Holds the mutable state for a single active game session.

    Attributes:
        session_id:     UUID string for this session.
        npc_name:       NPC character name.
        dialog_history: Ordered list of {"role": ..., "content": ...} dicts.
        current_state:  Most recent DialogState extracted from the dialog.
        turn_index:     Number of completed player turns.

    TODO: Serialise sessions to Redis or a SQLite database so the app can
          restart without losing active sessions.
          Use `pydantic.BaseModel` for the serialisation layer and
          `aiosqlite` for async SQLite writes.
    """

    def __init__(self, session_id: str, npc_name: str) -> None:
        self.session_id:     str             = session_id
        self.npc_name:       str             = npc_name
        self.dialog_history: list[dict]      = []
        self.current_state:  Optional[object] = None   # TODO: type as Optional[DialogState]
        self.turn_index:     int             = 0

    def add_turn(self, role: str, content: str) -> None:
        """
        Append a single dialog turn to the history.

        Args:
            role:    "player" or "npc".
            content: The utterance text.

        TODO: Enforce that turns alternate player/npc to catch logic errors.
        TODO: Trim history to the last 20 turns to bound memory usage.
        """
        # TODO: self.dialog_history.append({"role": role, "content": content})
        raise NotImplementedError


# ---------------------------------------------------------------------------
# NPC Engine
# ---------------------------------------------------------------------------

class NpcEngine:
    """
    Stateful engine managing all active game sessions and NPC interactions.

    This is a singleton instantiated once at FastAPI startup (via the
    lifespan context manager in main.py) and shared across all requests.

    Args:
        dialog_model:      Ollama model tag for the dialog (chat) model.
                           Example: "llama3:3b-instruct"
        extractor:         A configured StateExtractor instance.
        max_dialog_turns:  Maximum turns before a session is closed.

    TODO: Add an LRU cache with a max size to limit memory usage when many
          sessions are open simultaneously.  Use `cachetools.LRUCache`.
    """

    def __init__(
        self,
        dialog_model:     str = "llama3:3b-instruct",
        extractor:        Optional[object] = None,   # TODO: type as StateExtractor
        max_dialog_turns: int = 50,
    ) -> None:
        self.dialog_model     = dialog_model
        self.extractor        = extractor
        self.max_dialog_turns = max_dialog_turns
        self._sessions: dict[str, SessionState] = {}

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def create_session(self, npc_name: str) -> SessionState:
        """
        Initialise a new game session and return its SessionState.

        Args:
            npc_name: The NPC character for this session.

        Returns:
            A fresh SessionState with a new UUID session_id.

        TODO: Implement:
            session_id = str(uuid.uuid4())
            session = SessionState(session_id=session_id, npc_name=npc_name)
            # Set initial DialogState with neutral defaults
            session.current_state = DialogState(npc_name=npc_name)
            self._sessions[session_id] = session
            return session
        """
        raise NotImplementedError

    def get_session(self, session_id: str) -> SessionState:
        """
        Retrieve an existing session by ID.

        Args:
            session_id: UUID string.

        Returns:
            The SessionState for this session.

        Raises:
            KeyError: If the session_id is not found.

        TODO: Implement:
            if session_id not in self._sessions:
                raise KeyError(f"Session '{session_id}' not found.")
            return self._sessions[session_id]
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Dialog generation
    # ------------------------------------------------------------------

    def _build_npc_system_prompt(self, state: object) -> str:
        """
        Render the NPC system prompt with the current DialogState values.

        Args:
            state: The current DialogState for this session.

        Returns:
            Rendered system prompt string for the Ollama chat call.

        TODO: Implement by formatting NPC_SYSTEM_PROMPT with:
            - npc_name from state.npc_name
            - aggression from state.aggression.value
            - trust_score from state.trust_score
            - active_quest from state.active_quest or "None"
            - inventory_summary: ", ".join(f"{i.name} x{i.quantity}"
                                           for i in state.inventory) or "Empty"
            - player_intent from state.player_intent or "Unknown"
            - mood_note: f"Mood note: {state.npc_mood_note}" if state.npc_mood_note else ""
        """
        raise NotImplementedError

    def _call_dialog_model(
        self,
        system_prompt: str,
        dialog_history: list[dict],
        player_message: str,
    ) -> str:
        """
        Call the Ollama dialog model to generate the NPC's response.

        Args:
            system_prompt:  Rendered NPC system prompt.
            dialog_history: Previous turns for multi-turn context.
            player_message: The player's current message.

        Returns:
            NPC response string.

        TODO: Implement using ollama.chat:
            messages = [{"role": "system", "content": system_prompt}]
            # Convert dialog_history to OpenAI-style messages
            for turn in dialog_history[-10:]:   # last 10 turns for context
                role = "user" if turn["role"] == "player" else "assistant"
                messages.append({"role": role, "content": turn["content"]})
            messages.append({"role": "user", "content": player_message})

            response = ollama.chat(
                model=self.dialog_model,
                messages=messages,
                options={"temperature": 0.8, "num_predict": 200},
            )
            return response["message"]["content"].strip()

        TODO: Add streaming support using ollama.chat(stream=True) and yield
              tokens via a FastAPI StreamingResponse for better UX.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # State-to-snapshot conversion
    # ------------------------------------------------------------------

    def _state_to_snapshot(self, state: object) -> object:
        """
        Convert a DialogState to a UI-friendly StateSnapshot.

        Args:
            state: Current DialogState.

        Returns:
            A StateSnapshot Pydantic model for inclusion in NPCResponse.

        TODO: Implement:
            from app.models import StateSnapshot
            return StateSnapshot(
                aggression=state.aggression.value,
                trust_score=state.trust_score,
                inventory_summary=[
                    f"{i.name} x{i.quantity}" for i in state.inventory
                ],
                active_quest=state.active_quest,
                last_player_intent=state.player_intent,
            )
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Main public method
    # ------------------------------------------------------------------

    def process_turn(self, session_id: str, player_message: str) -> object:
        """
        Process one player turn end-to-end and return the NPC response.

        This is the central method called by the FastAPI POST /chat route.

        Args:
            session_id:     Active session UUID.
            player_message: The player's text input.

        Returns:
            An NPCResponse Pydantic model ready for JSON serialisation.

        Steps to implement:
            1. session = self.get_session(session_id)
            2. system_prompt = self._build_npc_system_prompt(session.current_state)
            3. npc_text = self._call_dialog_model(
                   system_prompt, session.dialog_history, player_message
               )
            4. session.add_turn("player", player_message)
            5. session.add_turn("npc", npc_text)
            6. session.turn_index += 1
            7. # Run extraction AFTER generating the response (non-blocking path)
               new_state = self.extractor.extract(
                   dialog_history=session.dialog_history,
                   npc_name=session.npc_name,
                   turn_index=session.turn_index,
               )
               session.current_state = new_state
            8. snapshot = self._state_to_snapshot(new_state)
            9. from app.models import NPCResponse
               return NPCResponse(
                   npc_message=npc_text,
                   session_id=session_id,
                   current_state=snapshot,
                   turn_index=session.turn_index,
               )

        TODO: Move step 7 (extraction) to an asyncio background task so that
              the HTTP response (steps 8-9) is returned to the client without
              waiting for extraction to finish.  Extraction latency on a 3B
              model is ~1-2 s; dialog generation is ~0.5 s.
        TODO: Enforce self.max_dialog_turns and return a graceful "session ended"
              message when the limit is reached.
        """
        raise NotImplementedError

    def get_opening_line(self, session_id: str) -> str:
        """
        Generate the NPC's opening greeting for a new session.

        Args:
            session_id: The session ID created by create_session.

        Returns:
            A short in-character greeting string.

        TODO: Implement a short Ollama call with a prompt like:
              "Introduce yourself as {npc_name} in one sentence."
              Use low temperature (0.5) for a stable greeting.
        """
        raise NotImplementedError
