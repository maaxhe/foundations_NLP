"""
models.py -- Pydantic request/response models for the FastAPI HTTP layer.

These models are SEPARATE from data_pipeline.schemas.DialogState intentionally:
    - API models define the contract with the frontend (stable, versioned).
    - DialogState defines the internal NLP schema (may evolve with the model).
A thin mapping layer in npc_engine.py converts between the two.

Dependencies:
    pip install pydantic>=2.0 fastapi
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

# TODO: from data_pipeline.schemas import AggressionLevel


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    """
    Payload sent by the frontend when the player submits a message.

    Attributes:
        player_message: The raw text typed by the player.
        session_id:     UUID identifying the current game session.
                        The server uses this to look up the current NPC state.
                        Generate on the client with `crypto.randomUUID()`.
        npc_name:       Which NPC the player is talking to.  Must match a name
                        known to the NpcEngine.

    TODO: Add a `timestamp: datetime` field and log it for latency analysis.
    TODO: Validate that session_id is a valid UUID4 with a Pydantic validator.
    """
    player_message: str = Field(..., min_length=1, max_length=2000)
    session_id:     str = Field(...)
    npc_name:       str = Field(default="Grimtooth") # Maybe change the default name to something funny, like... Mr. Meeseeks


class NewSessionRequest(BaseModel):
    """
    Payload to initialise a new game session with a specific NPC.

    Attributes:
        npc_name: The NPC character name.
        player_name: Optional player character name for personalised dialog.

    TODO: Add `scenario_id` to load a pre-defined quest context from a
          scenarios/ directory, allowing richer initial NPC state.
    """
    npc_name:    str            = Field(default="Grimtooth")
    player_name: Optional[str] = Field(default=None)


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class NPCResponse(BaseModel):
    """
    The server's response to a ChatRequest.

    Attributes:
        npc_message:   The NPC's dialog text shown to the player.
        session_id:    Echo of the request session_id.
        current_state: A simplified view of the NPC's internal state
                       for the UI state-panel widget.
        turn_index:    The turn number after processing the player's message.

    TODO: Add a `latency_ms: int` field populated server-side for
          performance monitoring in the frontend.
    """
    npc_message:   str        = Field(...)
    session_id:    str        = Field(...)
    current_state: "StateSnapshot" = Field(...)
    turn_index:    int        = Field(...)


class StateSnapshot(BaseModel):
    """
    A UI-friendly snapshot of the NPC's internal state.

    Deliberately flattened and simplified compared to DialogState so the
    frontend does not need to understand the full schema.

    Attributes:
        aggression:   String label for the aggression level.
        trust_score:  Float 0-1.
        inventory_summary: List of "Name x Qty" strings.
        active_quest: Name of the active quest or None.
        last_player_intent: Summary of what the player last wanted.

    TODO: Add a `mood_indicator: str` field derived from npc_mood_note
          for a visual emoji/icon in the UI state panel.
    """
    aggression:          str        = Field(...)
    trust_score:         float      = Field(...)
    inventory_summary:   list[str]  = Field(default_factory=list)
    active_quest:        Optional[str] = None
    last_player_intent:  str        = Field(default="")


class SessionInfo(BaseModel):
    """
    Returned by POST /session/new to confirm session creation.

    Attributes:
        session_id:   Server-assigned or client-provided session UUID.
        npc_name:     Confirmed NPC name for this session.
        opening_line: The NPC's first dialog line to display immediately.
    """
    session_id:   str = Field(...)
    npc_name:     str = Field(...)
    opening_line: str = Field(...)


# Resolve forward reference
NPCResponse.model_rebuild()
