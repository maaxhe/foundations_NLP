"""
schemas.py — Pydantic models defining the canonical Dialog State schema.

This module is the single source of truth for the JSON structure that the
fine-tuned model must produce.  Every downstream component (validator,
trainer prompt-builder, FastAPI response model) imports from here so that
a schema change propagates automatically.

Dependencies:
    pip install pydantic>=2.0
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class AggressionLevel(str, Enum):
    """
    Discrete aggression states for the NPC.

    TODO: Expand or collapse levels once you have labelled data and can
          measure inter-annotator agreement on boundary cases.
    """
    FRIENDLY   = "friendly"
    NEUTRAL    = "neutral"
    SUSPICIOUS = "suspicious"
    HOSTILE    = "hostile"


class QuestStatus(str, Enum):
    """
    Lifecycle of a quest tracked by the NPC.

    TODO: Add FAILED / EXPIRED if game design requires it.
    """
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED   = "completed"


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class InventoryItem(BaseModel):
    """
    A single item in the NPC's inventory.

    Attributes:
        name:     Human-readable item name, e.g. "Iron Sword".
        quantity: How many of this item the NPC currently holds.
        value_gp: Gold-piece value per unit; None if unknown / priceless.

    TODO: Add an `item_type` enum (WEAPON, ARMOR, CONSUMABLE, QUEST_ITEM)
          once the ontology is agreed upon with the game-design spec.
    """
    name:     str           = Field(..., description="Item name as it appears in lore.")
    quantity: int           = Field(default=1, ge=0)
    value_gp: Optional[int] = Field(default=None, ge=0)


class KnownFact(BaseModel):
    """
    A (subject, predicate, object) triple extracted from dialog.

    Attributes:
        subject:   The entity the fact is about (person, place, item).
        predicate: The relationship or property, e.g. "location", "ally_of".
        object_:   The value of the predicate.
        turn_idx:  Dialog turn index at which this fact was revealed.

    TODO: Consider replacing the flat triple with a networkx knowledge-graph
          node for multi-hop reasoning across turns.
    """
    subject:   str = Field(...)
    predicate: str = Field(...)
    object_:   str = Field(..., alias="object")
    turn_idx:  int = Field(default=0, ge=0)

    model_config = {"populate_by_name": True}


# ---------------------------------------------------------------------------
# Root Dialog State
# ---------------------------------------------------------------------------

class DialogState(BaseModel):
    """
    The complete internal state of the NPC after processing one dialog turn.

    This is the **target JSON schema** the fine-tuned Llama-3B model must
    produce.  The FastAPI `state_extractor` calls the model and validates
    the raw string output against this model before updating the NPC engine.

    Attributes:
        npc_name:      Identifier for the NPC (stable across turns).
        turn_index:    Zero-based index of the current dialog turn.
        aggression:    Current aggression level (see AggressionLevel).
        trust_score:   Float in [0, 1]; 1.0 = fully trusts the player.
        inventory:     List of items the NPC currently possesses.
        known_facts:   Structured facts extracted from the conversation.
        active_quest:  Name of the quest currently discussed, or None.
        quest_status:  Current status of active_quest, or None.
        player_intent: Short free-text summary of the player's last goal.
        npc_mood_note: Optional free-text note for the dialog generator
                       describing subtle mood nuances not captured by
                       the aggression enum.

    Example output the model must produce:
    ```json
    {
        "npc_name":      "Grimtooth",
        "turn_index":    3,
        "aggression":    "suspicious",
        "trust_score":   0.4,
        "inventory":     [{"name": "Health Potion", "quantity": 2, "value_gp": 50}],
        "known_facts":   [{"subject": "player", "predicate": "carries",
                           "object": "stolen map", "turn_idx": 2}],
        "active_quest":  "Retrieve the Crystal",
        "quest_status":  "in_progress",
        "player_intent": "Player is asking about the eastern pass route.",
        "npc_mood_note": "Nervous because the player mentioned the guild."
    }
    ```

    TODO: Add a `schema_version` field once the schema stabilises so that
          stored datasets can be migrated when the schema changes.
    """
    npc_name:      str                   = Field(...)
    turn_index:    int                   = Field(default=0, ge=0)
    aggression:    AggressionLevel       = AggressionLevel.NEUTRAL
    trust_score:   float                 = Field(default=0.5, ge=0.0, le=1.0)
    inventory:     list[InventoryItem]   = Field(default_factory=list)
    known_facts:   list[KnownFact]       = Field(default_factory=list)
    active_quest:  Optional[str]         = None
    quest_status:  Optional[QuestStatus] = None
    player_intent: str                   = Field(default="")
    npc_mood_note: Optional[str]         = None

    @field_validator("trust_score")
    @classmethod
    def clamp_trust(cls, v: float) -> float:
        """
        Clamp trust_score to [0, 1] to absorb minor generation noise.

        TODO: Emit a `logging.warning` when clamping fires so we can track
              how often the model produces out-of-range values at eval time.
        """
        # TODO: import logging; logging.warning("trust_score clamped: %s", v)
        return max(0.0, min(1.0, v))
