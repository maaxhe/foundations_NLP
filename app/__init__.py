"""
app -- FastAPI backend and browser UI for the interactive D&D NPC (Phase 3).

Modules:
    models          : Pydantic request/response models for the HTTP API.
    state_extractor : Calls the fine-tuned local model to extract DialogState JSON.
    npc_engine      : Manages NPC state across turns and generates NPC dialog.
    main            : FastAPI application, route definitions, and lifespan handler.
"""
