"""
main.py -- FastAPI application entry point (Phase 3).

Routes:
    POST /session/new       -- Create a new game session, get opening NPC line.
    POST /chat              -- Send a player message, get NPC response + state.
    GET  /session/{id}/state -- Fetch the current DialogState for a session.
    GET  /                  -- Serve the HTML chat UI.

Run with:
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

Dependencies:
    pip install fastapi uvicorn[standard] jinja2
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# TODO: from app.models import (
#           ChatRequest,
#           NewSessionRequest,
#           NPCResponse,
#           SessionInfo,
#       )
# TODO: from app.npc_engine import NpcEngine
# TODO: from app.state_extractor import StateExtractor


# ---------------------------------------------------------------------------
# Application lifespan
# ---------------------------------------------------------------------------

# TODO: Replace this placeholder with a real NpcEngine instance.
_engine: object = None   # TODO: type as NpcEngine


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    FastAPI lifespan context manager: runs startup and shutdown logic.

    Startup tasks:
        1. Load the StateExtractor (and optionally the HuggingFace model).
        2. Instantiate the NpcEngine with the extractor.
        3. Verify that Ollama is reachable by sending a test ping.

    Shutdown tasks:
        1. Flush any pending background extraction tasks.
        2. Close database connections if persistence is implemented.

    TODO: Implement startup:
        global _engine
        import yaml
        config = yaml.safe_load(open("configs/inference_config.yaml"))
        extractor = StateExtractor(
            backend=config["decoder_strategy"],
            ollama_model=config["extraction_model"],
        )
        _engine = NpcEngine(
            dialog_model=config["dialog_model"],
            extractor=extractor,
        )
        # Ping Ollama to fail fast if it is not running
        import ollama
        try:
            ollama.list()
        except Exception as e:
            raise RuntimeError(f"Ollama is not reachable: {e}")

    TODO: Add a startup banner logging the model names and config path.
    """
    # --- startup ---
    yield
    # --- shutdown ---
    # TODO: await pending_tasks (if background extraction is async)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="D&D NPC Dialog API",
    description=(
        "Interactive NPC powered by a local Llama-3B model via Ollama, "
        "with constrained dialog-state extraction."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# Static files and templates
# TODO: Uncomment once the static/ and templates/ directories have content.
# app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def serve_ui(request: Request) -> HTMLResponse:
    """
    Serve the main chat UI HTML page.

    Returns:
        Rendered index.html template.

    TODO: Pass any initial context (e.g. available NPC names) from
          configs/inference_config.yaml to the template via the context dict.
    """
    # TODO: return templates.TemplateResponse("index.html", {"request": request})
    raise HTTPException(status_code=501, detail="UI not yet implemented.")


@app.post("/session/new")  # TODO: add response_model=SessionInfo
async def new_session(body: object) -> dict:  # TODO: body: NewSessionRequest
    """
    Create a new game session and return the NPC's opening line.

    Args:
        body: NewSessionRequest with npc_name and optional player_name.

    Returns:
        SessionInfo with session_id, npc_name, and opening_line.

    TODO: Implement:
        session  = _engine.create_session(npc_name=body.npc_name)
        opening  = _engine.get_opening_line(session.session_id)
        return SessionInfo(
            session_id=session.session_id,
            npc_name=body.npc_name,
            opening_line=opening,
        )

    TODO: Add input validation: check that body.npc_name is in a whitelist
          of valid NPC names from the config to prevent prompt injection.
    """
    raise HTTPException(status_code=501, detail="Not yet implemented.")


@app.post("/chat")  # TODO: add response_model=NPCResponse
async def chat(body: object) -> dict:  # TODO: body: ChatRequest
    """
    Send a player message and receive the NPC's response with updated state.

    Args:
        body: ChatRequest with player_message and session_id.

    Returns:
        NPCResponse with npc_message, current_state snapshot, and turn_index.

    TODO: Implement:
        try:
            response = _engine.process_turn(
                session_id=body.session_id,
                player_message=body.player_message,
            )
            return response
        except KeyError:
            raise HTTPException(status_code=404,
                                detail=f"Session '{body.session_id}' not found.")

    TODO: Add rate-limiting per session_id (e.g. max 1 request/sec) to prevent
          spamming the local Ollama server.  Use `slowapi` (pip install slowapi).
    TODO: Sanitise player_message: strip control characters and truncate to
          max_length before passing to the NPC engine to prevent prompt injection.
    """
    raise HTTPException(status_code=501, detail="Not yet implemented.")


@app.get("/session/{session_id}/state")
async def get_state(session_id: str) -> dict:
    """
    Return the current raw DialogState for a session (for debugging / dev UI).

    Args:
        session_id: Active session UUID.

    Returns:
        The DialogState serialised as a JSON dict.

    TODO: Implement:
        try:
            session = _engine.get_session(session_id)
            return session.current_state.model_dump()
        except KeyError:
            raise HTTPException(status_code=404, detail="Session not found.")

    TODO: Guard this endpoint with an API key or disable it in production
          (it exposes internal model state that players should not see).
    """
    raise HTTPException(status_code=501, detail="Not yet implemented.")


@app.get("/health")
async def health_check() -> dict:
    """
    Liveness probe for container orchestration (Docker / Kubernetes).

    Returns:
        {"status": "ok"} always (even if the model is not loaded yet).

    TODO: Add a "ready" check that returns 503 until the NpcEngine is fully
          initialised, to prevent traffic before the model is warmed up.
    """
    return {"status": "ok"}
