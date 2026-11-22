"""Gift Concierge HTTP backend.

    POST /recommend   {"session_id": str, "message": str}  -> question or recommendations
    POST /reset       {"session_id": str}                   -> {"ok": true}
    GET  /health      component status (which LLM, retriever, ranker, KG, images)
    GET  /            the frontend (CODE/frontend), which then talks to this server

Run from the repo root:

    uvicorn CODE.backend.app:app --port 8000
    # or: python -m CODE.backend.app

then open http://localhost:8000. Settings: CODE/backend/config.py and README.md.
"""

import contextlib
import logging

from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from .assemble import to_ui_response
from .config import FRONTEND_DIR, Settings
from .engine import Engine, EngineUnavailable
from .sessions import SessionStore

log = logging.getLogger("gift_concierge.app")

MAX_MESSAGE_CHARS = 2000
MAX_SESSION_ID_CHARS = 100

settings = Settings.from_env()
state = {"engine": None, "error": None, "sessions": None}


def start_engine():
    try:
        engine = Engine(settings)
    except EngineUnavailable as exc:
        state["error"] = str(exc)
        log.error("Recommendation engine unavailable: %s", exc)
        return
    state["engine"] = engine
    state["sessions"] = SessionStore(engine.new_conversation,
                                     ttl_seconds=settings.session_ttl_minutes * 60,
                                     max_sessions=settings.max_sessions)


@contextlib.asynccontextmanager
async def lifespan(app):
    await run_in_threadpool(start_engine)   # loads models / indexes once
    yield


def error(status, message):
    return JSONResponse({"error": message}, status_code=status)


async def read_json(request):
    try:
        body = await request.json()
    except ValueError:
        return None
    return body if isinstance(body, dict) else None


def valid_session_id(value):
    return isinstance(value, str) and 0 < len(value) <= MAX_SESSION_ID_CHARS


def run_turn(session, message):
    with session.lock:   # one turn at a time per conversation
        raw = session.concierge.handle(message)
        response = to_ui_response(raw, state["engine"], debug=settings.debug,
                                  previous_results=session.has_results)
        if raw["type"] == "recommendations":
            session.has_results = True
        return response


async def recommend(request):
    if state["engine"] is None:
        return error(503, state["error"] or "The recommendation engine is starting.")

    body = await read_json(request)
    if body is None:
        return error(400, "Expected a JSON object.")

    session_id = body.get("session_id")
    message = body.get("message")
    if not valid_session_id(session_id):
        return error(400, "session_id must be a non-empty string.")
    if not isinstance(message, str) or not message.strip():
        return error(400, "message must be a non-empty string.")
    if len(message) > MAX_MESSAGE_CHARS:
        return error(400, f"message is longer than {MAX_MESSAGE_CHARS} characters.")

    session = state["sessions"].get(session_id)
    try:
        response = await run_in_threadpool(run_turn, session, message.strip())
    except Exception:   # provider outage, index error, ...: keep the conversation alive
        log.exception("Turn failed for session %s", session_id)
        return error(500, "The recommendation pipeline failed on this message.")
    return JSONResponse(response)


async def reset(request):
    body = await read_json(request) or {}
    session_id = body.get("session_id")
    if state["sessions"] is not None and valid_session_id(session_id):
        state["sessions"].drop(session_id)
    return JSONResponse({"ok": True})


async def health(request):
    engine = state["engine"]
    return JSONResponse({
        "service": "gift-concierge",
        "ready": engine is not None,
        "error": state["error"],
        "components": engine.status if engine else {},
        "active_sessions": len(state["sessions"]) if state["sessions"] else 0,
    })


app = Starlette(
    routes=[
        Route("/recommend", recommend, methods=["POST"]),
        Route("/reset", reset, methods=["POST"]),
        Route("/health", health, methods=["GET"]),
        Mount("/", app=StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend"),
    ],
    middleware=[
        Middleware(CORSMiddleware, allow_origins=settings.cors_origins,
                   allow_methods=["GET", "POST"], allow_headers=["Content-Type"]),
    ],
    lifespan=lifespan,
)


if __name__ == "__main__":
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Gift Concierge backend")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    uvicorn.run("CODE.backend.app:app", host=args.host, port=args.port)
