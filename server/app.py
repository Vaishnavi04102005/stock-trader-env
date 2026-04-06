"""
server/app.py — FastAPI server for the Stock Trader OpenEnv environment.
Exposes /reset, /step, /state HTTP endpoints + /ws WebSocket endpoint.
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from typing import Dict

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import StockAction, StockObservation, StockState, StepResult
from server.environment import StockTradingEnvironment

# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "Stock Trader OpenEnv",
    description = "A real-world RL environment for LLM-based stock trading agents.",
    version     = "1.0.0",
)

# Session manager — each WebSocket/client gets its own environment instance
_sessions: Dict[str, StockTradingEnvironment] = {}


def _get_or_create(session_id: str) -> StockTradingEnvironment:
    if session_id not in _sessions:
        _sessions[session_id] = StockTradingEnvironment()
    return _sessions[session_id]


# ── HTTP Endpoints ────────────────────────────────────────────────────────────

@app.get("/")
async def index():
    return HTMLResponse("""
    <html><body style="font-family:monospace;padding:2em;background:#111;color:#0f0">
    <h2>📉 Stock Trader OpenEnv</h2>
    <p>Environment: <b>stock-trader-v1</b></p>
    <p>Endpoints: POST /reset &nbsp;|&nbsp; POST /step &nbsp;|&nbsp; GET /state &nbsp;|&nbsp; WS /ws</p>
    <p><a href="/docs" style="color:#0af">API Docs →</a></p>
    </body></html>
    """)


@app.post("/reset", response_model=StockObservation)
async def reset(session_id: str = "default"):
    env = _get_or_create(session_id)
    obs = env.reset()
    return obs


@app.post("/step", response_model=StepResult)
async def step(action: StockAction, session_id: str = "default"):
    env = _get_or_create(session_id)
    try:
        result = env.step(action)
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state", response_model=StockState)
async def get_state(session_id: str = "default"):
    env = _get_or_create(session_id)
    return env.state


# ── WebSocket Endpoint ────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = str(uuid.uuid4())
    env = StockTradingEnvironment()
    _sessions[session_id] = env

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            cmd = msg.get("command")

            if cmd == "reset":
                obs = env.reset()
                await websocket.send_text(obs.model_dump_json())

            elif cmd == "step":
                action = StockAction(**msg.get("action", {}))
                result = env.step(action)
                await websocket.send_text(result.model_dump_json())

            elif cmd == "state":
                await websocket.send_text(env.state.model_dump_json())

            else:
                await websocket.send_text(
                    json.dumps({"error": f"Unknown command: {cmd}"})
                )

    except WebSocketDisconnect:
        _sessions.pop(session_id, None)


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    workers = int(os.getenv("WORKERS", "1"))
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, workers=workers)
