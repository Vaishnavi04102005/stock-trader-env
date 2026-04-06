"""
client.py — Python client for the Stock Trader OpenEnv environment.
Install directly from the HF Space:
    pip install git+https://huggingface.co/spaces/YOUR_USERNAME/stock-trader-env

Usage (async):
    from client import StockTraderEnv, StockAction
    import asyncio

    async def main():
        async with StockTraderEnv(base_url="https://YOUR_USERNAME-stock-trader-env.hf.space") as env:
            result = await env.reset()
            result = await env.step(StockAction(ticker="AAPL", decision="BUY", quantity=2))
            state  = await env.state()
    asyncio.run(main())

Usage (sync):
    with StockTraderEnv(base_url="...").sync() as env:
        result = env.reset()
"""

from __future__ import annotations
import asyncio
import json
from contextlib import asynccontextmanager, contextmanager
from typing import Optional

import httpx
import websockets

from models import StockAction, StockObservation, StockState, StepResult


class StockTraderEnv:
    """Async HTTP client for the Stock Trader OpenEnv server."""

    def __init__(self, base_url: str = "http://localhost:7860", session_id: str = "default"):
        self.base_url   = base_url.rstrip("/")
        self.session_id = session_id
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=30)
        return self

    async def __aexit__(self, *args):
        if self._client:
            await self._client.aclose()

    async def reset(self) -> StockObservation:
        r = await self._client.post("/reset", params={"session_id": self.session_id})
        r.raise_for_status()
        return StockObservation(**r.json())

    async def step(self, action: StockAction) -> StepResult:
        r = await self._client.post(
            "/step",
            json=action.model_dump(),
            params={"session_id": self.session_id},
        )
        r.raise_for_status()
        return StepResult(**r.json())

    async def state(self) -> StockState:
        r = await self._client.get("/state", params={"session_id": self.session_id})
        r.raise_for_status()
        return StockState(**r.json())

    def sync(self) -> "_SyncWrapper":
        return _SyncWrapper(self)


class _SyncWrapper:
    """Synchronous wrapper around the async client."""

    def __init__(self, env: StockTraderEnv):
        self._env = env
        self._loop = asyncio.new_event_loop()

    def __enter__(self):
        self._loop.run_until_complete(self._env.__aenter__())
        return self

    def __exit__(self, *args):
        self._loop.run_until_complete(self._env.__aexit__(*args))
        self._loop.close()

    def reset(self) -> StockObservation:
        return self._loop.run_until_complete(self._env.reset())

    def step(self, action: StockAction) -> StepResult:
        return self._loop.run_until_complete(self._env.step(action))

    def state(self) -> StockState:
        return self._loop.run_until_complete(self._env.state())
