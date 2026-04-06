"""
models.py — Type-safe contracts for the Stock Trader OpenEnv environment.
Defines Action, Observation, and State Pydantic models.
"""

from __future__ import annotations
from typing import Literal, Dict, List, Optional
from pydantic import BaseModel, Field


TICKERS = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
Decision = Literal["BUY", "SELL", "HOLD"]


# ── Action ────────────────────────────────────────────────────────────────────

class StockAction(BaseModel):
    ticker:   str      = Field("AAPL", description="Which ticker to trade")
    decision: Decision = Field("HOLD", description="BUY, SELL, or HOLD")
    quantity: int      = Field(1, ge=1, le=10, description="Number of shares")


# ── Candle (OHLCV bar) ────────────────────────────────────────────────────────

class Candle(BaseModel):
    date:   str
    open:   float
    high:   float
    low:    float
    close:  float
    volume: int


# ── Observation ───────────────────────────────────────────────────────────────

class StockObservation(BaseModel):
    step:            int
    max_steps:       int
    done:            bool
    cash:            float
    holdings:        Dict[str, int]
    prices:          Dict[str, float]
    candles:         Dict[str, List[Candle]]
    sentiment:       Dict[str, float]
    portfolio_value: float
    pnl_pct:         float


# ── State ─────────────────────────────────────────────────────────────────────

class StockState(BaseModel):
    step:          int
    done:          bool
    cash:          float
    holdings:      Dict[str, int]
    portfolio_value: float
    pnl_pct:       float
    total_trades:  int


# ── Step Result ───────────────────────────────────────────────────────────────

class StepResult(BaseModel):
    observation: StockObservation
    reward:      float
    done:        bool
    info:        dict
