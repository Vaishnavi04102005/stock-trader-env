"""
server/environment.py — Core Stock Trader RL environment logic.
Implements reset(), step(), state() as required by OpenEnv spec.
"""

from __future__ import annotations

import random
import math
from datetime import datetime, timedelta
from typing import Dict, List

import sys
import os

# Add parent to path so we can import models
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import (
    StockAction, StockObservation, StockState, StepResult, Candle, TICKERS
)

STARTING_CASH = 10_000.0
MAX_STEPS     = 30
LOOKBACK      = 30   # price history days before episode start


def _generate_price_series(ticker: str, days: int) -> List[Dict]:
    """Simulate realistic OHLCV data using seeded random walk."""
    random.seed(hash(ticker) % 9999)
    base = {"AAPL": 180, "GOOGL": 175, "MSFT": 415, "TSLA": 210, "AMZN": 190}[ticker]
    price = float(base)
    series = []
    start = datetime(2024, 1, 1)

    for i in range(days):
        change  = random.gauss(0.001, 0.018)
        price  *= (1 + change)
        high    = price * (1 + abs(random.gauss(0, 0.007)))
        low     = price * (1 - abs(random.gauss(0, 0.007)))
        volume  = int(random.uniform(5e6, 50e6))
        series.append({
            "date":   (start + timedelta(days=i)).strftime("%Y-%m-%d"),
            "open":   round(price * random.uniform(0.998, 1.002), 2),
            "high":   round(high, 2),
            "low":    round(low, 2),
            "close":  round(price, 2),
            "volume": volume,
        })
    return series


def _sentiment(ticker: str, day: int) -> float:
    random.seed(hash(ticker + str(day)) % 99999)
    return round(random.uniform(-1, 1), 3)


class StockTradingEnvironment:
    """
    Stock Trading RL environment following the OpenEnv spec.
    All state is instance-level — safe for concurrent sessions.
    """

    def __init__(self):
        self._price_history: Dict[str, List[Dict]] = {}
        self._step     = 0
        self._done     = False
        self._cash     = STARTING_CASH
        self._holdings : Dict[str, int]  = {t: 0 for t in TICKERS}
        self._trades   : List[dict]      = []

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self) -> StockObservation:
        self._price_history = {
            t: _generate_price_series(t, MAX_STEPS + LOOKBACK)
            for t in TICKERS
        }
        self._step     = 0
        self._done     = False
        self._cash     = STARTING_CASH
        self._holdings = {t: 0 for t in TICKERS}
        self._trades   = []
        return self._build_obs()

    def step(self, action: StockAction) -> StepResult:
        if self._done:
            raise RuntimeError("Episode finished — call reset() first.")

        current_step = self._step
        price        = self._price_at(action.ticker, current_step)
        prev_value   = self._portfolio_value(current_step)

        info: dict = {
            "ticker":   action.ticker,
            "decision": action.decision,
            "quantity": action.quantity,
            "price":    price,
            "step":     current_step,
        }

        if action.decision == "BUY":
            cost = price * action.quantity
            if cost <= self._cash:
                self._cash -= cost
                self._holdings[action.ticker] += action.quantity
                info["executed"] = True
            else:
                info["executed"] = False
                info["reason"]   = "insufficient_cash"

        elif action.decision == "SELL":
            if self._holdings[action.ticker] >= action.quantity:
                self._cash += price * action.quantity
                self._holdings[action.ticker] -= action.quantity
                info["executed"] = True
            else:
                info["executed"] = False
                info["reason"]   = "insufficient_holdings"
        else:
            info["executed"] = True   # HOLD

        self._trades.append(info)
        self._step += 1
        if self._step >= MAX_STEPS:
            self._done = True

        obs    = self._build_obs()
        reward = round(self._portfolio_value(self._step - 1) - prev_value, 4)

        return StepResult(observation=obs, reward=reward, done=self._done, info=info)

    @property
    def state(self) -> StockState:
        return StockState(
            step           = self._step,
            done           = self._done,
            cash           = round(self._cash, 2),
            holdings       = self._holdings,
            portfolio_value= round(self._portfolio_value(self._step), 2),
            pnl_pct        = round(
                (self._portfolio_value(self._step) / STARTING_CASH - 1) * 100, 3
            ),
            total_trades   = len(self._trades),
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _price_at(self, ticker: str, step: int) -> float:
        return self._price_history[ticker][step + LOOKBACK - 1]["close"]

    def _portfolio_value(self, step: int) -> float:
        cash = self._cash
        for t in TICKERS:
            cash += self._price_at(t, step) * self._holdings[t]
        return cash

    def _build_obs(self) -> StockObservation:
        s = self._step
        prices_today = {t: self._price_at(t, s) for t in TICKERS}
        portfolio    = self._cash + sum(
            prices_today[t] * self._holdings[t] for t in TICKERS
        )

        # Last 5 candles per ticker
        candles: Dict[str, list] = {}
        for t in TICKERS:
            window = self._price_history[t][s + LOOKBACK - 5: s + LOOKBACK]
            candles[t] = [Candle(**c) for c in window]

        return StockObservation(
            step            = s,
            max_steps       = MAX_STEPS,
            done            = self._done,
            cash            = round(self._cash, 2),
            holdings        = self._holdings.copy(),
            prices          = {t: round(v, 2) for t, v in prices_today.items()},
            candles         = candles,
            sentiment       = {t: _sentiment(t, s) for t in TICKERS},
            portfolio_value = round(portfolio, 2),
            pnl_pct         = round((portfolio / STARTING_CASH - 1) * 100, 3),
        )
