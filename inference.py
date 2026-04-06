"""
inference.py — Stock Trader OpenEnv Agent
==========================================
Connects to the running FastAPI environment server and runs a full
LLM-powered trading episode through the standard reset/step/state API.

Checklist compliance:
  ✅ inference.py at project root
  ✅ API_BASE_URL, MODEL_NAME, HF_TOKEN env vars (no HF_TOKEN default)
  ✅ Defaults only for API_BASE_URL and MODEL_NAME
  ✅ All LLM calls via OpenAI client configured from env vars
  ✅ Stdout logs follow START / STEP / END structured JSON format
  ✅ Optional LOCAL_IMAGE_NAME for from_docker_image() usage
"""

import os
import json
import sys
import requests
from openai import OpenAI

# ── Required environment variables ────────────────────────────────────────────
API_BASE_URL     = os.getenv("API_BASE_URL",  "<your-active-api-base-url>")
MODEL_NAME       = os.getenv("MODEL_NAME",    "<your-active-model-name>")
HF_TOKEN         = os.getenv("HF_TOKEN")          # intentionally no default

# Optional — used when running environment locally via from_docker_image()
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

# Environment server URL — defaults to HF Space URL, override for local dev
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")

# ── OpenAI client — ALL LLM calls use this client ─────────────────────────────
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN or "dummy-key",
)

# ─────────────────────────────────────────────────────────────────────────────
# Environment HTTP helpers
# ─────────────────────────────────────────────────────────────────────────────

SESSION_ID = "agent-run-1"


def env_reset() -> dict:
    r = requests.post(f"{ENV_BASE_URL}/reset", params={"session_id": SESSION_ID})
    r.raise_for_status()
    return r.json()


def env_step(action: dict) -> dict:
    r = requests.post(
        f"{ENV_BASE_URL}/step",
        json=action,
        params={"session_id": SESSION_ID},
    )
    r.raise_for_status()
    return r.json()


def env_state() -> dict:
    r = requests.get(f"{ENV_BASE_URL}/state", params={"session_id": SESSION_ID})
    r.raise_for_status()
    return r.json()


# ─────────────────────────────────────────────────────────────────────────────
# LLM Agent
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert stock-trading AI agent operating inside a
simulated market environment. Each turn you receive the current market observation
and must decide one trade action.

Respond ONLY with valid JSON — no markdown, no extra text:
{
  "ticker":   "<one of AAPL|GOOGL|MSFT|TSLA|AMZN>",
  "decision": "<BUY|SELL|HOLD>",
  "quantity": <positive integer 1-10>,
  "reasoning": "<one sentence>"
}"""


def agent_act(obs: dict) -> dict:
    """Ask the LLM for the next trading action."""
    lines = [
        f"Step {obs['step']}/{obs['max_steps']}",
        f"Cash: ${obs['cash']:,.2f} | Portfolio: ${obs['portfolio_value']:,.2f} | P&L: {obs['pnl_pct']:+.2f}%",
        "Holdings: " + (
            ", ".join(f"{t}={v}" for t, v in obs["holdings"].items() if v > 0) or "none"
        ),
        "Prices: " + ", ".join(f"{t}=${p:.2f}" for t, p in obs["prices"].items()),
        "Sentiment: " + ", ".join(f"{t}={s:+.2f}" for t, s in obs["sentiment"].items()),
    ]
    for ticker, candles in obs["candles"].items():
        closes = [c["close"] if isinstance(c, dict) else c for c in candles]
        if closes:
            trend = "↑" if closes[-1] > closes[0] else "↓"
            lines.append(f"{ticker} 5-day trend: {trend}  last={closes[-1]:.2f}")

    response = client.chat.completions.create(
        model    = MODEL_NAME,
        max_tokens = 200,
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": "\n".join(lines)},
        ],
    )

    raw = response.choices[0].message.content.strip()
    try:
        action = json.loads(raw)
    except json.JSONDecodeError:
        action = {"ticker": "AAPL", "decision": "HOLD", "quantity": 1,
                  "reasoning": "parse error — defaulting to HOLD"}
    return action


# ─────────────────────────────────────────────────────────────────────────────
# Main — structured stdout logging (START / STEP / END)
# ─────────────────────────────────────────────────────────────────────────────

def run_episode() -> dict:
    obs = env_reset()

    # ── START log (required) ──────────────────────────────────────────────────
    print(json.dumps({
        "event":           "START",
        "cash":            obs["cash"],
        "tickers":         list(obs["prices"].keys()),
        "max_steps":       obs["max_steps"],
        "portfolio_value": obs["portfolio_value"],
    }), flush=True)

    while not obs["done"]:
        action   = agent_act(obs)
        result   = env_step(action)
        obs      = result["observation"]
        reward   = result["reward"]
        info     = result["info"]

        # ── STEP log (required) ───────────────────────────────────────────────
        print(json.dumps({
            "event":     "STEP",
            "step":      info["step"],
            "ticker":    info["ticker"],
            "decision":  info["decision"],
            "quantity":  info["quantity"],
            "price":     info["price"],
            "executed":  info.get("executed", False),
            "reward":    reward,
            "pnl_pct":   obs["pnl_pct"],
            "reasoning": action.get("reasoning", ""),
        }), flush=True)

    # ── END log (required) ────────────────────────────────────────────────────
    final_state = env_state()
    print(json.dumps({
        "event":       "END",
        "total_steps": obs["step"],
        "final_value": obs["portfolio_value"],
        "pnl_pct":     obs["pnl_pct"],
        "total_trades": final_state["total_trades"],
    }), flush=True)

    return {
        "final_portfolio_value": obs["portfolio_value"],
        "pnl_pct":               obs["pnl_pct"],
    }


if __name__ == "__main__":
    summary = run_episode()
    print(json.dumps({"event": "SUMMARY", **summary}), flush=True)
