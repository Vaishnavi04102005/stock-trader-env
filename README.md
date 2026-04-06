---
title: Stock Trader OpenEnv
emoji: 📉
colorFrom: gray
colorTo: indigo
sdk: docker
pinned: false
license: mit
---

# Stock Trader OpenEnv 📉

A real-world OpenEnv environment where LLM agents learn to trade a portfolio of 5 tech stocks.

## Quickstart

```python
# Install client from this Space
pip install git+https://huggingface.co/spaces/YOUR_USERNAME/stock-trader-env

# Run the agent against the live Space
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=meta-llama/Meta-Llama-3-8B-Instruct
export HF_TOKEN=hf_your_token
export ENV_BASE_URL=https://YOUR_USERNAME-stock-trader-env.hf.space

python inference.py
```

## Environment

| Property | Value |
|---|---|
| ID | `stock-trader-v1` |
| Assets | AAPL · GOOGL · MSFT · TSLA · AMZN |
| Episode length | 30 steps (trading days) |
| Starting cash | $10,000 |
| Action | `{ ticker, decision: BUY/SELL/HOLD, quantity: 1–10 }` |
| Observation | 5-day OHLCV + sentiment + portfolio |
| Reward | Δ portfolio value per step |

## API

| Method | Endpoint | Description |
|---|---|---|
| POST | `/reset?session_id=...` | Start new episode |
| POST | `/step?session_id=...` | Execute action |
| GET  | `/state?session_id=...` | Current state |
| WS   | `/ws` | WebSocket (full session) |

## Required Secrets (set in Space Settings)

```
API_BASE_URL   — LLM endpoint  (default provided)
MODEL_NAME     — model to use  (default provided)
HF_TOKEN       — your HF token (set as Secret — no default)
```

## Project Structure

```
stock-trader-env/
├── inference.py          ← Main agent script (root, required)
├── models.py             ← Pydantic type contracts
├── client.py             ← Installable Python client
├── openenv.yaml          ← Environment manifest
├── pyproject.toml        ← Makes Space pip-installable
├── README.md             ← This file (HF Space card)
└── server/
    ├── app.py            ← FastAPI server
    ├── environment.py    ← RL environment logic
    ├── Dockerfile        ← Docker image definition
    └── requirements.txt  ← Server dependencies
```

## Checklist

- ✅ `inference.py` at project root, follows sample strictly
- ✅ `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN` env vars present
- ✅ Defaults only for `API_BASE_URL` and `MODEL_NAME` (not `HF_TOKEN`)
- ✅ All LLM calls use `OpenAI` client configured from env vars
- ✅ Stdout logs follow `START` / `STEP` / `END` structured JSON format
