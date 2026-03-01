# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Educational crypto trading bot for the Crypto + AI Le Wagon Course. The entire bot lives in a single class `CryptoBot` in `cryptobot/bot.py`. Most methods are stubbed with `# TODO: Implementar` — the course flow is for students to implement them incrementally.

## Setup & Commands

```bash
pip install -e .           # Install in dev mode
pip install -e ".[dev]"    # Install with dev deps (pytest, black, flake8)
pytest                     # Run tests
black cryptobot/           # Format code
flake8 cryptobot/          # Lint
```

## Architecture

**Single-class design**: `CryptoBot` in `cryptobot/bot.py` encapsulates the full pipeline. Methods must be called in order — each step has `_require_*()` guards that enforce the pipeline sequence:

```
fetch_data() → create_features() → detect_regime() → recommend_strategies()
    → select_strategy() → train_models() → get_signals() → backtest() / execute()
```

**Key dependencies and their roles**:
- `ccxt` — Exchange connectivity (Bybit default, USDT pairs)
- `ta` — Technical indicators (core: ~10, full: 86+ via `add_all_ta_features`)
- `scikit-learn` — GMM for regime detection, ML models (LogReg, SVM, RF), GridSearchCV with TimeSeriesSplit
- `xgboost` — Gradient boosting model option
- `backtesting` (backtesting.py) — Strategy backtesting engine; requires DataFrame with capitalized columns (Open, High, Low, Close, Volume)
- `plotly` — All visualizations use colors in COLOR_PALETTE dict
- `joblib` — Model persistence (save/load as .pkl)

**Strategy registry**: `STRATEGY_REGISTRY` dict maps strategy keys (`trend_following`, `mean_reversion`, `momentum`) to metadata including best/worst market regimes. Regime detection uses 3-class GMM: Bull, Bear, Sideways.

**Signals**: Integer convention — `1` (BUY), `-1` (SELL), `0` (HOLD).

**Paper trading only**: Live mode is explicitly disabled (`execute(mode="live")` raises ValueError). Testnet connection required for execution.

## Conventions

- All methods return `self` for method chaining (except void/query methods)
- Language: docstrings and user-facing messages are in Spanish
- Risk params: `max_position_pct` (10%), `stop_loss_pct` (5%), `take_profit_pct` (10%)
- Valid timeframes: `1h`, `4h`, `1d`
