# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CryptoBot is an educational cryptocurrency trading bot (Le Wagon Latam course). It provides a full pipeline: data fetching → feature engineering → market regime detection → ML model training → signal generation → backtesting. Python 3.8+, installable as a package.

## Commands

```bash
# Install (editable, with dev deps)
pip install -e ".[dev]"

# Run tests / format / lint
pytest
black .
flake8 .
```

## Architecture

**Mixin-based composition** — `CryptoBot` (in `bot.py`) inherits from 9 mixins, each in its own module:

```
CryptoBot(DataMixin, FeaturesMixin, RegimeMixin, ModelsMixin, SignalsMixin,
          BacktestMixin, VisualizationMixin, TradingMixin, PersistenceMixin, ScannerMixin)
```

**Pipeline flow:** `fetch_data()` → `create_features()` → `detect_regime()` → `select_strategy()` → `train_models()` → `get_signals()` → `backtest()`

All pipeline methods return `self` for method chaining. Lazy validation via `_require_*()` methods ensures steps are called in order.

### Key modules

| Module | Mixin | Role |
|---|---|---|
| `data.py` | DataMixin | OHLCV fetching via CCXT (10+ exchanges) |
| `features.py` | FeaturesMixin | Core (~10) or full (86+) technical indicators via `ta` library |
| `regime.py` | RegimeMixin | PCA + Gaussian Mixture Model for Bull/Bear/Sideways detection |
| `models.py` | ModelsMixin | 5 ML classifiers (LogReg, SVM, RF, XGBoost, AdaBoost), TimeSeriesSplit |
| `signals.py` | SignalsMixin | Regime-aware BUY/SELL/HOLD signal generation |
| `backtesting_.py` | BacktestMixin | backtesting.py integration with leverage/commission support |
| `visualization.py` | VisualizationMixin | Plotly interactive charts |
| `constants.py` | — | Timeframes, regime labels, strategy registry, color palette |

### Incomplete modules (stubs/TODO)

`trading.py` (paper trading), `persistence.py` (save/load), `scanner.py` (multi-symbol screening)

## Key Design Decisions

- **TimeSeriesSplit** for all cross-validation to prevent data leakage
- **6 strategies** defined in `STRATEGY_REGISTRY` (constants.py), each mapped to favorable/unfavorable regimes
- **Regime detection pipeline** (regime.py): generates indicators → excludes non-informative columns → VarianceThreshold → correlation filter (|r| > 0.95) → PCA (95% variance) → GMM (3 components)
- Feature modes: `"core"` for fast iteration, `"full"` for 86+ indicators
- Educational comments are in Spanish
