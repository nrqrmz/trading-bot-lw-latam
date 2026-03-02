# 🤖 CryptoBot — Crypto + AI Course

Trading bot educativo para criptomonedas. Integra data pipeline, feature engineering, detección de régimen de mercado, modelos ML, backtesting y paper trading.

## Instalación

```bash
pip install git+https://github.com/TU-USUARIO/trading-bot-lw-latam.git
```

Para desarrollo local:

```bash
git clone https://github.com/TU-USUARIO/trading-bot-lw-latam.git
cd trading-bot-lw-latam
pip install -e ".[dev]"    # Incluye pytest, black, flake8
```

## Quick Start

```python
from cryptobot import CryptoBot

# 1. Crear bot (todos los parámetros tienen defaults)
bot = CryptoBot(symbol="BTC", timeframe="1d", exchange="binanceus",
                max_position_pct=0.10, stop_loss_pct=0.05, take_profit_pct=0.10)

# 2. Cargar datos
bot.fetch_data(last_n=200)
# bot.fetch_data(start="2024-01-01", end="2024-12-31")
# bot.fetch_data(pair_symbol="ETH")  # necesario para stat_arb
bot.summary()
bot.plot_price()

# 3. Crear features
bot.create_features()              # core: ~10 indicadores
bot.create_features(mode="full")   # full: 86+ indicadores

# 4. Detectar régimen de mercado (GMM: Bull, Bear, Sideways)
bot.detect_regime(n_regimes=3)
bot.regime_report()

# 5. Elegir estrategia
bot.recommend_strategies()
bot.select_strategy("trend_following")

# 6. Entrenar modelos
bot.train_models(window="expanding", window_size=60, test_size=0.2)
bot.optimize_model()               # GridSearchCV con TimeSeriesSplit
bot.feature_importance(top_n=15)
bot.plot_feature_importance(top_n=15)

# 7. Generar señales (1=BUY, -1=SELL, 0=HOLD)
bot.get_signals()
bot.plot_signals()

# 8. Backtesting
bot.backtest(cash=10_000, commission=0.001, position_pct=100,
             leverage=1, scope="test")
bot.backtest_plot()
bot.plot_performance()

# ─── TODO stubs (para implementar en clase) ───

# 9. Paper Trading [TODO]
bot.connect_testnet(api_key="...", api_secret="...")
bot.execute(mode="paper")
bot.status()
bot.trade_history()

# 10. Guardar / Cargar [TODO]
bot.save("mi_bot_v1", path=".")
bot.load("mi_bot_v1", path=".")

# 11. Escanear mercado [TODO]
bot.scan(symbols=["BTC", "ETH", "SOL", "AVAX"])
```

## Estrategias Disponibles

| Estrategia | Key | Mejor Régimen | Peor Régimen |
|---|---|---|---|
| Trend Following (SMA Crossover) | `trend_following` | Bull, Bear | Sideways |
| Mean Reversion (Bollinger Bands) | `mean_reversion` | Sideways | Bull, Bear |
| Momentum (RSI + Volume) | `momentum` | Bull | Sideways |
| Breakout (Donchian + Squeeze) | `breakout` | Bull, Bear | Sideways |
| Statistical Arbitrage (Pairs) | `stat_arb` | Sideways, Bear | Bull |
| Volatility (Vol Mean Reversion) | `volatility` | Sideways, Bear | Bull |

> **Nota**: `breakout` requiere features con `mode="full"`. `stat_arb` requiere `fetch_data(pair_symbol="ETH")` para cargar datos del par secundario.

## Flujo Completo

```
fetch_data() → create_features() → detect_regime() → recommend_strategies()
     ↓                                                        ↓
  summary()                                          select_strategy()
  plot_price()                                               ↓
                                                       train_models()
                                                       optimize_model()
                                                       feature_importance()
                                                       plot_feature_importance()
                                                             ↓
                                                       get_signals()
                                                       plot_signals()
                                                             ↓
                                              ┌──────────────┴──────────────┐
                                          backtest()              connect_testnet() [TODO]
                                          backtest_plot()         execute()          [TODO]
                                          plot_performance()      status()           [TODO]
                                                                  trade_history()
                                                                       ↓
                                                               save() / load()      [TODO]
                                                               scan()               [TODO]
```

## Risk Management

| Parámetro | Default | Descripción |
|---|---|---|
| `max_position_pct` | 0.10 (10%) | Máximo del balance por trade |
| `stop_loss_pct` | 0.05 (5%) | Stop loss automático |
| `take_profit_pct` | 0.10 (10%) | Take profit automático |

⚠️ **Este bot es educativo. No operes con dinero real sin entender completamente los riesgos.**

## Crear tu propio bot (Fork)

1. Haz **fork** de este repositorio
2. Clona tu fork: `git clone https://github.com/TU-USUARIO/trading-bot-lw-latam.git`
3. Modifica `cryptobot/bot.py` con tus mejoras
4. Instala tu versión: `pip install git+https://github.com/TU-USUARIO/trading-bot-lw-latam.git`

```python
from cryptobot import CryptoBot

class MiBot(CryptoBot):
    def generate_signals(self):
        # Tu propia lógica
        pass
```

## Tecnologías

- [CCXT](https://github.com/ccxt/ccxt) — Conexión a exchanges (default: binanceus)
- [pandas](https://pandas.pydata.org/) / [NumPy](https://numpy.org/) — Manipulación de datos
- [ta](https://github.com/bukosabino/ta) — Indicadores técnicos
- [scikit-learn](https://scikit-learn.org/) — GMM, modelos ML, GridSearchCV
- [XGBoost](https://xgboost.readthedocs.io/) — Gradient Boosting
- [backtesting.py](https://kernc.github.io/backtesting.py/) — Backtesting
- [Plotly](https://plotly.com/) — Visualización interactiva
- [joblib](https://joblib.readthedocs.io/) — Persistencia de modelos (.pkl)

## Licencia

MIT — Usa, modifica y comparte libremente.
