# 🤖 CryptoBot — Crypto + AI Course

Trading bot educativo para criptomonedas. Integra data pipeline, feature engineering, detección de régimen de mercado, modelos ML, backtesting y paper trading.

## Instalación

```bash
pip install git+https://github.com/TU-USUARIO/trading-bot-lw-latam.git
```

## Quick Start

```python
from cryptobot import CryptoBot

# 0. Consultar exchanges disponibles
CryptoBot.supported_exchanges()              # lista todos los exchanges de CCXT
CryptoBot.is_exchange_supported("binance")   # True/False

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
bot.detect_regime()
bot.regime_report()

# 5. Elegir estrategia
bot.recommend_strategies()
bot.select_strategy("trend_following")

# 6. Entrenar modelos
bot.train_models()
bot.optimize_model()               # GridSearchCV con TimeSeriesSplit
bot.feature_importance()
bot.plot_feature_importance()

# 7. Generar señales (1=BUY, -1=SELL, 0=HOLD)
bot.get_signals()
bot.plot_signals()

# 8. Backtesting
bot.backtest()
bot.backtest_plot()
bot.plot_performance()

# 9. Guardar / cargar estado
bot.save()                          # guarda modelo, régimen, estrategia
bot.save("mi_bot_v1")
bot2 = CryptoBot(); bot2.load("mi_bot_v1")

# 10. Scanner multi-símbolo
bot.scan()                          # escanea BTC, ETH, SOL, BNB, XRP
bot.scan(symbols=["BTC", "ETH", "AVAX", "DOGE"])

# 11. Paper trading (testnet)
bot.connect_testnet(api_key="...", api_secret="...")
bot.execute()                       # ejecuta última señal en testnet
bot.status()                        # balance, posiciones, P&L
bot.trade_history()                 # historial como DataFrame
```

> **Nota**: `breakout` requiere features con `mode="full"`. `stat_arb` requiere `fetch_data(pair_symbol="ETH")` para cargar datos del par secundario.

## Timeframes disponibles

| Timeframe | Descripción |
|---|---|
| `15m` | 15 minutos |
| `30m` | 30 minutos |
| `1h` | 1 hora |
| `4h` | 4 horas |
| `1d` | 1 día (default) |

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
                                          backtest()              connect_testnet()
                                          backtest_plot()         execute()
                                          plot_performance()      status()
                                                                  trade_history()
                                                                       ↓
                                                               save() / load()
                                                               scan()
```


### ⚠️ **Este bot es educativo. No operes con dinero real sin entender completamente los riesgos.**

## Crear tu propio bot (Fork)

1. Haz **fork** de este repositorio
2. Clona tu fork: `git clone https://github.com/TU-USUARIO/trading-bot-lw-latam.git`
3. Modifica `cryptobot/bot.py` o el mixin respectivo con tus mejoras
4. Instala tu versión: `pip install git+https://github.com/TU-USUARIO/trading-bot-lw-latam.git`

```python
!pip install git+https://github.com/TU-USUARIO/trading-bot-lw-latam.git -q
```

```python
from cryptobot import CryptoBot

bot = CryptoBot()
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
