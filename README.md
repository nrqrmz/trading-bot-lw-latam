# 🤖 CryptoBot — Crypto + AI Course

Trading bot educativo para criptomonedas. Integra data pipeline, feature engineering, detección de régimen de mercado, modelos ML, backtesting y paper trading.

## Instalación

```bash
pip install git+https://github.com/TU-USUARIO/cryptobot-lewagon.git
```

## Quick Start

```python
from cryptobot_lewagon import CryptoBot

# 1. Crear bot
bot = CryptoBot(symbol="BTC", timeframe="1d", exchange="bybit")

# 2. Cargar datos
bot.fetch_data(last_days=90)
bot.summary()

# 3. Crear features
bot.create_features()              # core: ~10 indicadores
bot.create_features(mode="full")   # full: 86+ indicadores

# 4. Detectar régimen de mercado
bot.detect_regime()
bot.regime_report()

# 5. Elegir estrategia
bot.recommend_strategies()
bot.select_strategy("trend_following")

# 6. Entrenar modelos
bot.train_models()
bot.optimize_model()
bot.feature_importance()

# 7. Generar señales
bot.get_signals()

# 8. Backtesting
bot.backtest()
bot.backtest_plot()

# 9. Paper Trading
bot.connect_testnet(api_key="...", api_secret="...")
bot.execute(mode="paper")
bot.status()
bot.trade_history()

# 10. Escanear mercado
bot.scan(symbols=["BTC", "ETH", "SOL", "AVAX"])

# 11. Guardar / Cargar
bot.save("mi_bot_v1")
bot.load("mi_bot_v1")
```

## Flujo Completo

```
fetch_data() → create_features() → detect_regime() → recommend_strategies()
     ↓                                                        ↓
  summary()                                         select_strategy()
  plot_price()                                             ↓
                                                    train_models()
                                                    optimize_model()
                                                         ↓
                                                    get_signals()
                                                    plot_signals()
                                                         ↓
                                              ┌──────────┴──────────┐
                                          backtest()          connect_testnet()
                                          backtest_plot()     execute(mode="paper")
                                          plot_performance()  status()
                                                              trade_history()
```

## Crear tu propio bot (Fork)

1. Haz **fork** de este repositorio
2. Clona tu fork: `git clone https://github.com/TU-USUARIO/cryptobot-lewagon.git`
3. Modifica `cryptobot_lewagon/bot.py` con tus mejoras
4. Instala tu versión: `pip install git+https://github.com/TU-USUARIO/cryptobot-lewagon.git`

### Ideas para extender:

- Agregar nuevas estrategias al `STRATEGY_REGISTRY`
- Implementar nuevos indicadores técnicos en `create_features()`
- Agregar notificaciones (Telegram, email)
- Soportar múltiples pares simultáneos
- Heredar y personalizar:

```python
from cryptobot_lewagon import CryptoBot

class MiBot(CryptoBot):
    def generate_signals(self):
        # Tu propia lógica
        pass
```

## Risk Management

El bot incluye parámetros de gestión de riesgo:

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `max_position_pct` | 10% | Máximo del balance por trade |
| `stop_loss_pct` | 5% | Stop loss automático |
| `take_profit_pct` | 10% | Take profit automático |

⚠️ **Este bot es educativo. No operes con dinero real sin entender completamente los riesgos.**

## Tecnologías

- [CCXT](https://github.com/ccxt/ccxt) — Conexión a exchanges
- [pandas](https://pandas.pydata.org/) — Manipulación de datos
- [ta](https://github.com/bukosabino/ta) — Indicadores técnicos
- [scikit-learn](https://scikit-learn.org/) — Modelos ML
- [XGBoost](https://xgboost.readthedocs.io/) — Gradient Boosting
- [backtesting.py](https://kernc.github.io/backtesting.py/) — Backtesting
- [Plotly](https://plotly.com/) — Visualización interactiva

## Licencia

MIT — Usa, modifica y comparte libremente.
