# CryptoBot — Crypto + AI Course

Trading bot educativo para criptomonedas. Pipeline completo: datos → features → regimen de mercado → modelos ML → senales → backtesting.

> **Este bot es educativo. No operes con dinero real sin entender completamente los riesgos.**

## Indice

- [Instalacion](#instalacion)
- [Quick Start](#quick-start)
- [Flujo del Pipeline](#flujo-del-pipeline)
- [Documentacion Detallada](#documentacion-detallada)
  - [Parametros de CryptoBot](#parametros-de-cryptobot)
  - [Carga de Datos](#carga-de-datos)
  - [Feature Engineering](#feature-engineering)
  - [Fear & Greed Index (Sentimiento)](#fear--greed-index-sentimiento)
  - [Deteccion de Regimen de Mercado](#deteccion-de-regimen-de-mercado)
  - [Estrategias de Trading](#estrategias-de-trading)
  - [Modelos de Machine Learning](#modelos-de-machine-learning)
  - [Generacion de Senales](#generacion-de-senales)
  - [Backtesting](#backtesting)
  - [Scanner Multi-Simbolo](#scanner-multi-simbolo)
  - [Guardar y Cargar](#guardar-y-cargar)
  - [Paper Trading (Testnet)](#paper-trading-testnet)
  - [Visualizacion](#visualizacion)
- [Crear tu Propio Bot (Fork)](#crear-tu-propio-bot-fork)
- [Tecnologias](#tecnologias)

## Instalacion

```bash
pip install git+https://github.com/TU-USUARIO/trading-bot-lw-latam.git
```

[↑ Indice](#indice)

## Quick Start

```python
from cryptobot import CryptoBot

bot = CryptoBot(symbol="BTC", timeframe="1d", exchange="binanceus")

bot.fetch_data(last_n=200)                  # descargar datos OHLCV
bot.create_features()                       # indicadores tecnicos
bot.detect_regime()                         # Bull / Bear / Sideways
bot.recommend_strategies()                  # ver estrategias recomendadas
bot.select_strategy("trend_following")
bot.train_models()                          # entrena 5 modelos ML
bot.get_signals()                           # genera BUY / SELL / HOLD
bot.backtest()                              # simula estrategia
bot.plot_performance()                      # grafico de resultados
```

[↑ Indice](#indice)

## Flujo del Pipeline

```
fetch_data() → create_features() → detect_regime() → recommend_strategies()
     │                                                        │
  summary()                                          select_strategy()
  plot_price()                                               │
                                                       train_models()
                                                       optimize_model()
                                                       feature_importance()
                                                       plot_feature_importance()
                                                             │
                                                       get_signals()
                                                       plot_signals()
                                                             │
                                              ┌──────────────┴──────────────┐
                                          backtest()              connect_testnet()
                                          backtest_plot()         execute()
                                          plot_performance()      status()
                                                                  trade_history()
                                                                  disconnect_testnet()
                                                                       │
                                                              save() / load()
                                                              scan() / plot_scan()
```

[↑ Indice](#indice)

---

## Documentacion Detallada

### Parametros de CryptoBot

| Parametro | Default | Descripcion |
|---|---|---|
| `symbol` | `"BTC"` | Criptomoneda: BTC, ETH, SOL, BNB, etc. |
| `timeframe` | `"1d"` | Temporalidad: `15m`, `30m`, `1h`, `4h`, `1d` |
| `exchange` | `"binanceus"` | Exchange via CCXT |
| `max_position_pct` | `0.10` | Maximo % del balance por trade (10%) |
| `stop_loss_pct` | `0.05` | Stop loss (5%) |
| `take_profit_pct` | `0.10` | Take profit (10%) |

<details>
<summary><strong>Exchanges verificados desde Google Colab</strong></summary>

| Exchange | ID |
|---|---|
| Binance US | `binanceus` (default) |
| Kraken | `kraken` |
| Crypto.com | `cryptocom` |
| OKX | `okx` |
| Coinbase | `coinbase` |
| Bitget | `bitget` |
| KuCoin | `kucoin` |
| Gemini | `gemini` |
| Bitso | `bitso` |
| Poloniex | `poloniex` |

> **Nota:** `binance` y `bybit` estan bloqueados desde Colab (geo-block US).

```python
CryptoBot.supported_exchanges()              # lista todos los exchanges de CCXT
CryptoBot.is_exchange_supported("kraken")    # True/False
```

</details>

---

### Carga de Datos

```python
# Ultimas N velas (default)
bot.fetch_data(last_n=200)

# Rango de fechas
bot.fetch_data(start="2024-01-01", end="2024-12-31")

# Con par secundario (necesario para stat_arb)
bot.fetch_data(last_n=500, pair_symbol="ETH")

# Con Fear & Greed Index (sentimiento de mercado)
bot.fetch_data(last_n=200, fear_greed=True)

# Resumen y visualizacion
bot.summary()
bot.plot_price()
```

<details>
<summary><strong>Timeframes disponibles</strong></summary>

| Timeframe | Descripcion | `last_n=200` equivale a |
|---|---|---|
| `15m` | 15 minutos | ~2 dias |
| `30m` | 30 minutos | ~4 dias |
| `1h` | 1 hora | ~8 dias |
| `4h` | 4 horas | ~33 dias |
| `1d` | 1 dia (default) | ~6.5 meses |

</details>

---

### Feature Engineering

```python
bot.create_features()              # core: ~11 indicadores (+1 con FGI)
bot.create_features(mode="full")   # full: 86+ indicadores (necesario para breakout)
```

<details>
<summary><strong>Modo "core" — 11 indicadores</strong></summary>

| Indicador | Descripcion |
|---|---|
| `SMA_20` | Media movil simple (20 periodos) |
| `SMA_50` | Media movil simple (50 periodos) |
| `RSI_14` | Relative Strength Index |
| `MACD` | Moving Average Convergence Divergence |
| `MACD_signal` | Linea de senal del MACD |
| `BB_upper` | Bollinger Band superior |
| `BB_lower` | Bollinger Band inferior |
| `ATR_14` | Average True Range (volatilidad) |
| `volume_change` | Cambio % del volumen |
| `returns` | Retorno % diario |
| `volatility_20` | Volatilidad rolling 20 periodos |

</details>

<details>
<summary><strong>Modo "full" — 86+ indicadores (via ta library)</strong></summary>

| Categoria | Cantidad | Ejemplos |
|---|---|---|
| **Momentum** | 18 | RSI, Stochastic, KAMA, PPO, TSI, Williams %R, Ultimate Oscillator |
| **Tendencia** | 32 | ADX, Aroon, CCI, EMA, SMA, MACD, Ichimoku, Parabolic SAR, Vortex |
| **Volatilidad** | 21 | ATR, Bollinger Bands, Keltner Channel, Donchian Channel, Ulcer Index |
| **Volumen** | 10 | ADI, CMF, EMV, Force Index, MFI, NVI, OBV, VPT, VWAP |
| **Otros** | 3 | Cumulative Return, Daily Log Return, Daily Return |
| **Manuales** | 2 | returns, volatility_20 |

</details>

---

### Fear & Greed Index (Sentimiento)

El Fear & Greed Index (FGI) es un indicador de sentimiento de mercado que va de 0 (miedo extremo) a 100 (codicia extrema). Se basa en datos de BTC y proviene de la API de [alternative.me](https://alternative.me/crypto/fear-and-greed-index/).

| Rango | Clasificacion |
|---|---|
| 0–24 | Extreme Fear |
| 25–49 | Fear |
| 50–74 | Greed |
| 75–100 | Extreme Greed |

```python
# Activar FGI al cargar datos
bot.fetch_data(last_n=200, fear_greed=True)

# Funciona con cualquier simbolo, pero el FGI siempre refleja sentimiento de BTC
bot.fetch_data(last_n=200, fear_greed=True)  # ETH bot con sentimiento BTC
```

> **Nota:** El FGI siempre mide el sentimiento sobre BTC, sin importar el simbolo del bot. Para simbolos non-BTC se mostrara un warning informativo.

<details>
<summary><strong>Detalles tecnicos del FGI</strong></summary>

**Como fluye por el pipeline:**
1. `fetch_data(fear_greed=True)` descarga el historial del FGI y lo mergea con los datos OHLCV por fecha
2. `create_features()` detecta automaticamente la columna `fear_greed` y la incluye como feature adicional
3. Los modelos ML pueden usar el sentimiento como senal complementaria a los indicadores tecnicos

**Edge cases:**
- **API caida:** Si la API de alternative.me no responde, el bot muestra un warning y continua sin FGI
- **Non-BTC:** El FGI se basa exclusivamente en BTC. Para otros simbolos se aplica igual pero con un warning
- **Datos pre-2018:** El FGI existe desde febrero 2018. Para fechas anteriores los valores seran NaN y se manejan automaticamente

</details>

---

### Deteccion de Regimen de Mercado

El bot detecta automaticamente si el mercado esta en modo **Bull** (alcista), **Bear** (bajista) o **Sideways** (lateral) usando PCA + Gaussian Mixture Model.

```python
bot.detect_regime()         # detecta regimen actual
bot.regime_report()         # reporte detallado
```

<details>
<summary><strong>Pipeline tecnico de deteccion</strong></summary>

1. Genera 86+ features via `ta.add_all_ta_features()` + features custom
2. Excluye features no informativos (binarios, precio absoluto, acumulativos)
3. Limpieza: `ffill`, `inf→NaN`, drop columnas >50% NaN
4. `VarianceThreshold` para eliminar features near-constant
5. Filtro de correlacion (`|r| > 0.95`) — elimina redundancia
6. PCA retiene 95% de la varianza
7. GMM robusto (`n_init=10`, full covariance, 3 componentes)
8. Mapeo semantico de clusters por retorno promedio → Bull / Bear / Sideways

</details>

---

### Estrategias de Trading

| Estrategia | Tecnica | Mejor Regimen | Peor Regimen | Requisitos |
|---|---|---|---|---|
| `trend_following` | SMA Crossover | Bull, Bear | Sideways | — |
| `mean_reversion` | Bollinger Bands | Sideways | Bull, Bear | — |
| `momentum` | RSI + Volume | Bull | Sideways | — |
| `breakout` | Donchian + Squeeze | Bull, Bear | Sideways | `mode="full"` |
| `stat_arb` | Pairs Trading | Sideways, Bear | Bull | `pair_symbol` |
| `volatility` | Vol Mean Reversion | Sideways, Bear | Bull | — |

```python
bot.recommend_strategies()                  # recomendaciones segun regimen
bot.select_strategy("trend_following")      # seleccionar estrategia
```

<details>
<summary><strong>Por que cada estrategia funciona mejor en ciertos regimenes</strong></summary>

- **trend_following** — Funciona mejor en mercados con tendencia (Bull/Bear) porque las medias moviles capturan movimientos sostenidos sin generar senales falsas.
- **mean_reversion** — Funciona mejor en mercados laterales (Sideways) porque el precio oscila en un rango predecible, permitiendo comprar en soporte y vender en resistencia.
- **momentum** — Funciona mejor en mercados alcistas (Bull) porque los movimientos fuertes de precio con volumen alto tienden a continuar.
- **breakout** — Funciona mejor en mercados con tendencia (Bull/Bear) porque las rupturas generan movimientos direccionales fuertes. En Sideways produce falsos breakouts.
- **stat_arb** — Funciona mejor en Sideways/Bear porque es market-neutral y se beneficia de spreads estables. En Bull las tendencias fuertes pueden romper la cointegracion.
- **volatility** — Funciona mejor en Sideways/Bear donde la volatilidad es alta y mean-reverts. En Bull la vol suele ser baja y estable.

</details>

---

### Modelos de Machine Learning

Entrena y compara 5 clasificadores usando `TimeSeriesSplit` (sin data leakage). El mejor modelo se selecciona automaticamente por F1-score.

```python
bot.train_models()                                          # entrena y compara 5 modelos
bot.train_models(test_size=0.2)                             # 80% train / 20% test
bot.train_models(window="sliding", window_size=60)          # ventana deslizante
bot.optimize_model()                                        # GridSearchCV
bot.feature_importance()                                    # features mas importantes
bot.plot_feature_importance()                               # grafico de importancia
```

<details>
<summary><strong>Modelos disponibles y grids de hiperparametros</strong></summary>

| Modelo | Algoritmo | Hiperparametros optimizables |
|---|---|---|
| `logistic_regression` | Regresion Logistica | C, penalty, solver |
| `svm` | Support Vector Machine (RBF) | C, gamma, kernel |
| `random_forest` | Random Forest (100 arboles) | n_estimators, max_depth, min_samples_split |
| `xgboost` | XGBoost (100 arboles) | n_estimators, max_depth, learning_rate |
| `adaboost` | AdaBoost (100 arboles) | n_estimators, learning_rate, algorithm |

Todos los modelos usan un `Pipeline` con `StandardScaler` + estimador.

</details>

---

### Generacion de Senales

```python
bot.get_signals()                           # genera BUY (1), SELL (-1), HOLD (0)
bot.get_signals(confidence_threshold=0.5)   # ajustar umbral de confianza
bot.plot_signals()                          # grafico con senales marcadas
```

> Las senales respetan el regimen: si es desfavorable para la estrategia, todas seran HOLD.

---

### Backtesting

```python
# Basico (10,000 USDT, 0.1% comision)
bot.backtest()

# Personalizado
bot.backtest(cash=50_000, commission=0.001, leverage=5, position_pct=10)

# Scope: "test" (out-of-sample), "train" (in-sample), "all"
bot.backtest(scope="test")

# Visualizacion
bot.backtest_plot()                         # grafico interactivo de backtesting.py
bot.plot_performance()                      # equity curve + drawdown
```

<details>
<summary><strong>Parametros de backtest</strong></summary>

| Parametro | Default | Descripcion |
|---|---|---|
| `cash` | `10_000` | Capital inicial (USDT) |
| `commission` | `0.001` | Comision por trade (0.1%) |
| `position_pct` | `100` | % del capital por trade (0-100) |
| `leverage` | `1` | Apalancamiento: 1, 2, 3, 5, 10, 20, 50, 100 |
| `scope` | `"test"` | `"test"`, `"train"`, o `"all"` |

> **Exposure efectiva** = position_pct x leverage. Ej: position_pct=5, leverage=10 → 50%.

</details>

---

### Scanner Multi-Simbolo

```python
bot.scan()                                  # escanea BTC, ETH, SOL, BNB, XRP
bot.scan(symbols=["BTC", "ETH", "AVAX", "DOGE"])
bot.plot_scan()                             # grid visual con regimenes
bot.plot_scan(symbols=["BTC", "ETH", "SOL"], last_n=200)
```

---

### Guardar y Cargar

```python
bot.save()                                  # guarda modelo, regimen, estrategia
bot.save("mi_bot_v1")                       # nombre personalizado

bot2 = CryptoBot()
bot2.load("mi_bot_v1")                      # restaura estado
bot2.fetch_data()                           # datos frescos
bot2.get_signals()                          # usa modelo cargado
```

---

### Paper Trading (Testnet)

> **NUNCA uses API keys de tu cuenta real. Solo testnet.**

```python
bot.connect_testnet(api_key="...", api_secret="...")
bot.execute()                               # ejecuta ultima senal en testnet
bot.status()                                # balance, posiciones, P&L
bot.trade_history()                         # historial como DataFrame
bot.disconnect_testnet()                    # cierra conexion al testnet
```

---

### Visualizacion

| Metodo | Descripcion |
|---|---|
| `plot_price()` | Candlestick + volumen |
| `plot_signals()` | Precio con senales BUY/SELL |
| `plot_performance()` | Equity curve + drawdown |
| `plot_feature_importance()` | Top N features del modelo |
| `plot_scan()` | Grid multi-simbolo con regimenes |
| `backtest_plot()` | Grafico interactivo de backtesting.py |

---

<details>
<summary><strong>Personalizacion avanzada (config.py)</strong></summary>

Edita `cryptobot/config.py` para cambiar los defaults del bot:

**Bot**

| Variable | Default | Descripcion |
|---|---|---|
| `DEFAULT_SYMBOL` | `"BTC"` | Simbolo por defecto |
| `DEFAULT_TIMEFRAME` | `"1d"` | Temporalidad por defecto |

**Scanner**

| Variable | Default | Descripcion |
|---|---|---|
| `SCANNER_SYMBOLS` | `["BTC", "ETH", "SOL", "BNB", "XRP"]` | Simbolos del scanner |
| `SCANNER_LAST_N` | `100` | Velas por simbolo en scanner |

**Sentiment**

| Variable | Default | Descripcion |
|---|---|---|
| `FGI_API_URL` | `"https://api.alternative.me/fng/"` | URL de la API Fear & Greed |
| `FGI_API_TIMEOUT` | `10` | Timeout en segundos |
| `FGI_MAX_LIMIT` | `1000` | Maximo de registros historicos |

**Features**

| Variable | Default | Descripcion |
|---|---|---|
| `SMA_FAST_WINDOW` | `20` | Ventana SMA rapida |
| `SMA_SLOW_WINDOW` | `50` | Ventana SMA lenta |
| `RSI_PERIOD` | `14` | Periodo RSI |
| `ATR_PERIOD` | `14` | Periodo ATR |
| `VOLATILITY_WINDOW` | `20` | Ventana volatilidad rolling |

**Regimen**

| Variable | Default | Descripcion |
|---|---|---|
| `REGIME_SHORT_WINDOW` | `7` | Ventana corta (momentum rapido) |
| `REGIME_MEDIUM_WINDOW` | `21` | Ventana media (tendencia intermedia) |
| `REGIME_LONG_WINDOW` | `50` | Ventana larga (tendencia de fondo) |
| `REGIME_STRUCTURAL_SMA` | `200` | SMA estructural de largo plazo |
| `REGIME_CORRELATION_THRESHOLD` | `0.95` | Filtro de correlacion (0.80–0.99) |
| `REGIME_PCA_VARIANCE` | `0.95` | Varianza retenida por PCA (0.80–0.99) |
| `REGIME_NAN_THRESHOLD` | `0.5` | Max % NaN por columna (0.3–0.8) |

**Modelos ML**

| Variable | Default | Descripcion |
|---|---|---|
| `ML_N_ESTIMATORS` | `100` | Arboles para RF/XGBoost/AdaBoost |
| `ML_TEST_SIZE` | `0.2` | Proporcion test set |
| `ML_WINDOW_SIZE` | `60` | Tamano ventana sliding |
| `ML_CV_SPLITS` | `5` | Splits de TimeSeriesSplit |
| `ML_OVERFITTING_THRESHOLD` | `0.7` | Alerta si OOS < 70% del CV |

**Senales**

| Variable | Default | Descripcion |
|---|---|---|
| `SIGNAL_CONFIDENCE_THRESHOLD` | `0.6` | Umbral de confianza minimo (0.5–0.9) |

**Backtesting**

| Variable | Default | Descripcion |
|---|---|---|
| `BACKTEST_CASH` | `10_000` | Capital inicial (USDT) |
| `BACKTEST_COMMISSION` | `0.001` | Comision por trade (0.1%) |

**Visualizacion**

| Variable | Default | Descripcion |
|---|---|---|
| `CHART_HEIGHT_MAIN` | `600` | Altura graficos principales (px) |
| `CHART_HEIGHT_SECONDARY` | `500` | Altura graficos secundarios (px) |
| `CHART_ROW_HEIGHTS` | `[0.7, 0.3]` | Proporcion precio/volumen |
| `CHART_HEIGHT_SCAN` | `300` | Altura por fila del scanner (px) |

</details>

<details>
<summary><strong>Arquitectura del proyecto</strong></summary>

CryptoBot usa composicion via mixins. La clase principal hereda de 10 mixins:

```python
CryptoBot(DataMixin, FeaturesMixin, RegimeMixin, ModelsMixin, SignalsMixin,
          BacktestMixin, VisualizationMixin, TradingMixin, PersistenceMixin, ScannerMixin)
```

| Modulo | Mixin | Rol |
|---|---|---|
| `data.py` | DataMixin | OHLCV via CCXT (10+ exchanges) |
| `features.py` | FeaturesMixin | Indicadores tecnicos (core/full) |
| `regime.py` | RegimeMixin | PCA + GMM para regimen de mercado |
| `models.py` | ModelsMixin | 5 clasificadores ML, TimeSeriesSplit |
| `signals.py` | SignalsMixin | Senales BUY/SELL/HOLD |
| `backtesting_.py` | BacktestMixin | Backtesting con leverage |
| `visualization.py` | VisualizationMixin | Graficos Plotly |
| `trading.py` | TradingMixin | Paper trading en testnet |
| `persistence.py` | PersistenceMixin | Guardar/cargar estado |
| `scanner.py` | ScannerMixin | Escaneo multi-simbolo |
| `sentiment.py` | — | Fear & Greed Index (utilidad) |
| `config.py` | — | Variables configurables |
| `constants.py` | — | Timeframes, regimenes, estrategias, colores |

</details>

[↑ Indice](#indice)

---

## Crear tu Propio Bot (Fork)

1. Haz **fork** de este repositorio
2. Clona tu fork: `git clone https://github.com/TU-USUARIO/trading-bot-lw-latam.git`
3. Modifica `cryptobot/bot.py` o el mixin respectivo con tus mejoras
4. Instala tu version:

```bash
pip install git+https://github.com/TU-USUARIO/trading-bot-lw-latam.git
```

[↑ Indice](#indice)

## Tecnologias

- [CCXT](https://github.com/ccxt/ccxt) — Conexion a exchanges (default: binanceus)
- [pandas](https://pandas.pydata.org/) / [NumPy](https://numpy.org/) — Manipulacion de datos
- [ta](https://github.com/bukosabino/ta) — Indicadores tecnicos
- [scikit-learn](https://scikit-learn.org/) — GMM, modelos ML, GridSearchCV
- [XGBoost](https://xgboost.readthedocs.io/) — Gradient Boosting
- [backtesting.py](https://kernc.github.io/backtesting.py/) — Backtesting
- [Plotly](https://plotly.com/) — Visualizacion interactiva
- [joblib](https://joblib.readthedocs.io/) — Persistencia de modelos (.pkl)

[↑ Indice](#indice)

## Licencia

MIT — Usa, modifica y comparte libremente.
