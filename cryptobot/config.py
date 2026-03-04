"""
╔══════════════════════════════════════════════════════════════╗
║                   CONFIGURACION DEL BOT                      ║
║                                                              ║
║  Edita los valores en este archivo para personalizar tu bot. ║
║  Cada seccion tiene comentarios explicando que hace cada     ║
║  variable y que valores puedes usar.                         ║
║                                                              ║
║  Tip: Puedes editar este archivo directamente desde GitHub   ║
║  haciendo clic en el icono del lapiz (pencil icon).          ║
╚══════════════════════════════════════════════════════════════╝
"""

# ══════════════════════════════════════════════════════════════
#  BOT — Configuracion general
# ══════════════════════════════════════════════════════════════

# Simbolo por defecto al crear el bot (ej: "BTC", "ETH", "SOL")
DEFAULT_SYMBOL = "BTC"

# Temporalidad de las velas: "15m", "30m", "1h", "4h", "1d"
DEFAULT_TIMEFRAME = "1d"

# ══════════════════════════════════════════════════════════════
#  SCANNER — Escaneo de multiples criptos
# ══════════════════════════════════════════════════════════════

# Lista de criptos a escanear cuando ejecutas bot.scan()
SCANNER_SYMBOLS = ["BTC", "ETH", "SOL", "BNB", "XRP"]

# Numero de velas a descargar por cada cripto en el scanner
SCANNER_LAST_N = 100

# ══════════════════════════════════════════════════════════════
#  SENTIMENT — Fear & Greed Index
# ══════════════════════════════════════════════════════════════

# URL de la API de alternative.me (Fear & Greed Index crypto)
FGI_API_URL = "https://api.alternative.me/fng/"

# Timeout en segundos para la llamada a la API
FGI_API_TIMEOUT = 10

# Maximo de registros historicos que soporta la API
FGI_MAX_LIMIT = 1000

# ══════════════════════════════════════════════════════════════
#  FEATURES — Indicadores tecnicos (modo "core")
# ══════════════════════════════════════════════════════════════

# Ventana de la media movil rapida (SMA corta)
SMA_FAST_WINDOW = 20

# Ventana de la media movil lenta (SMA larga)
SMA_SLOW_WINDOW = 50

# Periodo del RSI (Relative Strength Index)
RSI_PERIOD = 14

# Periodo del ATR (Average True Range — mide volatilidad)
ATR_PERIOD = 14

# Ventana para calcular la volatilidad rolling
VOLATILITY_WINDOW = 20

# ══════════════════════════════════════════════════════════════
#  REGIMEN — Deteccion de mercado (Bull / Bear / Sideways)
# ══════════════════════════════════════════════════════════════

# Ventanas rolling para features del regimen
REGIME_SHORT_WINDOW = 7       # Ventana corta (momentum rapido)
REGIME_MEDIUM_WINDOW = 21     # Ventana media (tendencia intermedia)
REGIME_LONG_WINDOW = 50       # Ventana larga (tendencia de fondo)

# Ventana de la SMA estructural (referencia de largo plazo)
REGIME_STRUCTURAL_SMA = 200

# Filtro de correlacion: elimina features con |r| mayor a este valor
# Rango: 0.80 a 0.99 — mas bajo = filtrado mas agresivo
REGIME_CORRELATION_THRESHOLD = 0.95

# PCA: porcentaje de varianza a retener
# Rango: 0.80 a 0.99 — mas alto = mas componentes
REGIME_PCA_VARIANCE = 0.95

# Umbral de NaN: descarta columnas con mas de este % de valores nulos
# Rango: 0.3 a 0.8
REGIME_NAN_THRESHOLD = 0.5

# VarianceThreshold: elimina features casi constantes
REGIME_VARIANCE_THRESHOLD = 1e-5

# GMM: numero de inicializaciones (mas = mas robusto, mas lento)
REGIME_GMM_N_INIT = 10

# ══════════════════════════════════════════════════════════════
#  MODELOS — Machine Learning
# ══════════════════════════════════════════════════════════════

# Numero de arboles para Random Forest, XGBoost y AdaBoost
ML_N_ESTIMATORS = 100

# Proporcion de datos reservados para test (out-of-sample)
# Rango: 0.1 a 0.5 — ej: 0.2 = 20% para test
ML_TEST_SIZE = 0.2

# Tamano de ventana para modo "sliding" (en periodos/velas)
ML_WINDOW_SIZE = 60

# Numero de splits para TimeSeriesSplit (cross-validation temporal)
ML_CV_SPLITS = 5

# Alerta de overfitting: si F1 out-of-sample < este % del F1 cross-val
# Rango: 0.5 a 0.9 — ej: 0.7 = alerta si OOS es menor al 70% del CV
ML_OVERFITTING_THRESHOLD = 0.7

# ══════════════════════════════════════════════════════════════
#  SENALES — Generacion de senales BUY / SELL / HOLD
# ══════════════════════════════════════════════════════════════

# Umbral de confianza minima para generar senal
# Si la probabilidad del modelo es menor, la senal sera HOLD
# Rango: 0.5 a 0.9 — mas alto = menos senales pero mas seguras
SIGNAL_CONFIDENCE_THRESHOLD = 0.6

# ══════════════════════════════════════════════════════════════
#  BACKTESTING — Simulacion historica
# ══════════════════════════════════════════════════════════════

# Capital inicial para la simulacion (en USDT)
BACKTEST_CASH = 10_000

# Comision por trade (0.001 = 0.1%, tipico de exchanges crypto)
BACKTEST_COMMISSION = 0.001

# ══════════════════════════════════════════════════════════════
#  VISUALIZACION — Tamano de graficos
# ══════════════════════════════════════════════════════════════

# Altura de graficos principales (precio, performance) en pixeles
CHART_HEIGHT_MAIN = 600

# Altura de graficos secundarios (senales, regimen) en pixeles
CHART_HEIGHT_SECONDARY = 500

# Proporcion entre panel de precio y panel de volumen [precio, volumen]
CHART_ROW_HEIGHTS = [0.7, 0.3]

# Altura por fila del grid de scanner visual (plot_scan) en pixeles
CHART_HEIGHT_SCAN = 300
