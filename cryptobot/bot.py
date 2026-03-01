"""
CryptoBot — Clase principal del trading bot educativo.

Este módulo contiene la clase CryptoBot que integra:
- Conexión a exchanges via CCXT
- Feature engineering con indicadores técnicos
- Detección de régimen de mercado (GMM)
- Recomendación de estrategias según régimen
- Entrenamiento y comparación de modelos ML
- Backtesting con backtesting.py
- Paper trading en testnet
"""

import warnings
from datetime import datetime, timedelta
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════════

VALID_TIMEFRAMES = ["1h", "4h", "1d"]

REGIME_LABELS = {0: "Bear 🔴", 1: "Sideways 🟡", 2: "Bull 🟢"}

STRATEGY_REGISTRY = {
    "trend_following": {
        "name": "Trend Following (SMA Crossover)",
        "description": "Sigue la dirección de la tendencia usando cruces de medias móviles",
        "best_regimes": ["Bull", "Bear"],
        "worst_regimes": ["Sideways"],
    },
    "mean_reversion": {
        "name": "Mean Reversion (Bollinger Bands)",
        "description": "Apuesta a que el precio regresa a su media cuando se aleja demasiado",
        "best_regimes": ["Sideways"],
        "worst_regimes": ["Bull", "Bear"],
    },
    "momentum": {
        "name": "Momentum (RSI + Volume)",
        "description": "Identifica movimientos fuertes y se sube a ellos",
        "best_regimes": ["Bull"],
        "worst_regimes": ["Sideways"],
    },
}

BINANCE_COLORS = {
    "yellow": "#F0B90B",
    "dark": "#1E2329",
    "green": "#0ECB81",
    "red": "#F6465D",
    "gray": "#474D57",
}


class CryptoBot:
    """
    Trading bot educativo para criptomonedas.

    Integra data pipeline, feature engineering, detección de régimen,
    entrenamiento de modelos ML, backtesting y paper trading.

    Parameters
    ----------
    symbol : str
        Símbolo de la criptomoneda (e.g., "BTC", "ETH", "SOL").
        Se combina internamente con "/USDT" para CCXT.
    timeframe : str
        Temporalidad de las velas. Opciones: "1h", "4h", "1d" (default).
    exchange : str
        Exchange a usar via CCXT (default: "bybit").
    max_position_pct : float
        Máximo porcentaje del balance por trade (default: 0.10 = 10%).
    stop_loss_pct : float
        Stop loss como porcentaje (default: 0.05 = 5%).
    take_profit_pct : float
        Take profit como porcentaje (default: 0.10 = 10%).

    Attributes
    ----------
    data : pd.DataFrame or None
        OHLCV data. Columnas: Open, High, Low, Close, Volume.
    features : pd.DataFrame or None
        Data con indicadores técnicos agregados.
    regime : str or None
        Régimen de mercado actual ("Bull", "Bear", "Sideways").
    regime_probabilities : dict or None
        Probabilidades de cada régimen.
    selected_strategy : str or None
        Estrategia seleccionada por el usuario.
    model : object or None
        Mejor modelo entrenado.
    model_metrics : dict or None
        Métricas del modelo entrenado.
    signals : pd.Series or None
        Señales generadas: 1 (BUY), -1 (SELL), 0 (HOLD).
    backtest_results : object or None
        Resultados del backtesting.
    trades : list
        Historial de trades ejecutados.

    Examples
    --------
    >>> from cryptobot_lewagon import CryptoBot
    >>> bot = CryptoBot(symbol="BTC")
    >>> bot.fetch_data(last_days=90)
    >>> bot.create_features()
    >>> bot.detect_regime()
    >>> bot.recommend_strategies()
    >>> bot.select_strategy("trend_following")
    >>> bot.train_models()
    >>> bot.get_signals()
    >>> bot.backtest()
    """

    def __init__(
        self,
        symbol: str = "BTC",
        timeframe: str = "1d",
        exchange: str = "bybit",
        max_position_pct: float = 0.10,
        stop_loss_pct: float = 0.05,
        take_profit_pct: float = 0.10,
    ):
        # ── Validation ──────────────────────────────────
        if timeframe not in VALID_TIMEFRAMES:
            raise ValueError(
                f"Timeframe '{timeframe}' no válido. Opciones: {VALID_TIMEFRAMES}"
            )

        # ── Config ──────────────────────────────────────
        self.symbol = symbol.upper()
        self._pair = f"{self.symbol}/USDT"
        self.timeframe = timeframe
        self.exchange_id = exchange.lower()

        # ── Risk Management ─────────────────────────────
        self.max_position_pct = max_position_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

        # ── State (se llenan con los métodos) ───────────
        self.data: Optional[pd.DataFrame] = None
        self.features: Optional[pd.DataFrame] = None
        self.regime: Optional[str] = None
        self.regime_probabilities: Optional[dict] = None
        self.regime_model = None
        self.selected_strategy: Optional[str] = None
        self.model = None
        self.model_name: Optional[str] = None
        self.model_metrics: Optional[dict] = None
        self.model_comparison: Optional[pd.DataFrame] = None
        self.signals: Optional[pd.Series] = None
        self.backtest_results = None
        self._bt_object = None
        self.trades: list = []

        # ── Exchange Connection ─────────────────────────
        self._exchange = None
        self._testnet_connected = False

        # ── Initialize Exchange (public API only) ───────
        self._init_exchange()

    # ════════════════════════════════════════════════════════════
    #  PRIVATE HELPERS
    # ════════════════════════════════════════════════════════════

    def _init_exchange(self):
        """Inicializa conexión al exchange via CCXT (solo API pública)."""
        import ccxt

        exchange_class = getattr(ccxt, self.exchange_id, None)
        if exchange_class is None:
            raise ValueError(f"Exchange '{self.exchange_id}' no soportado por CCXT.")

        self._exchange = exchange_class({"enableRateLimit": True})
        print(f"✅ Conectado a {self.exchange_id.capitalize()} (API pública)")

    def _require_data(self):
        """Verifica que fetch_data() fue ejecutado."""
        if self.data is None:
            raise RuntimeError(
                "❌ No hay datos cargados. Ejecuta bot.fetch_data() primero."
            )

    def _require_features(self):
        """Verifica que create_features() fue ejecutado."""
        if self.features is None:
            raise RuntimeError(
                "❌ No hay features creados. Ejecuta bot.create_features() primero."
            )

    def _require_regime(self):
        """Verifica que detect_regime() fue ejecutado."""
        if self.regime is None:
            raise RuntimeError(
                "❌ No hay régimen detectado. Ejecuta bot.detect_regime() primero."
            )

    def _require_strategy(self):
        """Verifica que select_strategy() fue ejecutado."""
        if self.selected_strategy is None:
            raise RuntimeError(
                "❌ No hay estrategia seleccionada. Ejecuta bot.select_strategy() primero."
            )

    def _require_model(self):
        """Verifica que train_models() fue ejecutado."""
        if self.model is None:
            raise RuntimeError(
                "❌ No hay modelo entrenado. Ejecuta bot.train_models() primero."
            )

    def _require_signals(self):
        """Verifica que get_signals() fue ejecutado."""
        if self.signals is None:
            raise RuntimeError(
                "❌ No hay señales generadas. Ejecuta bot.get_signals() primero."
            )

    def _require_testnet(self):
        """Verifica que connect_testnet() fue ejecutado."""
        if not self._testnet_connected:
            raise RuntimeError(
                "❌ No conectado al testnet. Ejecuta bot.connect_testnet() primero."
            )

    # ════════════════════════════════════════════════════════════
    #  1. DATA PIPELINE
    # ════════════════════════════════════════════════════════════

    def fetch_data(
        self,
        last_days: int = 90,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> "CryptoBot":
        """
        Obtiene datos OHLCV del exchange via CCXT.

        Soporta dos modos:
        - Por cantidad de días: bot.fetch_data(last_days=90)
        - Por rango de fechas: bot.fetch_data(start="2024-01-01", end="2024-12-31")

        Las columnas del DataFrame siguen el formato requerido
        por backtesting.py: Open, High, Low, Close, Volume.

        Parameters
        ----------
        last_days : int, default 90
            Número de días hacia atrás desde hoy. Se ignora si start/end están definidos.
        start : str, optional
            Fecha de inicio en formato "YYYY-MM-DD".
        end : str, optional
            Fecha de fin en formato "YYYY-MM-DD". Default: hoy.

        Returns
        -------
        CryptoBot
            Retorna self para permitir method chaining.

        Examples
        --------
        >>> bot.fetch_data()
        >>> bot.fetch_data(last_days=180)
        >>> bot.fetch_data(start="2024-01-01", end="2024-06-30")
        """
        # TODO: Implementar
        # 1. Calcular since_timestamp basado en last_days o start/end
        # 2. Llamar self._exchange.fetch_ohlcv(self._pair, self.timeframe, since, limit)
        # 3. Paginar si es necesario (CCXT tiene límite por request)
        # 4. Construir DataFrame con columnas: Open, High, Low, Close, Volume
        # 5. Index: DatetimeIndex con nombre "Date"
        # 6. Guardar en self.data
        # 7. Print resumen: rango de fechas, registros, precio actual
        pass

    def summary(self) -> None:
        """
        Muestra resumen del dataset cargado.

        Incluye: rango de fechas, número de registros, precio actual,
        cambio porcentual, high/low del período, y estadísticas básicas.

        Raises
        ------
        RuntimeError
            Si no se ha ejecutado fetch_data() previamente.
        """
        self._require_data()
        # TODO: Implementar
        # 1. Rango de fechas (primer y último registro)
        # 2. Número de registros
        # 3. Precio actual (último close)
        # 4. Cambio % en el período
        # 5. High/Low del período
        # 6. self.data.describe() formateado
        pass

    # ════════════════════════════════════════════════════════════
    #  2. FEATURE ENGINEERING
    # ════════════════════════════════════════════════════════════

    def create_features(self, mode: str = "core") -> "CryptoBot":
        """
        Agrega indicadores técnicos al DataFrame.

        Dos modos disponibles:
        - "core": ~10 indicadores clave que se cubrieron en el curso.
          Ideal para exploración y comprensión.
        - "full": 86+ indicadores via ta.add_all_ta_features().
          Ideal para training de modelos donde feature importance
          determina qué indicadores son relevantes.

        Parameters
        ----------
        mode : str, default "core"
            "core" para indicadores esenciales, "full" para todos.

        Returns
        -------
        CryptoBot
            Retorna self para permitir method chaining.

        Core Features
        -------------
        - SMA_20, SMA_50 : Medias móviles simples
        - RSI_14 : Relative Strength Index
        - MACD, MACD_signal : Moving Average Convergence Divergence
        - BB_upper, BB_lower : Bollinger Bands
        - ATR_14 : Average True Range (volatilidad)
        - volume_change : Cambio porcentual del volumen
        - returns : Retorno porcentual diario
        - volatility_20 : Volatilidad rolling 20 períodos

        Raises
        ------
        RuntimeError
            Si no se ha ejecutado fetch_data() previamente.
        ValueError
            Si mode no es "core" o "full".
        """
        self._require_data()
        # TODO: Implementar
        # mode="core":
        #   1. Calcular cada indicador con la librería `ta`
        #   2. Agregar returns y volatility manualmente
        #   3. Dropear NaN rows iniciales
        #
        # mode="full":
        #   1. ta.add_all_ta_features(...)
        #   2. Dropear NaN rows iniciales
        #   3. Print cantidad de features agregados
        #
        # Guardar en self.features
        pass

    # ════════════════════════════════════════════════════════════
    #  3. MARKET INTELLIGENCE
    # ════════════════════════════════════════════════════════════

    def detect_regime(self, n_regimes: int = 3) -> "CryptoBot":
        """
        Detecta el régimen de mercado actual usando Gaussian Mixture Model.

        Clasifica el mercado en regímenes basados en:
        - Returns (retornos porcentuales)
        - Volatilidad (rolling std de returns)
        - Trend strength (pendiente de SMA)
        - Volume change (cambio porcentual de volumen)

        El GMM provee probabilidades suaves para cada régimen,
        no una clasificación binaria. Ejemplo: "72% Bull, 20% Sideways, 8% Bear".

        Parameters
        ----------
        n_regimes : int, default 3
            Número de regímenes a detectar.
            Default 3: Bull 🟢, Bear 🔴, Sideways 🟡.

        Returns
        -------
        CryptoBot
            Retorna self para permitir method chaining.

        Raises
        ------
        RuntimeError
            Si no se ha ejecutado create_features() previamente.
        """
        self._require_features()
        # TODO: Implementar
        # 1. Seleccionar features para clustering: returns, volatility, trend, volume_change
        # 2. Escalar con StandardScaler
        # 3. Fit GaussianMixture(n_components=n_regimes)
        # 4. Predecir régimen del último período
        # 5. Obtener probabilidades con .predict_proba()
        # 6. Mapear clusters a labels (Bull/Bear/Sideways) basado en mean returns
        # 7. Guardar en self.regime, self.regime_probabilities, self.regime_model
        # 8. Print régimen actual con confianza
        pass

    def regime_report(self) -> None:
        """
        Visualización detallada del régimen actual.

        Muestra:
        - Régimen actual con probabilidad
        - Distribución histórica de regímenes
        - Gráfico de precio coloreado por régimen
        - Métricas por régimen (return promedio, volatilidad, duración)

        Raises
        ------
        RuntimeError
            Si no se ha ejecutado detect_regime() previamente.
        """
        self._require_regime()
        # TODO: Implementar
        # 1. Tabla de probabilidades de cada régimen
        # 2. Plotly chart: precio con background coloreado por régimen
        # 3. Estadísticas por régimen: avg return, avg volatility, avg duration
        pass

    def recommend_strategies(self) -> None:
        """
        Recomienda estrategias de trading basadas en el régimen actual.

        Para cada estrategia del registry:
        1. Ejecuta un backtest rápido en datos del régimen actual
        2. Calcula Sharpe ratio, win rate, total return
        3. Rankea estrategias de mejor a peor
        4. Indica cuáles son recomendadas (🟢) y cuáles no (🔴)

        Mapping régimen → estrategia:
        - Bull: Trend Following, Momentum > Mean Reversion
        - Bear: Mean Reversion, Short Momentum > Trend Following
        - Sideways: Mean Reversion, Range Trading > Trend Following

        Raises
        ------
        RuntimeError
            Si no se ha ejecutado detect_regime() previamente.
        """
        self._require_regime()
        # TODO: Implementar
        # 1. Para cada estrategia en STRATEGY_REGISTRY:
        #    a. Crear señales basadas en la lógica de la estrategia
        #    b. Backtest rápido sobre períodos con el régimen actual
        #    c. Calcular métricas
        # 2. Rankear por Sharpe ratio
        # 3. Marcar como recomendada (🟢) o no recomendada (🔴)
        # 4. Print tabla formateada con resultados
        pass

    def select_strategy(self, strategy: str) -> "CryptoBot":
        """
        Selecciona una estrategia de trading.

        Parameters
        ----------
        strategy : str
            Nombre de la estrategia. Opciones:
            "trend_following", "mean_reversion", "momentum".

        Returns
        -------
        CryptoBot
            Retorna self para permitir method chaining.

        Raises
        ------
        ValueError
            Si la estrategia no está en el registry.
        """
        if strategy not in STRATEGY_REGISTRY:
            available = list(STRATEGY_REGISTRY.keys())
            raise ValueError(
                f"Estrategia '{strategy}' no válida. Opciones: {available}"
            )

        self.selected_strategy = strategy
        info = STRATEGY_REGISTRY[strategy]
        print(f"✅ Estrategia seleccionada: {info['name']}")
        print(f"   {info['description']}")
        return self

    # ════════════════════════════════════════════════════════════
    #  4. MODEL TRAINING
    # ════════════════════════════════════════════════════════════

    def train_models(
        self,
        window: str = "expanding",
        window_size: int = 60,
    ) -> "CryptoBot":
        """
        Entrena y compara múltiples modelos ML para la estrategia seleccionada.

        Modelos evaluados:
        - Logistic Regression
        - SVM (RBF kernel)
        - Random Forest
        - XGBoost

        Usa TimeSeriesSplit para validación temporal (sin data leakage).

        Parameters
        ----------
        window : str, default "expanding"
            Tipo de ventana de entrenamiento:
            - "expanding": usa todos los datos disponibles hasta t.
              Más estable, más datos de training.
            - "sliding": usa solo los últimos window_size períodos.
              Se adapta mejor a cambios de régimen.
        window_size : int, default 60
            Tamaño de la ventana para modo "sliding" (en períodos).

        Returns
        -------
        CryptoBot
            Retorna self para permitir method chaining.

        Notes
        -----
        El mejor modelo se auto-selecciona basado en F1-score.
        Resultados de comparación disponibles en self.model_comparison.

        Raises
        ------
        RuntimeError
            Si no se ha ejecutado select_strategy() previamente.
        """
        self._require_features()
        self._require_strategy()
        # TODO: Implementar
        # 1. Crear target (y) basado en la estrategia seleccionada
        #    - trend_following: ¿precio sube más de X% en N períodos?
        #    - mean_reversion: ¿precio regresa a SMA en N períodos?
        #    - momentum: ¿el momentum continúa en N períodos?
        # 2. Preparar X (features) e y (target)
        # 3. Split temporal según window type:
        #    - expanding: TimeSeriesSplit
        #    - sliding: rolling window manual
        # 4. Para cada modelo:
        #    a. Fit con pipeline (StandardScaler + modelo)
        #    b. Calcular métricas: accuracy, precision, recall, f1, auc
        # 5. Crear DataFrame comparativo (self.model_comparison)
        # 6. Auto-seleccionar mejor modelo por F1
        # 7. Guardar en self.model, self.model_name, self.model_metrics
        # 8. Print tabla comparativa
        pass

    def optimize_model(self, model_name: Optional[str] = None) -> "CryptoBot":
        """
        Optimiza hiperparámetros del modelo usando GridSearchCV.

        Parameters
        ----------
        model_name : str, optional
            Modelo a optimizar. Si None, optimiza el mejor modelo
            de train_models(). Opciones: "logistic_regression",
            "svm", "random_forest", "xgboost".

        Returns
        -------
        CryptoBot
            Retorna self para permitir method chaining.

        Notes
        -----
        Usa TimeSeriesSplit como cv para respetar temporalidad.
        Los grids de hiperparámetros están predefinidos por modelo.

        Raises
        ------
        RuntimeError
            Si no se ha ejecutado train_models() previamente.
        """
        self._require_model()
        # TODO: Implementar
        # 1. Definir param_grid según modelo:
        #    - random_forest: n_estimators, max_depth, min_samples_split
        #    - xgboost: n_estimators, max_depth, learning_rate
        #    - svm: C, gamma, kernel
        #    - logistic_regression: C, penalty
        # 2. GridSearchCV con TimeSeriesSplit
        # 3. Refit con mejores parámetros
        # 4. Actualizar self.model y self.model_metrics
        # 5. Print mejores parámetros y mejora en métricas
        pass

    def feature_importance(self, top_n: int = 15) -> pd.DataFrame:
        """
        Muestra las features más importantes del modelo entrenado.

        Parameters
        ----------
        top_n : int, default 15
            Número de features a mostrar.

        Returns
        -------
        pd.DataFrame
            DataFrame con columnas: feature, importance, ordenado descendente.

        Raises
        ------
        RuntimeError
            Si no se ha ejecutado train_models() previamente.
        """
        self._require_model()
        # TODO: Implementar
        # 1. Extraer importances del modelo:
        #    - Tree-based: .feature_importances_
        #    - Linear: .coef_
        #    - SVM: usar permutation_importance
        # 2. Crear DataFrame ordenado
        # 3. Retornar top_n
        pass

    def plot_feature_importance(self, top_n: int = 15) -> None:
        """
        Gráfico de barras horizontal con las features más importantes.

        Parameters
        ----------
        top_n : int, default 15
            Número de features a mostrar.

        Raises
        ------
        RuntimeError
            Si no se ha ejecutado train_models() previamente.
        """
        self._require_model()
        # TODO: Implementar
        # 1. Llamar self.feature_importance(top_n)
        # 2. Plotly horizontal bar chart
        # 3. Colores Binance
        pass

    # ════════════════════════════════════════════════════════════
    #  5. SIGNALS
    # ════════════════════════════════════════════════════════════

    def get_signals(self) -> "CryptoBot":
        """
        Genera señales de trading usando el modelo entrenado.

        Las señales respetan el régimen de mercado:
        - Si el régimen actual es desfavorable para la estrategia → HOLD
        - Si el modelo predice oportunidad → BUY (1) o SELL (-1)
        - Si no hay oportunidad → HOLD (0)

        Returns
        -------
        CryptoBot
            Retorna self para permitir method chaining.

        Notes
        -----
        Las señales se guardan en self.signals como pd.Series.
        Risk management (stop_loss, take_profit, position_sizing)
        se aplica en la etapa de ejecución, no en la generación de señales.

        Raises
        ------
        RuntimeError
            Si no se ha ejecutado train_models() previamente.
            Si no se ha ejecutado detect_regime() previamente.
        """
        self._require_model()
        self._require_regime()
        # TODO: Implementar
        # 1. Verificar si el régimen actual es favorable para la estrategia
        # 2. Si no es favorable: todas las señales = HOLD, con warning
        # 3. Si es favorable: predecir con self.model sobre datos recientes
        # 4. Convertir predicciones a señales: BUY (1), SELL (-1), HOLD (0)
        # 5. Guardar en self.signals
        # 6. Print resumen: última señal, confianza, fecha
        pass

    # ════════════════════════════════════════════════════════════
    #  6. BACKTESTING
    # ════════════════════════════════════════════════════════════

    def backtest(self, cash: float = 10_000, commission: float = 0.001) -> "CryptoBot":
        """
        Ejecuta backtest de la estrategia usando backtesting.py.

        Parameters
        ----------
        cash : float, default 10_000
            Capital inicial para la simulación.
        commission : float, default 0.001
            Comisión por trade (0.1% default, típico de exchanges crypto).

        Returns
        -------
        CryptoBot
            Retorna self para permitir method chaining.

        Notes
        -----
        Métricas disponibles en self.backtest_results:
        - Total Return, Sharpe Ratio, Max Drawdown
        - Win Rate, # Trades, Avg Trade Duration
        - Equity curve

        Raises
        ------
        RuntimeError
            Si no se han generado señales con get_signals().
        """
        self._require_signals()
        # TODO: Implementar
        # 1. Crear Strategy class de backtesting.py que use self.signals
        # 2. Instanciar Backtest(self.data, strategy, cash, commission)
        # 3. bt.run()
        # 4. Guardar stats en self.backtest_results
        # 5. Guardar bt object en self._bt_object (para plot)
        # 6. Print métricas clave formateadas
        pass

    def backtest_plot(self) -> None:
        """
        Muestra el gráfico interactivo de backtesting.py.

        Incluye: equity curve, drawdown, señales de entrada/salida,
        y precio del activo.

        Raises
        ------
        RuntimeError
            Si no se ha ejecutado backtest() previamente.
        """
        if self._bt_object is None:
            raise RuntimeError(
                "❌ No hay backtest ejecutado. Ejecuta bot.backtest() primero."
            )
        # TODO: Implementar
        # self._bt_object.plot()
        pass

    # ════════════════════════════════════════════════════════════
    #  7. VISUALIZATION
    # ════════════════════════════════════════════════════════════

    def plot_price(self) -> None:
        """
        Gráfico de velas japonesas (candlestick) con Plotly.

        Muestra OHLC data con volumen en panel inferior.
        Colores: Binance brand (verde/rojo).

        Raises
        ------
        RuntimeError
            Si no se ha ejecutado fetch_data() previamente.
        """
        self._require_data()
        # TODO: Implementar
        # 1. go.Candlestick con OHLC
        # 2. Panel de volumen abajo
        # 3. Colores Binance
        # 4. Layout responsive
        pass

    def plot_signals(self) -> None:
        """
        Gráfico de precio con señales de trading marcadas.

        BUY señales: triángulos verdes (▲)
        SELL señales: triángulos rojos (▼)
        Overlay sobre candlestick chart.

        Raises
        ------
        RuntimeError
            Si no se han generado señales con get_signals().
        """
        self._require_signals()
        # TODO: Implementar
        # 1. Base: candlestick chart
        # 2. Overlay: markers para BUY (green up triangle)
        # 3. Overlay: markers para SELL (red down triangle)
        pass

    def plot_performance(self) -> None:
        """
        Gráfico de equity curve y drawdown.

        Panel superior: equity curve vs buy-and-hold.
        Panel inferior: drawdown.

        Raises
        ------
        RuntimeError
            Si no se ha ejecutado backtest() previamente.
        """
        if self.backtest_results is None:
            raise RuntimeError(
                "❌ No hay backtest ejecutado. Ejecuta bot.backtest() primero."
            )
        # TODO: Implementar
        # 1. Equity curve del bot vs buy-and-hold
        # 2. Panel de drawdown
        # 3. Anotaciones: max drawdown, final return
        pass

    # ════════════════════════════════════════════════════════════
    #  8. PAPER TRADING (TESTNET)
    # ════════════════════════════════════════════════════════════

    def connect_testnet(self, api_key: str, api_secret: str) -> "CryptoBot":
        """
        Conecta al testnet del exchange para paper trading.

        Parameters
        ----------
        api_key : str
            API key del testnet (NO usar keys de producción).
        api_secret : str
            API secret del testnet.

        Returns
        -------
        CryptoBot
            Retorna self para permitir method chaining.

        Notes
        -----
        Cada exchange tiene su propio testnet:
        - Bybit: testnet.bybit.com
        - Binance: testnet.binance.vision
        - OKX: demo mode dentro de la plataforma

        ⚠️  NUNCA uses API keys de tu cuenta real aquí.
        """
        # TODO: Implementar
        # 1. Crear nueva instancia de exchange con sandbox=True
        # 2. Configurar api_key y api_secret
        # 3. Verificar conexión con fetch_balance()
        # 4. self._testnet_connected = True
        # 5. Print balance del testnet
        pass

    def execute(self, mode: str = "paper") -> dict:
        """
        Ejecuta la señal más reciente.

        Parameters
        ----------
        mode : str, default "paper"
            Modo de ejecución:
            - "paper": ejecuta en testnet (requiere connect_testnet())
            - "live": ⛔ DESHABILITADO — solo para referencia educativa

        Returns
        -------
        dict
            Información del trade ejecutado:
            {type, symbol, amount, price, stop_loss, take_profit, timestamp}

        Notes
        -----
        Risk management se aplica automáticamente:
        - Position size basado en max_position_pct
        - Stop loss basado en stop_loss_pct
        - Take profit basado en take_profit_pct

        Raises
        ------
        RuntimeError
            Si no hay señales o no está conectado al testnet.
        ValueError
            Si mode="live" (no permitido en este curso).
        """
        if mode == "live":
            raise ValueError(
                "⛔ Modo 'live' deshabilitado. Este bot es educativo. "
                "Usa mode='paper' con el testnet."
            )
        self._require_signals()
        self._require_testnet()
        # TODO: Implementar
        # 1. Obtener última señal
        # 2. Si HOLD: print "Sin acción" y retornar
        # 3. Si BUY o SELL:
        #    a. Calcular position size (balance * max_position_pct)
        #    b. Calcular stop_loss y take_profit prices
        #    c. Ejecutar orden via CCXT: create_market_order(...)
        #    d. Registrar trade en self.trades
        #    e. Print confirmación del trade
        # 4. Retornar dict con detalles del trade
        pass

    def status(self) -> None:
        """
        Muestra estado actual del bot en testnet.

        Incluye:
        - Balance actual (USDT + cripto)
        - Posiciones abiertas
        - Último trade ejecutado
        - P&L no realizado
        - P&L total

        Raises
        ------
        RuntimeError
            Si no está conectado al testnet.
        """
        self._require_testnet()
        # TODO: Implementar
        # 1. fetch_balance()
        # 2. Posiciones abiertas (si las hay)
        # 3. Último trade de self.trades
        # 4. Calcular P&L
        # 5. Print formateado
        pass

    # ════════════════════════════════════════════════════════════
    #  9. TRADE HISTORY & LOGGING
    # ════════════════════════════════════════════════════════════

    def trade_history(self) -> pd.DataFrame:
        """
        Retorna historial de trades como DataFrame.

        Returns
        -------
        pd.DataFrame
            Columnas: timestamp, type (BUY/SELL), symbol, amount,
            price, stop_loss, take_profit, pnl, status.

        Notes
        -----
        Exportable con: bot.trade_history().to_csv("mis_trades.csv")
        """
        if not self.trades:
            print("📭 No hay trades registrados aún.")
            return pd.DataFrame()

        return pd.DataFrame(self.trades)

    # ════════════════════════════════════════════════════════════
    #  10. MULTI-SYMBOL SCANNER
    # ════════════════════════════════════════════════════════════

    def scan(self, symbols: list[str] = None) -> pd.DataFrame:
        """
        Escanea múltiples criptomonedas y muestra régimen + señal de cada una.

        Parameters
        ----------
        symbols : list of str, optional
            Lista de símbolos a escanear.
            Default: ["BTC", "ETH", "SOL", "BNB", "XRP"]

        Returns
        -------
        pd.DataFrame
            Tabla con columnas: Symbol, Regime, Confidence,
            Signal, Price, Change_24h.

        Examples
        --------
        >>> bot.scan()
        >>> bot.scan(symbols=["BTC", "ETH", "SOL", "AVAX", "DOGE"])

        Notes
        -----
        Este método crea instancias temporales de CryptoBot para cada
        símbolo. Usa la misma configuración (timeframe, exchange) del
        bot actual pero NO modifica su estado.
        """
        if symbols is None:
            symbols = ["BTC", "ETH", "SOL", "BNB", "XRP"]

        # TODO: Implementar
        # 1. Para cada symbol:
        #    a. Crear CryptoBot temporal con misma config
        #    b. fetch_data (últimos 30 días para rapidez)
        #    c. create_features()
        #    d. detect_regime()
        #    e. Recopilar: symbol, regime, confidence, price, change_24h
        # 2. Construir DataFrame resumen
        # 3. Print tabla formateada
        # 4. Retornar DataFrame
        pass

    # ════════════════════════════════════════════════════════════
    #  11. PERSISTENCE
    # ════════════════════════════════════════════════════════════

    def save(self, name: str, path: str = ".") -> None:
        """
        Guarda el estado completo del bot a disco.

        Guarda: modelo entrenado, configuración, régimen, features,
        historial de trades, y métricas. NO guarda datos OHLCV crudos
        (se pueden re-descargar con fetch_data).

        Parameters
        ----------
        name : str
            Nombre del archivo (sin extensión).
            Se guarda como {name}.pkl
        path : str, default "."
            Directorio donde guardar. En Colab usa
            "/content/drive/MyDrive/" para persistencia.

        Examples
        --------
        >>> bot.save("mi_bot_v1")
        >>> bot.save("mi_bot_v1", path="/content/drive/MyDrive/bots/")
        """
        # TODO: Implementar
        # 1. Crear dict con estado completo:
        #    - config: symbol, timeframe, exchange, risk params
        #    - model: self.model, self.model_name, self.model_metrics
        #    - regime: self.regime, self.regime_model, self.regime_probabilities
        #    - strategy: self.selected_strategy
        #    - trades: self.trades
        #    - model_comparison: self.model_comparison
        # 2. joblib.dump(state, f"{path}/{name}.pkl")
        # 3. Print confirmación con tamaño del archivo
        pass

    def load(self, name: str, path: str = ".") -> "CryptoBot":
        """
        Carga estado previamente guardado.

        Parameters
        ----------
        name : str
            Nombre del archivo (sin extensión).
        path : str, default "."
            Directorio donde buscar.

        Returns
        -------
        CryptoBot
            Retorna self para permitir method chaining.

        Notes
        -----
        Después de cargar, aún necesitas ejecutar fetch_data() para
        obtener datos actualizados. El modelo y configuración se
        restauran automáticamente.

        Examples
        --------
        >>> bot = CryptoBot()
        >>> bot.load("mi_bot_v1")
        >>> bot.fetch_data()  # datos frescos
        >>> bot.get_signals()  # usa modelo cargado
        """
        # TODO: Implementar
        # 1. joblib.load(f"{path}/{name}.pkl")
        # 2. Restaurar todos los atributos del state
        # 3. Print resumen de lo que se cargó
        pass

    # ════════════════════════════════════════════════════════════
    #  12. DISPLAY
    # ════════════════════════════════════════════════════════════

    def __repr__(self) -> str:
        """Representación del bot para print/display."""
        status_parts = [
            f"CryptoBot('{self.symbol}', {self.timeframe}, {self.exchange_id})",
            f"  Data:     {'✅ ' + str(len(self.data)) + ' registros' if self.data is not None else '❌ No cargada'}",
            f"  Features: {'✅ ' + str(len(self.features.columns)) + ' columnas' if self.features is not None else '❌ No creados'}",
            f"  Régimen:  {self.regime if self.regime else '❌ No detectado'}",
            f"  Strategy: {self.selected_strategy if self.selected_strategy else '❌ No seleccionada'}",
            f"  Modelo:   {self.model_name if self.model_name else '❌ No entrenado'}",
            f"  Señales:  {'✅' if self.signals is not None else '❌ No generadas'}",
            f"  Testnet:  {'✅ Conectado' if self._testnet_connected else '❌ No conectado'}",
            f"  Trades:   {len(self.trades)} registrados",
        ]
        return "\n".join(status_parts)
