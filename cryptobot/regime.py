"""Mixin para market intelligence: régimen de mercado y estrategias."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import ta as ta_lib
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from .constants import COLOR_PALETTE, REGIME_LABELS, STRATEGY_REGISTRY


class RegimeMixin:
    """Métodos de market intelligence: detect_regime(), regime_report(), recommend_strategies(), select_strategy()."""

    def detect_regime(self, n_regimes: int = 3) -> "RegimeMixin":
        """
        Detecta el régimen de mercado actual usando pipeline PCA + GMM.

        Pipeline hedge-fund quality:
        1. 86+ features via ta.add_all_ta_features() + custom features
        2. Exclusión de features inútiles (binarias, precio absoluto, acumulativos)
        3. Limpieza (ffill, inf→NaN, drop columnas >50% NaN, dropna filas)
        4. VarianceThreshold para eliminar features near-constant
        5. Filtro de correlación (|r| > 0.95)
        6. PCA retiene 95% de varianza
        7. GMM robusto (n_init=10, full covariance)

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

        df = self.data.copy()
        n = len(df)

        # ── Constantes de ventana ────────────────────────
        SHORT, MEDIUM, LONG = 7, 21, 50
        STRUCTURAL_DESIRED = 200
        long_structural = max(min(STRUCTURAL_DESIRED, n - LONG), LONG)

        # Columnas a excluir del clustering
        EXCLUDE_COLS = {
            # Raw OHLCV
            "Open", "High", "Low", "Close", "Volume",
            # Binarias (0/1 — no informativas para clustering continuo)
            "volatility_bbhi", "volatility_bbli",
            "volatility_kchi", "volatility_kcli",
            "trend_psar_up_indicator", "trend_psar_down_indicator",
            # Precio absoluto (scale-dependent, redundante con %B/width)
            "trend_ema_fast", "trend_ema_slow",
            "trend_sma_fast", "trend_sma_slow",
            "volatility_bbh", "volatility_bbl", "volatility_bbm",
            "volatility_kch", "volatility_kcl", "volatility_kcc",
            "volatility_dch", "volatility_dcl", "volatility_dcm",
            "trend_ichimoku_conv", "trend_ichimoku_base",
            "trend_ichimoku_a", "trend_ichimoku_b",
            "trend_visual_ichimoku_a", "trend_visual_ichimoku_b",
            "trend_psar_up", "trend_psar_down",
            "trend_dpo", "momentum_kama",
            # Acumulativos (no estacionarios, sin techo)
            "volume_adi", "volume_obv", "volume_vpt",
            "volume_nvi", "others_cr", "volume_vwap",
        }

        # ── 1. Features desde ta library (86+) ──────────
        df_ta = ta_lib.add_all_ta_features(
            df, open="Open", high="High", low="Low", close="Close", volume="Volume"
        )

        # ── 2. Features custom (no incluidos en ta) ─────
        returns = df_ta["Close"].pct_change()
        df_ta["trend_7"] = returns.rolling(SHORT).mean()
        df_ta["trend_21"] = returns.rolling(MEDIUM).mean()
        df_ta["trend_50"] = returns.rolling(LONG).mean()
        df_ta["vol_7"] = returns.rolling(SHORT).std()
        df_ta["vol_21"] = returns.rolling(MEDIUM).std()
        df_ta["vol_50"] = returns.rolling(LONG).std()

        # Garman-Klass volatility
        log_hl = np.log(df_ta["High"] / df_ta["Low"]) ** 2
        log_co = np.log(df_ta["Close"] / df_ta["Open"]) ** 2
        df_ta["gk_volatility"] = np.sqrt(
            (0.5 * log_hl - (2 * np.log(2) - 1) * log_co).rolling(MEDIUM).mean()
        )

        df_ta["vol_ratio"] = df_ta["vol_7"] / df_ta["vol_50"]
        df_ta["volume_ratio"] = df_ta["Volume"] / df_ta["Volume"].rolling(MEDIUM).mean()

        sma_structural = df_ta["Close"].rolling(long_structural).mean()
        df_ta["dist_sma_structural"] = (df_ta["Close"] - sma_structural) / sma_structural

        rolling_high = df_ta["Close"].rolling(long_structural).max()
        df_ta["drawdown"] = (df_ta["Close"] - rolling_high) / rolling_high

        # Guardar returns para mapeo de clusters
        df_ta["_returns"] = returns

        # ── 3. Seleccionar features para GMM ─────────────
        exclude = EXCLUDE_COLS | {"_returns"}
        feature_cols = [c for c in df_ta.columns if c not in exclude]
        regime_features = df_ta[feature_cols].copy()

        # ── 4. Limpieza ─────────────────────────────────
        regime_features.ffill(inplace=True)
        regime_features.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Drop columnas con >50% NaN
        thresh = len(regime_features) * 0.5
        regime_features.dropna(axis=1, thresh=int(thresh), inplace=True)

        # Drop filas con NaN restantes (warmup de indicadores)
        regime_features.dropna(inplace=True)

        n_raw = regime_features.shape[1]

        if len(regime_features) < n_regimes * 5:
            import warnings
            warnings.warn(
                f"Solo {len(regime_features)} filas válidas para {n_regimes} regímenes. "
                f"Considere usar más datos (last_days más alto)."
            )

        # ── 5. Curación automatizada ─────────────────────
        # 5a. Escalar
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(regime_features)

        # 5b. VarianceThreshold — elimina features near-constant
        var_selector = VarianceThreshold(threshold=1e-10)
        X_selected = var_selector.fit_transform(X_scaled)

        # 5c. Correlación — de cada par con |r| > 0.95, eliminar uno
        corr = pd.DataFrame(X_selected).corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
        X_curated = np.delete(X_selected, to_drop, axis=1)

        n_after_curation = X_curated.shape[1]

        # ── 6. PCA — retener 95% varianza ───────────────
        pca = PCA(n_components=0.95, random_state=42)
        X_pca = pca.fit_transform(X_curated)

        # ── 7. GMM robusto ───────────────────────────────
        gmm = GaussianMixture(
            n_components=n_regimes,
            covariance_type="full",
            n_init=10,
            random_state=42,
        )
        gmm.fit(X_pca)

        # ── 8. Predecir y mapear clusters → labels ──────
        labels = gmm.predict(X_pca)

        # Usar returns alineados para mapeo semántico
        aligned_returns = df_ta.loc[regime_features.index, "_returns"]
        cluster_returns = pd.DataFrame({"cluster": labels, "returns": aligned_returns.values})
        mean_returns = cluster_returns.groupby("cluster")["returns"].mean()

        sorted_clusters = mean_returns.sort_values().index.tolist()
        cluster_to_regime = {
            sorted_clusters[0]: 0,   # Bear
            sorted_clusters[1]: 1,   # Sideways
            sorted_clusters[2]: 2,   # Bull
        }

        # Asignar régimen mapeado a self.features (alineación por índice)
        mapped_labels = pd.Series(
            [cluster_to_regime[c] for c in labels],
            index=regime_features.index,
        )
        self.features["regime"] = np.nan
        common_idx = self.features.index.intersection(mapped_labels.index)
        self.features.loc[common_idx, "regime"] = mapped_labels.loc[common_idx].values

        # ── 9. Probabilidades del último período ─────────
        last_point = X_pca[-1].reshape(1, -1)
        proba = gmm.predict_proba(last_point)[0]

        regime_probs = {}
        for cluster_id, regime_id in cluster_to_regime.items():
            label = REGIME_LABELS[regime_id]
            regime_probs[label] = round(proba[cluster_id], 4)

        # ── 10. Guardar estado ───────────────────────────
        last_regime_id = cluster_to_regime[labels[-1]]
        regime_label = REGIME_LABELS[last_regime_id]
        self.regime = regime_label.split()[0]  # "Bull", "Bear", o "Sideways"
        self.regime_probabilities = regime_probs
        self.regime_model = gmm

        # ── 11. Print informativo ────────────────────────
        confidence = proba[labels[-1]]
        print(f"📊 Régimen detectado: {regime_label} (confianza: {confidence:.1%})")
        print(f"   Pipeline: {n_raw} features → {n_after_curation} curados → {X_pca.shape[1]} PCA")
        print(f"   Períodos analizados: {len(regime_features)} de {n} disponibles")
        print(f"   BIC: {gmm.bic(X_pca):.0f}")

        return self

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

        # 1. Tabla de probabilidades
        print("=" * 50)
        print("📊 REPORTE DE RÉGIMEN DE MERCADO")
        print("=" * 50)
        emoji = {"Bull": "🟢", "Bear": "🔴", "Sideways": "🟡"}.get(self.regime, "")
        print(f"\nRégimen actual: {self.regime} {emoji}")
        print(f"\n{'Régimen':<20} {'Probabilidad':>12}")
        print("-" * 34)
        for label, prob in self.regime_probabilities.items():
            bar = "█" * int(prob * 20)
            print(f"{label:<20} {prob:>10.1%}  {bar}")

        # 2. Estadísticas por régimen
        df = self.features.dropna(subset=["regime"]).copy()
        print(f"\n{'Régimen':<20} {'Retorno Prom':>14} {'Volatilidad':>14} {'Períodos':>10} {'Duración Prom':>15}")
        print("-" * 75)

        for regime_id, label in REGIME_LABELS.items():
            mask = df["regime"] == regime_id
            subset = df[mask]
            if len(subset) == 0:
                continue

            avg_return = subset["returns"].mean()
            avg_vol = subset["volatility_20"].mean()
            n_periods = len(subset)

            # Duración promedio (rachas consecutivas)
            regime_series = mask.astype(int)
            changes = regime_series.diff().fillna(0) != 0
            groups = changes.cumsum()
            streaks = regime_series.groupby(groups).sum()
            streaks = streaks[streaks > 0]
            avg_duration = streaks.mean() if len(streaks) > 0 else 0

            print(
                f"{label:<20} {avg_return:>13.4%} {avg_vol:>13.4f} {n_periods:>10} {avg_duration:>14.1f}"
            )

        # 3. Gráfico Plotly: precio con background coloreado por régimen
        regime_colors = {
            0: COLOR_PALETTE["red"],     # Bear
            1: COLOR_PALETTE["yellow"],  # Sideways
            2: COLOR_PALETTE["green"],   # Bull
        }

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df["Close"],
            mode="lines",
            name="Precio",
            line=dict(color="white", width=1.5),
        ))

        # Agregar rectángulos coloreados por régimen
        regime_col = df["regime"].dropna()
        if len(regime_col) > 0:
            current_regime = regime_col.iloc[0]
            start_idx = regime_col.index[0]

            for i in range(1, len(regime_col)):
                if regime_col.iloc[i] != current_regime or i == len(regime_col) - 1:
                    end_idx = regime_col.index[i]
                    fig.add_vrect(
                        x0=start_idx,
                        x1=end_idx,
                        fillcolor=regime_colors.get(int(current_regime), COLOR_PALETTE["gray"]),
                        opacity=0.15,
                        line_width=0,
                    )
                    current_regime = regime_col.iloc[i]
                    start_idx = regime_col.index[i]

        fig.update_layout(
            title=f"Precio con Regímenes de Mercado — {getattr(self, 'symbol', '')}",
            xaxis_title="Fecha",
            yaxis_title="Precio (USDT)",
            template="plotly_dark",
            plot_bgcolor=COLOR_PALETTE["dark"],
            paper_bgcolor=COLOR_PALETTE["dark"],
            height=500,
        )

        fig.show()

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

        df = self.features.dropna(subset=["regime"]).copy()

        # 1. Generar señales desde OHLCV completo (self.data tiene más historia)
        ohlcv = self.data.copy()
        strategy_signals = {}

        # Trend Following — SMA Crossover
        sma_short = ohlcv["Close"].rolling(20).mean()
        sma_long = ohlcv["Close"].rolling(50).mean()
        tf_signal = pd.Series(0, index=ohlcv.index)
        tf_signal[sma_short > sma_long] = 1
        tf_signal[sma_short < sma_long] = -1
        strategy_signals["trend_following"] = tf_signal.reindex(df.index).fillna(0)

        # Mean Reversion — Bollinger Bands
        bb_mid = ohlcv["Close"].rolling(20).mean()
        bb_std = ohlcv["Close"].rolling(20).std()
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std
        mr_signal = pd.Series(0, index=ohlcv.index)
        mr_signal[ohlcv["Close"] < bb_lower] = 1
        mr_signal[ohlcv["Close"] > bb_upper] = -1
        strategy_signals["mean_reversion"] = mr_signal.reindex(df.index).fillna(0)

        # Momentum — RSI + Volume
        delta = ohlcv["Close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        volume_change = ohlcv["Volume"].pct_change()
        mom_signal = pd.Series(0, index=ohlcv.index)
        mom_signal[(rsi > 50) & (volume_change > 0)] = 1
        mom_signal[(rsi < 50) & (volume_change < 0)] = -1
        strategy_signals["momentum"] = mom_signal.reindex(df.index).fillna(0)

        # 2. Filtrar a períodos del régimen actual
        regime_name_to_id = {"Bull": 2, "Bear": 0, "Sideways": 1}
        current_regime_id = regime_name_to_id[self.regime]
        regime_mask = df["regime"] == current_regime_id
        returns = df["returns"]

        # 3. Calcular métricas por estrategia
        results = []
        for key, info in STRATEGY_REGISTRY.items():
            signals = strategy_signals[key]

            # Filtrar solo períodos del régimen actual
            regime_signals = signals[regime_mask]
            regime_returns = returns[regime_mask]

            strategy_returns = regime_signals.shift(1) * regime_returns
            strategy_returns = strategy_returns.dropna()

            # Sharpe Ratio (anualizado para crypto: 365 días)
            if strategy_returns.std() != 0 and len(strategy_returns) > 0:
                sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(365)
            else:
                sharpe = 0.0

            # Win Rate
            active_returns = strategy_returns[strategy_returns != 0]
            if len(active_returns) > 0:
                win_rate = (active_returns > 0).sum() / len(active_returns)
            else:
                win_rate = 0.0

            # Total Return
            total_return = (1 + strategy_returns).cumprod().iloc[-1] - 1 if len(strategy_returns) > 0 else 0.0

            # Recomendación basada en régimen
            if self.regime in info["best_regimes"]:
                recommendation = "🟢 Recomendada"
            elif self.regime in info["worst_regimes"]:
                recommendation = "🔴 No recomendada"
            else:
                recommendation = "🟡 Neutral"

            results.append({
                "key": key,
                "name": info["name"],
                "sharpe": sharpe,
                "win_rate": win_rate,
                "total_return": total_return,
                "recommendation": recommendation,
            })

        # 4. Rankear por Sharpe ratio (descendente)
        results.sort(key=lambda x: x["sharpe"], reverse=True)

        # 5. Print tabla formateada
        print("=" * 85)
        print(f"📈 ESTRATEGIAS RECOMENDADAS — Régimen: {self.regime}")
        print("=" * 85)
        print(f"\n{'#':<4} {'Estrategia':<35} {'Sharpe':>8} {'Win Rate':>10} {'Return':>10} {'Señal'}")
        print("-" * 85)

        for i, r in enumerate(results, 1):
            print(
                f"{i:<4} {r['name']:<35} {r['sharpe']:>8.2f} {r['win_rate']:>9.1%} "
                f"{r['total_return']:>9.2%}  {r['recommendation']}"
            )

    def select_strategy(self, strategy: str) -> "RegimeMixin":
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
