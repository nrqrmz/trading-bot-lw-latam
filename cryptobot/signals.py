"""Mixin para generación de señales de trading."""

import numpy as np
import pandas as pd

from .config import SIGNAL_CONFIDENCE_THRESHOLD
from .constants import STRATEGY_REGISTRY


class SignalsMixin:
    """Métodos de señales: get_signals()."""

    def get_signals(self) -> "SignalsMixin":
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

        strategy_info = STRATEGY_REGISTRY[self.selected_strategy]

        # ── 1. Régimen desfavorable → todas las señales HOLD ──
        if self.regime in strategy_info["worst_regimes"]:
            self.signals = pd.Series(0, index=self.features.index, name="signal")
            print(f"⚠️ Régimen {self.regime} desfavorable para {strategy_info['name']}")
            print("   Todas las señales = HOLD")
            return self

        # ── 2. Preparar features ──────────────────────────────
        X = self.features[self._feature_cols].copy()
        valid_mask = X.dropna().index
        X_clean = X.loc[valid_mask]

        # ── 3. Predecir con el modelo ─────────────────────────
        predictions = self.model.predict(X_clean.values)

        # ── 4. Convertir predicciones a señales ───────────────
        # predict 1 → BUY (1), predict 0 → SELL (-1)
        signals = np.where(predictions == 1, 1, -1)

        # ── 5. Umbral de confianza ────────────────────────────
        last_confidence = None
        if hasattr(self.model, "predict_proba"):
            probas = self.model.predict_proba(X_clean.values)
            confidence = probas.max(axis=1)
            signals = np.where(confidence < SIGNAL_CONFIDENCE_THRESHOLD, 0, signals)
            last_confidence = confidence[-1] if len(confidence) > 0 else None

        # ── 6. Crear Series alineada al index completo ────────
        self.signals = pd.Series(0, index=self.features.index, name="signal")
        self.signals.loc[valid_mask] = signals

        # ── 7. Print resumen ──────────────────────────────────
        n_buy = (self.signals == 1).sum()
        n_sell = (self.signals == -1).sum()
        n_hold = (self.signals == 0).sum()

        last_signal = self.signals.iloc[-1]
        last_date = self.signals.index[-1]
        signal_map = {1: "BUY 🟢", -1: "SELL 🔴", 0: "HOLD 🟡"}

        regime_status = "favorable" if self.regime in strategy_info["best_regimes"] else "neutral"

        print(f"📡 Señales generadas para {strategy_info['name']}")
        print(f"   BUY: {n_buy} | SELL: {n_sell} | HOLD: {n_hold}")
        print(f"   Última señal: {signal_map[last_signal]} ({last_date})")
        if last_confidence is not None:
            print(f"   Confianza última señal: {last_confidence:.1%}")
        print(f"   Régimen: {self.regime} ({regime_status})")

        return self
