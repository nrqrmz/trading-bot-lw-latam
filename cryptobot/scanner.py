"""Mixin para escaneo multi-símbolo."""

import io
import sys

import pandas as pd

from .config import SCANNER_LAST_N, SCANNER_SYMBOLS
from .constants import STRATEGY_REGISTRY


class ScannerMixin:
    """Métodos de scanner: scan()."""

    def scan(self, symbols: list = None, last_n: int = SCANNER_LAST_N) -> pd.DataFrame:
        """
        Escanea múltiples criptomonedas y muestra régimen + estrategia recomendada.

        Parameters
        ----------
        symbols : list of str, optional
            Lista de símbolos a escanear.
            Default: ["BTC", "ETH", "SOL", "BNB", "XRP"]
        last_n : int, default 100
            Número de velas a descargar por símbolo.

        Returns
        -------
        pd.DataFrame
            Tabla con columnas: Symbol, Price, Change_24h,
            Regime, Confidence, Strategy.

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
            symbols = list(SCANNER_SYMBOLS)

        # Importar aquí para evitar circular import
        from .bot import CryptoBot

        results = []

        print("=" * 70)
        print("🔍 SCANNER — Escaneando mercado")
        print("=" * 70)

        for symbol in symbols:
            try:
                # Suprimir output de los sub-bots
                old_stdout = sys.stdout
                sys.stdout = io.StringIO()

                try:
                    temp_bot = CryptoBot(
                        symbol=symbol,
                        timeframe=self.timeframe,
                        exchange=self.exchange_id,
                    )
                    temp_bot.fetch_data(last_n=last_n)
                    temp_bot.create_features()
                    temp_bot.detect_regime()
                finally:
                    sys.stdout = old_stdout

                # Extraer datos
                price = temp_bot.data["Close"].iloc[-1]

                # Calcular cambio 24h (últimas velas según timeframe)
                tf_hours = {"15m": 0.25, "30m": 0.5, "1h": 1, "4h": 4, "1d": 24}
                hours_per_candle = tf_hours[self.timeframe]
                candles_24h = int(24 / hours_per_candle)
                if len(temp_bot.data) > candles_24h + 1:
                    price_24h_ago = temp_bot.data["Close"].iloc[-(candles_24h + 1)]
                    change_24h = (price - price_24h_ago) / price_24h_ago * 100
                else:
                    change_24h = 0.0

                # Régimen y confianza
                regime = temp_bot.regime
                confidence = max(temp_bot.regime_probabilities.values())

                # Estrategia recomendada (primera que sea best para el régimen)
                recommended = "—"
                for key, info in STRATEGY_REGISTRY.items():
                    if regime in info["best_regimes"]:
                        recommended = key
                        break

                results.append({
                    "Symbol": symbol,
                    "Price": price,
                    "Change_24h": change_24h,
                    "Regime": regime,
                    "Confidence": confidence,
                    "Strategy": recommended,
                })

                emoji = {"Bull": "🟢", "Bear": "🔴", "Sideways": "🟡"}.get(regime, "")
                print(f"  ✅ {symbol:<6} ${price:>10,.2f}  {change_24h:>+6.1f}%  {regime} {emoji}")

            except Exception as e:
                sys.stdout = old_stdout
                print(f"  ❌ {symbol:<6} Error: {e}")
                results.append({
                    "Symbol": symbol,
                    "Price": None,
                    "Change_24h": None,
                    "Regime": "Error",
                    "Confidence": None,
                    "Strategy": "—",
                })

        print("=" * 70)

        df = pd.DataFrame(results)
        return df
