"""Mixin para data pipeline: descarga y resumen de datos OHLCV."""

from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from .sentiment import fetch_fear_greed_index


class DataMixin:
    """Métodos de data pipeline: fetch_data() y summary()."""

    def _fetch_ohlcv_paginated(self, pair: str, since_ts: int, end_ts: int) -> list:
        """
        Fetch paginado de OHLCV para un par dado.

        Parameters
        ----------
        pair : str
            Par de trading (e.g., "BTC/USDT").
        since_ts : int
            Timestamp de inicio en milisegundos.
        end_ts : int
            Timestamp de fin en milisegundos.

        Returns
        -------
        list
            Lista de candles [timestamp, open, high, low, close, volume].
        """
        all_candles = []
        since = since_ts
        limit = 1000

        while since < end_ts:
            batch = self._exchange.fetch_ohlcv(
                pair, self.timeframe, since=since, limit=limit
            )
            if not batch:
                break
            batch = [c for c in batch if c[0] <= end_ts]
            all_candles.extend(batch)
            if len(batch) < limit:
                break
            since = batch[-1][0] + 1

        return all_candles

    @staticmethod
    def _candles_to_dataframe(candles: list) -> pd.DataFrame:
        """Convierte lista de candles OHLCV a DataFrame indexado por Date."""
        df = pd.DataFrame(
            candles, columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"]
        )
        df["Date"] = pd.to_datetime(df["Timestamp"], unit="ms")
        df = df.set_index("Date").drop(columns=["Timestamp"])
        df = df[~df.index.duplicated(keep="last")].sort_index()
        return df

    def fetch_data(
        self,
        last_n: int = 200,
        start: Optional[str] = None,
        end: Optional[str] = None,
        pair_symbol: Optional[str] = None,
        fear_greed: bool = False,
    ) -> "DataMixin":
        """
        Obtiene datos OHLCV del exchange via CCXT.

        Soporta dos modos:
        - Por número de velas: bot.fetch_data(last_n=200)
        - Por rango de fechas: bot.fetch_data(start="2024-01-01", end="2024-12-31")

        La ventana temporal se calcula automáticamente según el timeframe:
        - last_n=200 con timeframe="1h" → últimas 200 horas (~8 días)
        - last_n=200 con timeframe="4h" → últimas 800 horas (~33 días)
        - last_n=200 con timeframe="1d" → últimos 200 días (~6.5 meses)

        Las columnas del DataFrame siguen el formato requerido
        por backtesting.py: Open, High, Low, Close, Volume.

        Parameters
        ----------
        last_n : int, default 200
            Número de velas hacia atrás desde hoy. Se ignora si start/end están definidos.
        start : str, optional
            Fecha de inicio en formato "YYYY-MM-DD".
        end : str, optional
            Fecha de fin en formato "YYYY-MM-DD". Default: hoy.
        pair_symbol : str, optional
            Símbolo del par secundario para stat_arb (e.g., "ETH").
            Se construye como "{pair_symbol}/USDT".
        fear_greed : bool, default False
            Si True, descarga el Fear & Greed Index de alternative.me
            y lo agrega como columna ``fgi_value`` (0-100).

        Returns
        -------
        CryptoBot
            Retorna self para permitir method chaining.

        Examples
        --------
        >>> bot.fetch_data()
        >>> bot.fetch_data(last_n=500)
        >>> bot.fetch_data(start="2024-01-01", end="2024-06-30")
        >>> bot.fetch_data(last_n=500, pair_symbol="ETH")
        """
        # ── Calcular timestamps ────────────────────────
        if start is not None:
            since_timestamp = int(
                datetime.strptime(start, "%Y-%m-%d").timestamp() * 1000
            )
            if end is not None:
                end_timestamp = int(
                    datetime.strptime(end, "%Y-%m-%d").timestamp() * 1000
                )
            else:
                end_timestamp = int(datetime.now().timestamp() * 1000)
        else:
            end_timestamp = int(datetime.now().timestamp() * 1000)
            tf_hours = {"15m": 0.25, "30m": 0.5, "1h": 1, "4h": 4, "1d": 24}
            hours = last_n * tf_hours[self.timeframe]
            since_timestamp = int(
                (datetime.now() - timedelta(hours=hours)).timestamp() * 1000
            )

        # ── Fetch principal (paginado) ────────────────
        all_candles = self._fetch_ohlcv_paginated(
            self._pair, since_timestamp, end_timestamp
        )

        # ── Caso sin datos ────────────────────────────
        if not all_candles:
            print(f"❌ No se obtuvieron datos para {self._pair}")
            return self

        # ── Construir DataFrame ───────────────────────
        df = self._candles_to_dataframe(all_candles)
        self.data = df

        # ── Resumen ───────────────────────────────────
        print(f"📊 {self._pair} | {self.timeframe}")
        print(f"   Período: {df.index[0]:%Y-%m-%d} → {df.index[-1]:%Y-%m-%d}")
        print(f"   Registros: {len(df)}")
        print(f"   Precio actual: ${df['Close'].iloc[-1]:,.2f}")

        # ── Fear & Greed Index (opcional) ────────────
        if fear_greed:
            if self.symbol != "BTC":
                print(f"⚠️ FGI está basado en BTC — se usa como proxy para {self.symbol}")

            # Calcular días necesarios del rango de datos
            days_needed = (df.index[-1] - df.index[0]).days + 2
            fgi_df = fetch_fear_greed_index(limit=days_needed)

            if fgi_df is not None and not fgi_df.empty:
                # merge_asof: alinear datos diarios → cualquier timeframe
                # direction="backward" evita look-ahead bias
                df = df.sort_index()
                fgi_df = fgi_df.sort_index()
                df = pd.merge_asof(
                    df,
                    fgi_df,
                    left_index=True,
                    right_index=True,
                    direction="backward",
                )
                self.data = df
                self.fear_greed_enabled = True

                # Clasificación del FGI actual
                current_fgi = int(df["fgi_value"].iloc[-1])
                fgi_labels = {
                    (0, 25): "Extreme Fear",
                    (25, 50): "Fear",
                    (50, 75): "Greed",
                    (75, 101): "Extreme Greed",
                }
                fgi_label = next(
                    label for (lo, hi), label in fgi_labels.items()
                    if lo <= current_fgi < hi
                )
                print(f"😱 Fear & Greed Index: {current_fgi}/100 ({fgi_label})")
            else:
                print("⚠️ No se pudo obtener FGI — continuando sin sentimiento")

        # ── Fetch par secundario (stat_arb) ───────────
        if pair_symbol is not None:
            pair_pair = f"{pair_symbol.upper()}/USDT"
            pair_candles = self._fetch_ohlcv_paginated(
                pair_pair, since_timestamp, end_timestamp
            )

            if not pair_candles:
                print(f"❌ No se obtuvieron datos para par secundario {pair_pair}")
            else:
                pair_df = self._candles_to_dataframe(pair_candles)
                self.pair_data = pair_df
                self.pair_symbol = pair_symbol.upper()

                print(f"📊 Par secundario: {pair_pair}")
                print(f"   Período: {pair_df.index[0]:%Y-%m-%d} → {pair_df.index[-1]:%Y-%m-%d}")
                print(f"   Registros: {len(pair_df)}")
                print(f"   Precio actual: ${pair_df['Close'].iloc[-1]:,.2f}")

        return self

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

        df = self.data
        first_close = df["Close"].iloc[0]
        last_close = df["Close"].iloc[-1]
        change_pct = (last_close - first_close) / first_close * 100
        period_high = df["High"].max()
        period_low = df["Low"].min()

        print(f"\n{'═' * 50}")
        print(f"  📊 Resumen: {self._pair} | {self.timeframe}")
        print(f"{'═' * 50}")
        print(f"  Período:    {df.index[0]:%Y-%m-%d} → {df.index[-1]:%Y-%m-%d}")
        print(f"  Registros:  {len(df)}")
        print(f"  Precio actual: ${last_close:,.2f}")
        print(f"  Cambio:     {change_pct:+.2f}%")
        print(f"  High:       ${period_high:,.2f}")
        print(f"  Low:        ${period_low:,.2f}")
        print(f"{'─' * 50}")
        print(df.describe())
