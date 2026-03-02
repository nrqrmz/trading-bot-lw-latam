"""Mixin para backtesting de estrategias."""

VALID_LEVERAGE = (1, 2, 3, 5, 10, 20, 50, 100)


class BacktestMixin:
    """Métodos de backtesting: backtest() y backtest_plot()."""

    def backtest(self, cash: float = 10_000, commission: float = 0.001,
                 position_pct: float = 100, leverage: int = 1) -> "BacktestMixin":
        """
        Ejecuta backtest de la estrategia usando backtesting.py.

        Parameters
        ----------
        cash : float, default 10_000
            Capital inicial para la simulación.
        commission : float, default 0.001
            Comisión por trade (0.1% default, típico de exchanges crypto).
        position_pct : float, default 100
            Porcentaje del capital a usar por trade (0-100).
            Ej: 5 = 5% del capital por trade.
        leverage : int, default 1
            Apalancamiento a usar. Debe ser uno de: 1, 2, 3, 5, 10, 20, 50, 100.
            Ej: 10 = 10x leverage (margin = 10%).

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

        Exposure efectiva = position_pct × leverage.
        Ej: position_pct=5, leverage=10 → exposure 50% del capital.

        Raises
        ------
        RuntimeError
            Si no se han generado señales con get_signals().
        ValueError
            Si position_pct no está entre 0 y 100, o leverage no es válido.
        """
        self._require_signals()

        # --- Validaciones ---
        if not (0 < position_pct <= 100):
            raise ValueError(
                f"❌ position_pct debe ser mayor a 0 y menor o igual a 100, "
                f"recibido: {position_pct}"
            )
        if leverage not in VALID_LEVERAGE:
            raise ValueError(
                f"❌ leverage debe ser uno de {VALID_LEVERAGE}, "
                f"recibido: {leverage}"
            )

        from backtesting import Backtest, Strategy
        from .constants import STRATEGY_REGISTRY

        # Convertir parámetros para backtesting.py
        size = position_pct / 100       # ej: 5 → 0.05
        if size >= 1:
            size = 0.9999  # Asegurar interpretación como fracción del equity
        margin = 1 / leverage           # ej: 10x → 0.1

        # Alinear datos OHLCV con el índice de señales
        bt_data = self.data.loc[self.signals.index].copy()

        # Capturar señales para la inner class
        signal_values = self.signals.values

        class SignalStrategy(Strategy):
            signal_array = signal_values
            _size = size

            def init(self):
                pass

            def next(self):
                idx = len(self.data) - 1
                if idx < len(self.signal_array):
                    signal = self.signal_array[idx]
                    if signal == 1 and not self.position:
                        self.buy(size=self._size)
                    elif signal == -1 and self.position:
                        self.sell()
                # Cerrar posición abierta en la última barra
                if idx == len(self.signal_array) - 1 and self.position:
                    self.position.close()

        avg_price = bt_data["Close"].mean()
        use_fractional = avg_price > cash

        if use_fractional:
            from backtesting.lib import FractionalBacktest
            bt = FractionalBacktest(
                bt_data, SignalStrategy,
                cash=cash, commission=commission, margin=margin,
                exclusive_orders=True,
                finalize_trades=True,
            )
        else:
            bt = Backtest(
                bt_data, SignalStrategy,
                cash=cash, commission=commission, margin=margin,
                exclusive_orders=True,
                finalize_trades=True,
            )
        stats = bt.run()

        self.backtest_results = stats
        self._bt_object = bt

        # --- Imprimir métricas ---
        strategy_name = STRATEGY_REGISTRY.get(
            self.selected_strategy, {}
        ).get("name", self.selected_strategy)

        exposure = position_pct * leverage

        print("=" * 60)
        print(f"📈 BACKTEST — {strategy_name}")
        print("=" * 60)

        print(f"\n  Capital inicial:   ${cash:,.2f}")
        print(f"  Posición/trade:    {position_pct:.1f}%")
        print(f"  Leverage:          {leverage}x")
        print(f"  Exposure efectiva: {exposure:.1f}%")
        print(f"  Capital final:     ${stats['Equity Final [$]']:,.2f}")
        print(f"  Retorno total:     {stats['Return [%]']:+.2f}%")
        print(f"  Buy & Hold:        {stats['Buy & Hold Return [%]']:+.2f}%")
        print(f"  Sharpe Ratio:      {stats['Sharpe Ratio']:.2f}")
        print(f"  Max Drawdown:      {stats['Max. Drawdown [%]']:.2f}%")

        n_trades = stats["# Trades"]
        print(f"\n  Trades:            {n_trades}")

        if n_trades > 0:
            print(f"  Win Rate:          {stats['Win Rate [%]']:.1f}%")
            print(f"  Mejor trade:       {stats['Best Trade [%]']:+.2f}%")
            print(f"  Peor trade:        {stats['Worst Trade [%]']:+.2f}%")
            print(f"  Duración promedio: {stats['Avg. Trade Duration']}")
            print(f"  Profit Factor:     {stats['Profit Factor']:.2f}")

        print("=" * 60)

        return self

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
        self._bt_object.plot(open_browser=True)
