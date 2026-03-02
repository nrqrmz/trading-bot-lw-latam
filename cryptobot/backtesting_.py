"""Mixin para backtesting de estrategias."""


class BacktestMixin:
    """Métodos de backtesting: backtest() y backtest_plot()."""

    def backtest(self, cash: float = 10_000, commission: float = 0.001) -> "BacktestMixin":
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

        from backtesting import Strategy
        from backtesting.lib import FractionalBacktest
        from .constants import STRATEGY_REGISTRY

        # Alinear datos OHLCV con el índice de señales
        bt_data = self.data.loc[self.signals.index].copy()

        # Capturar señales para la inner class
        signal_values = self.signals.values

        class SignalStrategy(Strategy):
            signal_array = signal_values

            def init(self):
                pass

            def next(self):
                idx = len(self.data) - 1
                if idx < len(self.signal_array):
                    signal = self.signal_array[idx]
                    if signal == 1 and not self.position:
                        self.buy()
                    elif signal == -1 and self.position:
                        self.sell()
                # Cerrar posición abierta en la última barra
                if idx == len(self.signal_array) - 1 and self.position:
                    self.position.close()

        bt = FractionalBacktest(
            bt_data, SignalStrategy,
            cash=cash, commission=commission,
            exclusive_orders=True,
        )
        stats = bt.run()

        self.backtest_results = stats
        self._bt_object = bt

        # --- Imprimir métricas ---
        strategy_name = STRATEGY_REGISTRY.get(
            self.strategy, {}
        ).get("name", self.strategy)

        print("=" * 60)
        print(f"📈 BACKTEST — {strategy_name}")
        print("=" * 60)

        print(f"\n  Capital inicial:   ${cash:,.2f}")
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
