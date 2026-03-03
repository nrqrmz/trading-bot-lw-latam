"""Mixin para visualización: gráficos de precio, señales y performance."""

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .config import CHART_HEIGHT_MAIN, CHART_HEIGHT_SECONDARY, CHART_ROW_HEIGHTS
from .constants import COLOR_PALETTE


class VisualizationMixin:
    """Métodos de visualización: plot_price(), plot_signals(), plot_performance()."""

    def plot_price(self) -> None:
        """
        Gráfico de velas japonesas (candlestick) con Plotly.

        Muestra OHLC data con volumen en panel inferior.
        Colores: paleta definida en COLOR_PALETTE.

        Raises
        ------
        RuntimeError
            Si no se ha ejecutado fetch_data() previamente.
        """
        self._require_data()
        df = self.data

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            row_heights=CHART_ROW_HEIGHTS,
            vertical_spacing=0.05,
        )

        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                increasing_line_color=COLOR_PALETTE["green"],
                decreasing_line_color=COLOR_PALETTE["red"],
                name="OHLC",
            ),
            row=1,
            col=1,
        )

        colors = [
            COLOR_PALETTE["green"] if c >= o else COLOR_PALETTE["red"]
            for c, o in zip(df["Close"], df["Open"])
        ]
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df["Volume"],
                marker_color=colors,
                name="Volumen",
                showlegend=False,
            ),
            row=2,
            col=1,
        )

        fig.update_layout(
            title=f"Precio — {getattr(self, 'symbol', '')}",
            yaxis_title="Precio (USDT)",
            yaxis2_title="Volumen",
            template="plotly_dark",
            plot_bgcolor=COLOR_PALETTE["dark"],
            paper_bgcolor=COLOR_PALETTE["dark"],
            xaxis_rangeslider_visible=False,
            height=CHART_HEIGHT_MAIN,
        )

        fig.show()

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
        df = self.data
        signals = self.signals

        common_idx = df.index.intersection(signals.index)
        signals = signals.loc[common_idx]
        df = df.loc[common_idx]

        fig = go.Figure()

        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                increasing_line_color=COLOR_PALETTE["green"],
                decreasing_line_color=COLOR_PALETTE["red"],
                name="OHLC",
            )
        )

        buy_idx = signals[signals == 1].index
        if len(buy_idx) > 0:
            fig.add_trace(
                go.Scatter(
                    x=buy_idx,
                    y=df.loc[buy_idx, "Low"],
                    mode="markers",
                    marker=dict(
                        symbol="triangle-up",
                        size=12,
                        color=COLOR_PALETTE["green"],
                    ),
                    name="BUY",
                )
            )

        sell_idx = signals[signals == -1].index
        if len(sell_idx) > 0:
            fig.add_trace(
                go.Scatter(
                    x=sell_idx,
                    y=df.loc[sell_idx, "High"],
                    mode="markers",
                    marker=dict(
                        symbol="triangle-down",
                        size=12,
                        color=COLOR_PALETTE["red"],
                    ),
                    name="SELL",
                )
            )

        fig.update_layout(
            title=f"Señales de Trading — {getattr(self, 'symbol', '')}",
            yaxis_title="Precio (USDT)",
            template="plotly_dark",
            plot_bgcolor=COLOR_PALETTE["dark"],
            paper_bgcolor=COLOR_PALETTE["dark"],
            xaxis_rangeslider_visible=False,
            height=CHART_HEIGHT_SECONDARY,
        )

        fig.show()

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

        eq = self.backtest_results["_equity_curve"]
        stats = self.backtest_results

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            row_heights=CHART_ROW_HEIGHTS,
            vertical_spacing=0.05,
        )

        fig.add_trace(
            go.Scatter(
                x=eq.index,
                y=eq["Equity"],
                mode="lines",
                name="Bot",
                line=dict(color=COLOR_PALETTE["yellow"], width=2),
            ),
            row=1,
            col=1,
        )

        close = self.data["Close"].loc[eq.index[0] : eq.index[-1]]
        initial_equity = eq["Equity"].iloc[0]
        bh = close / close.iloc[0] * initial_equity
        fig.add_trace(
            go.Scatter(
                x=bh.index,
                y=bh,
                mode="lines",
                name="Buy & Hold",
                line=dict(color="white", width=1.5, dash="dash"),
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=eq.index,
                y=eq["DrawdownPct"] * -100,
                mode="lines",
                fill="tozeroy",
                name="Drawdown",
                line=dict(color=COLOR_PALETTE["red"], width=1),
                fillcolor="rgba(246, 70, 93, 0.3)",
            ),
            row=2,
            col=1,
        )

        max_dd = stats.get("Max. Drawdown [%]", 0)
        final_return = stats.get("Return [%]", 0)

        fig.update_layout(
            title=f"Performance — {getattr(self, 'symbol', '')}",
            yaxis_title="Equity (USDT)",
            yaxis2_title="Drawdown (%)",
            template="plotly_dark",
            plot_bgcolor=COLOR_PALETTE["dark"],
            paper_bgcolor=COLOR_PALETTE["dark"],
            height=CHART_HEIGHT_MAIN,
            annotations=[
                dict(
                    xref="paper",
                    yref="paper",
                    x=0.01,
                    y=0.95,
                    text=f"Retorno: {final_return:.1f}%",
                    showarrow=False,
                    font=dict(color=COLOR_PALETTE["yellow"], size=14),
                ),
                dict(
                    xref="paper",
                    yref="paper",
                    x=0.01,
                    y=0.88,
                    text=f"Max Drawdown: {max_dd:.1f}%",
                    showarrow=False,
                    font=dict(color=COLOR_PALETTE["red"], size=14),
                ),
            ],
        )

        fig.show()
