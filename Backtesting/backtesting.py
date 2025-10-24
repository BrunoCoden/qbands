"""
Backtesting engine for the Q-bands strategy driven by TouchUpperQ/TouchLowerQ.

This version fetches raw candles from Binance (both stream interval and channel
interval), rebuilds the channel values, simulates the order logic, reports the
performance, and optionally renders a price chart with all closed trades.

Usage example:
    python Backtesting/backtesting.py --weeks 8 --plot-path Backtesting/trades.png
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys
from typing import Iterable, List, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from zoneinfo import ZoneInfo

# Allow imports from repo root when executed as a script.
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from paginado_binance import INTERVAL_MS, fetch_klines_paginado

load_dotenv()

# Environment-driven defaults (mirrors the live streamer settings)
SYMBOL_DISPLAY = os.getenv("SYMBOL", "ETHUSDT.P").strip()
STREAM_INTERVAL = os.getenv("STREAM_INTERVAL", "1m").strip()
CHANNEL_INTERVAL = os.getenv("CHANNEL_INTERVAL", "30m").strip()
TZ_NAME = os.getenv("TZ", "America/Argentina/Buenos_Aires").strip()
RB_MULTI = float(os.getenv("RB_MULTI", "4.0"))
RB_INIT_BAR = int(os.getenv("RB_INIT_BAR", "301"))

DEFAULT_MONTHS = 2
TREND_CONFIRM_BARS = 30  # candles that must respect the 100-period trend
BREAKEVEN_TRIGGER_PCT = 0.02  # move stop to breakeven after 2% in favor


def _slugify_label(label: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", label).strip("-_")
    return slug.lower()


def _resolve_run_label(initial: Optional[str]) -> str:
    label = (initial or "").strip()
    if not label and sys.stdin.isatty():
        try:
            label = input("Nombre para la versión de salida (enter usa timestamp): ").strip()
        except EOFError:
            label = ""
    if not label:
        label = datetime.now().strftime("%Y%m%d-%H%M%S")
    return label


def _labelled_path(path: Path, label: str) -> Path:
    if not label:
        return path
    return path.with_name(f"{path.stem}_{label}{path.suffix}")


def _to_float(value) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _linebreak_like(series: pd.Series) -> pd.Series:
    s = series.copy()
    prev = s.shift(1)
    s[(~prev.isna()) & (s != prev)] = float("nan")
    return s


def _style_tv_dark():
    mc = mpf.make_marketcolors(
        up="lime",
        down="red",
        edge="inherit",
        wick="white",
        volume="in",
    )
    return mpf.make_mpf_style(
        marketcolors=mc,
        base_mpf_style="nightclouds",
        facecolor="black",
        edgecolor="black",
        gridcolor="#333333",
        gridstyle="--",
        rc={
            "axes.labelcolor": "white",
            "xtick.color": "white",
            "ytick.color": "white",
        },
        y_on_right=False,
    )


def _has_data(series: Optional[pd.Series]) -> bool:
    if series is None or series.empty:
        return False
    try:
        arr = pd.to_numeric(series, errors="coerce").to_numpy()
    except Exception:
        return False
    if arr.size == 0:
        return False
    return np.isfinite(arr).any()


# === Channel computation (same logic as the live script) =====================
def _rma(x: pd.Series, length: int) -> pd.Series:
    alpha = 1.0 / length
    return x.ewm(alpha=alpha, adjust=False).mean()


def _atr(df: pd.DataFrame, length: int) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    prev_c = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return _rma(tr, length)


def compute_channels(df: pd.DataFrame, multi: float, init_bar: int) -> pd.DataFrame:
    df = df.copy()
    df["hl2"] = (df["High"] + df["Low"]) / 2.0
    atr200 = _atr(df, 200)
    width = atr200.rolling(100, min_periods=1).mean() * multi

    n = len(df)
    value = np.full(n, np.nan)
    vup = np.full(n, np.nan)
    vlo = np.full(n, np.nan)
    umid = np.full(n, np.nan)
    lmid = np.full(n, np.nan)

    highs = df["High"].values
    lows = df["Low"].values
    hl2 = df["hl2"].values
    w = width.values

    def crossed_up(prev_val, curr_val, prev_lvl, curr_lvl):
        return (prev_val <= prev_lvl) and (curr_val > curr_lvl)

    def crossed_dn(prev_val, curr_val, prev_lvl, curr_lvl):
        return (prev_val >= prev_lvl) and (curr_val < curr_lvl)

    count = 0
    for i in range(n):
        if i == init_bar:
            value[i] = hl2[i]
            vup[i] = hl2[i] + w[i]
            vlo[i] = hl2[i] - w[i]
            umid[i] = (value[i] + vup[i]) / 2.0
            lmid[i] = (value[i] + vlo[i]) / 2.0
        else:
            if i > 0:
                value[i] = value[i - 1]
                vup[i] = vup[i - 1]
                vlo[i] = vlo[i - 1]
                umid[i] = umid[i - 1]
                lmid[i] = lmid[i - 1]

        if i < max(init_bar, 1):
            continue

        cross_up = crossed_up(lows[i - 1], lows[i], vup[i - 1], vup[i])
        cross_down = crossed_dn(highs[i - 1], highs[i], vlo[i - 1], vlo[i])

        if not np.isnan(vup[i]) and not np.isnan(vlo[i]):
            if (lows[i] > vup[i]) or (highs[i] < vlo[i]):
                count += 1

        if cross_up or cross_down or (count == 100):
            count = 0
            value[i] = hl2[i]
            vup[i] = hl2[i] + w[i]
            vlo[i] = hl2[i] - w[i]
            umid[i] = (value[i] + vup[i]) / 2.0
            lmid[i] = (value[i] + vlo[i]) / 2.0

    upper_q = (umid + vup) / 2.0
    lower_q = (lmid + vlo) / 2.0

    return pd.DataFrame(
        {
            "Value": value,
            "ValueUpper": vup,
            "ValueLower": vlo,
            "UpperMid": umid,
            "LowerMid": lmid,
            "UpperQ": upper_q,
            "LowerQ": lower_q,
        },
        index=df.index,
    )


def _align_channels_to_stream(ch: pd.DataFrame, idx1m: pd.DatetimeIndex) -> pd.DataFrame:
    if ch is None or ch.empty:
        return pd.DataFrame(index=idx1m)
    want = ["Value", "ValueUpper", "ValueLower", "UpperMid", "LowerMid", "UpperQ", "LowerQ"]
    for col in want:
        if col not in ch.columns:
            ch[col] = pd.NA
    out = ch[want].reindex(idx1m.union(ch.index)).sort_index().ffill()
    return out.reindex(idx1m)


# === Data acquisition ========================================================
def _to_utc(ts_like, tz: ZoneInfo) -> pd.Timestamp:
    ts = pd.to_datetime(ts_like)
    if ts.tzinfo is None:
        ts = ts.tz_localize(tz)
    return ts.tz_convert("UTC")


def _calc_total_bars(interval: str, start_ms: int, end_ms: int, buffer: int = 5) -> int:
    ms_per = INTERVAL_MS.get(interval)
    if ms_per is None:
        raise ValueError(f"Intervalo no soportado: {interval}")
    diff = max(end_ms - start_ms, ms_per)
    return int(diff // ms_per) + 1 + buffer


def fetch_history(
    api_symbol: str,
    *,
    months: Optional[int],
    weeks: Optional[int],
    stream_interval: str,
    channel_interval: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    tz = ZoneInfo(TZ_NAME)

    end_utc = pd.Timestamp.now(tz="UTC")
    if end:
        end_utc = _to_utc(end, tz)

    if start:
        start_utc = _to_utc(start, tz)
    else:
        weeks = weeks if weeks and weeks > 0 else None
        months = months if months and months > 0 else None
        if weeks:
            start_utc = end_utc - pd.DateOffset(weeks=weeks)
        else:
            month_span = months if months else DEFAULT_MONTHS
            start_utc = end_utc - pd.DateOffset(months=month_span)

    if start_utc >= end_utc:
        raise ValueError("La fecha inicial debe ser menor que la final.")

    start_ms = int(start_utc.timestamp() * 1000)
    end_ms = int(end_utc.timestamp() * 1000)

    stream_bars = _calc_total_bars(stream_interval, start_ms, end_ms, buffer=25)
    stream_df = fetch_klines_paginado(
        api_symbol,
        stream_interval,
        stream_bars,
        start_ms=start_ms,
        end_ms=end_ms,
    ).sort_index()

    if stream_df.empty:
        raise RuntimeError("No se pudieron descargar velas del stream interval.")

    local_start = start_utc.tz_convert(tz)
    local_end = end_utc.tz_convert(tz)
    stream_df = stream_df[(stream_df.index >= local_start) & (stream_df.index <= local_end)]

    if stream_df.empty:
        raise RuntimeError("No hay velas en el rango solicitado.")

    channel_ms = INTERVAL_MS.get(channel_interval)
    if channel_ms is None:
        raise ValueError(f"Intervalo de canales no soportado: {channel_interval}")

    warmup_bars = RB_INIT_BAR + 50
    channel_start_ms = max(0, start_ms - warmup_bars * channel_ms)
    channel_bars = _calc_total_bars(channel_interval, channel_start_ms, end_ms, buffer=25)

    channel_df = fetch_klines_paginado(
        api_symbol,
        channel_interval,
        channel_bars,
        start_ms=channel_start_ms,
        end_ms=end_ms,
    ).sort_index()

    if channel_df.empty:
        raise RuntimeError("No se pudieron descargar velas del canal.")

    channel_local = channel_df.index
    channel_cut = channel_df[channel_local <= local_end]
    ohlc = channel_cut[["Open", "High", "Low", "Close", "Volume"]]
    channels = compute_channels(ohlc, multi=RB_MULTI, init_bar=RB_INIT_BAR)

    aligned = _align_channels_to_stream(channels, stream_df.index)
    trend_close = channel_cut["Close"]
    trend_sma = trend_close.rolling(window=100, min_periods=1).mean()
    trend_sma_aligned = trend_sma.reindex(stream_df.index).ffill().bfill()

    df = stream_df.copy()
    df["Date"] = df["CloseTimeDT"]
    df["Value"] = aligned.get("Value")
    for col in ["UpperMid", "ValueUpper", "LowerMid", "ValueLower", "UpperQ", "LowerQ"]:
        df[col] = aligned.get(col)
    df["TrendSMA"] = trend_sma_aligned

    upperq = df["UpperQ"]
    lowerq = df["LowerQ"]
    lows = df["Low"]
    highs = df["High"]

    df["TouchUpperQ"] = ((upperq.notna()) & (lows <= upperq) & (upperq <= highs)).astype(int)
    df["TouchLowerQ"] = ((lowerq.notna()) & (lows <= lowerq) & (lowerq <= highs)).astype(int)

    wanted_cols = [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Value",
        "UpperMid",
        "ValueUpper",
        "LowerMid",
        "ValueLower",
        "UpperQ",
        "LowerQ",
        "TrendSMA",
        "TouchUpperQ",
        "TouchLowerQ",
    ]

    return df[wanted_cols].reset_index(drop=True)


# === Backtester core =========================================================
@dataclass
class LimitOrder:
    side: str
    price: float
    context: str  # "upper" or "lower"
    created_at: pd.Timestamp


@dataclass
class Position:
    side: str  # "long" or "short"
    context: str
    entry_price: float
    entry_time: pd.Timestamp
    profit_pct: float  # decimal (e.g. 0.012 → 1.2%)
    stop_pct: float    # decimal
    tp_price: float
    sl_price: float
    bars_in_trade: int = 0
    breakeven_active: bool = False


@dataclass
class TradeResult:
    side: str
    context: str
    entry_price: float
    exit_price: float
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    exit_reason: str  # "target_hit" or "stop_hit"
    profit_pct: float
    stop_pct: float
    pnl_pct: float
    bars_held: int


class Backtester:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        trend_series = self.df.get('TrendSMA')
        if trend_series is not None:
            trend_series = pd.to_numeric(trend_series, errors="coerce")
        self._trend_sma = trend_series
        close_series = self.df.get("Close")
        if close_series is not None:
            close_series = pd.to_numeric(close_series, errors="coerce")
        self._close_series = close_series
        self.pending_orders: List[LimitOrder] = []
        self.open_positions: List[Position] = []
        self.closed_trades: List[TradeResult] = []

    def run(self) -> None:
        for idx, row in self.df.iterrows():
            ts = row["Date"]
            high = _to_float(row.get("High"))
            low = _to_float(row.get("Low"))

            if high is None or low is None:
                continue

            row_ctx = {
                "timestamp": ts,
                "high": high,
                "low": low,
                "upper_mid": _to_float(row.get("UpperMid")),
                "lower_mid": _to_float(row.get("LowerMid")),
                "value_upper": _to_float(row.get("ValueUpper")),
                "value_lower": _to_float(row.get("ValueLower")),
                "touch_upper": _to_int(row.get("TouchUpperQ")),
                "touch_lower": _to_int(row.get("TouchLowerQ")),
                "row_index": idx,
            }

            self._evaluate_positions(row_ctx)
            self._evaluate_orders(row_ctx, row)
            self._maybe_replace_orders(row_ctx)

    def _close_conflicting_positions(self, new_side: str, price: float, timestamp: pd.Timestamp) -> None:
        remaining: List[Position] = []
        for pos in self.open_positions:
            if pos.side != new_side:
                trade = self._force_close_position(pos, price, timestamp, reason="opposite_signal")
                self.closed_trades.append(trade)
            else:
                remaining.append(pos)
        self.open_positions = remaining

    def _force_close_position(self, pos: Position, exit_price: float, timestamp: pd.Timestamp, reason: str) -> TradeResult:
        if pos.side == "long":
            pnl_pct = ((exit_price - pos.entry_price) / pos.entry_price) * 100.0
        else:
            pnl_pct = ((pos.entry_price - exit_price) / pos.entry_price) * 100.0
        bars_held = pos.bars_in_trade + 1
        return TradeResult(
            side=pos.side,
            context=pos.context,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            entry_time=pos.entry_time,
            exit_time=timestamp,
            exit_reason=reason,
            profit_pct=pos.profit_pct * 100.0,
            stop_pct=pos.stop_pct * 100.0,
            pnl_pct=pnl_pct,
            bars_held=bars_held,
        )

    def _evaluate_positions(self, row_ctx: dict) -> None:
        still_open: List[Position] = []
        for pos in self.open_positions:
            self._maybe_move_stop_to_breakeven(pos, row_ctx)
            exit_event = self._position_exit(pos, row_ctx)
            if exit_event:
                self.closed_trades.append(exit_event)
            else:
                pos.bars_in_trade += 1
                still_open.append(pos)
        self.open_positions = still_open

    def _maybe_move_stop_to_breakeven(self, pos: Position, row_ctx: dict) -> None:
        if pos.breakeven_active:
            return
        high = row_ctx["high"]
        low = row_ctx["low"]
        if pos.side == "long":
            trigger_price = pos.entry_price * (1.0 + BREAKEVEN_TRIGGER_PCT)
            if high >= trigger_price:
                pos.sl_price = max(pos.sl_price, pos.entry_price)
                pos.breakeven_active = True
        else:
            trigger_price = pos.entry_price * (1.0 - BREAKEVEN_TRIGGER_PCT)
            if low <= trigger_price:
                pos.sl_price = min(pos.sl_price, pos.entry_price)
                pos.breakeven_active = True

    def _position_exit(self, pos: Position, row_ctx: dict) -> Optional[TradeResult]:
        high = row_ctx["high"]
        low = row_ctx["low"]
        ts = row_ctx["timestamp"]

        if pos.side == "long":
            stop_hit = low <= pos.sl_price
            target_hit = high >= pos.tp_price
            if stop_hit and target_hit:
                exit_price = pos.sl_price
                reason = "stop_hit"
            elif stop_hit:
                exit_price = pos.sl_price
                reason = "stop_hit"
            elif target_hit:
                exit_price = pos.tp_price
                reason = "target_hit"
            else:
                return None
            pnl_pct = ((exit_price - pos.entry_price) / pos.entry_price) * 100.0
        else:
            stop_hit = high >= pos.sl_price
            target_hit = low <= pos.tp_price
            if stop_hit and target_hit:
                exit_price = pos.sl_price
                reason = "stop_hit"
            elif stop_hit:
                exit_price = pos.sl_price
                reason = "stop_hit"
            elif target_hit:
                exit_price = pos.tp_price
                reason = "target_hit"
            else:
                return None
            pnl_pct = ((pos.entry_price - exit_price) / pos.entry_price) * 100.0

        bars_held = pos.bars_in_trade + 1
        return TradeResult(
            side=pos.side,
            context=pos.context,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            entry_time=pos.entry_time,
            exit_time=ts,
            exit_reason=reason,
            profit_pct=pos.profit_pct * 100.0,
            stop_pct=pos.stop_pct * 100.0,
            pnl_pct=pnl_pct,
            bars_held=bars_held,
        )

    def _evaluate_orders(self, row_ctx: dict, row: pd.Series) -> None:
        active_next: List[LimitOrder] = []
        for order in self.pending_orders:
            if self._order_filled(order, row_ctx):
                pos = self._open_position(order, row_ctx, row)
                if pos:
                    self.open_positions.append(pos)
            else:
                active_next.append(order)
        self.pending_orders = active_next

    def _order_filled(self, order: LimitOrder, row_ctx: dict) -> bool:
        high = row_ctx["high"]
        low = row_ctx["low"]
        price = order.price
        if price is None:
            return False
        return low <= price <= high

    def _open_position(self, order: LimitOrder, row_ctx: dict, row: pd.Series) -> Optional[Position]:
        profit_pct = 0.03
        stop_pct = 0.02

        if not self._is_trend_aligned(row, order, row_ctx.get("row_index")):
            return None

        entry_price = order.price
        entry_time = row_ctx["timestamp"]

        side = "long" if order.side == "buy" else "short"
        tp_price = entry_price * (1.0 + profit_pct) if side == "long" else entry_price * (1.0 - profit_pct)
        sl_price = entry_price * (1.0 - stop_pct) if side == "long" else entry_price * (1.0 + stop_pct)

        self._close_conflicting_positions(side, entry_price, entry_time)

        return Position(
            side=side,
            context=order.context,
            entry_price=entry_price,
            entry_time=entry_time,
            profit_pct=profit_pct,
            stop_pct=stop_pct,
            tp_price=tp_price,
            sl_price=sl_price,
        )

    def _is_trend_aligned(self, row: pd.Series, order: LimitOrder, row_index: Optional[int] = None) -> bool:
        trend_series = self._trend_sma
        if trend_series is None or trend_series.empty:
            return True
        price_close = _to_float(row.get('Close'))
        price_sma = _to_float(row.get('TrendSMA'))
        if price_close is None or price_sma is None:
            return True
        is_buy = order.side == 'buy'
        if is_buy and price_close < price_sma:
            return False
        if (not is_buy) and price_close > price_sma:
            return False

        close_series = self._close_series
        if (
            row_index is None
            or close_series is None
            or close_series.empty
        ):
            return True

        if row_index + 1 < TREND_CONFIRM_BARS:
            return False

        start = row_index - TREND_CONFIRM_BARS + 1
        closes_window = close_series.iloc[start:row_index + 1]
        trend_window = trend_series.iloc[start:row_index + 1]

        if closes_window.isna().any() or trend_window.isna().any():
            return False

        if is_buy:
            return bool((closes_window >= trend_window).all())
        return bool((closes_window <= trend_window).all())

    def _range_pct(self, row: pd.Series, context: str) -> Optional[float]:
        upper_mid = _to_float(row.get("UpperMid"))
        lower_mid = _to_float(row.get("LowerMid"))
        if upper_mid is None or lower_mid is None:
            return None
        diff = upper_mid - lower_mid
        if context == "upper":
            denom = abs(lower_mid) if lower_mid else None
        else:
            denom = abs(upper_mid) if upper_mid else None
        if not denom:
            return None
        return abs(diff / denom)

    def _maybe_replace_orders(self, row_ctx: dict) -> None:
        touches: List[LimitOrder] = []
        ts = row_ctx["timestamp"]

        if row_ctx["touch_upper"] == 1:
            touches.extend(
                self._build_orders_from_signal(
                    ts,
                    "upper",
                    [
                        ("buy", row_ctx["value_upper"]),
                        ("sell", row_ctx["upper_mid"]),
                    ],
                )
            )

        if row_ctx["touch_lower"] == 1:
            touches.extend(
                self._build_orders_from_signal(
                    ts,
                    "lower",
                    [
                        ("buy", row_ctx["lower_mid"]),
                        ("sell", row_ctx["value_lower"]),
                    ],
                )
            )

        if touches:
            self.pending_orders = touches

    def _build_orders_from_signal(
        self,
        ts: pd.Timestamp,
        context: str,
        definitions: Iterable[tuple],
    ) -> List[LimitOrder]:
        orders: List[LimitOrder] = []
        seen_prices = set()
        for side, price in definitions:
            price_f = _to_float(price)
            if price_f is None:
                continue
            rounded = round(price_f, 6)
            if rounded in seen_prices:
                continue
            if self._has_open_position_at_price(price_f):
                continue
            orders.append(LimitOrder(side=side, price=price_f, context=context, created_at=ts))
            seen_prices.add(rounded)
        return orders

    def _has_open_position_at_price(self, price: float) -> bool:
        for pos in self.open_positions:
            if abs(pos.entry_price - price) <= 1e-6:
                return True
        return False

    def summary(self) -> dict:
        trades = self.closed_trades
        total = len(trades)
        wins = sum(1 for t in trades if t.exit_reason == "target_hit")
        losses = sum(1 for t in trades if t.exit_reason == "stop_hit")
        pnl_total = sum(t.pnl_pct for t in trades)
        avg = pnl_total / total if total else 0.0

        return {
            "candles": len(self.df),
            "trades": total,
            "wins": wins,
            "losses": losses,
            "win_rate": (wins / total * 100.0) if total else 0.0,
            "total_return_pct": pnl_total,
            "avg_return_pct": avg,
            "open_positions": len(self.open_positions),
            "pending_orders": len(self.pending_orders),
        }

    def trades_dataframe(self) -> pd.DataFrame:
        if not self.closed_trades:
            return pd.DataFrame(
                columns=[
                    "side",
                    "context",
                    "entry_time",
                    "exit_time",
                    "entry_price",
                    "exit_price",
                    "exit_reason",
                    "profit_target_pct",
                    "stop_pct",
                    "pnl_pct",
                    "bars_held",
                ]
            )
        data = [
            {
                "side": t.side,
                "context": t.context,
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "exit_reason": t.exit_reason,
                "profit_target_pct": t.profit_pct,
                "stop_pct": t.stop_pct,
                "pnl_pct": t.pnl_pct,
                "bars_held": t.bars_held,
            }
            for t in self.closed_trades
        ]
        return pd.DataFrame(data)


# === Plotting ================================================================
def plot_trades(price_df: pd.DataFrame, trades_df: pd.DataFrame, output_path: Path) -> Optional[Path]:
    if trades_df.empty:
        print("No hay trades cerrados, omito la generación del gráfico.")
        return None

    if "Date" not in price_df.columns:
        raise ValueError("El DataFrame de precios debe incluir la columna 'Date'.")

    staged = price_df.copy()
    staged["Date"] = pd.to_datetime(staged["Date"])
    staged = staged.set_index("Date").sort_index()

    # OHLC data for mplfinance
    ohlc_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in ohlc_cols:
        if col not in staged.columns:
            staged[col] = np.nan
    ohlc = staged[ohlc_cols]

    addplots = []

    def add_if(series: Optional[pd.Series], **kwargs) -> None:
        if _has_data(series):
            addplots.append(mpf.make_addplot(series, **kwargs))

    value_upper = staged.get("ValueUpper")
    value_mid = staged.get("Value")
    value_lower = staged.get("ValueLower")
    upper_mid = staged.get("UpperMid")
    lower_mid = staged.get("LowerMid")
    upper_q = staged.get("UpperQ")
    lower_q = staged.get("LowerQ")

    if isinstance(value_upper, pd.Series):
        add_if(_linebreak_like(value_upper), color="#1dac70", width=1)
    if isinstance(value_mid, pd.Series):
        add_if(_linebreak_like(value_mid), color="gray", width=1)
    if isinstance(value_lower, pd.Series):
        add_if(_linebreak_like(value_lower), color="#df3a79", width=1)

    if isinstance(upper_mid, pd.Series):
        add_if(_linebreak_like(upper_mid), color="gray", width=1, alpha=0.5)
    if isinstance(lower_mid, pd.Series):
        add_if(_linebreak_like(lower_mid), color="gray", width=1, alpha=0.5)

    if isinstance(upper_q, pd.Series):
        add_if(_linebreak_like(upper_q), color="yellow", width=1, linestyle=":")
    if isinstance(lower_q, pd.Series):
        add_if(_linebreak_like(lower_q), color="yellow", width=1, linestyle=":")

    # Mark touches
    touch_upper = (
        (staged["Low"] <= upper_q) & (staged["High"] >= upper_q)
    ) if isinstance(upper_q, pd.Series) else pd.Series(False, index=staged.index)
    touch_lower = (
        (staged["Low"] <= lower_q) & (staged["High"] >= lower_q)
    ) if isinstance(lower_q, pd.Series) else pd.Series(False, index=staged.index)
    suq = pd.Series(np.nan, index=staged.index)
    slq = pd.Series(np.nan, index=staged.index)
    if isinstance(upper_q, pd.Series) and touch_upper.any():
        suq.loc[touch_upper] = upper_q.loc[touch_upper]
    if isinstance(lower_q, pd.Series) and touch_lower.any():
        slq.loc[touch_lower] = lower_q.loc[touch_lower]
    add_if(suq, type="scatter", marker="o", markersize=40, color="white")
    add_if(slq, type="scatter", marker="o", markersize=40, color="white")

    idx_tz = staged.index.tz

    def normalize_ts(value) -> pd.Timestamp:
        ts = pd.to_datetime(value)
        if idx_tz is not None:
            if ts.tzinfo is None:
                ts = ts.tz_localize(idx_tz)
            else:
                ts = ts.tz_convert(idx_tz)
        else:
            if ts.tzinfo is not None:
                ts = ts.tz_convert("UTC").tz_localize(None)
        return ts

    entries_win = pd.Series(np.nan, index=staged.index)
    exits_win = pd.Series(np.nan, index=staged.index)
    entries_loss = pd.Series(np.nan, index=staged.index)
    exits_loss = pd.Series(np.nan, index=staged.index)

    for trade in trades_df.itertuples():
        entry_time = normalize_ts(trade.entry_time)
        exit_time = normalize_ts(trade.exit_time)
        if entry_time not in staged.index or exit_time not in staged.index:
            continue
        if trade.exit_reason == "target_hit":
            entries_win.loc[entry_time] = trade.entry_price
            exits_win.loc[exit_time] = trade.exit_price
        else:
            entries_loss.loc[entry_time] = trade.entry_price
            exits_loss.loc[exit_time] = trade.exit_price

    add_if(entries_win, type="scatter", marker="^", color="#3fff9d", markersize=80)
    add_if(exits_win, type="scatter", marker="v", color="#3fff9d", markersize=80)
    add_if(entries_loss, type="scatter", marker="^", color="#ff4d4d", markersize=80)
    add_if(exits_loss, type="scatter", marker="v", color="#ff4d4d", markersize=80)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, _ = mpf.plot(
        ohlc,
        type="candle",
        style=_style_tv_dark(),
        addplot=addplots,
        volume=False,
        title=f"{SYMBOL_DISPLAY} {STREAM_INTERVAL} — Backtesting trades ({len(trades_df)} operaciones)",
        datetime_format="%Y-%m-%d %H:%M",
        warn_too_much_data=100000,
        returnfig=True,
        figratio=(16, 9),
        figscale=1.0,
        tight_layout=True,
    )
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path


# === CLI =====================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Backtesting de la estrategia Q-bands con datos en vivo de Binance.")
    parser.add_argument("--months", type=int, default=None, help="Meses de historia a descargar (si no se indica fecha de inicio).")
    parser.add_argument("--weeks", type=int, default=None, help="Semanas de historia a descargar (alternativa a meses).")
    parser.add_argument("--start", type=str, default=None, help="Fecha inicial (YYYY-MM-DD).")
    parser.add_argument("--end", type=str, default=None, help="Fecha final (YYYY-MM-DD).")
    parser.add_argument("--symbol", type=str, default=None, help="Símbolo a testear (por defecto usa SYMBOL env).")
    parser.add_argument("--stream-interval", type=str, default=None, help="Intervalo de stream (ej: 1m).")
    parser.add_argument("--channel-interval", type=str, default=None, help="Intervalo de canales (ej: 30m).")
    parser.add_argument("--plot-path", type=Path, default=Path("Backtesting/trades.png"), help="Ruta donde guardar el gráfico de trades.")
    parser.add_argument("--export-trades", type=Path, default=Path("Backtesting/trades_table.csv"), help="Ruta para exportar el detalle de trades.")
    parser.add_argument("--no-export-trades", action="store_true", help="No exportar la tabla de trades.")
    parser.add_argument("--summary-path", type=Path, default=Path("Backtesting/summary.csv"), help="Ruta para exportar métricas resumidas.")
    parser.add_argument("--no-summary", action="store_true", help="No exportar la tabla de resumen.")
    parser.add_argument("--run-label", type=str, default=None, help="Etiqueta para los archivos de salida (se solicitará si no se indica).")
    args = parser.parse_args()

    if args.months and args.months <= 0:
        parser.error("--months debe ser mayor que 0.")
    if args.weeks and args.weeks <= 0:
        parser.error("--weeks debe ser mayor que 0.")
    if args.months and args.weeks:
        parser.error("Elegí semanas o meses, no ambos al mismo tiempo.")

    symbol_display = (args.symbol or SYMBOL_DISPLAY).strip()
    api_symbol = symbol_display.replace(".P", "")
    stream_interval = (args.stream_interval or STREAM_INTERVAL).strip()
    channel_interval = (args.channel_interval or CHANNEL_INTERVAL).strip()

    history = fetch_history(
        api_symbol,
        months=args.months if args.months is not None else None,
        weeks=args.weeks if args.weeks is not None else None,
        stream_interval=stream_interval,
        channel_interval=channel_interval,
        start=args.start,
        end=args.end,
    )

    bt = Backtester(history)
    bt.run()

    run_label = _resolve_run_label(args.run_label)
    run_slug = _slugify_label(run_label)
    if not run_slug:
        run_slug = datetime.now().strftime("%Y%m%d-%H%M%S")
    print(f"Usando etiqueta de salida: {run_slug}")

    export_trades_path: Optional[Path] = None
    if (not args.no_export_trades) and args.export_trades:
        export_trades_path = _labelled_path(args.export_trades, run_slug)

    summary_path: Optional[Path] = None
    if not args.no_summary and args.summary_path:
        summary_path = _labelled_path(args.summary_path, run_slug)

    plot_path: Optional[Path] = None
    if args.plot_path:
        plot_path = _labelled_path(args.plot_path, run_slug)

    summary = bt.summary()
    print("=== Resumen Backtest ===")
    for key in (
        "candles",
        "trades",
        "wins",
        "losses",
        "win_rate",
        "total_return_pct",
        "avg_return_pct",
    ):
        value = summary[key]
        if isinstance(value, float):
            print(f"{key:>18}: {value:.2f}")
        else:
            print(f"{key:>18}: {value}")
    print(f"{'open_positions':>18}: {summary['open_positions']}")
    print(f"{'pending_orders':>18}: {summary['pending_orders']}")

    trades_df = bt.trades_dataframe()
    if export_trades_path:
        export_trades_path.parent.mkdir(parents=True, exist_ok=True)
        trades_df.to_csv(export_trades_path, index=False)
        print(f"Tabla de trades exportada a {export_trades_path}")

    if summary_path:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([summary]).to_csv(summary_path, index=False)
        print(f"Resumen exportado a {summary_path}")

    if plot_path:
        path = plot_trades(history, trades_df, plot_path)
        if path:
            print(f"Gráfico generado en {path}")


if __name__ == "__main__":
    main()
