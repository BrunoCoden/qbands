# grafico_live.py
# ---------------------------------------------------------
# Dibuja velas + Range Breakout en vivo (solo una corrida).
# - Carga .env
# - Baja datos de Binance
# - Reusa la lógica de velas_TV_sin_sdk.py (bandas + flechas, sin círculos)
# ---------------------------------------------------------

import os
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import mplfinance as mpf

# Importá las funciones tal como las definiste en tu script principal
from velas_tv_sin_sdk import (
    fetch_klines,
    compute_range_breakout,
    TZ_NAME,        # para coherencia de zona horaria
    SYMBOL_DISPLAY, # si querés forzar desde .env, abajo se sobrescribe
)

# ================== Config (desde .env) ==================
load_dotenv()

SYMBOL_DISPLAY = os.getenv("SYMBOL", SYMBOL_DISPLAY)  # por si querés sobreescribir
API_SYMBOL     = SYMBOL_DISPLAY.replace(".P", "")
INTERVAL       = os.getenv("INTERVAL", "30m")
LIMIT          = int(os.getenv("LIMIT", "800"))

RB_MULTI        = float(os.getenv("RB_MULTI", "4.0"))
RB_LB           = int(os.getenv("RB_LB", "10"))
RB_FILTER_TREND = os.getenv("RB_FILTER_TREND", "false").lower() == "true"

# ================== Helpers (idénticos a velas_TV_sin_sdk.py) ==================
def _linebreak_like(series: pd.Series) -> pd.Series:
    """Imita plot.style_linebr: solo mantiene tramos horizontales continuos."""
    s = series.copy()
    prev = s.shift(1)
    s[(~prev.isna()) & (s != prev)] = np.nan
    return s

def _series_from_points(index, points):
    """Convierte lista de puntos [(ts, price), ...] a Series indexada para scatter."""
    s = pd.Series(np.nan, index=index)
    for ts, price in points:
        # buscar índice más cercano
        pos = s.index.get_indexer([ts], method='nearest')[0]
        s.iloc[pos] = price
    return s

def _style_tv():
    mc = mpf.make_marketcolors(up='green', down='red', edge='inherit', wick='inherit', volume='in')
    return mpf.make_mpf_style(marketcolors=mc, gridstyle='--', y_on_right=False)

# ================== Plot ==================
def build_addplots(df: pd.DataFrame, indi: dict):
    """
    Construye la misma superposición que en velas_TV_sin_sdk.py:
    - value_upper (verde), value (gris), value_lower (magenta)
    - upper_mid / lower_mid (gris semi), flechas buy/sell
    """
    ap = [
        mpf.make_addplot(_linebreak_like(indi['value_upper']), color='#1dac70', width=1),
        mpf.make_addplot(_linebreak_like(indi['value']),       color='gray',    width=1),
        mpf.make_addplot(_linebreak_like(indi['value_lower']), color='#df3a79', width=1),
        mpf.make_addplot(_linebreak_like(indi['upper_mid']),   color='gray',    width=1, alpha=0.5),
        mpf.make_addplot(_linebreak_like(indi['lower_mid']),   color='gray',    width=1, alpha=0.5),
    ]

    # SOLO flechas (sin círculos), como en la versión final de velas_TV_sin_sdk.py
    buy_s  = _series_from_points(df.index, indi['arrows_buy'])
    sell_s = _series_from_points(df.index, indi['arrows_sell'])

    ap += [
        mpf.make_addplot(buy_s,  type='scatter', marker='^', markersize=60, color='#1dac70'),
        mpf.make_addplot(sell_s, type='scatter', marker='v', markersize=60, color='#df3a79'),
    ]
    return ap

def plot_live():
    """Descarga velas y grafica Range Breakout con la misma lógica que velas_TV_sin_sdk.py."""
    # Bajamos datos crudos
    df = fetch_klines(API_SYMBOL, INTERVAL, LIMIT)
    # Calculamos indicador sobre OHLCV
    indi = compute_range_breakout(
        df[["Open","High","Low","Close","Volume"]],
        multi=RB_MULTI, lb=RB_LB, filter_trend=RB_FILTER_TREND
    )

    # Armamos overlays con la MISMA estética/lógica del script principal
    ap = build_addplots(df, indi)

    # Plot
    mpf.plot(
        df[["Open","High","Low","Close","Volume"]],
        type="candle",
        style=_style_tv(),
        addplot=ap,
        figsize=(12,6),
        datetime_format="%Y-%m-%d %H:%M",
        warn_too_much_data=len(df)+1,
        title=f"{SYMBOL_DISPLAY} {INTERVAL} — Range Breakout (live)"
    )

if __name__ == "__main__":
    plot_live()
