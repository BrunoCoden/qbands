# grafico_de_script_principal_solo_bandas.py
import os
import numpy as np
import pandas as pd
import mplfinance as mpf

from script_principal_de_velas_solo_bandas import (
    compute_channels,
    SYMBOL_DISPLAY, API_SYMBOL,
    CHANNEL_INTERVAL, STREAM_INTERVAL,
    RB_MULTI, RB_INIT_BAR
)

# NUEVO: paginador
from paginado_binance import fetch_klines_paginado

# Cuántas velas pedir (configurable por .env)
PLOT_STREAM_BARS  = int(os.getenv("PLOT_STREAM_BARS",  "5000"))   # 1m → ~3.5 días
PLOT_CHANNEL_BARS = int(os.getenv("PLOT_CHANNEL_BARS", "2000"))   # 30m → ~41 días
WARN_TOO_MUCH = int(os.getenv("WARN_TOO_MUCH", "5000"))

def _linebreak_like(s: pd.Series) -> pd.Series:
    s = s.copy()
    prev = s.shift(1)
    s[(~prev.isna()) & (s != prev)] = float('nan')
    return s

def _style_tv_dark():
    mc = mpf.make_marketcolors(
        up='lime',
        down='red',
        edge='inherit',
        wick='white',
        volume='in'
    )
    return mpf.make_mpf_style(
        marketcolors=mc,
        base_mpf_style='nightclouds',
        facecolor='black',
        edgecolor='black',
        gridcolor='#333333',
        gridstyle='--',
        rc={'axes.labelcolor':'white', 'xtick.color':'white', 'ytick.color':'white'},
        y_on_right=False
    )

def _align_channels_to_stream(ch30: pd.DataFrame, idx1m: pd.DatetimeIndex) -> pd.DataFrame:
    want = ["Value","ValueUpper","ValueLower","UpperMid","LowerMid","UpperQ","LowerQ"]
    ch30 = ch30.copy()
    for c in want:
        if c not in ch30.columns:
            ch30[c] = pd.NA
    out = ch30[want].reindex(idx1m.union(ch30.index)).sort_index().ffill()
    return out.reindex(idx1m)

def _has_data(s: pd.Series) -> bool:
    if s is None or len(s) == 0:
        return False
    try:
        arr = pd.to_numeric(s, errors="coerce").to_numpy()
        if arr.size == 0:
            return False
        return np.isfinite(arr).any()
    except Exception:
        return False

def main():
    # 1) Traigo mucha historia con paginado
    df1  = fetch_klines_paginado(API_SYMBOL, STREAM_INTERVAL,  PLOT_STREAM_BARS)
    df30 = fetch_klines_paginado(API_SYMBOL, CHANNEL_INTERVAL, PLOT_CHANNEL_BARS)

    ohlc1  = df1[["Open","High","Low","Close","Volume"]]
    ohlc30 = df30[["Open","High","Low","Close","Volume"]]

    # 2) Calculo canales en 30m y los extiendo a grilla 1m
    chans30 = compute_channels(ohlc30, multi=RB_MULTI, init_bar=RB_INIT_BAR)
    chans1  = _align_channels_to_stream(chans30, ohlc1.index)

    # 3) Si no hay datos suficientes para canales, ploteo solo velas 1m
    cols_chk = ["Value","ValueUpper","ValueLower","UpperMid","LowerMid","UpperQ","LowerQ"]
    all_nan_cols = [c for c in cols_chk if not _has_data(chans1.get(c, pd.Series(dtype=float)))]
    if len(all_nan_cols) == len(cols_chk):
        print(f"[WARN] Canales/Q vacíos. Subí PLOT_CHANNEL_BARS (>= {RB_INIT_BAR+50}) o revisá datos en {CHANNEL_INTERVAL}.")
        mpf.plot(
            ohlc1, type='candle',
            style=_style_tv_dark(),
            addplot=[],
            figsize=(12,6),
            datetime_format='%Y-%m-%d %H:%M',
            title=f"{SYMBOL_DISPLAY} {STREAM_INTERVAL} — sin canales (historia insuficiente)",
            warn_too_much_data=WARN_TOO_MUCH
        )
        return

    # 4) Toques de Q sobre 1m
    touch_uq = (ohlc1['Low'] <= chans1['UpperQ']) & (ohlc1['High'] >= chans1['UpperQ'])
    touch_lq = (ohlc1['Low'] <= chans1['LowerQ']) & (ohlc1['High'] >= chans1['LowerQ'])
    suq = pd.Series(float('nan'), index=ohlc1.index)
    slq = pd.Series(float('nan'), index=ohlc1.index)
    if touch_uq.any():
        suq.loc[touch_uq] = chans1['UpperQ'].loc[touch_uq]
    if touch_lq.any():
        slq.loc[touch_lq] = chans1['LowerQ'].loc[touch_lq]

    # 5) Overlays con guardas
    ap = []
    def add_if(series, **kwargs):
        if _has_data(series):
            ap.append(mpf.make_addplot(series, **kwargs))

    add_if(_linebreak_like(chans1.get('ValueUpper')), color='#1dac70', width=1)
    add_if(_linebreak_like(chans1.get('Value')),      color='gray',    width=1)
    add_if(_linebreak_like(chans1.get('ValueLower')), color='#df3a79', width=1)

    add_if(_linebreak_like(chans1.get('UpperMid')),   color='gray',    width=1, alpha=0.5)
    add_if(_linebreak_like(chans1.get('LowerMid')),   color='gray',    width=1, alpha=0.5)

    add_if(_linebreak_like(chans1.get('UpperQ')),     color='yellow',  width=1, linestyle=':')
    add_if(_linebreak_like(chans1.get('LowerQ')),     color='yellow',  width=1, linestyle=':')

    if _has_data(suq):
        ap.append(mpf.make_addplot(suq, type='scatter', marker='o', markersize=40, color='white'))
    if _has_data(slq):
        ap.append(mpf.make_addplot(slq, type='scatter', marker='o', markersize=40, color='white'))

    # 6) Plot
    mpf.plot(
        ohlc1, type='candle',
        style=_style_tv_dark(),
        addplot=ap, figsize=(12,6),
        datetime_format='%Y-%m-%d %H:%M',
        title=f"{SYMBOL_DISPLAY} {STREAM_INTERVAL} — canales/Q de {CHANNEL_INTERVAL} (histórico paginado)",
        warn_too_much_data=WARN_TOO_MUCH
    )

if __name__ == "__main__":
    main()
