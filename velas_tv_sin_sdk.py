# velas_TV_sin_sdk.py
# ---------------------------------------------------------
# Velas + Range Breakout (canal + flechas, sin círculos)
# - SOLO VELAS CERRADAS → 1 fila por vela en CSV
# - Evita duplicados con CloseTimeMs (UTC, numérico)
# - Bandas: SMA(ATR(200),100) * multi (réplica Pine)
# - RB_INIT_BAR para alinear con bar_index de TradingView (default 301)
# - plot_from_csv(): ver gráfico desde el CSV (en otro proceso)
# ---------------------------------------------------------

import os
import time
import numpy as np
import pandas as pd
import mplfinance as mpf
from zoneinfo import ZoneInfo
from datetime import datetime, timezone
from dotenv import load_dotenv

# Cargar .env
load_dotenv()

try:
    from binance.um_futures import UMFutures
except Exception:
    print("ERROR: Falta el conector de Futuros de Binance.")
    print("Instalá con:  pip install binance-futures-connector")
    raise

# ================== Config ==================
SYMBOL_DISPLAY   = os.getenv("SYMBOL", "ETHUSDT.P")
API_SYMBOL       = SYMBOL_DISPLAY.replace(".P", "")
INTERVAL         = os.getenv("INTERVAL", "30m")
LIMIT            = int(os.getenv("LIMIT", "800"))
TZ_NAME          = os.getenv("TZ", "America/Argentina/Buenos_Aires")

RB_MULTI         = float(os.getenv("RB_MULTI", "4.0"))
RB_LB            = int(os.getenv("RB_LB", "10"))
RB_FILTER_TREND  = os.getenv("RB_FILTER_TREND", "false").lower() == "true"
RB_INIT_BAR      = int(os.getenv("RB_INIT_BAR", "301"))   # para alinear con TradingView

CSV_PATH         = os.getenv("CSV_PATH", "stream_table.csv").strip()
SAVEFIG_PATH     = os.getenv("SAVEFIG", "").strip()       # usado sólo por plot_from_csv()
SLEEP_FALLBACK   = int(os.getenv("SLEEP_FALLBACK", "10"))
WARN_TOO_MUCH    = 5000
DEBUG_SIGNALS    = os.getenv("DEBUG_SIGNALS", "0") == "1" # imprime flags de la última vela

BINANCE_INTERVAL_SECONDS = {
    "1m":60, "3m":180, "5m":300, "15m":900, "30m":1800, "1h":3600, "2h":7200,
    "4h":14400, "6h":21600, "8h":28800, "12h":43200, "1d":86400, "3d":259200,
    "1w":604800, "1M":2592000
}
def interval_seconds(s: str) -> int:
    return BINANCE_INTERVAL_SECONDS.get(s, SLEEP_FALLBACK)

def fmt_ts(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y-%m-%d %H:%M:%S")

# ================== Datos ==================
def get_binance_client():
    base_url = os.getenv("BINANCE_UM_BASE_URL", "https://fapi.binance.com")
    return UMFutures(base_url=base_url)

def fetch_klines(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    client = get_binance_client()
    data = client.klines(symbol=symbol, interval=interval, limit=limit)
    rows = []
    for k in data:
        rows.append({
            "OpenTime": int(k[0]),
            "Open": float(k[1]),
            "High": float(k[2]),
            "Low": float(k[3]),
            "Close": float(k[4]),
            "Volume": float(k[5]),
            "CloseTime": int(k[6]),  # fin de vela en ms (UTC)
        })
    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["OpenTime"], unit="ms", utc=True)
    df["CloseTimeDT"] = pd.to_datetime(df["CloseTime"], unit="ms", utc=True)
    tz = ZoneInfo(TZ_NAME)
    df = df.set_index(df["Date"].dt.tz_convert(tz)).sort_index()
    df["CloseTimeDT"] = df["CloseTimeDT"].dt.tz_convert(tz)
    return df[["Open","High","Low","Close","Volume","CloseTime","CloseTimeDT"]]

# ================== Indicador (réplica Pine) ==================
def _rma(x: pd.Series, length: int) -> pd.Series:
    alpha = 1.0 / length
    return x.ewm(alpha=alpha, adjust=False).mean()

def _atr(df: pd.DataFrame, length: int) -> pd.Series:
    h, l, c = df['High'], df['Low'], df['Close']
    prev_c = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return _rma(tr, length)

def compute_range_breakout(df: pd.DataFrame,
                           multi: float = 4.0,
                           lb: int = 10,
                           filter_trend: bool = False):
    """Canal + medias intermedias + flechas; anclaje de flechas en i (misma vela que dispara)."""
    df = df.copy()
    df['hl2'] = (df['High'] + df['Low']) / 2.0

    # Ancho del canal: SMA(ATR(200),100) * multi
    atr200 = _atr(df, 200)
    width = atr200.rolling(100, min_periods=1).mean() * multi

    n = len(df)
    value = np.full(n, np.nan)
    vup   = np.full(n, np.nan)
    vlo   = np.full(n, np.nan)
    umid  = np.full(n, np.nan)
    lmid  = np.full(n, np.nan)

    plot_buy  = np.full(n, False)
    plot_sell = np.full(n, False)

    t = False
    count = 0
    arrows_buy, arrows_sell = [], []

    highs = df['High'].values
    lows  = df['Low'].values
    hl2   = df['hl2'].values
    w     = width.values
    idx   = df.index

    # helpers: igual que ta.crossover/under (usa prev y actual del valor y del nivel)
    def crossed_up(prev_val, curr_val, prev_lvl, curr_lvl):
        return (prev_val <= prev_lvl) and (curr_val > curr_lvl)
    def crossed_dn(prev_val, curr_val, prev_lvl, curr_lvl):
        return (prev_val >= prev_lvl) and (curr_val < curr_lvl)

    for i in range(n):
        # Inicialización alineable con TV
        if i == RB_INIT_BAR:
            value[i] = hl2[i]
            vup[i]   = hl2[i] + w[i]
            vlo[i]   = hl2[i] - w[i]
            umid[i]  = (value[i] + vup[i]) / 2.0
            lmid[i]  = (value[i] + vlo[i]) / 2.0
        else:
            if i > 0:
                value[i] = value[i-1]; vup[i] = vup[i-1]; vlo[i] = vlo[i-1]
                umid[i]  = umid[i-1];  lmid[i] = lmid[i-1]

        if i < max(RB_INIT_BAR, 1):
            continue

        # Rupturas (usar nivel previo y actual)
        cross_up   = crossed_up(lows[i-1],  lows[i],  vup[i-1], vup[i])
        cross_down = crossed_dn(highs[i-1], highs[i], vlo[i-1], vlo[i])

        # Conteo de barras fuera del canal
        if not np.isnan(vup[i]) and not np.isnan(vlo[i]):
            if (lows[i] > vup[i]) or (highs[i] < vlo[i]):
                count += 1

        # Reset del canal
        channel_changed = False
        if cross_up or cross_down or count == 100:
            count = 0
            value[i] = hl2[i]; vup[i] = hl2[i] + w[i]; vlo[i] = hl2[i] - w[i]
            umid[i]  = (value[i] + vup[i]) / 2.0; lmid[i] = (value[i] + vlo[i]) / 2.0
            channel_changed = True

        # Tendencia por ruptura
        if cross_up:   t = True
        if cross_down: t = False

        chage = not channel_changed  # igual a not(value != value[1]) en Pine

        # Cruce de medias (usar nivel previo/actual; anclado en i)
        if chage and not np.isnan(lmid[i]) and not np.isnan(umid[i]):
            buy_cross  = crossed_up(lows[i-1],  lows[i],  lmid[i-1], lmid[i])
            sell_cross = crossed_dn(highs[i-1], highs[i], umid[i-1], umid[i])

            lb_ok_buy  = (lb == 0 or (i - lb >= 0 and lows[i - lb]  > lmid[i]))
            lb_ok_sell = (lb == 0 or (i - lb >= 0 and highs[i - lb] < umid[i]))

            if buy_cross and lb_ok_buy and (t if filter_trend else True):
                plot_buy[i] = True
                arrows_buy.append((idx[i], lows[i]))
            if sell_cross and lb_ok_sell and ((not t) if filter_trend else True):
                plot_sell[i] = True
                arrows_sell.append((idx[i], highs[i]))

        # Señales por ruptura (ancladas en i)
        if cross_up and (t if filter_trend else True):
            plot_buy[i] = True
            price_anchor = vup[i] if not np.isnan(vup[i]) else lows[i]
            arrows_buy.append((idx[i], price_anchor))
        if cross_down and ((not t) if filter_trend else True):
            plot_sell[i] = True
            price_anchor = vlo[i] if not np.isnan(vlo[i]) else highs[i]
            arrows_sell.append((idx[i], price_anchor))

    buy_flag  = pd.Series(plot_buy,  index=df.index)
    sell_flag = pd.Series(plot_sell, index=df.index)

    return {
        'value': pd.Series(value, index=df.index),
        'value_upper': pd.Series(vup, index=df.index),
        'value_lower': pd.Series(vlo, index=df.index),
        'upper_mid': pd.Series(umid, index=df.index),
        'lower_mid': pd.Series(lmid, index=df.index),
        'buy_flag': buy_flag,
        'sell_flag': sell_flag,
        'arrows_buy': arrows_buy,
        'arrows_sell': arrows_sell,
    }

# ================== Helpers Plot ==================
def _unique_sorted_index(index: pd.Index) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(pd.to_datetime(index))
    idx = idx.sort_values()
    return idx[~idx.duplicated(keep="last")]

def _linebreak_like(series: pd.Series) -> pd.Series:
    s = series.copy(); prev = s.shift(1)
    s[(~prev.isna()) & (s != prev)] = np.nan
    return s

def _series_from_points(index, points):
    idx = _unique_sorted_index(index)
    s = pd.Series(np.nan, index=idx, dtype="float64")
    for ts, price in points:
        try:
            ts = pd.to_datetime(ts)
            if hasattr(idx, "tz") and idx.tz is not None:
                if ts.tzinfo is None: ts = ts.tz_localize(idx.tz)
                else:                  ts = ts.tz_convert(idx.tz)
            pos = idx.get_indexer([ts], method='nearest')[0]
            s.iloc[pos] = float(price)
        except Exception:
            continue
    return s

def _build_overlays(df: pd.DataFrame, indi: dict):
    return [
        mpf.make_addplot(_linebreak_like(indi['value_upper']), color='#1dac70', width=1),
        mpf.make_addplot(_linebreak_like(indi['value']),       color='gray',    width=1),
        mpf.make_addplot(_linebreak_like(indi['value_lower']), color='#df3a79', width=1),
        mpf.make_addplot(_linebreak_like(indi['upper_mid']),   color='gray',    width=1, alpha=0.5),
        mpf.make_addplot(_linebreak_like(indi['lower_mid']),   color='gray',    width=1, alpha=0.5),
        mpf.make_addplot(_series_from_points(df.index, indi['arrows_buy']),
                         type='scatter', marker='^', markersize=60, color='#1dac70'),
        mpf.make_addplot(_series_from_points(df.index, indi['arrows_sell']),
                         type='scatter', marker='v', markersize=60, color='#df3a79'),
    ]

def plot_with_overlays(df: pd.DataFrame, indi: dict, title="Range Breakout"):
    ap = _build_overlays(df, indi)
    fig, _ = mpf.plot(
        df[["Open","High","Low","Close","Volume"]],
        type='candle', style=mpf.make_mpf_style(),
        addplot=ap, returnfig=True, figsize=(12,6),
        datetime_format='%Y-%m-%d %H:%M', title=title, warn_too_much_data=WARN_TOO_MUCH
    )
    if SAVEFIG_PATH:
        try:
            fig.savefig(SAVEFIG_PATH, dpi=130)
            print(f"[OK] Gráfico guardado en: {SAVEFIG_PATH}")
        except Exception:
            pass

# ================== CSV ==================
CSV_COLUMNS = ["CloseTimeMs","Date","Open","High","Low","Close","Volume","Buy","Sell"]

def ensure_csv_header(path: str):
    if not os.path.exists(path):
        pd.DataFrame(columns=CSV_COLUMNS).to_csv(path, index=False, encoding="utf-8")

def append_row_to_csv(path: str, row: dict):
    pd.DataFrame([row]).to_csv(path, mode="a", header=False, index=False, encoding="utf-8")

# ================== plot_from_csv ==================
def plot_from_csv():
    """Usar desde OTRO proceso o consola. El loop del bot no se detiene."""
    if not os.path.exists(CSV_PATH):
        print(f"[WARN] No existe CSV '{CSV_PATH}' todavía.")
        return
    df = pd.read_csv(CSV_PATH, parse_dates=["Date"])
    if df.empty:
        print("[WARN] CSV vacío.")
        return
    df = df.sort_values("CloseTimeMs").drop_duplicates(subset=["CloseTimeMs"], keep="last")
    df = df.set_index("Date")
    df = df.rename(columns={c:c.capitalize() for c in df.columns})
    for c in ["Open","High","Low","Close","Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    indi = compute_range_breakout(df[["Open","High","Low","Close","Volume"]],
                                  multi=RB_MULTI, lb=RB_LB, filter_trend=RB_FILTER_TREND)
    plot_with_overlays(df, indi, title=f"{SYMBOL_DISPLAY} {INTERVAL} (desde CSV)")

# ================== Loop (sin gráficos) ==================
def run_loop():
    print(f"[INIT] {SYMBOL_DISPLAY} {INTERVAL} | TZ={TZ_NAME}")
    ensure_csv_header(CSV_PATH)

    last_logged_ms = None
    if os.path.exists(CSV_PATH):
        try:
            tail = pd.read_csv(CSV_PATH, usecols=["CloseTimeMs"]).tail(1)
            if not tail.empty: last_logged_ms = int(tail["CloseTimeMs"].iloc[0])
        except Exception:
            pass

    while True:
        try:
            df = fetch_klines(API_SYMBOL, INTERVAL, LIMIT)
            now_utc_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

            # Solo velas CERRADAS
            df_closed = df[df["CloseTime"] <= now_utc_ms]
            if df_closed.empty:
                time.sleep(SLEEP_FALLBACK); continue

            last_row = df_closed.iloc[-1]
            last_idx = df_closed.index[-1]        # tz-aware
            last_ms  = int(last_row["CloseTime"]) # clave numérica

            if (last_logged_ms is None) or (last_ms > last_logged_ms):
                indi = compute_range_breakout(
                    df_closed[["Open","High","Low","Close","Volume"]],
                    multi=RB_MULTI, lb=RB_LB, filter_trend=RB_FILTER_TREND
                )

                # tomar flags por índice exacto de la vela cerrada
                buy_flag  = bool(indi['buy_flag'].loc[last_idx])
                sell_flag = bool(indi['sell_flag'].loc[last_idx])

                if DEBUG_SIGNALS:
                    print(f"[DEBUG] close={fmt_ts(last_idx)} buy={buy_flag} sell={sell_flag}")

                row = {
                    "CloseTimeMs": last_ms,
                    "Date":   fmt_ts(last_idx),
                    "Open":   round(float(last_row["Open"]), 2),
                    "High":   round(float(last_row["High"]), 2),
                    "Low":    round(float(last_row["Low"]),  2),
                    "Close":  round(float(last_row["Close"]),2),
                    "Volume": round(float(last_row["Volume"]),2),
                    "Buy":    int(buy_flag),
                    "Sell":   int(sell_flag),
                }
                append_row_to_csv(CSV_PATH, row)

                sig = "▲ BUY" if buy_flag and not sell_flag else "▼ SELL" if sell_flag and not buy_flag else " "
                print(f"[{row['Date']}]  O:{row['Open']:>8}  H:{row['High']:>8}  L:{row['Low']:>8}  C:{row['Close']:>8}  Vol:{row['Volume']:>10}   Sig:{sig}")

                last_logged_ms = last_ms

            # Dormir hasta el próximo cierre de vela
            next_close_ms = int(df.iloc[-1]["CloseTime"])
            now_utc = datetime.now(timezone.utc).timestamp()
            eta = max(2, int((next_close_ms/1000) - now_utc) + 1)
            time.sleep(eta)

        except KeyboardInterrupt:
            print("\n[EXIT] Cortado por usuario."); break
        except Exception as e:
            print(f"\n[WARN] {type(e).__name__}: {e}")
            time.sleep(SLEEP_FALLBACK)

# ================== Main ==================
def main():
    print(f"[INFO] Loop de velas → CSV='{CSV_PATH}'")
    run_loop()

if __name__ == "__main__":
    main()
