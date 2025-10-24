# ---------------------------------------------------------
# Velas + Canales Range Breakout (solo canales, sin buy/sell)
# Dual TF:
#   - Canales + Q-lines en CHANNEL_INTERVAL (p.ej. 5m/30m)
#   - Stream/CSV por vela CERRADA en STREAM_INTERVAL (1m)
#
# CSV: una fila por cada vela 1m cerrada
# Columns: Date, Open, High, Low, Close,
#          UpperMid, ValueUpper, LowerMid, ValueLower,
#          TouchUpperQ, TouchLowerQ
#  - Precios truncados a 3 decimales (no redondeo)
#  - Salida: tablaQ.csv (encabezado fijo, columnas estables)
# ---------------------------------------------------------

import os
import time
import io
import re
import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo
from datetime import datetime, timezone
from dotenv import load_dotenv

# ================== Config/Entorno ==================
load_dotenv()

SYMBOL_DISPLAY   = os.getenv("SYMBOL", "ETHUSDT.P")
API_SYMBOL       = SYMBOL_DISPLAY.replace(".P", "")

# Dual timeframe
CHANNEL_INTERVAL = os.getenv("CHANNEL_INTERVAL", "30m").strip()
STREAM_INTERVAL  = os.getenv("STREAM_INTERVAL",  "1m").strip()

# Historia larga para igualar al gráfico
PLOT_STREAM_BARS  = int(os.getenv("PLOT_STREAM_BARS",  "5000"))
PLOT_CHANNEL_BARS = int(os.getenv("PLOT_CHANNEL_BARS", "2000"))

USE_PAGINADO_CHANNEL = int(os.getenv("USE_PAGINADO_CHANNEL", "1"))
USE_PAGINADO_STREAM  = int(os.getenv("USE_PAGINADO_STREAM",  "0"))

# Helpers ints
def _parse_int_like(val, default):
    try:
        if val is None:
            return int(default)
        if isinstance(val, (int, np.integer)):
            return int(val)
        s = str(val).strip()
        if s == "":
            return int(default)
        s = re.sub(r"[,_\s]", "", s)
        return int(s)
    except Exception:
        return int(default)

def _env_int_clamped(name, default, lo=1, hi=1500):
    raw = os.getenv(name)
    x = _parse_int_like(raw, default)
    if x < lo or x > hi:
        print(f"[WARN] {name} fuera de rango ({x}). Clampeo a [{lo}..{hi}].")
    return max(lo, min(hi, x))

# Límites fetch corto
LIMIT_CHANNEL    = _env_int_clamped("LIMIT_CHANNEL", 800, 1, 1500)
LIMIT_STREAM     = _env_int_clamped("LIMIT_STREAM", 1500, 1, 1500)

TZ_NAME          = os.getenv("TZ", "America/Argentina/Buenos_Aires")

RB_MULTI         = float(os.getenv("RB_MULTI", "4.0"))
RB_INIT_BAR      = int(os.getenv("RB_INIT_BAR", "301"))

# Salida CSV (nuevo nombre y columnas fijas)
TABLE_CSV_PATH = os.getenv("TABLE_CSV_PATH", "tablaQ.csv").strip()
TABLE_COLUMNS  = [
    "Date","Open","High","Low","Close",
    "Volume","Value","UpperMid","ValueUpper","LowerMid","ValueLower",
    "UpperQ","LowerQ","TrendSMA",
    "TouchUpperQ","TouchLowerQ"
]

SLEEP_FALLBACK = int(os.getenv("SLEEP_FALLBACK", "5"))

def fmt_ts(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y-%m-%d %H:%M:%S")

def trunc3(x):
    """Trunca a 3 decimales (sin redondear)."""
    try:
        fx = float(x)
        return float(np.trunc(fx * 1000.0) / 1000.0)
    except Exception:
        return np.nan

# ================== Binance client ==================
try:
    from binance.um_futures import UMFutures
    try:
        from binance.error import ClientError
    except Exception:
        class ClientError(Exception): ...
except Exception:
    print("ERROR: Falta el conector de Futuros de Binance.")
    print("Instalá con:  pip install binance-futures-connector")
    raise

def get_binance_client():
    base_url = os.getenv("BINANCE_UM_BASE_URL", "https://fapi.binance.com").strip()
    return UMFutures(base_url=base_url)

def _fetch_klines_raw(client, symbol: str, interval: str, limit: int):
    return client.klines(symbol=symbol, interval=interval, limit=limit)

def fetch_klines(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    """ Fetch corto: Sanea 'limit' (1..1500) y maneja -1130 reintentando. """
    client = get_binance_client()
    tries = []
    lim0 = _env_int_clamped("X_TMP_LIMIT_IGNORE_THIS", limit, 1, 1500)
    tries.append(lim0)
    for fallback in (1500, 1000, 500):
        if fallback not in tries:
            tries.append(fallback)

    last_err = None
    for lim in tries:
        try:
            data = _fetch_klines_raw(client, symbol, interval, lim)
            if not data:
                raise ValueError("Respuesta vacía de klines.")
            if lim != limit:
                print(f"[INFO] Usando limit={lim} (pedido={limit}) para {symbol} {interval}")
            rows = []
            for k in data:
                rows.append({
                    "OpenTime": int(k[0]),
                    "Open": float(k[1]),
                    "High": float(k[2]),
                    "Low": float(k[3]),
                    "Close": float(k[4]),
                    "Volume": float(k[5]),
                    "CloseTime": int(k[6]),
                })
            df = pd.DataFrame(rows)
            df["DateUTC"] = pd.to_datetime(df["OpenTime"], unit="ms", utc=True)
            df["CloseTimeDT_UTC"] = pd.to_datetime(df["CloseTime"], unit="ms", utc=True)

            tz = ZoneInfo(TZ_NAME)
            df = df.set_index(df["DateUTC"].dt.tz_convert(tz)).sort_index()
            df["CloseTimeDT"] = df["CloseTimeDT_UTC"].dt.tz_convert(tz)
            return df[["Open","High","Low","Close","Volume","CloseTime","CloseTimeDT"]]
        except ClientError as ce:
            msg = f"{ce}"
            if "parameter 'limit'" in msg or "-1130" in msg:
                print(f"[WARN] Binance rechazó limit={lim} para {symbol} {interval}. Reintento con otro valor...")
                last_err = ce
                time.sleep(0.2)
                continue
            last_err = ce
            break
        except Exception as e:
            last_err = e
            break

    raise last_err

# ================== Paginado (historia larga) ==================
_fetch_paginado_ok = False
try:
    from paginado_binance import fetch_klines_paginado
    _fetch_paginado_ok = True
except Exception:
    print("[WARN] No se pudo importar 'paginado_binance.fetch_klines_paginado'.")
    print("       El CSV usará historia CORTA. Ubicá 'paginado_binance.py' para igualar al gráfico.")

def fetch_hist(symbol: str, interval: str, bars: int, prefer_paginado: bool) -> pd.DataFrame:
    """
    Devuelve DataFrame OHLCV con índice tz-aware, columnas:
    Open, High, Low, Close, Volume, CloseTime, CloseTimeDT
    """
    if prefer_paginado and _fetch_paginado_ok:
        df = fetch_klines_paginado(symbol, interval, bars)
        if "CloseTime" not in df.columns:
            dt_idx = pd.DatetimeIndex(df.index)
            close_dt = dt_idx.shift(-1, freq=None)
            close_ms = (close_dt.view('int64') // 1_000_000).astype('int64')
            df["CloseTime"] = close_ms
        if "CloseTimeDT" not in df.columns:
            df["CloseTimeDT"] = pd.to_datetime(df["CloseTime"], unit="ms", utc=True).dt.tz_convert(ZoneInfo(TZ_NAME))
        keep = ["Open","High","Low","Close","Volume","CloseTime","CloseTimeDT"]
        for k in keep:
            if k not in df.columns:
                if k == "Volume": df[k] = 0.0
                elif k == "CloseTime": df[k] = 0
                elif k == "CloseTimeDT": df[k] = pd.to_datetime(df.index, utc=True)
                else: df[k] = np.nan
        df = df[keep].sort_index()
        return df
    limit = bars if bars <= 1500 else 1500
    return fetch_klines(symbol, interval, limit)

# ================== Indicador (canales) ==================
def _rma(x: pd.Series, length: int) -> pd.Series:
    alpha = 1.0 / length
    return x.ewm(alpha=alpha, adjust=False).mean()

def _atr(df: pd.DataFrame, length: int) -> pd.Series:
    h, l, c = df['High'], df['Low'], df['Close']
    prev_c = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return _rma(tr, length)

def compute_channels(df: pd.DataFrame, multi: float = 4.0, init_bar: int = 301) -> pd.DataFrame:
    df = df.copy()
    df['hl2'] = (df['High'] + df['Low']) / 2.0
    atr200 = _atr(df, 200)
    width  = atr200.rolling(100, min_periods=1).mean() * multi

    n = len(df)
    value = np.full(n, np.nan)
    vup   = np.full(n, np.nan)
    vlo   = np.full(n, np.nan)
    umid  = np.full(n, np.nan)
    lmid  = np.full(n, np.nan)

    highs = df['High'].values
    lows  = df['Low'].values
    hl2   = df['hl2'].values
    w     = width.values

    def crossed_up(prev_val, curr_val, prev_lvl, curr_lvl):
        return (prev_val <= prev_lvl) and (curr_val > curr_lvl)

    def crossed_dn(prev_val, curr_val, prev_lvl, curr_lvl):
        return (prev_val >= prev_lvl) and (curr_val < curr_lvl)

    count = 0
    for i in range(n):
        if i == init_bar:
            value[i] = hl2[i]
            vup[i]   = hl2[i] + w[i]
            vlo[i]   = hl2[i] - w[i]
            umid[i]  = (value[i] + vup[i]) / 2.0
            lmid[i]  = (value[i] + vlo[i]) / 2.0
        else:
            if i > 0:
                value[i] = value[i-1]
                vup[i]   = vup[i-1]
                vlo[i]   = vlo[i-1]
                umid[i]  = umid[i-1]
                lmid[i]  = lmid[i-1]

        if i < max(init_bar, 1):
            continue

        cross_up   = crossed_up(lows[i-1],  lows[i],  vup[i-1], vup[i])
        cross_down = crossed_dn(highs[i-1], highs[i], vlo[i-1], vlo[i])

        if not np.isnan(vup[i]) and not np.isnan(vlo[i]):
            if (lows[i] > vup[i]) or (highs[i] < vlo[i]):
                count += 1

        if cross_up or cross_down or (count == 100):
            count   = 0
            value[i] = hl2[i]
            vup[i]   = hl2[i] + w[i]
            vlo[i]   = hl2[i] - w[i]
            umid[i]  = (value[i] + vup[i]) / 2.0
            lmid[i]  = (value[i] + vlo[i]) / 2.0

    upper_q = (umid + vup) / 2.0
    lower_q = (lmid + vlo) / 2.0

    return pd.DataFrame({
        'Value': value,
        'ValueUpper': vup,
        'ValueLower': vlo,
        'UpperMid': umid,
        'LowerMid': lmid,
        'UpperQ': upper_q,
        'LowerQ': lower_q,
    }, index=df.index)

# ================== CSV helpers ==================
def ensure_table_csv_header_strict(path: str):
    """
    Encabezado fijo garantizado:
    - Si no existe o está vacío: crea con header correcto.
    - Si existe pero el header no coincide: respalda y recrea con header correcto.
    """
    want = TABLE_COLUMNS
    header_line = ",".join(want)

    if not os.path.exists(path) or os.path.getsize(path) == 0:
        pd.DataFrame(columns=want).to_csv(path, index=False, encoding="utf-8")
        return

    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            first = (f.readline() or "").strip()
    except Exception:
        first = ""

    if first.replace(" ", "") != header_line.replace(" ", ""):
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        bak = f"{path}.bak.{ts}"
        try:
            os.replace(path, bak)
            print(f"[WARN] {os.path.basename(path)} tenía encabezado inválido. Backup: {bak}")
        except Exception as e:
            print(f"[WARN] No se pudo respaldar {path}: {e}")
            try:
                os.remove(path)
            except Exception:
                pass
        pd.DataFrame(columns=want).to_csv(path, index=False, encoding="utf-8")

def append_row_to_table(path: str, row: dict):
    """Agrega fila con columnas fijas y formato a 3 decimales."""
    out = {k: row.get(k, np.nan) for k in TABLE_COLUMNS}
    pd.DataFrame([out], columns=TABLE_COLUMNS).to_csv(
        path,
        mode="a",
        header=False,
        index=False,
        encoding="utf-8",
        float_format="%.3f",
        na_rep=""
    )

# ================== Alineación CH → 1m ==================
def _align_channels_to_stream(ch: pd.DataFrame, idx1m: pd.DatetimeIndex) -> pd.DataFrame:
    """Extiende canales (CHANNEL_INTERVAL) a grilla 1m por forward-fill."""
    if ch is None or ch.empty:
        return pd.DataFrame(index=idx1m)
    want = ["Value","ValueUpper","ValueLower","UpperMid","LowerMid","UpperQ","LowerQ","TrendSMA"]
    for c in want:
        if c not in ch.columns:
            ch[c] = pd.NA
    out = ch[want].reindex(idx1m.union(ch.index)).sort_index().ffill()
    return out.reindex(idx1m)

# ================== Loop Dual TF (cache + realineo SIEMPRE) ==================
def run_loop_dual_tf():
    print(f"[INFO] Loop dual TF → CSV único '{TABLE_CSV_PATH}'")
    print(f"[INIT] {SYMBOL_DISPLAY} CH={CHANNEL_INTERVAL} | ST={STREAM_INTERVAL} | TZ={TZ_NAME}")
    print(f"[INIT] Historia larga: CHANNEL_BARS={PLOT_CHANNEL_BARS} (paginado={USE_PAGINADO_CHANNEL}), STREAM_BARS={PLOT_STREAM_BARS} (paginado={USE_PAGINADO_STREAM})")

    # Header fijo sí o sí
    ensure_table_csv_header_strict(TABLE_CSV_PATH)

    last_logged_ms = None  # solo para dedupe in-memory

    chans_1m = pd.DataFrame()
    chans_cached = pd.DataFrame()
    last_ch_closed_key = None

    while True:
        try:
            # 1) Stream 1m (incluye vela en curso)
            if USE_PAGINADO_STREAM:
                df1 = fetch_hist(API_SYMBOL, STREAM_INTERVAL, PLOT_STREAM_BARS, prefer_paginado=True)
            else:
                df1 = fetch_klines(API_SYMBOL, STREAM_INTERVAL, LIMIT_STREAM)

            now_utc_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
            df1_closed = df1[df1["CloseTime"] <= now_utc_ms]
            if df1_closed.empty:
                time.sleep(SLEEP_FALLBACK)
                continue

            # 2) Canales: recomputar SOLO si cambió la última CH cerrada (clave corta)
            dfCH_key = fetch_klines(API_SYMBOL, CHANNEL_INTERVAL, max(3, min(LIMIT_CHANNEL, 1500)))
            if len(dfCH_key) >= 2:
                keyCH = int(dfCH_key.iloc[-2]["CloseTime"])  # última CH CERRADA
            else:
                keyCH = int(dfCH_key.iloc[-1]["CloseTime"])

            if (last_ch_closed_key is None) or (keyCH != last_ch_closed_key) or chans_cached.empty:
                dfCH_hist = fetch_hist(API_SYMBOL, CHANNEL_INTERVAL, PLOT_CHANNEL_BARS, prefer_paginado=bool(USE_PAGINADO_CHANNEL))
                ohlcCH = dfCH_hist[["Open","High","Low","Close","Volume"]]
                chans_cached = compute_channels(ohlcCH, multi=RB_MULTI, init_bar=RB_INIT_BAR)
                trend_close = ohlcCH["Close"].astype(float)
                trend_sma = trend_close.rolling(window=100, min_periods=1).mean()
                chans_cached = chans_cached.copy()
                chans_cached["TrendSMA"] = trend_sma.reindex(chans_cached.index).ffill()
                last_ch_closed_key = keyCH
                print(f"[INFO] Recalculados canales con historia larga ({len(ohlcCH)} barras {CHANNEL_INTERVAL}).")

            # 2.b) Realinear canales a 1m
            chans_1m = _align_channels_to_stream(chans_cached, df1_closed.index)

            # 3) Última 1m CERRADA → CSV
            last_row = df1_closed.iloc[-1]
            last_idx = df1_closed.index[-1]        # tz-aware
            last_ms  = int(last_row["CloseTime"])  # clave en memoria

            if (last_logged_ms is None) or (last_ms > last_logged_ms):
                ch = chans_1m.loc[last_idx] if (not chans_1m.empty and last_idx in chans_1m.index) else pd.Series()

                # Toques Q en 1m
                touch_uq = int(pd.notna(ch.get("UpperQ")) and (last_row["Low"] <= ch["UpperQ"] <= last_row["High"]))
                touch_lq = int(pd.notna(ch.get("LowerQ")) and (last_row["Low"] <= ch["LowerQ"] <= last_row["High"]))

                def _t(x): return trunc3(x)

                trow = {
                    "Date":        fmt_ts(last_idx),
                    "Open":        _t(last_row["Open"]),
                    "High":        _t(last_row["High"]),
                    "Low":         _t(last_row["Low"]),
                    "Close":       _t(last_row["Close"]),
                    "Volume":      _t(last_row["Volume"]),
                    "Value":       _t(ch.get("Value")),
                    "UpperMid":    _t(ch.get("UpperMid")),
                    "ValueUpper":  _t(ch.get("ValueUpper")),
                    "LowerMid":    _t(ch.get("LowerMid")),
                    "ValueLower":  _t(ch.get("ValueLower")),
                    "UpperQ":      _t(ch.get("UpperQ")),
                    "LowerQ":      _t(ch.get("LowerQ")),
                    "TrendSMA":    _t(ch.get("TrendSMA")),
                    "TouchUpperQ": touch_uq,
                    "TouchLowerQ": touch_lq,
                }
                append_row_to_table(TABLE_CSV_PATH, trow)

                print(f"[{trow['Date']}] "
                      f"Open:{trow['Open']:.3f} High:{trow['High']:.3f} Low:{trow['Low']:.3f} Close:{trow['Close']:.3f} "
                      f"Vol:{trow['Volume']:.3f} Val:{trow['Value']:.3f} Trend:{trow['TrendSMA']:.3f} "
                      f"| UMid:{trow['UpperMid']:.3f} VUp:{trow['ValueUpper']:.3f} "
                      f"LMid:{trow['LowerMid']:.3f} VLo:{trow['ValueLower']:.3f} "
                      f"UQ:{trow['UpperQ']:.3f} LQ:{trow['LowerQ']:.3f} "
                      f"| UQ:{trow['TouchUpperQ']}  LQ:{trow['TouchLowerQ']}")

                last_logged_ms = last_ms

            # 4) Dormir hasta el próximo cierre de 1m
            next_close_ms = int(df1.iloc[-1]["CloseTime"])
            now_utc = datetime.now(timezone.utc).timestamp()
            eta = max(1, int((next_close_ms/1000) - now_utc) + 1)
            time.sleep(eta)

        except KeyboardInterrupt:
            print("\n[EXIT] Cortado por usuario.")
            break
        except Exception as e:
            print(f"\n[WARN] {type(e).__name__}: {e}")
            time.sleep(SLEEP_FALLBACK)

# ================== Main ==================
def main():
    run_loop_dual_tf()

if __name__ == "__main__":
    main()
