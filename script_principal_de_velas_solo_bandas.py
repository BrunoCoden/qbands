# ---------------------------------------------------------
# Velas + Canales Range Breakout (solo canales, sin buy/sell)
# Dual TF:
#   - Canales + Q-lines en CHANNEL_INTERVAL (p.ej. 5m/30m)
#   - Stream/CSV por vela CERRADA en STREAM_INTERVAL (1m)
#
# CSV: una fila por cada vela 1m cerrada
# Columns: CloseTimeMs, Date, Open, High, Low, Close, Volume,
#          UpperMid, ValueUpper, LowerMid, ValueLower,
#          TouchUpperQ, TouchLowerQ
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

# Helpers para leer ints de env de forma robusta
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

LIMIT_CHANNEL    = _env_int_clamped("LIMIT_CHANNEL", 800, 1, 1500)
LIMIT_STREAM     = _env_int_clamped("LIMIT_STREAM", 1500, 1, 1500)

TZ_NAME          = os.getenv("TZ", "America/Argentina/Buenos_Aires")

RB_MULTI         = float(os.getenv("RB_MULTI", "4.0"))
RB_INIT_BAR      = int(os.getenv("RB_INIT_BAR", "301"))

# ÚNICA salida CSV (por vela 1m)
TABLE_CSV_PATH = os.getenv("TABLE_CSV_PATH", "tabla.csv").strip()
TABLE_COLUMNS  = [
    "CloseTimeMs","Date","Open","High","Low","Close","Volume",
    "UpperMid","ValueUpper","LowerMid","ValueLower",
    "TouchUpperQ","TouchLowerQ"
]

SLEEP_FALLBACK = int(os.getenv("SLEEP_FALLBACK", "5"))

def fmt_ts(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y-%m-%d %H:%M:%S")

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
    """ Sanea 'limit' (1..1500) y maneja -1130 reintentando. """
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

# ================== Indicador: SOLO canales ==================
def _rma(x: pd.Series, length: int) -> pd.Series:
    alpha = 1.0 / length
    return x.ewm(alpha=alpha, adjust=False).mean()

def _atr(df: pd.DataFrame, length: int) -> pd.Series:
    h, l, c = df['High'], df['Low'], df['Close']
    prev_c = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return _rma(tr, length)

def compute_channels(df: pd.DataFrame, multi: float = 4.0, init_bar: int = 301) -> pd.DataFrame:
    """ Devuelve Value, ValueUpper, ValueLower, UpperMid, LowerMid, UpperQ, LowerQ """
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
def ensure_table_csv_header(path: str):
    if not os.path.exists(path):
        pd.DataFrame(columns=TABLE_COLUMNS).to_csv(path, index=False, encoding="utf-8")

def append_row_to_table(path: str, row: dict):
    out = {k: row.get(k, np.nan) for k in TABLE_COLUMNS}
    pd.DataFrame([out]).to_csv(path, mode="a", header=False, index=False, encoding="utf-8")

def _append_row_dedup_fast(csv_path: str, close_time_ms: int, trow: dict):
    """Append rápido evitando duplicados por CloseTimeMs sin leer todo el CSV."""
    header_needed = not os.path.exists(csv_path)
    if not header_needed:
        try:
            with open(csv_path, "rb") as f:
                f.seek(0, io.SEEK_END)
                size = f.tell()
                f.seek(max(0, size - 4096))
                tail = f.read().decode("utf-8", errors="ignore")
            last_line = tail.strip().splitlines()[-1]
            if "CloseTimeMs" in last_line and close_time_ms is not None:
                if str(close_time_ms) in last_line:
                    return
        except Exception:
            pass
    append_row_to_table(csv_path, trow)

# ================== Alineación CH → 1m ==================
def _align_channels_to_stream(ch: pd.DataFrame, idx1m: pd.DatetimeIndex) -> pd.DataFrame:
    """Extiende canales (CHANNEL_INTERVAL) a grilla 1m por forward-fill."""
    if ch is None or ch.empty:
        return pd.DataFrame(index=idx1m)
    want = ["Value","ValueUpper","ValueLower","UpperMid","LowerMid","UpperQ","LowerQ"]
    for c in want:
        if c not in ch.columns:
            ch[c] = pd.NA
    out = ch[want].reindex(idx1m.union(ch.index)).sort_index().ffill()
    return out.reindex(idx1m)

# ================== Loop Dual TF (cache + realineo SIEMPRE) ==================
def run_loop_dual_tf():
    print(f"[INFO] Loop dual TF → CSV único '{TABLE_CSV_PATH}'")
    print(f"[INIT] {SYMBOL_DISPLAY} CH={CHANNEL_INTERVAL} | ST={STREAM_INTERVAL} | TZ={TZ_NAME}")

    ensure_table_csv_header(TABLE_CSV_PATH)

    last_logged_ms = None
    if os.path.exists(TABLE_CSV_PATH):
        try:
            tail = pd.read_csv(TABLE_CSV_PATH, usecols=["CloseTimeMs"]).tail(1)
            if not tail.empty:
                last_logged_ms = int(tail["CloseTimeMs"].iloc[0])
        except Exception:
            pass

    chans_1m = pd.DataFrame()
    chans_cached = pd.DataFrame()
    last_ch_closed_key = None

    while True:
        try:
            # 1) Stream 1m (incluye vela en curso)
            df1 = fetch_klines(API_SYMBOL, STREAM_INTERVAL, LIMIT_STREAM)
            now_utc_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
            df1_closed = df1[df1["CloseTime"] <= now_utc_ms]
            if df1_closed.empty:
                time.sleep(SLEEP_FALLBACK)
                continue

            # 2) Canales: recomputar SOLO si cambió la última CH cerrada
            dfCH = fetch_klines(API_SYMBOL, CHANNEL_INTERVAL, LIMIT_CHANNEL)
            if len(dfCH) >= 2:
                keyCH = int(dfCH.iloc[-2]["CloseTime"])  # última CH CERRADA
            else:
                keyCH = int(dfCH.iloc[-1]["CloseTime"])

            if (last_ch_closed_key is None) or (keyCH != last_ch_closed_key) or chans_cached.empty:
                ohlc = dfCH[["Open","High","Low","Close","Volume"]]
                chans_cached = compute_channels(ohlc, multi=RB_MULTI, init_bar=RB_INIT_BAR)
                last_ch_closed_key = keyCH

            # 2.b) SIEMPRE re-alinear canales a la grilla 1m actual
            chans_1m = _align_channels_to_stream(chans_cached, df1_closed.index)

            # 3) Última 1m CERRADA → CSV
            last_row = df1_closed.iloc[-1]
            last_idx = df1_closed.index[-1]        # tz-aware
            last_ms  = int(last_row["CloseTime"])  # clave numérica

            if (last_logged_ms is None) or (last_ms > last_logged_ms):
                ch = chans_1m.loc[last_idx] if (not chans_1m.empty and last_idx in chans_1m.index) else pd.Series()

                # Señales: toque a Q-lines en la vela 1m cerrada (sin EPS)
                touch_uq = int(pd.notna(ch.get("UpperQ")) and (last_row["Low"] <= ch["UpperQ"] <= last_row["High"]))
                touch_lq = int(pd.notna(ch.get("LowerQ")) and (last_row["Low"] <= ch["LowerQ"] <= last_row["High"]))

                def _num(x):
                    try:
                        return round(float(x), 6)
                    except Exception:
                        return np.nan

                trow = {
                    "CloseTimeMs": last_ms,
                    "Date":   fmt_ts(last_idx),
                    "Open":   round(float(last_row["Open"]),  6),
                    "High":   round(float(last_row["High"]),  6),
                    "Low":    round(float(last_row["Low"]),   6),
                    "Close":  round(float(last_row["Close"]), 6),
                    "Volume": round(float(last_row["Volume"]),6),
                    "UpperMid":   _num(ch.get("UpperMid")),
                    "ValueUpper": _num(ch.get("ValueUpper")),
                    "LowerMid":   _num(ch.get("LowerMid")),
                    "ValueLower": _num(ch.get("ValueLower")),
                    "TouchUpperQ": touch_uq,
                    "TouchLowerQ": touch_lq,
                }
                _append_row_dedup_fast(TABLE_CSV_PATH, last_ms, trow)

                print(f"[{trow['Date']}] Open:{trow['Open']:>10} High:{trow['High']:>10} "
                      f"Low:{trow['Low']:>10} Close:{trow['Close']:>10} "
                      f"| UMid:{trow['UpperMid']} VUp:{trow['ValueUpper']} "
                      f"LMid:{trow['LowerMid']} VLo:{trow['ValueLower']} "
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
