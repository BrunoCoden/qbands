# paginado_binance.py
# ---------------------------------------------------------
# Descarga histórico de klines con PAGINADO (hasta N velas).
# Devuelve un DataFrame con columnas:
#   ["Open","High","Low","Close","Volume","CloseTime","CloseTimeDT"]
# Index: DatetimeIndex en tu TZ (por env TZ).
#
# ENV opcionales:
#   BINANCE_UM_BASE_URL=https://fapi.binance.com
#   TZ=America/Argentina/Buenos_Aires
#   PAGINATE_PAGE_LIMIT=1500      # tope por request (máx 1500 en Binance)
#   PAGE_SLEEP_SEC=0.2            # pausa entre requests
# ---------------------------------------------------------

import os
import time
import pandas as pd
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

# Cliente Binance
try:
    from binance.um_futures import UMFutures
except Exception:
    print("ERROR: Falta 'binance-futures-connector'. Instalá:")
    print("  pip install binance-futures-connector")
    raise

def _get_client():
    base_url = os.getenv("BINANCE_UM_BASE_URL", "https://fapi.binance.com")
    return UMFutures(base_url=base_url)

# ms por intervalo (los más comunes)
INTERVAL_MS = {
    "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
    "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000, "6h": 21_600_000,
    "8h": 28_800_000, "12h": 43_200_000, "1d": 86_400_000
}

def fetch_klines_paginado(
    symbol: str,
    interval: str,
    total_bars: int,
    *,
    start_ms: int | None = None,
    end_ms: int | None = None,
    page_limit: int | None = None,
    sleep_sec: float | None = None,
    tz_name: str | None = None,
) -> pd.DataFrame:
    """
    Baja hasta total_bars velas paginando contra Binance.
    Si no especificás start/end, descarga hacia atrás desde 'now'.

    Retorna DataFrame con columnas:
      Open, High, Low, Close, Volume, CloseTime, CloseTimeDT
    Index TZ-aware (según TZ).
    """
    if interval not in INTERVAL_MS:
        raise ValueError(f"Interval no soportado para paginar: {interval}")

    page_limit = int(page_limit or os.getenv("PAGINATE_PAGE_LIMIT", 1500))
    page_limit = max(1, min(page_limit, 1500))  # Binance no da más de 1500
    sleep_sec = float(sleep_sec or os.getenv("PAGE_SLEEP_SEC", 0.2))
    tz_name = tz_name or os.getenv("TZ", "America/Argentina/Buenos_Aires")

    ms_per = INTERVAL_MS[interval]
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    end_ms = end_ms or now_ms
    start_ms = start_ms or (end_ms - total_bars * ms_per)

    client = _get_client()
    out = []
    fetched = 0
    curr_start = start_ms

    while fetched < total_bars:
        batch_limit = min(page_limit, total_bars - fetched)
        data = client.klines(
            symbol=symbol,
            interval=interval,
            startTime=curr_start,
            endTime=end_ms,
            limit=batch_limit
        )
        if not data:
            break

        out.extend(data)
        fetched += len(data)

        last_close = int(data[-1][6])
        next_start = last_close + 1
        # Si no avanzamos, evitamos loop infinito
        if next_start <= curr_start:
            break
        curr_start = next_start

        time.sleep(sleep_sec)

    if not out:
        return pd.DataFrame(columns=["Open","High","Low","Close","Volume","CloseTime","CloseTimeDT"])

    # Ensamble
    rows = []
    # Nos quedamos con las ÚLTIMAS total_bars por si juntamos de más
    for k in out[-total_bars:]:
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

    tz = ZoneInfo(tz_name)
    df = df.set_index(df["DateUTC"].dt.tz_convert(tz)).sort_index()
    df["CloseTimeDT"] = df["CloseTimeDT_UTC"].dt.tz_convert(tz)

    return df[["Open","High","Low","Close","Volume","CloseTime","CloseTimeDT"]]
