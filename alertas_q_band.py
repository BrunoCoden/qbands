# ---------------------------------------------------------
# Lee el CSV (1m) y dispara alertas por toques de Q.
# Enriquecido con niveles del canal (si faltan, los calcula).
# Incluye backfill al iniciar, dedupe y auto-test de Telegram.
# ---------------------------------------------------------

import os
import time
import json
import platform
from typing import Optional, List, Tuple, Set
import pandas as pd

# ---------- ENV ----------
def _load_envs():
    try:
        from dotenv import load_dotenv
        if os.path.exists(".env"):
            load_dotenv(".env")
        if os.path.exists("alerts.env"):
            load_dotenv("alerts.env")
    except Exception:
        pass
_load_envs()

CSV_PATH        = os.getenv("CSV_PATH", os.getenv("TABLE_CSV_PATH", "tabla.csv")).strip()
SYMBOL          = os.getenv("SYMBOL", "ETHUSDT.P")
POLL_SEC        = int(os.getenv("ALERT_POLL_SEC", "2"))

ENABLE_BEEP     = os.getenv("ENABLE_BEEP", "1") == "1"
ENABLE_TOAST    = os.getenv("ENABLE_TOAST", "0") == "1"
WEBHOOK_URL     = os.getenv("ALERT_WEBHOOK_URL", "").strip()

TG_TOKEN        = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TG_CHAT_ID      = os.getenv("TELEGRAM_CHAT_ID", "").strip()
TG_BROADCAST    = os.getenv("TELEGRAM_BROADCAST_ALL", "0") == "1"
TG_TARGETS_PATH = os.getenv("TELEGRAM_TARGETS_PATH", "telegram_targets.json").strip()
TG_REFRESH_SEC  = int(os.getenv("TELEGRAM_REFRESH_UPDATES_SEC", "60"))
ALERT_TEST      = os.getenv("ALERT_TEST", "0") == "1"

# Backfill y dedupe
BACKFILL_N      = int(os.getenv("ALERT_BACKFILL_N", "50"))
BACKFILL_SEND   = os.getenv("ALERT_BACKFILL_SEND", "1") == "1"

TF_LABEL        = os.getenv("ALERT_TF_LABEL", os.getenv("STREAM_INTERVAL", "1m"))
TZ_NAME         = os.getenv("TZ", "America/Argentina/Buenos_Aires")

# ---------- Red opcional ----------
try:
    import requests
except Exception:
    requests = None

# ---------- Reusar lógica del principal (para calcular niveles si faltan) ----------
_USE_IMPORT = True
try:
    from script_principal_de_velas_solo_bandas import (
        fetch_klines, compute_channels, API_SYMBOL,
        CHANNEL_INTERVAL, LIMIT_CHANNEL, RB_MULTI, RB_INIT_BAR
    )
except Exception:
    _USE_IMPORT = False
    API_SYMBOL       = SYMBOL.replace(".P", "")
    CHANNEL_INTERVAL = os.getenv("CHANNEL_INTERVAL", "30m")
    LIMIT_CHANNEL    = int(os.getenv("LIMIT_CHANNEL", "800"))
    RB_MULTI         = float(os.getenv("RB_MULTI", "4.0"))
    RB_INIT_BAR      = int(os.getenv("RB_INIT_BAR", "301"))
    try:
        from binance.um_futures import UMFutures
        from zoneinfo import ZoneInfo
        from datetime import datetime, timezone
        import numpy as np
        def _get_binance_client():
            base_url = os.getenv("BINANCE_UM_BASE_URL", "https://fapi.binance.com").strip()
            return UMFutures(base_url=base_url)
        def fetch_klines(symbol: str, interval: str, limit: int) -> pd.DataFrame:
            client = _get_binance_client()
            data = client.klines(symbol=symbol, interval=interval, limit=max(1, min(int(limit), 1500)))
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
    except Exception:
        pass

# ---------- Utils ----------
def _fmt_row(r) -> str:
    date  = r.get("Date", "")
    open_ = r.get("Open", "")
    high  = r.get("High", "")
    low   = r.get("Low", "")
    close = r.get("Close", "")
    vol   = r.get("Volume", "")
    tuq   = r.get("TouchUpperQ", "")
    tlq   = r.get("TouchLowerQ", "")
    return (
        f"Date: {date} | "
        f"Open: {open_} | High: {high} | Low: {low} | Close: {close} | "
        f"Volume: {vol} | TouchUpperQ: {tuq} | TouchLowerQ: {tlq}"
    )

def _fmt_price(x) -> str:
    try:
        return f"{float(x):.6f}"
    except Exception:
        return "n/a"

def _beep():
    if not ENABLE_BEEP:
        return
    try:
        if platform.system().lower().startswith("win"):
            import winsound
            winsound.Beep(880, 200)
            winsound.Beep(660, 150)
        else:
            print("\a", end="", flush=True)
    except Exception:
        pass

def _toast(title: str, msg: str):
    if not ENABLE_TOAST:
        return
    try:
        from win10toast import ToastNotifier
        ToastNotifier().show_toast(title, msg, duration=4, threaded=True)
    except Exception:
        pass

def _post_webhook(payload: dict):
    if not WEBHOOK_URL or not requests:
        return
    try:
        requests.post(WEBHOOK_URL, json=payload, timeout=5)
    except Exception:
        pass

def _to01(x) -> int:
    """Normaliza cualquier cosa parecida a 1/0 a enteros 1/0."""
    try:
        if x is None:
            return 0
        if isinstance(x, str):
            s = x.strip().lower()
            if s in ("true", "t", "yes", "y"): return 1
            return 1 if float(s) >= 1 else 0
        return 1 if float(x) >= 1 else 0
    except Exception:
        return 0

# ---------- Telegram ----------
class TelegramTargets:
    def __init__(self, path: str):
        self.path = path
        self.data = {"last_update_id": 0, "targets": {}}
        self._load()
    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
                self.data.setdefault("last_update_id", 0)
                self.data.setdefault("targets", {})
            except Exception:
                self.data = {"last_update_id": 0, "targets": {}}
    def save(self):
        try:
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.path)
        except Exception:
            pass
    @property
    def last_update_id(self) -> int:
        return int(self.data.get("last_update_id", 0))
    @last_update_id.setter
    def last_update_id(self, v: int):
        self.data["last_update_id"] = int(v)
    def add_chat(self, chat_id: int | str, chat_type: str, title: str):
        chat_id = str(chat_id)
        if chat_id not in self.data["targets"]:
            self.data["targets"][chat_id] = {"type": chat_type, "title": title or ""}
        else:
            self.data["targets"][chat_id]["type"] = chat_type
            if title:
                self.data["targets"][chat_id]["title"] = title
    def list_chat_ids(self) -> List[str]:
        return list(self.data.get("targets", {}).keys())

def tg_api(method: str, payload: dict):
    if not (TG_TOKEN and requests):
        print("[WARN] Telegram desactivado: falta TOKEN o 'requests'.")
        return None
    url = f"https://api.telegram.org/bot{TG_TOKEN}/{method}"
    try:
        r = requests.post(url, json=payload, timeout=10)
        if r.status_code != 200:
            print(f"[WARN] Telegram HTTP {r.status_code}: {r.text}")
            return None
        return r.json()
    except Exception as e:
        print("[WARN] Telegram request error:", e)
        return None

def tg_send_text(chat_id: str | int, text: str):
    resp = tg_api("sendMessage", {"chat_id": chat_id, "text": text})
    if not resp or not resp.get("ok"):
        print("[WARN] Telegram no envió. resp=", resp)

def tg_self_test():
    if not (TG_TOKEN and TG_CHAT_ID and requests):
        print("[INFO] Telegram self-test omitido (sin token/chat/requests).")
        return
    # getMe
    gm = tg_api("getMe", {})
    print("[INFO] getMe:", gm)
    # test send
    tg_send_text(TG_CHAT_ID, f"✅ Alert bot listo | SYMBOL={SYMBOL} | TF={TF_LABEL}")

def tg_collect_updates(store: TelegramTargets):
    offset = store.last_update_id + 1 if store.last_update_id else None
    payload = {"timeout": 0, "allowed_updates": ["message", "my_chat_member", "chat_member", "channel_post"]}
    if offset:
        payload["offset"] = offset
    resp = tg_api("getUpdates", payload)
    if not resp or not resp.get("ok", False):
        return
    max_update_id = store.last_update_id
    for upd in resp.get("result", []):
        max_update_id = max(max_update_id, int(upd.get("update_id", 0)))
        for field in ["message", "channel_post", "my_chat_member", "chat_member"]:
            msg = upd.get(field)
            if msg and "chat" in msg:
                chat = msg["chat"]
                chat_id = chat.get("id")
                chat_type = chat.get("type")
                title = chat.get("title") or chat.get("username") or chat.get("first_name") or ""
                if chat_id is not None:
                    store.add_chat(chat_id, chat_type, title)
    if max_update_id > store.last_update_id:
        store.last_update_id = max_update_id
        store.save()

# ---------- CSV ----------
def _load_df_safe() -> Optional[pd.DataFrame]:
    if not os.path.exists(CSV_PATH):
        return None
    try:
        return pd.read_csv(CSV_PATH)
    except Exception:
        return None

# ---------- Canal provider (lookup por timestamp) ----------
from zoneinfo import ZoneInfo
def _parse_row_ts(row: pd.Series):
    try:
        ts = pd.to_datetime(row.get("Date"))
        if ts.tzinfo is None:
            ts = ts.tz_localize(ZoneInfo(TZ_NAME))
        return ts
    except Exception:
        return None

class ChannelLevelsProvider:
    """ Mantiene canales y permite consultar niveles ffill. """
    def __init__(self):
        self.last_key = None
        self.chans = pd.DataFrame()

    def _refresh_if_needed(self):
        df_ch = fetch_klines(API_SYMBOL, os.getenv("CHANNEL_INTERVAL", "30m"), LIMIT_CHANNEL)
        if df_ch is None or df_ch.empty:
            return
        if len(df_ch) >= 2:
            key = int(df_ch.iloc[-2]["CloseTime"])  # última CH cerrada
        else:
            key = int(df_ch.iloc[-1]["CloseTime"])
        if (self.last_key is None) or (self.last_key != key) or self.chans.empty:
            ohlc = df_ch[["Open","High","Low","Close","Volume"]]
            self.chans = compute_channels(ohlc, multi=RB_MULTI, init_bar=RB_INIT_BAR)
            self.last_key = key

    def get_levels(self, ts) -> Optional[pd.Series]:
        try:
            self._refresh_if_needed()
            if self.chans is None or self.chans.empty or ts is None:
                return None
            sel = self.chans.loc[:ts]
            if sel.empty:
                return None
            c = sel.iloc[-1]
            return c[["UpperMid","ValueUpper","LowerMid","ValueLower"]]
        except Exception:
            return None

_levels = ChannelLevelsProvider()

def _levels_from_csv_or_compute(row: pd.Series) -> Tuple[Optional[float],Optional[float],Optional[float],Optional[float]]:
    def _num(x):
        try:
            return float(x)
        except Exception:
            return None
    um = _num(row.get("UpperMid"))
    vu = _num(row.get("ValueUpper"))
    lm = _num(row.get("LowerMid"))
    vl = _num(row.get("ValueLower"))
    if all(v is not None for v in [um, vu, lm, vl]):
        return um, vu, lm, vl
    ts = _parse_row_ts(row)
    levels = _levels.get_levels(ts)
    if levels is None or levels.empty:
        return um, vu, lm, vl
    um = um if um is not None else _num(levels.get("UpperMid"))
    vu = vu if vu is not None else _num(levels.get("ValueUpper"))
    lm = lm if lm is not None else _num(levels.get("LowerMid"))
    vl = vl if vl is not None else _num(levels.get("ValueLower"))
    return um, vu, lm, vl

# ---------- Notificación ----------
def _notify_all(signal: str, row: pd.Series):
    um, vu, lm, vl = _levels_from_csv_or_compute(row)
    header = f"{SYMBOL} — {signal} | TF: {TF_LABEL}"
    base   = _fmt_row(row)

    extra = ""
    if "UpperQ" in signal:
        extra = f" | UpperMid: {_fmt_price(um)} | ValueUpper: {_fmt_price(vu)}"
    elif "LowerQ" in signal or "ambos Q" in signal:
        extra = f" | LowerMid: {_fmt_price(lm)} | ValueLower: {_fmt_price(vl)}"
        if "ambos Q" in signal:
            extra += f" | UpperMid: {_fmt_price(um)} | ValueUpper: {_fmt_price(vu)}"

    text = f"{header}\n{base}{extra}"

    print(f"[ALERTA] {header} | {base}{extra}")
    _beep()
    _toast(header, base + extra)
    _post_webhook({"symbol": SYMBOL, "signal": signal, "tf": TF_LABEL, "data": dict(row), "levels": {
        "UpperMid": um, "ValueUpper": vu, "LowerMid": lm, "ValueLower": vl
    }})

    if not (TG_TOKEN and requests):
        return
    if TG_CHAT_ID:
        try:
            tg_send_text(TG_CHAT_ID, text)
        except Exception as e:
            print("[WARN] tg_send_text error:", e)
    if TG_BROADCAST:
        store = TelegramTargets(TG_TARGETS_PATH)
        try:
            tg_collect_updates(store)
        except Exception:
            pass
        for cid in store.list_chat_ids():
            if TG_CHAT_ID and str(cid) == str(TG_CHAT_ID):
                continue
            try:
                tg_send_text(cid, text)
            except Exception:
                continue

# ---------- CSV ----------
def _load_df_ordered() -> Optional[pd.DataFrame]:
    df = _load_df_safe()
    if df is None or df.empty or "CloseTimeMs" not in df.columns:
        return None
    df["CloseTimeMs"] = pd.to_numeric(df["CloseTimeMs"], errors="coerce")
    df = df.dropna(subset=["CloseTimeMs"]).sort_values("CloseTimeMs")
    return df

# ---------- Loop ----------
def run_alerts():
    print(f"[INFO] Alertas CSV: {CSV_PATH} | SYMBOL={SYMBOL} | TF_LABEL={TF_LABEL} | poll={POLL_SEC}s")
    if TG_TOKEN:
        print(f"[INFO] Telegram bot activo. Broadcast={'ON' if TG_BROADCAST else 'OFF'} | targets={TG_TARGETS_PATH}")

    # Self-test Telegram explícito
    tg_self_test()

    # Test manual de alerta si se pidió
    if ALERT_TEST:
        dummy = pd.Series({"Date": "TEST", "Open": 0, "High": 0, "Low": 0, "Close": 0, "Volume": 0,
                           "TouchUpperQ": 1, "TouchLowerQ": 0})
        _notify_all("🚀 TEST ALERT (UpperQ)", dummy)

    sent_keys: Set[int] = set()
    last_key = None

    # -------- Backfill inicial --------
    df0 = _load_df_ordered()
    if df0 is not None:
        try:
            if BACKFILL_N > 0:
                tail = df0.tail(BACKFILL_N)
                print(f"[INFO] Backfill inicial sobre {len(tail)} filas")
                for _, r in tail.iterrows():
                    ck = int(r["CloseTimeMs"])
                    if ck in sent_keys:
                        continue
                    tuq = _to01(r.get("TouchUpperQ"))
                    tlq = _to01(r.get("TouchLowerQ"))
                    print(f"[DEBUG] backfill ck={ck} tuq={tuq} tlq={tlq}")
                    if BACKFILL_SEND:
                        if tuq == 1 and tlq != 1:
                            _notify_all("TOUCH UpperQ (backfill)", r)
                            sent_keys.add(ck)
                        elif tlq == 1 and tuq != 1:
                            _notify_all("TOUCH LowerQ (backfill)", r)
                            sent_keys.add(ck)
                        elif tuq == 1 and tlq == 1:
                            _notify_all("TOUCH ambos Q (backfill)", r)
                            sent_keys.add(ck)
            last_key = int(df0["CloseTimeMs"].iloc[-1])
        except Exception as e:
            print("[WARN] Backfill falló:", e)

    last_refresh = 0.0
    while True:
        try:
            df = _load_df_ordered()
            if df is None:
                time.sleep(POLL_SEC)
                continue

            new_rows = df.tail(1) if last_key is None else df[df["CloseTimeMs"] > last_key]

            for _, r in new_rows.iterrows():
                ck = int(r["CloseTimeMs"])
                tuq = _to01(r.get("TouchUpperQ"))
                tlq = _to01(r.get("TouchLowerQ"))
                print(f"[DEBUG] procesando ck={ck} tuq={tuq} tlq={tlq}")

                if ck not in sent_keys:
                    if tuq == 1 and tlq != 1:
                        _notify_all("TOUCH UpperQ", r)
                        sent_keys.add(ck)
                    elif tlq == 1 and tuq != 1:
                        _notify_all("TOUCH LowerQ", r)
                        sent_keys.add(ck)
                    elif tuq == 1 and tlq == 1:
                        _notify_all("TOUCH ambos Q (Upper & Lower)", r)
                        sent_keys.add(ck)

                last_key = ck

            now = time.time()
            if TG_TOKEN and TG_BROADCAST and requests and (now - last_refresh >= TG_REFRESH_SEC):
                try:
                    store = TelegramTargets(TG_TARGETS_PATH)
                    tg_collect_updates(store)
                except Exception:
                    pass
                last_refresh = now

            time.sleep(POLL_SEC)
        except KeyboardInterrupt:
            print("\n[EXIT] Cortado por usuario.")
            break
        except Exception as e:
            print(f"[WARN] {type(e).__name__}: {e}")
            time.sleep(max(2, POLL_SEC))

if __name__ == "__main__":
    run_alerts()
