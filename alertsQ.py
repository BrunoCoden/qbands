# alertsQ.py
# Envía alertas a Telegram cuando hay toque en UpperQ/LowerQ.
# Se usa como módulo desde script_principal_de_velas_solo_bandas.py

import os
import json
from datetime import datetime
from dotenv import load_dotenv

# Sin dependencias externas: preferimos requests si está, sino urllib
try:
    import requests
    _USE_REQUESTS = True
except Exception:
    import urllib.request
    _USE_REQUESTS = False

load_dotenv()

# Config desde .env
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "").strip()          # chat o user id
TELEGRAM_THREAD_ID = os.getenv("TELEGRAM_THREAD_ID", "").strip() or None # para topics (opcional)
TELEGRAM_PARSE_MODE = os.getenv("TELEGRAM_PARSE_MODE", "MarkdownV2").strip()  # MarkdownV2 o HTML
ALERTS_ENABLED = os.getenv("ALERTS_ENABLED", "1").strip() not in ("0", "false", "False")

SYMBOL_DISPLAY = os.getenv("SYMBOL", "ETHUSDT.P").strip()
STREAM_INTERVAL = os.getenv("STREAM_INTERVAL", "1m").strip()
CHANNEL_INTERVAL = os.getenv("CHANNEL_INTERVAL", "30m").strip()

def _require_cfg():
    if not ALERTS_ENABLED:
        return "alerts disabled"
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return "missing token/chat_id"
    return None

def _post(url: str, payload: dict):
    data = json.dumps(payload).encode("utf-8")
    if _USE_REQUESTS:
        r = requests.post(url, json=payload, timeout=10)
        r.raise_for_status()
        return r.json()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))

def _escape_md(s: str) -> str:
    # Escapado basico para MarkdownV2 de Telegram
    if TELEGRAM_PARSE_MODE != "MarkdownV2":
        return s
    # caracteres a escapar según docs de Telegram
    for ch in r"_*[]()~`>#+-=|{}.!":
        s = s.replace(ch, "\\" + ch)
    return s

def _trunc3(x):
    try:
        return int(float(x) * 1000) / 1000.0
    except Exception:
        return x

def _fmt_price(x):
    try:
        return f"{_trunc3(x):.3f}"
    except Exception:
        return "-"

def _build_message(side: str, trow: dict, symbol: str = None) -> str:
    sym = symbol or SYMBOL_DISPLAY
    date = trow.get("Date", "")
    o = _fmt_price(trow.get("Open"))
    h = _fmt_price(trow.get("High"))
    l = _fmt_price(trow.get("Low"))
    c = _fmt_price(trow.get("Close"))
    umid = _fmt_price(trow.get("UpperMid"))
    vup  = _fmt_price(trow.get("ValueUpper"))
    lmid = _fmt_price(trow.get("LowerMid"))
    vlo  = _fmt_price(trow.get("ValueLower"))

    # MarkdownV2 seguro
    title = _escape_md(f"{sym} {STREAM_INTERVAL} | Touch {side}Q")
    line1 = _escape_md(f"{date}")
    line2 = _escape_md(f"O {o}  H {h}  L {l}  C {c}")
    line3 = _escape_md(f"Bands({CHANNEL_INTERVAL}):  VUp {vup}  UMid {umid}  |  VLo {vlo}  LMid {lmid}")

    return f"*{title}*\n{line1}\n{line2}\n{line3}"

def send_touch_alert(side: str, trow: dict, symbol: str = None):
    """
    Envía una alerta a Telegram.
    Parámetros:
      - side: "Upper" o "Lower"
      - trow: dict con las claves Date, Open, High, Low, Close, UpperMid, ValueUpper, LowerMid, ValueLower
      - symbol: opcional, para override de símbolo
    """
    err = _require_cfg()
    if err:
        # Silencioso si faltan credenciales o deshabilitado
        return {"status": "skipped", "reason": err}

    msg = _build_message(side, trow, symbol)
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": msg,
        "parse_mode": TELEGRAM_PARSE_MODE,
        "disable_web_page_preview": True,
    }
    if TELEGRAM_THREAD_ID:
        payload["message_thread_id"] = int(TELEGRAM_THREAD_ID)

    try:
        res = _post(url, payload)
        return {"status": "ok", "result": res}
    except Exception as e:
        return {"status": "error", "error": str(e)}

# Modo CLI opcional: enviar prueba rápida
if __name__ == "__main__":
    sample = {
        "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Open":  1234.5678,
        "High":  1250.4321,
        "Low":   1222.1111,
        "Close": 1240.9999,
        "UpperMid":   1300.12,
        "ValueUpper": 1320.66,
        "LowerMid":   1200.45,
        "ValueLower": 1180.33,
    }
    print(send_touch_alert("Upper", sample))
