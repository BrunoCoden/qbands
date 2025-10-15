# alertsQ.py
# Envía alertas a Telegram cuando hay toque en UpperQ/LowerQ.
# Se usa como módulo desde script_principal_de_velas_solo_bandas.py

import os
import json
import time
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
TELEGRAM_BROADCAST_ALL = os.getenv("TELEGRAM_BROADCAST_ALL", "0").strip() in ("1", "true", "True")
TELEGRAM_TARGETS_PATH = os.getenv("TELEGRAM_TARGETS_PATH", "telegram_targets.json").strip()
TELEGRAM_EXTRA_CHAT_IDS = [cid.strip() for cid in os.getenv("TELEGRAM_EXTRA_CHAT_IDS", "").split(",") if cid.strip()]
TELEGRAM_REFRESH_UPDATES_SEC = int(os.getenv("TELEGRAM_REFRESH_UPDATES_SEC", "60") or "60")
TELEGRAM_BROADCAST_PRIVATE_ONLY = os.getenv("TELEGRAM_BROADCAST_PRIVATE_ONLY", "1").strip() not in ("0", "false", "False")
ALERTS_ENABLED = os.getenv("ALERTS_ENABLED", "1").strip() not in ("0", "false", "False")

SYMBOL_DISPLAY = os.getenv("SYMBOL", "ETHUSDT.P").strip()
STREAM_INTERVAL = os.getenv("STREAM_INTERVAL", "1m").strip()
CHANNEL_INTERVAL = os.getenv("CHANNEL_INTERVAL", "30m").strip()


class TelegramTargets:
    """Persist and manage unique chat ids collected via getUpdates."""

    def __init__(self, path: str):
        self.path = path
        self.data = {
            "last_update_id": 0,
            "targets": {},
            "last_fetch_ts": 0.0,
        }
        self.changed = False
        self._load()

    def _load(self) -> None:
        if not self.path:
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                incoming = json.load(f)
            if isinstance(incoming, dict):
                self.data.update(incoming)
        except FileNotFoundError:
            return
        except Exception:
            # ignore malformed file; keep defaults
            return
        finally:
            self.data.setdefault("last_update_id", 0)
            self.data.setdefault("targets", {})
            self.data.setdefault("last_fetch_ts", 0.0)

    @property
    def last_update_id(self) -> int:
        try:
            return int(self.data.get("last_update_id", 0))
        except Exception:
            return 0

    @last_update_id.setter
    def last_update_id(self, value: int) -> None:
        self.data["last_update_id"] = int(value)
        self.changed = True

    def should_refresh(self, interval_sec: int) -> bool:
        if interval_sec <= 0:
            return True
        last = float(self.data.get("last_fetch_ts", 0.0) or 0.0)
        return (time.time() - last) >= max(5, interval_sec)

    def mark_refreshed(self) -> None:
        self.data["last_fetch_ts"] = time.time()
        self.changed = True

    def add_chat(self, chat_id, chat_type: str, title: str) -> None:
        if chat_id is None:
            return
        chat_id_str = str(chat_id)
        chat_type = (chat_type or "").strip()
        title = title or ""
        existing = self.data.setdefault("targets", {}).get(chat_id_str)
        if not existing:
            self.data["targets"][chat_id_str] = {"type": chat_type, "title": title}
            self.changed = True
            return
        needs_update = (
            existing.get("type") != chat_type
            or (title and existing.get("title") != title)
        )
        if needs_update:
            existing["type"] = chat_type
            if title:
                existing["title"] = title
            self.changed = True

    def list_chat_ids(self, chat_types=None):
        targets = self.data.get("targets", {})
        if not chat_types:
            return list(targets.keys())
        allowed = set(chat_types)
        return [cid for cid, info in targets.items() if info.get("type") in allowed]

    def save(self) -> None:
        if not self.changed or not self.path:
            self.changed = False
            return
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        tmp_path = self.path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, self.path)
        self.changed = False


def _telegram_api(method: str, payload: dict):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/{method}"
    return _post(url, payload)


def _collect_targets(store: TelegramTargets) -> None:
    if not TELEGRAM_BOT_TOKEN:
        return
    offset = store.last_update_id + 1 if store.last_update_id else None
    payload = {
        "timeout": 0,
        "allowed_updates": [
            "message",
            "edited_message",
            "channel_post",
            "edited_channel_post",
            "my_chat_member",
            "chat_member",
        ],
    }
    if offset:
        payload["offset"] = offset

    try:
        resp = _telegram_api("getUpdates", payload)
    except Exception:
        store.mark_refreshed()
        store.save()
        return

    if not resp or not resp.get("ok"):
        store.mark_refreshed()
        store.save()
        return

    max_update_id = store.last_update_id
    for upd in resp.get("result", []):
        try:
            max_update_id = max(max_update_id, int(upd.get("update_id", 0)))
        except Exception:
            pass
        for field in (
            "message",
            "edited_message",
            "channel_post",
            "edited_channel_post",
            "my_chat_member",
            "chat_member",
        ):
            msg = upd.get(field)
            if not msg:
                continue
            chat = msg.get("chat") or {}
            chat_id = chat.get("id")
            chat_type = chat.get("type") or ""
            title = (
                chat.get("title")
                or chat.get("username")
                or chat.get("first_name")
                or ""
            )
            store.add_chat(chat_id, chat_type, title)
    store.last_update_id = max_update_id
    store.mark_refreshed()
    store.save()


def _resolve_recipients():
    recipients = []

    def _add(cid):
        if cid:
            recipients.append(str(cid))

    _add(TELEGRAM_CHAT_ID)
    for cid in TELEGRAM_EXTRA_CHAT_IDS:
        _add(cid)

    if TELEGRAM_BROADCAST_ALL and TELEGRAM_TARGETS_PATH:
        store = TelegramTargets(TELEGRAM_TARGETS_PATH)
        if store.should_refresh(TELEGRAM_REFRESH_UPDATES_SEC):
            _collect_targets(store)
        chat_types = ("private",) if TELEGRAM_BROADCAST_PRIVATE_ONLY else None
        for cid in store.list_chat_ids(chat_types=chat_types):
            _add(cid)

    ordered = []
    seen = set()
    for cid in recipients:
        if cid and cid not in seen:
            seen.add(cid)
            ordered.append(cid)
    return ordered

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

    base_payload = {
        "text": msg,
        "parse_mode": TELEGRAM_PARSE_MODE,
        "disable_web_page_preview": True,
    }

    recipients = _resolve_recipients()
    if not recipients:
        return {"status": "skipped", "reason": "no recipients"}

    results = []
    for chat_id in recipients:
        payload = dict(base_payload)
        payload["chat_id"] = chat_id
        if TELEGRAM_THREAD_ID and str(chat_id) == str(TELEGRAM_CHAT_ID):
            try:
                payload["message_thread_id"] = int(TELEGRAM_THREAD_ID)
            except Exception:
                pass
        try:
            resp = _telegram_api("sendMessage", payload)
            if isinstance(resp, dict) and resp.get("ok") is False:
                results.append({
                    "chat_id": chat_id,
                    "status": "error",
                    "error": resp.get("description", "telegram response not ok"),
                    "response": resp,
                })
            else:
                results.append({
                    "chat_id": chat_id,
                    "status": "ok",
                    "response": resp,
                })
        except Exception as e:
            results.append({
                "chat_id": chat_id,
                "status": "error",
                "error": str(e),
            })

    overall = "ok" if any(r.get("status") == "ok" for r in results) else "error"
    return {"status": overall, "results": results}

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
