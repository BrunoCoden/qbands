# alerts_watcher_Q.py
# Observa tablaQ.csv y envía alertas a Telegram cuando TouchUpperQ/TouchLowerQ == 1
# No modifica el script principal. Se apoya en alertsQ.py y en la salida tablaQ.csv.

import os
import time
import json
import csv
import io
from datetime import datetime
from typing import Dict, Any, Optional

from dotenv import load_dotenv

# Requiere el módulo de envío a Telegram que ya hicimos
import alertsQ

load_dotenv()

# ===================== Config =====================
TABLE_CSV_PATH   = os.getenv("TABLE_CSV_PATH", "tablaQ.csv").strip()
POLL_SECONDS     = float(os.getenv("ALERTS_POLL_SECONDS", "2.0"))  # frecuencia de chequeo
STATE_PATH       = os.getenv("ALERTS_STATE_PATH", ".alertsQ.state.json").strip()
# Si querés filtrar por símbolo o por ventanas, podrías ampliar acá sin tocar el principal.

# Columnas esperadas (header fijo del CSV)
EXPECTED_COLUMNS = [
    "Date","Open","High","Low","Close",
    "UpperMid","ValueUpper","LowerMid","ValueLower",
    "TouchUpperQ","TouchLowerQ"
]

# ===================== Estado =====================
def _load_state() -> Dict[str, Any]:
    if not os.path.exists(STATE_PATH):
        return {"file": None, "offset": 0, "last_date": None}
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"file": None, "offset": 0, "last_date": None}

def _save_state(state: Dict[str, Any]) -> None:
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    os.replace(tmp, STATE_PATH)

def _file_id(path: str) -> Optional[str]:
    try:
        st = os.stat(path)
        # Usamos inodo + tamaño como id de rotación simple
        return f"{st.st_dev}:{st.st_ino}"
    except Exception:
        return None

# ===================== Utils =====================
def _parse_int(x: Any) -> int:
    try:
        return int(float(x))
    except Exception:
        return 0

def _read_header(fp) -> Optional[list]:
    pos = fp.tell()
    first = fp.readline()
    if not first:
        return None
    # retroceder si hace falta
    # parseo CSV robusto por si hay comas en algún futuro campo
    fp.seek(pos)
    reader = csv.reader(io.TextIOWrapper(fp.buffer, encoding="utf-8", newline=""))
    try:
        header = next(reader)
    except StopIteration:
        return None
    # normalizar espacios
    return [h.strip() for h in header]

def _iterate_new_rows(fp, start_offset: int):
    """
    Itera las filas nuevas a partir de start_offset.
    Devuelve (nuevas_filas_yielded, nuevo_offset)
    """
    fp.seek(start_offset)
    # Si estamos en medio de línea, avanzar a la siguiente
    if start_offset > 0:
        fp.readline()

    start_pos = fp.tell()

    # Leemos el header cuando start_offset == 0
    reader = csv.DictReader(io.TextIOWrapper(fp.buffer, encoding="utf-8", newline=""))
    if reader.fieldnames is None:
        return 0, start_pos

    yielded = 0
    for row in reader:
        yielded += 1
        yield row
    # calcular nuevo offset al final
    fp.seek(0, os.SEEK_END)
    end_off = fp.tell()
    return yielded, end_off

def _validate_header(path: str) -> bool:
    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
        header = [h.strip() for h in header]
        return header == EXPECTED_COLUMNS
    except Exception:
        return False

def _trow_from_row(row: Dict[str, Any]) -> Dict[str, Any]:
    # Asegurar claves y tipos mínimos
    def _safe_get(k):
        v = row.get(k, "")
        return v if v is not None else ""
    t = {k: _safe_get(k) for k in EXPECTED_COLUMNS}
    # Convertir toques a int 0/1 por si vienen como strings
    t["TouchUpperQ"] = _parse_int(t.get("TouchUpperQ"))
    t["TouchLowerQ"] = _parse_int(t.get("TouchLowerQ"))
    return t

# ===================== Core loop =====================
def process_once(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Procesa una pasada: detecta rotación/cambio de archivo, lee filas nuevas y envía alertas.
    Devuelve el estado actualizado.
    """
    if not os.path.exists(TABLE_CSV_PATH):
        print(f"[WARN] No existe {TABLE_CSV_PATH}. Nada que hacer.")
        return state

    # Validar encabezado
    if not _validate_header(TABLE_CSV_PATH):
        print(f"[WARN] El encabezado de {TABLE_CSV_PATH} no coincide con las columnas esperadas.")
        print(f"       Esperado: {', '.join(EXPECTED_COLUMNS)}")
        # igual no abortamos; podrías elegir abortar si querés
    fid = _file_id(TABLE_CSV_PATH)

    current_size = os.path.getsize(TABLE_CSV_PATH)
    if state.get("offset", 0) > current_size:
        state["offset"] = 0

    try:
        with open(TABLE_CSV_PATH, "rb") as fbin:
            # Si cambió el archivo (rotación/tamaño), reiniciar offset a línea 2 (después del header)
            if state.get("file") != fid:
                state["file"] = fid
                state["offset"] = 0
                print("[INFO] Nuevo archivo o rotacion detectada. Reinicio offset al inicio.")
                return state

            # Continuar desde el offset guardado
            fbin.seek(state.get("offset", 0))
            # A partir de acá, leer filas nuevas
            tfp = io.TextIOWrapper(fbin, encoding="utf-8", newline="")
            if state.get("offset", 0) == 0:
                reader = csv.DictReader(tfp)
            else:
                reader = csv.DictReader(tfp, fieldnames=EXPECTED_COLUMNS)
            new_count = 0
            last_date_seen = state.get("last_date")

            for row in reader:
                new_count += 1
                trow = _trow_from_row(row)

                # De-dup adicional por Date (opcional)
                # Si el archivo se reescribe, offset puede engañar. Comparamos con la última fecha procesada.
                date_str = trow.get("Date") or ""
                if last_date_seen and date_str and date_str <= last_date_seen:
                    continue
                # Enviar alertas
                if trow["TouchUpperQ"] == 1 or trow["TouchLowerQ"] == 1:
                    pass

                last_date_seen = date_str

            # Guardar nuevo offset al final del archivo
            fbin.seek(0, os.SEEK_END)
            end_off = fbin.tell()
            if new_count > 0:
                print(f"[INFO] Procesadas {new_count} filas nuevas. ultimo offset={end_off}")
            state["offset"] = 0
            if last_date_seen:
                state["last_date"] = last_date_seen

            return state
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print(f"[WARN] {type(e).__name__}: {e}")
        return state

def main():
    print(f"[INIT] Watcher de alertas sobre: {TABLE_CSV_PATH}")
    print(f"[INIT] Estado en: {STATE_PATH} | poll cada {POLL_SECONDS}s")

    sample_now = datetime.now().isoformat(timespec='seconds')
    alertsQ.send_trade_open_alert(
        side="long",
        context="lower",
        entry_time=sample_now,
        entry_price=1234.5,
        reference_mid=1225.0,
        reference_value=1200.0,
    )
    alertsQ.send_trade_close_alert(
        {
            "side": "short",
            "context": "upper",
            "entry_time": sample_now,
            "exit_time": datetime.now().isoformat(timespec='seconds'),
            "entry_price": 1300.0,
            "exit_price": 1257.0,
            "exit_reason": "target_hit",
            "profit_target_pct": 3.0,
            "stop_pct": 2.0,
            "pnl_pct": 3.31,
            "bars_held": 5,
            "reference_mid": 1280.0,
            "reference_value": 1305.0,
        }
    )

    state = _load_state()
    # Primer pasada: si el archivo cambió, reinicia offset tras header
    state = process_once(state)
    _save_state(state)

    while True:
        try:
            time.sleep(POLL_SECONDS)
            state = process_once(state)
            _save_state(state)
        except KeyboardInterrupt:
            print("\n[EXIT] Cortado por usuario.")
            break

if __name__ == "__main__":
    main()
