# Backtesting Q-Bands – Guía Rápida

## 1. Preparación
- Repositorio localizado en `~/bot 5 octubre/binance-candles-bot-nuevo/binance-candles-bot`.
- Asegurate de tener el entorno virtual creado (`python -m venv .venv`) y actualizado (`pip install -r requirements.txt`).
- Variables de entorno en `.env` con credenciales de Binance y Telegram (`TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, etc.).

## 2. Arranque Express
```bash
cd "~/bot 5 octubre/binance-candles-bot-nuevo/binance-candles-bot"
./run_live.sh
```
El script levanta cuatro terminales:
1. `script_principal_de_velas_solo_bandas.py` – genera/actualiza `tablaQ.csv`.
2. `alerts_watcher_Q.py` – monitorea el CSV y envía alertas de operaciones (arranca con dos alertas de prueba).
3. `Backtesting/realtime_backtest.py --print-summary` – recalcula backtesting sobre las velas vivas y actualiza `Backtesting/live_summary.csv` + `Backtesting/live_trades.csv`.
4. `python -m http.server 9010` dentro de `Backtesting/` – sirve el dashboard.

## 3. Dashboard en Vivo
1. Abrí `http://localhost:9010/dashboard/`.
2. Dejá activado “Auto refrescar cada 30s”.
3. Paneles:
   - **Resumen en vivo**: toma la última fila de `live_summary.csv`.
   - **Tabla / Gráficos**: se llenan cuando hay operaciones en `live_trades.csv`.

## 4. Flujo Manual (si no usás `run_live.sh`)
```bash
source .venv/bin/activate
python script_principal_de_velas_solo_bandas.py        # genera tablaQ.csv
python alerts_watcher_Q.py                             # alertas Telegram (envía pruebas al iniciar)
python Backtesting/realtime_backtest.py --print-summary
cd Backtesting && python -m http.server 9010
```

## 5. Alertas Telegram
- **Apertura**: mensaje con símbolo, sentido (LONG/SHORT), Q origen, precio de entrada y bandas (`Value`/`Mid`).
- **Cierre**: incluye motivo (`target_hit`/`stop_hit`), resultado (`PnL`), TP/SL configurados, barras que estuvo abierta y bandas de referencia.
- El watcher envía siempre dos alertas de prueba (apertura + cierre ficticios) al iniciar; sirven para verificar conectividad.

## 6. Archivos y Limpieza
- Resultados del backtest en vivo:
  - `Backtesting/live_summary.csv`
  - `Backtesting/live_trades.csv`
- Si necesitás “limpiar” el repo, detené los procesos y elimina esos archivos; el backtester los recreará en el próximo arranque.

## 7. Depuración Rápida
- **Sin velas nuevas** → revisar `script_principal_de_velas_solo_bandas.py`.
- **Sin alertas** → chequeá `.env` (token/chat), `telegram_targets.json` y los logs del watcher.
- **Sin trades en dashboard** → corroborá `live_trades.csv`; aparece solo cuando se cierra la primera operación.

Listo. Con estos pasos tenés el backtesting en tiempo real, alertas y visualización funcionando en minutos.
