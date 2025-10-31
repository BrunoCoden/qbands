#!/usr/bin/env bash

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="$REPO_DIR/.venv/bin/activate"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

find_terminal() {
  for term in gnome-terminal xfce4-terminal mate-terminal konsole xterm; do
    if command -v "$term" >/dev/null 2>&1; then
      echo "$term"
      return 0
    fi
  done
  return 1
}

launch_in_terminal() {
  local title="$1"
  local workdir="$2"
  local body="$3"

  local term
  if ! term="$(find_terminal)"; then
    return 1
  fi

  case "$term" in
    gnome-terminal|mate-terminal)
      "$term" --title="$title" -- bash -lc "cd \"$workdir\" && $body; echo; read -rp 'Proceso finalizado. Pulse Enter para cerrar...' _" &
      ;;
    xfce4-terminal)
      "$term" --title="$title" --hold -x bash -lc "cd \"$workdir\" && $body" &
      ;;
    konsole)
      "$term" --new-tab -p tabtitle="$title" -e bash -lc "cd \"$workdir\" && $body; echo; read -rp 'Proceso finalizado. Pulse Enter para cerrar...' _" &
      ;;
    xterm)
      xterm -T "$title" -e bash -lc "cd \"$workdir\" && $body; echo; read -rp 'Proceso finalizado. Pulse Enter para cerrar...' _" &
      ;;
    *)
      return 1
      ;;
  esac
}

check_venv() {
  if [[ ! -f "$VENV_PATH" ]]; then
    log "ERROR: No se encontró $VENV_PATH. Creá el entorno con 'python -m venv .venv'"
    exit 1
  fi
}

main() {
  log "Iniciando run_live.sh"
  check_venv

  local setup_cmd="source \"$VENV_PATH\""

  declare -A TASKS=(
    ["Velas en vivo"]="${setup_cmd}; python script_principal_de_velas_solo_bandas.py"
    ["Watcher alertas"]="${setup_cmd}; python alerts_watcher_Q.py"
    ["Backtest realtime"]="${setup_cmd}; python Backtesting/realtime_backtest.py --print-summary"
    ["Dashboard backtesting"]="${setup_cmd}; cd Backtesting && python -m http.server 9010"
  )

  local term
  if ! term="$(find_terminal)"; then
    log "No se encontró un emulador de terminal compatible. Ejecutando todo en esta terminal."
    for title in "${!TASKS[@]}"; do
      log "Ejecutando ${title} en segundo plano"
      ( cd "$REPO_DIR" && bash -lc "${TASKS[$title]}" ) &
      log "${title} lanzado (PID $!)"
    done
  else
    for title in "${!TASKS[@]}"; do
      log "Abriendo terminal para: ${title}"
      if launch_in_terminal "$title" "$REPO_DIR" "${TASKS[$title]}"; then
        log "${title}: terminal lanzada correctamente"
      else
        log "${title}: error al abrir terminal; ejecutando en esta ventana"
        ( cd "$REPO_DIR" && bash -lc "${TASKS[$title]}" ) &
        log "${title} lanzado (PID $!)"
      fi
    done
  fi

  log "Todos los procesos fueron lanzados. Revisá las terminales para el detalle."
}

main "$@"
