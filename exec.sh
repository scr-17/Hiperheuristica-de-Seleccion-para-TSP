#!/usr/bin/env bash
clear
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="$SCRIPT_DIR/.venv"
VENV_PYTHON="$VENV_DIR/bin/python"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Error: no se encontró '$PYTHON_BIN'. Instala Python 3.12 o superior." >&2
    exit 1
fi

if ! "$PYTHON_BIN" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)'; then
    echo "Error: se requiere Python 3.12 o superior (detectado: $("$PYTHON_BIN" --version 2>&1))." >&2
    exit 1
fi

if [ ! -x "$VENV_PYTHON" ]; then
    echo "[1/4] Creando entorno virtual en $VENV_DIR ..."
    "$PYTHON_BIN" -m venv "$VENV_DIR"
else
    echo "[1/4] Reutilizando entorno virtual existente en $VENV_DIR"
fi

echo "[2/4] Instalando dependencias desde requirements.txt ..."
"$VENV_PYTHON" -m pip install --quiet --upgrade pip
"$VENV_PYTHON" -m pip install --quiet -r "$SCRIPT_DIR/requirements.txt"

echo "[3/4] Generando etiquetas (data/train_labeled.csv y data/test_labeled.csv) ..."
"$VENV_PYTHON" -m src.generate_labels

echo "[4/4] Entrenando y evaluando la hiperheurística ..."
"$VENV_PYTHON" -m src.hyperheuristic

echo
echo "Proceso completado. Revisa los resultados en data/"
