#!/usr/bin/env bash
# VEXPACE installer for macOS / Linux: creates a private virtual environment in ./.venv
set -e
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
if ! command -v $PY >/dev/null 2>&1; then
  echo "Python 3 not found. Install it from https://www.python.org/downloads/ (3.9 or newer)."; exit 1
fi
echo "Using $($PY --version)"
$PY -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[gui]"
read -r -p "Install MPI support (mpi4py + mpich, optional)? [y/N] " ans
if [[ "$ans" == "y" || "$ans" == "Y" ]]; then python -m pip install mpich mpi4py || echo "MPI install failed (optional)"; fi
python -m picstudio info
echo "Pre-compiling kernels (first run only)..."
python -m picstudio run examples/child_langmuir_1d.pic.json --out /tmp/picstudio_warmup --steps 20 >/dev/null 2>&1 || true
echo
echo "Done.  Start the GUI with:   ./install/run_gui.sh   (or double-click install/VEXPACE.command on macOS)"
