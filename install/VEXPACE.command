#!/usr/bin/env bash
# macOS double-click launcher
cd "$(dirname "$0")/.."
if [ ! -d .venv ]; then bash install/install_mac_linux.sh; fi
source .venv/bin/activate
python -m picstudio gui
