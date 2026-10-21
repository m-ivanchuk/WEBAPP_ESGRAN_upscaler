#!/usr/bin/env bash
# Start the page when a Codespace opens. Skip if it is already listening.
set -euo pipefail
cd "$(dirname "$0")/.."

if python -c "import socket; s=socket.create_connection(('127.0.0.1', 8765), 2)"; then
  echo "Upscale already running: http://127.0.0.1:8765"
  exit 0
fi

exec python web/server.py --no-browser --host 0.0.0.0 --port 8765
