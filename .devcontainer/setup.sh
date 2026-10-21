#!/usr/bin/env bash
# One-time Codespaces setup: ffmpeg and a CPU build of PyTorch.
set -euo pipefail
cd "$(dirname "$0")/.."

sudo apt-get update
sudo apt-get install -y ffmpeg

python -m pip install -U pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
