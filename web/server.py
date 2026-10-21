#!/usr/bin/env python3
"""Local page for batch Real-ESRGAN image and video upscale.

    python web/server.py

Then open http://127.0.0.1:8765
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from web.engine import (  # noqa: E402
    ALPHA_MODES,
    MODELS,
    SCALES,
    TILES,
    Job,
    describe_media,
    ffmpeg_available,
    list_media,
    runtime_status,
    weights_present,
)

INDEX = Path(__file__).with_name("index.html")
JOB = Job()
RUNTIME = {"value": None, "lock": threading.Lock()}


def meta() -> dict:
    with RUNTIME["lock"]:
        if RUNTIME["value"] is None:
            RUNTIME["value"] = runtime_status()
        info = dict(RUNTIME["value"])
    info.update(
        models=[{key: item[key] for key in ("id", "label", "blurb", "netscale", "denoise")} for item in MODELS],
        scales=list(SCALES),
        alpha_modes=list(ALPHA_MODES),
        tiles=list(TILES),
        weights=weights_present(),
        ffmpeg=ffmpeg_available(),
    )
    return info


def scan(payload: dict) -> dict:
    raw = str(payload.get("input_dir") or "").strip()
    if not raw:
        return {"ok": False, "error": "Укажите папку."}
    folder = Path(raw).expanduser().resolve()
    if not folder.is_dir():
        return {"ok": False, "error": "Папка не найдена."}
    files = list_media(folder, bool(payload.get("recursive")))
    info = describe_media(files)
    info.update(ok=True, path=str(folder))
    return info


def pick_folder(prompt: str) -> dict:
    if sys.platform != "darwin":
        return {"ok": False, "error": "Кнопка выбора работает на macOS. Вставьте путь вручную."}
    try:
        result = subprocess.run(
            ["osascript", "-e", f'POSIX path of (choose folder with prompt "{prompt}")'],
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "Окно выбора закрыто по таймауту."}
    if result.returncode != 0:
        return {"ok": True, "cancelled": True, "path": ""}
    path = result.stdout.strip().rstrip("/")
    return {"ok": True, "cancelled": False, "path": path}


def reveal(path_raw: str) -> dict:
    folder = Path(str(path_raw or "")).expanduser()
    if not folder.is_dir():
        return {"ok": False, "error": "Папка ещё не создана."}
    if sys.platform == "darwin":
        command = ["open", str(folder)]
    elif sys.platform == "win32":
        command = ["explorer", str(folder)]
    else:
        command = ["xdg-open", str(folder)]
    subprocess.run(command, check=False)
    return {"ok": True}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            body = INDEX.read_bytes()
            self._send(200, "text/html; charset=utf-8", body)
            return
        if path == "/api/meta":
            self._json(meta())
            return
        if path == "/api/status":
            self._json(JOB.status())
            return
        self._send(404, "text/plain; charset=utf-8", b"not found")

    def do_POST(self):
        path = urlparse(self.path).path
        payload = self._read_json()
        if path == "/api/scan":
            self._json(scan(payload))
            return
        if path == "/api/pick":
            which = payload.get("which")
            prompt = "Папка для результатов" if which == "output" else "Папка с фото или видео"
            self._json(pick_folder(prompt))
            return
        if path == "/api/start":
            error = JOB.start(payload)
            if error:
                code = 409 if error.startswith("Уже идёт") else 400
                self._json({"ok": False, "error": error}, code)
                return
            self._json({"ok": True})
            return
        if path == "/api/stop":
            JOB.stop()
            self._json({"ok": True})
            return
        if path == "/api/reveal":
            self._json(reveal(payload.get("path")))
            return
        self._send(404, "text/plain; charset=utf-8", b"not found")

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        if length > 1_000_000:
            return {}
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            return {}
        return data if isinstance(data, dict) else {}

    def _json(self, data: dict, code: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self._send(code, "application/json; charset=utf-8", body)

    def _send(self, code: int, content_type: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def main():
    parser = argparse.ArgumentParser(description="Local Real-ESRGAN image and video upscaler")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1", help="Bind address. Use 0.0.0.0 in GitHub Codespaces.")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://127.0.0.1:{args.port}"
    print(f"Upscale: {url}", flush=True)
    if not args.no_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstop")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
