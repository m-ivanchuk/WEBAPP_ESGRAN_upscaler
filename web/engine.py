"""Batch image and video upscale on top of RealESRGANer.

Torch and BasicSR are imported only when a job actually starts, so the
local page can open before the ML stack is installed.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

WEIGHTS_DIR = ROOT / "weights"

# Same networks and weight URLs as inference_realesrgan.py.
MODELS = [
    {
        "id": "RealESRGAN_x4plus",
        "label": "RealESRGAN_x4plus — универсальная",
        "blurb": "Основная модель для обычных картинок. Сеть увеличивает в 4 раза.",
        "netscale": 4,
        "denoise": False,
        "kind": "rrdb",
        "blocks": 23,
        "conv": None,
        "urls": ["https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"],
    },
    {
        "id": "RealESRGAN_x2plus",
        "label": "RealESRGAN_x2plus — универсальная ×2",
        "blurb": "Сеть сразу делает ×2. Лучший вариант, если итог нужен ровно в два раза.",
        "netscale": 2,
        "denoise": False,
        "kind": "rrdb",
        "blocks": 23,
        "conv": None,
        "urls": ["https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"],
    },
    {
        "id": "RealESRNet_x4plus",
        "label": "RealESRNet_x4plus — мягкая",
        "blurb": "Без GAN: ровнее и меньше выдуманной текстуры, если x4plus слишком резкая.",
        "netscale": 4,
        "denoise": False,
        "kind": "rrdb",
        "blocks": 23,
        "conv": None,
        "urls": ["https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth"],
    },
    {
        "id": "RealESRGAN_x4plus_anime_6B",
        "label": "Anime 6B — иллюстрации",
        "blurb": "Для аниме и рисованных картинок. Меньше и быстрее, чем универсальная ×4.",
        "netscale": 4,
        "denoise": False,
        "kind": "rrdb",
        "blocks": 6,
        "conv": None,
        "urls": [
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
        ],
    },
    {
        "id": "realesr-animevideov3",
        "label": "AnimeVideo v3 — быстрая",
        "blurb": "Самая лёгкая модель для аниме. Быстро и мало памяти, детали слабее.",
        "netscale": 4,
        "denoise": False,
        "kind": "srvgg",
        "blocks": None,
        "conv": 16,
        "urls": ["https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth"],
    },
    {
        "id": "realesr-general-x4v3",
        "label": "General v3 — компактная",
        "blurb": "Маленькая универсальная модель. Есть сила шумоподавления: 0 оставляет шум, 1 сильнее сглаживает.",
        "netscale": 4,
        "denoise": True,
        "kind": "srvgg",
        "blocks": None,
        "conv": 32,
        "urls": [
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth",
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
        ],
    },
]

MODEL_BY_ID = {item["id"]: item for item in MODELS}
SCALES = (2, 3, 4)
ALPHA_MODES = ("realesrgan", "bicubic")
TILES = (0, 128, 256, 512)
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi"}


class Cancelled(Exception):
    """The user pressed stop while a file was still running."""


def list_media(root: Path, recursive: bool) -> list[Path]:
    """PNG, JPEG, and video files. Hidden files (including macOS AppleDouble) are skipped."""
    if recursive:
        found = [path for path in root.rglob("*") if _is_media(path)]
    else:
        found = [path for path in root.iterdir() if _is_media(path)]
    return sorted(found, key=lambda path: str(path.relative_to(root)).lower())


def describe_media(files: list[Path]) -> dict:
    images = sum(1 for path in files if _is_image(path))
    videos = sum(1 for path in files if _is_video(path))
    preview = [path.name for path in files[:6]]
    if not files:
        summary = "В папке нет PNG, JPEG или видео."
        label = "Обработать"
    else:
        parts = []
        if images:
            parts.append(f"{images} фото")
        if videos:
            parts.append(f"{videos} видео")
        summary = "Найдено: " + ", ".join(parts)
        if preview:
            summary += " — " + ", ".join(preview)
        label = "Обработать " + _ru_files(len(files))
    return {
        "count": len(files),
        "images": images,
        "videos": videos,
        "preview": preview,
        "summary": summary,
        "label": label,
    }


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _is_image(path: Path) -> bool:
    return path.is_file() and not path.name.startswith(".") and path.suffix.lower() in IMAGE_SUFFIXES


def _is_video(path: Path) -> bool:
    return path.is_file() and not path.name.startswith(".") and path.suffix.lower() in VIDEO_SUFFIXES


def _is_media(path: Path) -> bool:
    return _is_image(path) or _is_video(path)


def _ru_files(count: int) -> str:
    mod10 = count % 10
    mod100 = count % 100
    if mod10 == 1 and mod100 != 11:
        word = "файл"
    elif mod10 in (2, 3, 4) and mod100 not in (12, 13, 14):
        word = "файла"
    else:
        word = "файлов"
    return f"{count} {word}"


def weights_present() -> dict[str, bool]:
    return {item["id"]: all((WEIGHTS_DIR / url.rsplit("/", 1)[-1]).is_file() for url in item["urls"]) for item in MODELS}


def _patch_basicsr_torchvision() -> None:
    """basicsr 1.4.2 still imports a torchvision module removed in current releases."""
    import sys
    import types

    name = "torchvision.transforms.functional_tensor"
    if name in sys.modules:
        return
    try:
        __import__(name)
    except ModuleNotFoundError:
        from torchvision.transforms.functional import rgb_to_grayscale

        module = types.ModuleType(name)
        module.rgb_to_grayscale = rgb_to_grayscale
        sys.modules[name] = module


def runtime_status() -> dict:
    """Whether the ML stack imports, and which compute devices exist."""
    status = {
        "deps": False,
        "deps_message": "",
        "cuda": False,
        "mps": False,
        "device_label": "не проверено",
    }
    try:
        import torch
        _patch_basicsr_torchvision()
    except Exception as exc:  # noqa: BLE001 — report any install problem to the page
        status["deps_message"] = (
            "PyTorch не установлен, апскейл не запустится. "
            "Из корня репозитория: pip install -r requirements.txt"
        )
        status["device_label"] = "нет PyTorch"
        status["import_error"] = str(exc)
        return status

    missing = []
    for module_name in ("cv2", "basicsr"):
        try:
            __import__(module_name)
        except Exception as exc:  # noqa: BLE001
            missing.append(f"{module_name} ({exc})")
    if missing:
        status["deps_message"] = (
            "Не хватает зависимостей: " + ", ".join(missing) + ". Команда: pip install -r requirements.txt"
        )
        status["device_label"] = "зависимости не собраны"
        return status

    cuda = bool(torch.cuda.is_available())
    mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    if cuda:
        label = "NVIDIA CUDA"
    elif mps:
        label = "Apple MPS"
    else:
        label = "CPU"
    status.update(deps=True, deps_message="", cuda=cuda, mps=mps, device_label=label)
    return status


class Job:
    """One batch at a time. The page polls status()."""

    def __init__(self):
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None
        self._state = _idle_state()

    def status(self) -> dict:
        with self._lock:
            return dict(self._state)

    def start(self, options: dict) -> str | None:
        """Validate options and start a worker. Returns an error string, or None."""
        try:
            prepared = _prepare(options)
        except ValueError as exc:
            return str(exc)

        with self._lock:
            if self._state["running"]:
                return "Уже идёт обработка. Дождитесь конца или нажмите «Остановить»."
            self._cancel.clear()
            self._state = {
                "running": True,
                "phase": "model",
                "total": len(prepared["files"]),
                "done": 0,
                "ok": 0,
                "failed": 0,
            "current": "",
            "frame": 0,
            "frames": 0,
            "message": "Готовлю модель…",
                "lines": [],
                "output_dir": str(prepared["output_dir"]),
                "error": "",
            }
            self._thread = threading.Thread(target=self._worker, args=(prepared,), daemon=True)
            self._thread.start()
        return None

    def stop(self) -> None:
        self._cancel.set()
        self._set(message="Остановка после текущего кадра…")

    def _worker(self, prepared: dict) -> None:
        try:
            upsampler = self._load_model(prepared)
            if upsampler is None:
                return
            self._set(phase="files", message="Обрабатываю файлы…")
            for index, src in enumerate(prepared["files"], start=1):
                if self._cancel.is_set():
                    self._finish_stopped()
                    return
                rel = src.relative_to(prepared["input_dir"])
                dest = _result_path(src, prepared["input_dir"], prepared["output_dir"])
                shown = dest.relative_to(prepared["output_dir"])
                self._set(current=str(shown), frame=0, frames=0, message=f"{index} / {prepared['total']}: {shown}")
                try:
                    if _is_video(src):
                        note = _upscale_video(
                            upsampler,
                            src,
                            dest,
                            outscale=prepared["outscale"],
                            cancel=self._cancel,
                            on_frame=lambda frame, total, name=shown, index=index, count=prepared["total"]: self._set(
                                frame=frame,
                                frames=total,
                                message=f"{index} / {count}: {name} — кадр {frame}" + (f"/{total}" if total else ""),
                            ),
                        )
                    else:
                        note = ""
                        _upscale_image(
                            upsampler,
                            src,
                            dest,
                            outscale=prepared["outscale"],
                            alpha=prepared["alpha"],
                        )
                except Cancelled:
                    self._finish_stopped()
                    return
                except Exception as exc:  # noqa: BLE001 — keep the batch going
                    self._mark_file(ok=False, line=f"ошибка  {shown} — {exc}")
                else:
                    line = f"готово   {shown}"
                    if note:
                        line += f" ({note})"
                    self._mark_file(ok=True, line=line)
            if self._cancel.is_set():
                self._finish_stopped()
                return
            with self._lock:
                ok = self._state["ok"]
                failed = self._state["failed"]
            message = f"Готово: {ok} файлов"
            if failed:
                message += f", ошибок: {failed}"
            self._set(running=False, phase="done", current="", message=message)
        except Exception as exc:  # noqa: BLE001
            self._set(running=False, phase="error", current="", error=str(exc), message=str(exc))

    def _load_model(self, prepared: dict):
        self._set(message="Загружаю модель. При первом запуске веса скачаются в weights/…")
        if self._cancel.is_set():
            self._finish_stopped()
            return None
        try:
            return build_upsampler(
                model_id=prepared["model_id"],
                tile=prepared["tile"],
                denoise=prepared["denoise"],
                device_choice=prepared["device"],
                half=prepared["half"],
            )
        except ImportError:
            text = (
                "Не установлены зависимости Real-ESRGAN (PyTorch, BasicSR, OpenCV). "
                "Из корня репозитория: pip install -r requirements.txt"
            )
            self._set(running=False, phase="error", error=text, message=text)
            return None
        except Exception as exc:  # noqa: BLE001
            text = str(exc)
            lowered = text.lower()
            if "memory" in lowered or "mps" in lowered:
                text += " Попробуйте тайл 256 или 128."
            self._set(running=False, phase="error", error=text, message=text)
            return None

    def _mark_file(self, ok: bool, line: str) -> None:
        with self._lock:
            self._state["done"] += 1
            if ok:
                self._state["ok"] += 1
            else:
                self._state["failed"] += 1
            lines = self._state["lines"] + [line]
            self._state["lines"] = lines[-400:]

    def _finish_stopped(self) -> None:
        with self._lock:
            ok = self._state["ok"]
        self._set(
            running=False,
            phase="done",
            current="",
            message=f"Остановлено. Успело сохраниться файлов: {ok}",
        )

    def _set(self, **fields) -> None:
        with self._lock:
            self._state.update(fields)


def build_upsampler(model_id: str, tile: int, denoise: float, device_choice: str, half: bool):
    import os

    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    import torch

    _patch_basicsr_torchvision()
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from basicsr.utils.download_util import load_file_from_url

    # Load the upscaler modules without realesrgan/__init__.py: that file also
    # imports the training stack and a generated version.py that is not in git.
    SRVGGNetCompact, RealESRGANer = _load_upscaler_classes()

    spec = MODEL_BY_ID[model_id]
    if spec["kind"] == "rrdb":
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=spec["blocks"],
            num_grow_ch=32,
            scale=spec["netscale"],
        )
    else:
        model = SRVGGNetCompact(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_conv=spec["conv"],
            upscale=spec["netscale"],
            act_type="prelu",
        )

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    for url in spec["urls"]:
        filename = url.rsplit("/", 1)[-1]
        if not (WEIGHTS_DIR / filename).is_file():
            load_file_from_url(url=url, model_dir=str(WEIGHTS_DIR), progress=True, file_name=filename)

    model_path: str | list[str] = str(WEIGHTS_DIR / f"{model_id}.pth")
    dni_weight = None
    if model_id == "realesr-general-x4v3" and denoise != 1:
        wdn_path = model_path.replace("realesr-general-x4v3", "realesr-general-wdn-x4v3")
        model_path = [model_path, wdn_path]
        dni_weight = [denoise, 1 - denoise]

    device = _resolve_device(torch, device_choice)
    use_half = bool(half and device.type == "cuda")
    if half and device.type != "cuda":
        use_half = False

    return RealESRGANer(
        scale=spec["netscale"],
        model_path=model_path,
        dni_weight=dni_weight,
        model=model,
        tile=tile,
        tile_pad=10,
        pre_pad=0,
        half=use_half,
        device=device,
    )


def _load_upscaler_classes():
    import importlib
    import types

    package_dir = ROOT / "realesrgan"
    if "realesrgan" not in sys.modules:
        package = types.ModuleType("realesrgan")
        package.__path__ = [str(package_dir)]
        package.__package__ = "realesrgan"
        sys.modules["realesrgan"] = package
    archs_name = "realesrgan.archs"
    if archs_name not in sys.modules:
        archs = types.ModuleType(archs_name)
        archs.__path__ = [str(package_dir / "archs")]
        archs.__package__ = archs_name
        sys.modules[archs_name] = archs
    srvgg = importlib.import_module("realesrgan.archs.srvgg_arch")
    utils = importlib.import_module("realesrgan.utils")
    return srvgg.SRVGGNetCompact, utils.RealESRGANer


def _resolve_device(torch, choice: str):
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA недоступна на этой машине.")
        return torch.device("cuda")
    if choice == "mps":
        if not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
            raise RuntimeError("Apple MPS недоступен. Оставьте устройство «авто» или выберите CPU.")
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _result_path(src: Path, input_dir: Path, output_dir: Path) -> Path:
    relative = src.relative_to(input_dir)
    if _is_video(src):
        relative = relative.with_suffix(".mp4")
    return output_dir / relative


def _upscale_image(upsampler, src: Path, dest: Path, outscale: float, alpha: str) -> None:
    import cv2

    image = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError("файл не читается")
    output, _mode = upsampler.enhance(image, outscale=outscale, alpha_upsampler=alpha)
    dest.parent.mkdir(parents=True, exist_ok=True)
    suffix = dest.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        params = [cv2.IMWRITE_JPEG_QUALITY, 95]
        fail = "не удалось записать JPEG"
    else:
        params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
        fail = "не удалось записать PNG"
    if not cv2.imwrite(str(dest), output, params):
        raise RuntimeError(fail)


def _upscale_video(upsampler, src: Path, dest: Path, outscale: float, cancel: threading.Event, on_frame) -> str:
    """Upscale a video frame by frame. Returns a short note, such as a missing audio track."""
    if not ffmpeg_available():
        raise RuntimeError("для видео нужен ffmpeg (на Mac: brew install ffmpeg)")

    import numpy as np

    meta = _probe_video(src)
    out_w = int(meta["width"] * outscale)
    out_h = int(meta["height"] * outscale)
    out_w -= out_w % 2
    out_h -= out_h % 2
    if out_w < 2 or out_h < 2:
        raise RuntimeError("слишком маленький кадр")

    dest.parent.mkdir(parents=True, exist_ok=True)
    silent = dest.with_suffix(".silent.mp4")
    frame_bytes = meta["width"] * meta["height"] * 3
    reader = _ffmpeg_reader(src)
    writer = _ffmpeg_writer(silent, out_w, out_h, meta["fps"])
    frame_index = 0
    try:
        while True:
            if cancel.is_set():
                raise Cancelled()
            raw = _read_exact(reader.stdout, frame_bytes)
            if raw is None:
                break
            frame = np.frombuffer(raw, np.uint8).reshape(meta["height"], meta["width"], 3)
            output, _mode = upsampler.enhance(frame, outscale=outscale)
            output = np.ascontiguousarray(output[:out_h, :out_w])
            if output.shape[0] != out_h or output.shape[1] != out_w:
                raise RuntimeError("модель вернула кадр неожиданного размера")
            writer.stdin.write(output.tobytes())
            frame_index += 1
            on_frame(frame_index, meta["frames"])
        _finish_ffmpeg(reader, "ffmpeg не смог прочитать видео")
        _finish_ffmpeg(writer, "ffmpeg не смог записать видео")
    except Exception:
        _kill_ffmpeg(reader)
        _kill_ffmpeg(writer)
        silent.unlink(missing_ok=True)
        raise

    if frame_index == 0:
        silent.unlink(missing_ok=True)
        raise RuntimeError("в видео нет кадров")
    if not meta["audio"]:
        silent.replace(dest)
        return ""
    note = _mux_audio(silent, src, dest)
    silent.unlink(missing_ok=True)
    return note


def _probe_video(path: Path) -> dict:
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,width,height,avg_frame_rate,nb_frames",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or "").strip() or "ffprobe завершился с ошибкой"
        raise RuntimeError(detail)
    import json

    payload = json.loads(proc.stdout or "{}")
    streams = payload.get("streams") or []
    video = next((item for item in streams if item.get("codec_type") == "video"), None)
    if video is None:
        raise RuntimeError("в файле нет видеодорожки")
    width = int(video.get("width") or 0)
    height = int(video.get("height") or 0)
    if width < 1 or height < 1:
        raise RuntimeError("не удалось узнать размер кадра")
    frames_text = str(video.get("nb_frames") or "")
    frames = int(frames_text) if frames_text.isdigit() else 0
    return {
        "width": width,
        "height": height,
        "fps": _frame_rate(str(video.get("avg_frame_rate") or "")),
        "frames": frames,
        "audio": any(item.get("codec_type") == "audio" for item in streams),
    }


def _frame_rate(value: str) -> float:
    if "/" in value:
        num, den = value.split("/", 1)
        try:
            denominator = float(den)
            rate = float(num) / denominator if denominator else 0
        except ValueError:
            rate = 0
    else:
        try:
            rate = float(value)
        except ValueError:
            rate = 0
    return rate if rate > 0 else 24.0


def _ffmpeg_reader(path: Path):
    return subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )


def _ffmpeg_writer(path: Path, width: int, height: int, fps: float):
    return subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            f"{fps:.6f}",
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            str(path),
        ],
        stdin=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )


def _read_exact(stream, size: int) -> bytes | None:
    chunks = []
    remaining = size
    while remaining:
        piece = stream.read(remaining)
        if not piece:
            break
        chunks.append(piece)
        remaining -= len(piece)
    if not chunks:
        return None
    data = b"".join(chunks)
    if len(data) != size:
        return None
    return data


def _finish_ffmpeg(proc, message: str) -> None:
    if proc.stdin:
        proc.stdin.close()
    if proc.stdout:
        proc.stdout.read()
    code = proc.wait()
    if code != 0:
        raise RuntimeError(message)


def _kill_ffmpeg(proc) -> None:
    if proc.poll() is None:
        proc.kill()
        proc.wait()


def _mux_audio(video: Path, source: Path, dest: Path) -> str:
    proc = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(video),
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c",
            "copy",
            "-shortest",
            str(dest),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode == 0:
        return ""
    video.replace(dest)
    return "без звука"


def _prepare(options: dict) -> dict:
    model_id = str(options.get("model") or "")
    if model_id not in MODEL_BY_ID:
        raise ValueError("Неизвестная модель.")

    try:
        outscale = int(options.get("scale"))
    except (TypeError, ValueError) as exc:
        raise ValueError("Масштаб должен быть 2, 3 или 4.") from exc
    if outscale not in SCALES:
        raise ValueError("Масштаб должен быть 2, 3 или 4.")

    alpha = str(options.get("alpha") or "realesrgan")
    if alpha not in ALPHA_MODES:
        raise ValueError("Неизвестный режим альфа-канала.")

    try:
        tile = int(options.get("tile"))
    except (TypeError, ValueError) as exc:
        raise ValueError("Неизвестный размер тайла.") from exc
    if tile not in TILES:
        raise ValueError("Тайл должен быть 0, 128, 256 или 512.")

    try:
        denoise = float(options.get("denoise", 0.5))
    except (TypeError, ValueError) as exc:
        raise ValueError("Сила шумоподавления должна быть числом от 0 до 1.") from exc
    if not 0 <= denoise <= 1:
        raise ValueError("Сила шумоподавления должна быть от 0 до 1.")

    device = str(options.get("device") or "auto")
    if device not in ("auto", "cpu", "mps", "cuda"):
        raise ValueError("Неизвестное устройство.")

    input_dir = _as_dir(options.get("input_dir"), "Папка с исходниками")
    output_text = str(options.get("output_dir") or "").strip()
    if not output_text:
        raise ValueError("Папка для результатов: укажите путь.")
    output_dir = Path(output_text).expanduser().resolve()
    if input_dir == output_dir or _is_inside(output_dir, input_dir):
        raise ValueError("Папка результатов должна быть вне папки с исходниками, чтобы не затереть файлы.")

    files = list_media(input_dir, bool(options.get("recursive")))
    if not files:
        raise ValueError("В папке нет PNG, JPEG или видео.")
    output_dir = _as_dir(output_dir, "Папка для результатов", create=True)

    return {
        "model_id": model_id,
        "outscale": float(outscale),
        "alpha": alpha,
        "tile": tile,
        "denoise": denoise,
        "device": device,
        "half": bool(options.get("half")),
        "input_dir": input_dir,
        "output_dir": output_dir,
        "files": files,
        "total": len(files),
    }


def _as_dir(raw, title: str, create: bool = False) -> Path:
    text = str(raw or "").strip()
    if not text:
        raise ValueError(f"{title}: укажите путь.")
    path = Path(text).expanduser().resolve()
    if create:
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ValueError(f"{title}: не удалось создать папку.") from exc
    if not path.is_dir():
        raise ValueError(f"{title}: папка не найдена.")
    return path


def _is_inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _idle_state() -> dict:
    return {
        "running": False,
        "phase": "idle",
        "total": 0,
        "done": 0,
        "ok": 0,
        "failed": 0,
        "current": "",
        "frame": 0,
        "frames": 0,
        "message": "",
        "lines": [],
        "output_dir": "",
        "error": "",
    }
