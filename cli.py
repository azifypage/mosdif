"""mosdif Command Line Interface (CLI).

Run video mosaic restoration directly from the terminal without Gradio or Web UI.
Supports single video processing, batch directory processing, model status checks,
and automatic model downloading.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import threading
from pathlib import Path

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

from mosaicdiff.comfy_setup import comfy_ready, discover_comfy
from mosaicdiff.fetch import ensure_weights, FetchError
from mosaicdiff.paths import MODEL_LABELS, default_output_dir, models_dir
from mosaicdiff.settings import Settings


class Cancelled(Exception):
    pass


VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}


def print_system_info() -> None:
    print("=" * 65)
    print("mosdif CLI - Video Mosaic Restoration Pipeline")
    print("Engine: BasicVSR++ & LTX-2.5 Uncensored Turbo GGUF")
    print("=" * 65)
    if HAS_TORCH and torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU Device : {device_name} ({vram_gb:.2f} GB VRAM)")
    else:
        print("GPU Device : Tidak terdeteksi NVIDIA GPU (CUDA tidak tersedia!)")
    print(f"Platform   : {sys.platform}")
    print("-" * 65)


def check_models(settings: Settings) -> bool:
    print_system_info()
    print("STATUS MODEL & DEPENDENSI:")
    all_ok = True
    for key in ("detector", "vsr", "unet", "lora", "clip", "vae"):
        resolved = settings.resolved(key)
        name = MODEL_LABELS.get(key, key)
        if resolved.is_file():
            size_mb = resolved.stat().st_size / (1024 * 1024)
            size_str = f"{size_mb / 1024:.2f} GB" if size_mb >= 1024 else f"{size_mb:.1f} MB"
            print(f"  [OK]       {name:<30} -> {resolved.name} ({size_str})")
        elif key == "lora":
            print(f"  [OPTIONAL] {name:<30} -> Baked-in pada LTX-2.5 GGUF checkpoint")
        else:
            print(f"  [MISSING]  {name:<30} -> Belum ada ({resolved})")
            all_ok = False

    print("\nSTATUS COMFYUI BACKEND:")
    discover_comfy(settings, lambda _: None)
    c_root = settings.resolved("comfy_root")
    c_py = settings.resolved("comfy_python")
    c_ok = comfy_ready(settings)
    print(f"  Folder ComfyUI : {c_root} ({'OK' if c_root.is_dir() else 'TIDAK DITEMUKAN'})")
    print(f"  Python ComfyUI : {c_py} ({'OK' if c_py.is_file() else 'TIDAK DITEMUKAN'})")
    if not c_ok:
        all_ok = False
        print("  Status Backend : Belum Siap (Pastikan ComfyUI dan ComfyUI-GGUF terpasang)")
    else:
        print("  Status Backend : Terhubung dan Siap")

    print("=" * 65)
    return all_ok


def download_models(settings: Settings) -> None:
    print_system_info()
    print("Memulai pemeriksaan dan pengunduhan bobot model yang kurang...")
    cancel = threading.Event()

    def log_cb(msg: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    def progress_cb(fraction: float) -> None:
        bar_len = 30
        filled = int(round(bar_len * fraction))
        bar = "=" * filled + "-" * (bar_len - filled)
        sys.stdout.write(f"\r  [{bar}] {fraction * 100:.1f}%")
        sys.stdout.flush()

    try:
        ensure_weights(settings, log_cb, cancel, progress_cb)
        print("\n\nSemua bobot model telah terpasang dan siap digunakan!")
    except KeyboardInterrupt:
        cancel.set()
        print("\nPengunduhan dibatalkan.")
    except Exception as exc:
        print(f"\nError saat mengunduh model: {exc}")


def process_single_video(
    source: Path,
    output_path: Path,
    settings: Settings,
    cancel_event: threading.Event,
) -> bool:
    print(f"\n>>> Memproses: {source.name}")
    print(f"    Output   : {output_path}")
    print(f"    Settings : Steps={settings.ltx_steps}, Res={settings.ltx_resolution}px, Window={settings.ltx_seconds}s")

    last_desc = [""]

    def log_cb(msg: str) -> None:
        sys.stdout.write("\n" if last_desc[0] else "")
        last_desc[0] = ""
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    def progress_cb(fraction: float, desc: str = "") -> None:
        bar_len = 25
        frac = max(0.0, min(1.0, float(fraction)))
        filled = int(round(bar_len * frac))
        bar = "#" * filled + "-" * (bar_len - filled)
        d_str = f" - {desc}" if desc else ""
        sys.stdout.write(f"\r  [{bar}] {frac * 100:5.1f}%{d_str:<35}")
        sys.stdout.flush()
        last_desc[0] = desc

    start_time = time.time()
    try:
        from mosaicdiff.pipeline import Cancelled as PipelineCancelled, process_video
        written = process_video(
            source=source,
            destination=output_path,
            settings=settings,
            log=log_cb,
            progress=progress_cb,
            cancel=cancel_event,
        )
        sys.stdout.write("\n")
        elapsed = time.time() - start_time
        print(f"[OK] Selesai dalam {elapsed:.1f} detik: {written.resolve()}")
        return True
    except (Cancelled, PipelineCancelled):
        sys.stdout.write("\n")
        print(f"[STOP] Pemrosesan {source.name} dihentikan pengguna.")
        return False
    except Exception as exc:
        sys.stdout.write("\n")
        print(f"[FAIL] Gagal memproses {source.name}: {exc}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="mosdif CLI - Video Mosaic Restoration (Terminal Mode)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=None,
        help="File video (.mp4, .mkv, .mov, dll) atau folder yang berisi video untuk diproses.",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Path file output atau folder tujuan hasil restorasi.\n(Default: subfolder 'output/' di direktori saat ini)",
    )
    parser.add_argument(
        "-s", "--steps",
        type=int,
        default=None,
        help="Jumlah sampling steps LTX-2.5 Turbo (default: 4, rekomendasi: 4-8).",
    )
    parser.add_argument(
        "-r", "--resolution",
        type=int,
        default=None,
        help="Resolusi area inpainting LTX-2.5 (default: 512, snapped ke kelipatan 32).",
    )
    parser.add_argument(
        "-w", "--seconds",
        type=int,
        default=None,
        help="Panjang waktu sampel video per window LTX-2.5 dalam detik (default: 3).",
    )
    parser.add_argument(
        "--check-models",
        action="store_true",
        help="Cek ketersediaan semua file model dan status ComfyUI di terminal.",
    )
    parser.add_argument(
        "--download-models",
        action="store_true",
        help="Otomatis unduh semua file model LTX-2.5 & VSR yang kurang dari HuggingFace.",
    )
    parser.add_argument(
        "--comfy-root",
        default=None,
        help="Override path folder root ComfyUI (misal: /content/ComfyUI).",
    )
    parser.add_argument(
        "--comfy-python",
        default=None,
        help="Override path executable Python ComfyUI (misal: /usr/bin/python3).",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        default=None,
        help="Prompt teks deskriptif untuk inpainting LTX-2.5.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed acak untuk inpainting LTX-2.5 (default: 42).",
    )

    args = parser.parse_args()

    settings = Settings.load()
    if args.comfy_root:
        settings.paths["comfy_root"] = args.comfy_root.strip()
    if args.comfy_python:
        settings.paths["comfy_python"] = args.comfy_python.strip()
    if args.steps is not None:
        settings.ltx_steps = max(1, args.steps)
    if args.resolution is not None:
        snapped = int(round(args.resolution / 32.0) * 32)
        settings.ltx_resolution = min(1280, max(384, snapped))
    if args.seconds is not None:
        settings.ltx_seconds = max(1, min(15, args.seconds))
    if args.prompt:
        settings.prompt = args.prompt.strip()
    if args.seed is not None:
        settings.seed = args.seed
    settings.save()

    # Mode 1: Check models
    if args.check_models:
        check_models(settings)
        return

    # Mode 2: Download models
    if args.download_models:
        download_models(settings)
        return

    # Mode 3: Video processing
    if not args.input:
        parser.print_help()
        print("\nContoh penggunaan:")
        print("  python cli.py video.mp4 -o hasil.mp4")
        print("  python cli.py video.mp4 --steps 4 --resolution 512")
        print("  python cli.py /path/to/folder_video/ -o /path/to/output_folder/")
        print("  python cli.py --check-models")
        print("  python cli.py --download-models")
        return

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input path '{input_path}' tidak ditemukan!")
        sys.exit(1)

    # Cek model sebelum mulai
    discover_comfy(settings, lambda _: None)
    missing = settings.missing()
    if missing:
        print("\nPerhatian: Ada model yang belum terdeteksi:")
        for lbl, pth in missing:
            print(f"  - {lbl}: {pth}")
        print("\nJalankan perintah berikut untuk mengunduh model:")
        print("  python cli.py --download-models")
        sys.exit(1)

    # Kumpulkan daftar video
    videos: list[Path] = []
    if input_path.is_file():
        if input_path.suffix.lower() not in VIDEO_EXTS:
            print(f"Peringatan: Ekstensi {input_path.suffix} mungkin bukan format video.")
        videos.append(input_path)
    elif input_path.is_dir():
        for file in sorted(input_path.iterdir()):
            if file.is_file() and file.suffix.lower() in VIDEO_EXTS:
                videos.append(file)
        if not videos:
            print(f"Tidak ada file video yang ditemukan di direktori: {input_path}")
            sys.exit(1)

    # Tentukan output
    out_arg = Path(args.output) if args.output else default_output_dir()
    if len(videos) == 1 and not out_arg.is_dir() and out_arg.suffix:
        out_is_dir = False
        out_arg.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_is_dir = True
        out_arg.mkdir(parents=True, exist_ok=True)

    print_system_info()
    print(f"Total video yang akan diproses: {len(videos)}")

    cancel_event = threading.Event()

    def signal_handler():
        cancel_event.set()
        print("\n[SIGINT] Sinyal batal diterima. Menghentikan proses saat checkpoint aman...")

    try:
        for idx, vid in enumerate(videos, 1):
            if cancel_event.is_set():
                break
            print(f"\n[{idx}/{len(videos)}]")
            if out_is_dir:
                dest = out_arg / f"{vid.stem}_restored.mp4"
            else:
                dest = out_arg
            success = process_single_video(vid, dest, settings, cancel_event)
            if not success and cancel_event.is_set():
                break
    except KeyboardInterrupt:
        signal_handler()

    print("\nProses selesai.")


if __name__ == "__main__":
    main()
