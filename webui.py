"""MosaicDiff Web Service Interface using Gradio.

Allows running MosaicDiff as a web service on both local machine and Google Colab.
Supports reverse tunneling (AWS VM, SSH, ngrok, cloudflared) and local testing.
"""

from __future__ import annotations

import os
import sys
import time
import shutil
import threading
from pathlib import Path
from fractions import Fraction

import cv2
import gradio as gr
try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

from mosaicdiff.comfy_setup import comfy_ready, discover_comfy
from mosaicdiff.fetch import ensure_weights, FetchError
from mosaicdiff.paths import BUNDLED_NAMES, MODEL_LABELS, default_output_dir, models_dir
from mosaicdiff.settings import Settings
from mosaicdiff.videoio import VideoWriter, open_capture


class Cancelled(Exception):
    pass


_cancel_event = threading.Event()


def get_system_info() -> str:
    if not HAS_TORCH:
        return f"**Platform:** `{sys.platform}` | **PyTorch:** *Belum terinstall di environment ini (Mode Simulasi / Dry Run aktif)*"
    cuda_avail = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_avail else "No NVIDIA GPU (CPU only)"
    vram_info = ""
    if cuda_avail:
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        vram_info = f" | Total VRAM: {total_vram:.2f} GB"
    
    return f"**Platform:** `{sys.platform}` | **Device:** `{device_name}`{vram_info}"


def check_models_status() -> str:
    settings = Settings.load()
    lines = ["### Status File Model:"]
    for key in ("detector", "vsr", "unet", "lora", "clip", "vae"):
        resolved_path = settings.resolved(key)
        name = MODEL_LABELS.get(key, key)
        if resolved_path.exists():
            size_mb = resolved_path.stat().st_size / (1024 * 1024)
            size_str = f"{size_mb / 1024:.2f} GB" if size_mb >= 1024 else f"{size_mb:.1f} MB"
            lines.append(f"-  **{name}**: `{resolved_path.name}` ({size_str})")
        else:
            lines.append(f"-  **{name}**: *Missing* (`{resolved_path}`)")
    
    # ComfyUI status
    comfy_ok = comfy_ready(settings)
    c_root = settings.resolved("comfy_root")
    c_py = settings.resolved("comfy_python")
    lines.append(f"\n### ComfyUI Status: {' Terhubung' if comfy_ok else ' Belum Siap'}")
    lines.append(f"- Folder: `{c_root}` ({'OK' if c_root.exists() else 'Tidak Ditemukan'})")
    lines.append(f"- Python: `{c_py}` ({'OK' if c_py.exists() else 'Tidak Ditemukan'})")
    
    return "\n".join(lines)


def run_download_weights(progress=gr.Progress()):
    settings = Settings.load()
    logs = []

    def log_cb(msg: str):
        logs.append(msg)

    try:
        progress(0.1, desc="Memeriksa / Mengunduh model...")
        ensure_weights(settings, log_cb, _cancel_event)
        progress(1.0, desc="Selesai")
        return "\n".join(logs) + "\n Pengunduhan model selesai!", check_models_status()
    except Exception as exc:
        return f"Gagal mengunduh: {exc}", check_models_status()


def simulate_dry_run(video_path: Path, output_path: Path, log_fn, progress_fn, cancel_event):
    """Simulates restoration pipeline to test Gradio UI, progress, and video rendering."""
    log_fn("[DRY RUN] Mode simulasi diaktifkan. Memvalidasi video input...")
    cap, width, height, fps, count = open_capture(video_path)
    total_frames = max(count, 30)
    
    log_fn(f"[DRY RUN] Input resolusi: {width}x{height}, FPS: {fps:.2f}, Total frames: {total_frames}")
    
    writer = VideoWriter(output_path, width, height, Fraction(int(fps), 1))
    
    stages = [
        (0.2, "Tahap 1/4: Deteksi sensor mosaik (RF-DETR)..."),
        (0.5, "Tahap 2/4: Rekonstruksi temporal BasicVSR++..."),
        (0.8, "Tahap 3/4: MiniMax H3 diffusion sampling..."),
        (0.95, "Tahap 4/4: Blending patch & encoding output video..."),
    ]
    
    current_p = 0.0
    for target_p, desc in stages:
        if cancel_event.is_set():
            writer.close()
            raise Cancelled("Dibatalkan oleh pengguna.")
        log_fn(f"[DRY RUN] {desc}")
        step_increment = (target_p - current_p) / 5
        for _ in range(5):
            current_p += step_increment
            progress_fn(current_p, desc)
            time.sleep(0.3)
    
    # Process a few frames to write a valid video
    written = 0
    while written < min(total_frames, 90):
        ok, frame = cap.read()
        if not ok:
            break
        # Draw a small test watermark badge
        cv2.putText(
            frame,
            "MosaicDiff [Dry Run Tested]",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 128),
            2,
            cv2.LINE_AA,
        )
        writer.write(frame)
        written += 1
    
    cap.release()
    writer.close()
    progress_fn(1.0, "Selesai")
    log_fn(f"[DRY RUN] Video output simulasi berhasil dibuat di: {output_path}")


def process_video_web(
    video_file,
    h3_seconds: int,
    h3_resolution: int,
    generate_compare: bool,
    dry_run_mode: bool,
    progress=gr.Progress(track_tqdm=True),
):
    if not video_file:
        return None, None, " Silakan unggah file video terlebih dahulu!"

    _cancel_event.clear()
    logs = []

    def log_cb(msg: str):
        logs.append(f"[{time.strftime('%H:%M:%S')}] {msg}")

    src_path = Path(video_file)
    out_dir = default_output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    dst_path = out_dir / f"{src_path.stem}_restored.mp4"

    settings = Settings.load()
    settings.h3_seconds = int(h3_seconds)
    settings.h3_resolution = int(h3_resolution)
    settings.compare = bool(generate_compare)
    settings.output_dir = str(out_dir)

    log_cb(f"Memulai pemrosesan video: {src_path.name}")
    log_cb(f"Parameter: H3 Window={settings.h3_seconds}s, Resolution={settings.h3_resolution}px, Compare={settings.compare}")

    try:
        if dry_run_mode:
            simulate_dry_run(
                src_path,
                dst_path,
                log_cb,
                lambda p, desc: progress(p, desc=desc),
                _cancel_event,
            )
        else:
            if not HAS_TORCH:
                raise RuntimeError("PyTorch belum terinstall di environment ini. Gunakan centang 'Dry Run / Simulation Mode' untuk pengujian lokal.")
            from mosaicdiff.pipeline import process_video
            discover_comfy(settings, log_cb)
            missing = settings.missing()
            if missing:
                missing_str = ", ".join(lbl for lbl, _ in missing)
                raise FileNotFoundError(
                    f"Model berikut belum tersedia: {missing_str}.\n"
                    "Silakan unduh atau tempatkan file di folder 'models/'."
                )
            
            process_video(
                source=src_path,
                destination=dst_path,
                settings=settings,
                log=log_cb,
                progress=lambda p, desc: progress(p, desc=desc),
                cancel=_cancel_event,
            )

        log_cb(" Pemrosesan berhasil selesai!")
        
        compare_path = dst_path.with_name(f"{dst_path.stem}_compare{dst_path.suffix}")
        comp_result = str(compare_path) if generate_compare and compare_path.exists() else None

        return str(dst_path), comp_result, "\n".join(logs)

    except Cancelled:
        log_cb("⏹️ Proses dibatalkan oleh pengguna.")
        return None, None, "\n".join(logs)
    except Exception as exc:
        log_cb(f" Error: {str(exc)}")
        return None, None, "\n".join(logs)


def cancel_processing():
    _cancel_event.set()
    return "Mengirim sinyal pembatalan..."


def save_comfy_config(root_path: str, py_path: str):
    settings = Settings.load()
    settings.paths["comfy_root"] = root_path.strip()
    settings.paths["comfy_python"] = py_path.strip()
    settings.save()
    return " Pengaturan ComfyUI berhasil disimpan!", check_models_status()


def create_ui() -> gr.Blocks:
    settings = Settings.load()
    # Discover ComfyUI if possible
    discover_comfy(settings, lambda _: None)

    with gr.Blocks(title="MosaicDiff Web Service") as demo:
        gr.Markdown(
            """
            #  MosaicDiff Web Service
            **Restorasi Sensor Mosaik Video menggunakan BasicVSR++ dan MiniMax H3**
            """
        )
        sys_info = gr.Markdown(get_system_info())

        with gr.Tabs():
            with gr.TabItem(" Pemrosesan Video"):
                with gr.Row():
                    with gr.Column(scale=1):
                        video_input = gr.Video(label="Unggah Video Input", sources=["upload"])

                        with gr.Group():
                            gr.Markdown("#### Pengaturan Parameter")
                            h3_sec = gr.Slider(
                                minimum=1,
                                maximum=15,
                                value=settings.h3_seconds,
                                step=1,
                                label="H3 Sample Window (Detik)",
                                info="Panjang cuplikan yang diproses MiniMax H3 sekaligus",
                            )
                            h3_res = gr.Slider(
                                minimum=512,
                                maximum=1280,
                                value=settings.h3_resolution,
                                step=32,
                                label="H3 Resolution (Pixels)",
                                info="Resolusi generasi model H3",
                            )
                            chk_compare = gr.Checkbox(
                                value=settings.compare,
                                label="Buat video perbandingan (Side-by-Side Compare)",
                            )
                            chk_dry_run = gr.Checkbox(
                                value=True,
                                label="🧪 Dry Run / Simulation Mode (Test UI & Alur tanpa GPU/Model Berat)",
                                info="Centang opsi ini untuk pengujian lokal tanpa membutuhkan GPU besar atau download 37GB model.",
                            )

                        with gr.Row():
                            btn_start = gr.Button(" Mulai Restorasi", variant="primary", scale=2)
                            btn_cancel = gr.Button("⏹️ Batal", variant="stop", scale=1)

                    with gr.Column(scale=1):
                        video_output = gr.Video(label="Hasil Video Restorasi")
                        video_compare = gr.Video(label="Video Perbandingan (Side-by-side)")
                        logs_box = gr.Textbox(
                            label="Terminal Log Realtime",
                            lines=12,
                            max_lines=18,
                            autoscroll=True,
                            interactive=False,
                        )

            with gr.TabItem(" Status Model & Pengaturan ComfyUI"):
                with gr.Row():
                    with gr.Column():
                        models_status_box = gr.Markdown(check_models_status())
                        btn_check_models = gr.Button(" Refresh Status Model")
                        btn_download_models = gr.Button(" Unduh Model Publik (Hugging Face)")
                        download_log = gr.Textbox(label="Log Pengunduhan", lines=5, interactive=False)

                    with gr.Column():
                        gr.Markdown("#### Konfigurasi Path ComfyUI")
                        comfy_root_input = gr.Textbox(
                            label="ComfyUI Folder Path",
                            value=str(settings.resolved("comfy_root")),
                            placeholder="/content/ComfyUI atau C:\\ComfyUI",
                        )
                        comfy_py_input = gr.Textbox(
                            label="ComfyUI Python Path",
                            value=str(settings.resolved("comfy_python")),
                            placeholder="/usr/bin/python3 atau C:\\ComfyUI\\venv\\Scripts\\python.exe",
                        )
                        btn_save_comfy = gr.Button(" Simpan Path ComfyUI")
                        save_status = gr.Label(visible=False)

        # Event Handlers
        btn_start.click(
            fn=process_video_web,
            inputs=[video_input, h3_sec, h3_res, chk_compare, chk_dry_run],
            outputs=[video_output, video_compare, logs_box],
        )

        btn_cancel.click(
            fn=cancel_processing,
            outputs=logs_box,
        )

        btn_check_models.click(
            fn=check_models_status,
            outputs=models_status_box,
        )

        btn_download_models.click(
            fn=run_download_weights,
            outputs=[download_log, models_status_box],
        )

        btn_save_comfy.click(
            fn=save_comfy_config,
            inputs=[comfy_root_input, comfy_py_input],
            outputs=[download_log, models_status_box],
        )

    return demo


def main():
    import argparse
    parser = argparse.ArgumentParser(description="MosaicDiff Gradio Web Service")
    parser.add_argument("--port", type=int, default=7860, help="Port untuk Web Service (default: 7860)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host binding (default: 0.0.0.0)")
    parser.add_argument("--share", action="store_true", help="Buat Gradio public share link")
    args = parser.parse_args()

    demo = create_ui()
    theme = gr.themes.Soft(
        primary_hue="amber",
        secondary_hue="zinc",
        neutral_hue="slate",
    )
    print(f"Starting MosaicDiff Web Service on http://{args.host}:{args.port}", flush=True)
    demo.queue().launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        theme=theme,
    )


if __name__ == "__main__":
    main()
