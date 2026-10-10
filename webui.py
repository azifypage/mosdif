"""mosdif Web Service Interface using Gradio.

Designed for Google Colab and AWS VM Reverse Tunneling.
Headless-friendly: Upload, Live Progress & Logs, Notification, and Direct File Download.
No video preview players or comparison videos.
"""

from __future__ import annotations

import os
import sys
import time
import shutil
import threading
from pathlib import Path

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


class Cancelled(Exception):
    pass


_cancel_event = threading.Event()


def get_system_info() -> str:
    if not HAS_TORCH:
        return f"**Platform:** `{sys.platform}` | **PyTorch:** *Tidak terdeteksi*"
    cuda_avail = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_avail else "No NVIDIA GPU (CPU only)"
    vram_info = ""
    if cuda_avail:
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        vram_info = f" | Total VRAM: {total_vram:.2f} GB"
    
    return f"**Platform:** `{sys.platform}` | **Device:** `{device_name}`{vram_info}"


def check_models_status() -> str:
    settings = Settings.load()
    lines = ["### Status Deteksi Model:"]
    for key in ("detector", "vsr", "unet", "lora", "clip", "vae"):
        resolved_path = settings.resolved(key)
        name = MODEL_LABELS.get(key, key)
        if resolved_path.exists():
            size_mb = resolved_path.stat().st_size / (1024 * 1024)
            size_str = f"{size_mb / 1024:.2f} GB" if size_mb >= 1024 else f"{size_mb:.1f} MB"
            lines.append(f"-  **{name}**: `{resolved_path}` ({size_str})")
        elif key == "lora":
            lines.append(f"- ℹ️ **{name}**: *Opsional (Sudah terpasang di dalam GGUF checkpoint)*")
        else:
            lines.append(f"- ❌ **{name}**: *Belum ditemukan* (`{resolved_path}`)")
    
    # ComfyUI status
    comfy_ok = comfy_ready(settings)
    c_root = settings.resolved("comfy_root")
    c_py = settings.resolved("comfy_python")
    lines.append(f"\n### ComfyUI Status: {' Terhubung' if comfy_ok else ' Belum Siap'}")
    lines.append(f"- Folder: `{c_root}` ({'OK' if c_root.exists() else 'Tidak Ditemukan'})")
    lines.append(f"- Python: `{c_py}` ({'OK' if c_py.exists() else 'Tidak Ditemukan'})")
    
    return "\n".join(lines)


def process_video_web(
    video_file,
    h3_seconds: int,
    h3_resolution: int,
    h3_steps: int,
    progress=gr.Progress(track_tqdm=True),
):
    if not video_file:
        return None, " Silakan unggah file video terlebih dahulu!", ""

    _cancel_event.clear()
    logs = []

    def log_cb(msg: str):
        logs.append(f"[{time.strftime('%H:%M:%S')}] {msg}")

    # Handle file input whether string path or UploadedFile
    if hasattr(video_file, "name"):
        src_path = Path(video_file.name)
    else:
        src_path = Path(str(video_file))

    out_dir = default_output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    dst_path = out_dir / f"{src_path.stem}_restored.mp4"

    settings = Settings.load()
    settings.h3_seconds = int(h3_seconds)
    settings.h3_resolution = int(h3_resolution)
    settings.h3_steps = int(h3_steps)
    settings.compare = False
    settings.output_dir = str(out_dir)

    log_cb(f"Memulai restorasi video: {src_path.name}")
    log_cb(f"Pengaturan: H3 Window={settings.h3_seconds}s, Resolution={settings.h3_resolution}px, Steps={settings.h3_steps}")

    try:
        discover_comfy(settings, log_cb)
        missing = settings.missing()
        if missing:
            missing_str = ", ".join(lbl for lbl, _ in missing)
            raise FileNotFoundError(
                f"Model berikut belum terdeteksi di sistem: {missing_str}.\n"
                "Pastikan file model berada di folder 'models/' atau ComfyUI models."
            )

        log_cb("Semua model terdeteksi. Memulai pipeline...")
        from mosaicdiff.pipeline import process_video
        process_video(
            source=src_path,
            destination=dst_path,
            settings=settings,
            log=log_cb,
            progress=lambda p, desc: progress(p, desc=desc),
            cancel=_cancel_event,
        )

        notification = f" Restorasi Selesai! File tersimpan di: {dst_path.resolve()}"
        log_cb(notification)
        return str(dst_path), notification, "\n".join(logs)

    except Cancelled:
        msg = "⏹️ Proses dibatalkan oleh pengguna."
        log_cb(msg)
        return None, msg, "\n".join(logs)
    except Exception as exc:
        err_msg = f" Error: {str(exc)}"
        log_cb(err_msg)
        return None, err_msg, "\n".join(logs)


def cancel_processing():
    _cancel_event.set()
    return "Mengirim sinyal pembatalan..."


def save_comfy_config(root_path: str, py_path: str):
    settings = Settings.load()
    settings.paths["comfy_root"] = root_path.strip()
    settings.paths["comfy_python"] = py_path.strip()
    settings.save()
    return " Pengaturan ComfyUI disimpan!", check_models_status()


def create_ui() -> gr.Blocks:
    settings = Settings.load()
    discover_comfy(settings, lambda _: None)

    with gr.Blocks(title="mosdif Web Service") as demo:
        gr.Markdown(
            """
            #  mosdif Web Service (Google Colab)
            **Video Mosaic Restoration Pipeline (BasicVSR++ & LTX-2.5 Uncensored Turbo GGUF)**
            """
        )
        sys_info = gr.Markdown(get_system_info())

        with gr.Tabs():
            with gr.TabItem(" Restorasi Video"):
                with gr.Row():
                    with gr.Column(scale=1):
                        video_input = gr.File(
                            label="Unggah File Video (.mp4, .mkv, .avi, .mov)",
                            file_types=["video"],
                        )

                        with gr.Group():
                            gr.Markdown("#### Parameter Restorasi & Kecepatan")
                            h3_steps_slider = gr.Slider(
                                minimum=2,
                                maximum=12,
                                value=getattr(settings, "ltx_steps", getattr(settings, "h3_steps", 4)),
                                step=1,
                                label="⚡ Sampling Steps (Kecepatan)",
                                info="4 steps = Turbo Super Cepat (Rekomendasi). 8 steps = Standar.",
                            )
                            h3_res = gr.Slider(
                                minimum=384,
                                maximum=1024,
                                value=settings.h3_resolution,
                                step=32,
                                label="📐 LTX-2.5 Generation Resolution (Pixels)",
                                info="512px = Sangat Cepat & Tajam (Rekomendasi). 768px = Resolusi Tinggi.",
                            )
                            h3_sec = gr.Slider(
                                minimum=1,
                                maximum=10,
                                value=settings.h3_seconds,
                                step=1,
                                label="⏱️ LTX-2.5 Sample Window (Detik)",
                                info="Panjang sampel video per window (3-4s optimal).",
                            )

                        with gr.Row():
                            btn_start = gr.Button(" Mulai Restorasi", variant="primary", scale=2)
                            btn_cancel = gr.Button("⏹️ Batal", variant="stop", scale=1)

                    with gr.Column(scale=1):
                        notification_box = gr.Textbox(
                            label="Status / Notifikasi",
                            lines=2,
                            interactive=False,
                            value="Siap memproses video.",
                        )
                        file_download = gr.File(
                            label="Download Hasil Restorasi",
                            interactive=False,
                        )
                        logs_box = gr.Textbox(
                            label="Live Terminal Logs",
                            lines=14,
                            max_lines=20,
                            autoscroll=True,
                            interactive=False,
                        )

            with gr.TabItem(" Status Model & Pengaturan"):
                with gr.Row():
                    with gr.Column():
                        models_status_box = gr.Markdown(check_models_status())
                        btn_check_models = gr.Button(" Periksa Ulang Lokasi Model")

                    with gr.Column():
                        gr.Markdown("#### Konfigurasi Path ComfyUI")
                        comfy_root_input = gr.Textbox(
                            label="ComfyUI Folder Path",
                            value=str(settings.resolved("comfy_root")),
                            placeholder="/content/ComfyUI",
                        )
                        comfy_py_input = gr.Textbox(
                            label="ComfyUI Python Path",
                            value=str(settings.resolved("comfy_python")),
                            placeholder="/usr/bin/python3",
                        )
                        btn_save_comfy = gr.Button(" Simpan Path ComfyUI")
                        save_status = gr.Textbox(label="Status Simpan", lines=1, interactive=False)

        # Event Handlers
        btn_start.click(
            fn=process_video_web,
            inputs=[video_input, h3_sec, h3_res, h3_steps_slider],
            outputs=[file_download, notification_box, logs_box],
        )

        btn_cancel.click(
            fn=cancel_processing,
            outputs=notification_box,
        )

        btn_check_models.click(
            fn=check_models_status,
            outputs=models_status_box,
        )

        btn_save_comfy.click(
            fn=save_comfy_config,
            inputs=[comfy_root_input, comfy_py_input],
            outputs=[save_status, models_status_box],
        )

    return demo


def main():
    import argparse
    parser = argparse.ArgumentParser(description="mosdif Web Service")
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
    print(f"Starting mosdif Web Service on http://{args.host}:{args.port}", flush=True)
    demo.queue().launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        theme=theme,
    )


if __name__ == "__main__":
    main()
