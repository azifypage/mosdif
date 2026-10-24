"""ComfyUI Headless API Runner for BFS Head Swap Workflow.

Executes the official workflow_ltx2.5_v1.1_head_swap_drag_and_drop.json directly
via ComfyUI's REST & WebSocket API without requiring a web browser interface.
Normalizes all inputs to 24 FPS CFR, locks onto a single target actor, dynamically
sizes the crop rectangle to cover all hair (target + source), and composites the
generated rectangle back onto the full 24 FPS video.
"""

from __future__ import annotations

from fractions import Fraction
import json
import math
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

from mosaicdiff.geometry import align_up

TARGET_FPS = 24


class ComfyAPIError(RuntimeError):
    pass


# Stores head-crop metadata between patch_bfs_prompt() and run_comfy_workflow()
_ACTIVE_CROP_CONTEXT: dict[str, dict] = {}


REQUIRED_BFS_MODELS = [
    (
        "diffusion_models",
        "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors",
        "https://huggingface.co/osantinello/LTX25_Models/resolve/main/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors",
    ),
    (
        "loras",
        "head_swap_ltx25_r128_v1.1.safetensors",
        "https://huggingface.co/Alissonerdx/BFS-Best-Face-Swap-Video/resolve/main/ltx-2.5/head_swap_ltx25_r128_v1.1.safetensors",
    ),
    (
        "text_encoders",
        "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
        "https://huggingface.co/osantinello/LTX25_Models/resolve/main/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
    ),
    (
        "vae",
        "ltx-2.5-video-vae-bf16.safetensors",
        "https://huggingface.co/osantinello/LTX25_Models/resolve/main/ltx-2.5-video-vae-bf16.safetensors",
    ),
    (
        "vae",
        "ltx-2.5-audio-vae-bf16.safetensors",
        "https://huggingface.co/osantinello/LTX25_Models/resolve/main/ltx-2.5-audio-vae-bf16.safetensors",
    ),
]

SECOND_PASS_MODELS = [
    (
        "latent_upscale_models",
        "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
        "https://huggingface.co/osantinello/LTX25_Models/resolve/main/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
    ),
]


def is_comfy_running(host: str = "127.0.0.1", port: int = 8188) -> bool:
    """Check if ComfyUI server is actively responding on port 8188."""
    url = f"http://{host}:{port}/system_stats"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "MosaicDiff-Runner"})
        with urllib.request.urlopen(req, timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False


def stop_comfy_daemon(
    host: str = "127.0.0.1",
    port: int = 8188,
    log_cb=print,
) -> None:
    """Stop running ComfyUI daemon so it can refresh and re-index models."""
    if not is_comfy_running(host, port):
        return
    log_cb("Menghentikan ComfyUI daemon untuk re-indexing model...")
    if sys.platform != "win32":
        try:
            subprocess.run(f"fuser -k -9 {port}/tcp 2>/dev/null || true", shell=True)
            subprocess.run(["pkill", "-9", "-f", "main.py"], stderr=subprocess.DEVNULL)
        except Exception:
            pass
    time.sleep(2.5)


def is_model_indexed_by_comfy(
    class_type: str,
    input_name: str,
    model_name: str,
    host: str = "127.0.0.1",
    port: int = 8188,
) -> bool:
    """Check if ComfyUI's object_info API lists model_name for a given node class."""
    url = f"http://{host}:{port}/object_info/{class_type}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "MosaicDiff-Runner"})
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        inputs = data.get(class_type, {}).get("input", {}).get("required", {})
        choices = inputs.get(input_name, [[]])[0]
        return model_name in choices
    except Exception:
        return False


def ensure_bfs_models(
    comfy_root: str | Path = "/content/ComfyUI",
    enable_second_pass: bool = False,
    log_cb=print,
    **kwargs,
) -> bool:
    """Verify and auto-download any missing official BFS model weights."""
    c_root = Path(comfy_root).resolve()
    models_dir = c_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    targets = list(REQUIRED_BFS_MODELS)
    if enable_second_pass:
        targets.extend(SECOND_PASS_MODELS)

    search_roots = [
        Path("/content/mosdif/models"),
        Path(__file__).resolve().parent.parent / "models",
        Path.home() / ".cache" / "huggingface" / "hub",
    ]

    downloaded_any = False

    for subf, fname, url in targets:
        dest_dir = models_dir / subf
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_file = dest_dir / fname

        if dest_file.is_file() and dest_file.stat().st_size > 1024 * 1024:
            continue

        found_local = False
        for s_root in search_roots:
            cand = s_root / subf / fname
            if not cand.is_file():
                cand = s_root / fname
            if cand.is_file() and cand.stat().st_size > 1024 * 1024:
                log_cb(f"Linking local model {fname} -> models/{subf}/...")
                try:
                    if dest_file.is_symlink() or dest_file.is_file():
                        dest_file.unlink()
                    dest_file.symlink_to(cand)
                    found_local = True
                    downloaded_any = True
                    break
                except Exception:
                    shutil.copy2(cand, dest_file)
                    found_local = True
                    downloaded_any = True
                    break

        if found_local:
            continue

        log_cb(f"⚡ Mengunduh model pendukung ({fname})...")
        downloaded_any = True

        aria2_bin = shutil.which("aria2c")
        if aria2_bin:
            cmd = [
                aria2_bin,
                "-c",
                "-x", "16",
                "-s", "16",
                "-k", "1M",
                "-d", str(dest_dir),
                "-o", fname,
                url,
            ]
            res = subprocess.run(cmd)
            if res.returncode != 0:
                raise ComfyAPIError(f"Gagal mengunduh {fname} via aria2c (exit code {res.returncode})")
        else:
            req = urllib.request.Request(url, headers={"User-Agent": "MosaicDiff-Downloader"})
            with urllib.request.urlopen(req) as resp, open(dest_file, "wb") as out_f:
                shutil.copyfileobj(resp, out_f)

        if not dest_file.is_file() or dest_file.stat().st_size < 1024 * 1024:
            raise ComfyAPIError(f"Model {fname} gagal diunduh atau ukuran file korup.")

        size_mb = dest_file.stat().st_size / (1024 * 1024)
        log_cb(f"[OK] Model siap: {fname} ({size_mb:.1f} MB)")

    return downloaded_any


def start_comfy_daemon(
    comfy_root: str | Path = "/content/ComfyUI",
    python_bin: str = sys.executable,
    host: str = "127.0.0.1",
    port: int = 8188,
    log_cb=print,
    timeout_sec: int = 60,
    restart: bool = False,
) -> subprocess.Popen | None:
    """Ensure ComfyUI backend server is running in background."""
    if restart:
        stop_comfy_daemon(host, port, log_cb=log_cb)

    if is_comfy_running(host, port):
        log_cb(f"ComfyUI backend already running on http://{host}:{port}")
        return None

    comfy_dir = Path(comfy_root).resolve()
    main_py = comfy_dir / "main.py"
    if not main_py.is_file():
        raise ComfyAPIError(f"ComfyUI main.py tidak ditemukan di: {main_py}")

    log_cb(f"Starting headless ComfyUI daemon on http://{host}:{port}...")
    log_file = comfy_dir / "comfy_daemon.log"
    out_f = open(log_file, "a", encoding="utf-8")

    cmd = [
        str(python_bin),
        str(main_py),
        "--listen",
        host,
        "--port",
        str(port),
        "--disable-auto-launch",
    ]

    proc = subprocess.Popen(
        cmd,
        cwd=str(comfy_dir),
        stdout=out_f,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    start_time = time.time()
    while time.time() - start_time < timeout_sec:
        if is_comfy_running(host, port):
            log_cb(f"ComfyUI daemon is ready and responding on http://{host}:{port}!")
            return proc
        if proc.poll() is not None:
            out_f.close()
            err_log = log_file.read_text(encoding="utf-8", errors="replace")[-2000:]
            raise ComfyAPIError(f"ComfyUI daemon exited prematurely:\n{err_log}")
        time.sleep(1.5)

    out_f.close()
    raise ComfyAPIError(f"ComfyUI daemon failed to respond within {timeout_sec}s.")


def compute_dynamic_generation_dims(
    crop_w: int,
    crop_h: int,
    resolution_mp: float = 1.0,
) -> tuple[int, int]:
    """Compute dynamic (width, height) matching the exact rectangle aspect ratio (crop_w : crop_h).

    Snapped to multiples of 64 so that first-pass half-resolution (a/2) in LTX-2.5 is a
    strict multiple of 32 for the Video VAE.
    """
    mp = max(0.35, min(2.5, float(resolution_mp)))
    target_pixels = mp * 1024.0 * 1024.0
    ratio = max(0.25, min(4.0, float(crop_w) / max(1.0, float(crop_h))))

    raw_h = math.sqrt(target_pixels / ratio)
    raw_w = raw_h * ratio

    gen_w = max(256, min(1920, int(round(raw_w / 64.0) * 64)))
    gen_h = max(256, min(1920, int(round(raw_h / 64.0) * 64)))
    return gen_w, gen_h


def prepare_head_crop_video(
    video_input_path: str | Path,
    head_image_path: str | Path | None = None,
    comfy_root: str | Path = "/content/ComfyUI",
    resolution_mp: float = 1.0,
    show_paste_box: bool = False,
    log_cb=print,
) -> dict:
    """Normalize input video to 24 FPS CFR, lock onto 1 primary actor, dynamically size the
    head+hair rectangle (covering both target hair down to chest/stomach and source hair),
    lock aspect ratio to the LTX-2.5 latent grid, and export a 24 FPS cropped guide video.
    """
    import cv2
    import numpy as np
    from mosaicdiff.head_detector import (
        HeadDetector,
        draw_paste_box_outline,
        lock_single_actor_trajectory,
        smooth_head_trajectory,
    )
    from mosaicdiff.videoio import VideoWriter, normalize_video_to_24fps, open_capture

    raw_src_path = Path(video_input_path).resolve()
    if not raw_src_path.is_file():
        raise FileNotFoundError(f"Input video file tidak ditemukan: {raw_src_path}")

    c_in = Path(comfy_root).resolve() / "input"
    c_in.mkdir(parents=True, exist_ok=True)

    # 1. Always normalize input video to strict 24.0 FPS CFR first
    norm_24_path = c_in / f"master_24fps_{raw_src_path.stem}.mp4"
    src_path = normalize_video_to_24fps(raw_src_path, norm_24_path, log_cb=log_cb)

    capture, width, height, _fps, count = open_capture(src_path)
    detector = HeadDetector()

    # 2. Analyze source head photo hair extent (4 sides: left, top, right, bottom)
    src_ratios = (0.35, 0.48, 0.35, 0.18)
    if head_image_path is not None and Path(head_image_path).is_file():
        src_img = cv2.imread(str(head_image_path))
        if src_img is not None:
            src_cands = detector.detect_all_faces(src_img)
            if src_cands:
                best_src = max(
                    src_cands,
                    key=lambda c: (c["face_box"][2] - c["face_box"][0]) * (c["face_box"][3] - c["face_box"][1]),
                )
                src_ratios = detector.estimate_hair_expansion_ratios(
                    src_img, best_src["face_box"], landmarks=best_src.get("landmarks")
                )

    if log_cb:
        log_cb(f"🔍 Mendeteksi wajah & mengunci 1 aktor utama pada video 24 FPS ({width}x{height})...")

    all_frame_candidates: list[list[dict]] = []
    frames_read = 0
    sample_stride = max(1, (count or 120) // 16)
    sampled_frames: dict[int, np.ndarray] = {}

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            cands = detector.detect_all_faces(frame)
            all_frame_candidates.append(cands)
            if frames_read % sample_stride == 0 and len(sampled_frames) < 20:
                sampled_frames[frames_read] = frame.copy()
            frames_read += 1
    finally:
        capture.release()

    if frames_read == 0:
        detector.close()
        raise ComfyAPIError(f"Video '{src_path.name}' kosong atau tidak dapat dibaca.")

    if not any(len(c) > 0 for c in all_frame_candidates):
        detector.close()
        raise ComfyAPIError(
            f"Tidak terdeteksi area kepala/wajah pada video '{raw_src_path.name}'! "
            "Pastikan wajah/kepala target terlihat pada video."
        )

    # 3. Lock onto ONLY 1 primary actor across the entire video
    locked_actor = lock_single_actor_trajectory(all_frame_candidates, width, height)

    # 4. Estimate full head + back-of-head + chest/stomach hair extent (left, top, right, bottom) on sampled keyframes
    target_left_list = [0.35]
    target_top_list = [0.48]
    target_right_list = [0.35]
    target_bot_list = [0.22]
    for f_idx, f_img in sampled_frames.items():
        if f_idx < len(locked_actor) and locked_actor[f_idx] is not None:
            det_kf = locked_actor[f_idx]
            lr, tr, rr, br = detector.estimate_hair_expansion_ratios(
                f_img, det_kf["face_box"], landmarks=det_kf.get("landmarks")
            )
            target_left_list.append(lr)
            target_top_list.append(tr)
            target_right_list.append(rr)
            target_bot_list.append(br)

    detector.close()

    t_left = float(np.percentile(target_left_list, 95))
    t_top = float(np.percentile(target_top_list, 95))
    t_right = float(np.percentile(target_right_list, 95))
    t_bot = float(np.percentile(target_bot_list, 95))

    # Union of Target full-head/chest-stomach hair extent and Source photo head/hair extent (4 independent sides)
    final_ratios = (
        max(t_left, src_ratios[0]),
        max(t_top, src_ratios[1]),
        max(t_right, src_ratios[2]),
        max(t_bot, src_ratios[3]),
    )
    if log_cb:
        log_cb(
            f"📐 Ekspansi Penuh Kepala + Belakang Kepala + Rambut Dada/Perut (+Margin Feather): "
            f"Kiri=+{final_ratios[0]*100:.0f}%, Atas=+{final_ratios[1]*100:.0f}%, "
            f"Kanan=+{final_ratios[2]*100:.0f}%, Bawah=+{final_ratios[3]*100:.0f}%"
        )

    # Apply the 4-side expansion ratios + yaw back-of-head expansion to the locked actor
    for idx, det in enumerate(locked_actor):
        if det is not None:
            fx1, fy1, fx2, fy2 = det["face_box"]
            det["head_box"] = detector._expand_to_full_head(
                fx1,
                fy1,
                fx2,
                fy2,
                width,
                height,
                ratios=final_ratios,
                landmarks=det.get("landmarks"),
            )

    # 5. Smooth trajectory and build a 100% STATIC (LOCKED) Union Rectangle across the entire clip
    # so the crop window NEVER moves/wobbles following the face per-frame!
    smoothed = smooth_head_trajectory(locked_actor, max_gap=120, fill_edges=True, window=5)
    head_boxes: list[tuple[int, int, int, int] | None] = [
        d["head_box"] if d is not None else None for d in smoothed
    ]
    present_boxes = [b for b in head_boxes if b is not None]

    union_x1 = min(b[0] for b in present_boxes)
    union_y1 = min(b[1] for b in present_boxes)
    union_x2 = max(b[2] for b in present_boxes)
    union_y2 = max(b[3] for b in present_boxes)

    union_w = max(64, union_x2 - union_x1)
    union_h = max(64, union_y2 - union_y1)
    pad_x = max(18, int(round(union_w * 0.06)))
    pad_y = max(18, int(round(union_h * 0.06)))

    cx1 = max(0, union_x1 - pad_x)
    cy1 = max(0, union_y1 - pad_y)
    cx2 = min(width, union_x2 + pad_x)
    cy2 = min(height, union_y2 + pad_y)
    cx1 -= cx1 % 2
    cy1 -= cy1 % 2
    if (cx2 - cx1) % 2 != 0:
        cx2 = cx2 + 1 if cx2 < width else cx2 - 1
    if (cy2 - cy1) % 2 != 0:
        cy2 = cy2 + 1 if cy2 < height else cy2 - 1

    crop_w = max(64, cx2 - cx1)
    crop_h = max(64, cy2 - cy1)

    # Lock (crop_w, crop_h) aspect ratio to match (gen_w, gen_h) so ComfyUI ImageResizeKJv2
    # performs 100% uniform isotropic scaling with zero horizontal/vertical stretch distortion
    gen_w, gen_h = compute_dynamic_generation_dims(crop_w, crop_h, resolution_mp=resolution_mp)
    target_ar = float(gen_w) / float(max(1, gen_h))
    cur_ar = float(crop_w) / float(max(1, crop_h))
    box_cx = (cx1 + cx2) / 2.0
    box_cy = (cy1 + cy2) / 2.0

    if abs(cur_ar - target_ar) > 0.005:
        if cur_ar < target_ar:
            # Expand width around center to match target_ar
            ideal_w = min(width, int(round(crop_h * target_ar)))
            ideal_w -= ideal_w % 2
            if ideal_w > crop_w:
                nx1 = max(0, min(width - ideal_w, int(round(box_cx - ideal_w / 2.0))))
                nx1 -= nx1 % 2
                cx1, cx2 = nx1, nx1 + ideal_w
                crop_w = ideal_w
        else:
            # Expand height around center to match target_ar
            ideal_h = min(height, int(round(crop_w / target_ar)))
            ideal_h -= ideal_h % 2
            if ideal_h > crop_h:
                ny1 = max(0, min(height - ideal_h, int(round(box_cy - ideal_h / 2.0))))
                ny1 -= ny1 % 2
                cy1, cy2 = ny1, ny1 + ideal_h
                crop_h = ideal_h

    static_crop_box = (cx1, cy1, cx1 + crop_w, cy1 + crop_h)
    crop_boxes = [static_crop_box] * frames_read
    mode_desc = f"static locked rectangle ({cx1},{cy1})-({cx1 + crop_w},{cy1 + crop_h}) mencakup kepala + rambut dada/perut"

    # Save a preview image showing the "Kotak Tempel" outline on a representative frame
    preview_box_path = c_in / f"paste_box_preview_{raw_src_path.stem}.jpg"
    if sampled_frames:
        mid_key = sorted(sampled_frames.keys())[len(sampled_frames) // 2]
        prev_frame = sampled_frames[mid_key]
        prev_hbox = head_boxes[mid_key] if mid_key < len(head_boxes) else None
        outlined_prev = draw_paste_box_outline(prev_frame, static_crop_box, prev_hbox)
        cv2.imwrite(str(preview_box_path), outlined_prev)

    # Pad frame count to LTX-2.5 requirement: (padded_N - 1) % 8 == 0
    padded_frames = align_up(frames_read)
    crop_filename = f"head_crop_{raw_src_path.stem}.mp4"
    crop_video_path = c_in / crop_filename

    writer = VideoWriter(crop_video_path, crop_w, crop_h, Fraction(TARGET_FPS, 1))
    capture2, _, _, _, _ = open_capture(src_path)
    last_crop_frame = None
    idx = 0
    try:
        while True:
            ok, frame = capture2.read()
            if not ok or idx >= frames_read:
                break
            cx1, cy1, cx2, cy2 = crop_boxes[idx]
            patch = frame[cy1:cy2, cx1:cx2]
            if patch.shape[1] != crop_w or patch.shape[0] != crop_h:
                patch = cv2.resize(patch, (crop_w, crop_h), interpolation=cv2.INTER_AREA)
            writer.write(patch)
            last_crop_frame = patch
            idx += 1
        if last_crop_frame is not None and padded_frames > idx:
            for _ in range(padded_frames - idx):
                writer.write(last_crop_frame)
    finally:
        capture2.release()
        writer.close()

    area_pct = (crop_w * crop_h) / max(1, width * height) * 100.0
    if log_cb:
        log_cb(
            f"✂️ Rectangle Kepala+Rambut berhasil di-crop: {crop_w}x{crop_h} px @ {TARGET_FPS} FPS "
            f"({area_pct:.1f}% dari frame utuh {width}x{height} px, {mode_desc})"
        )

    return {
        "orig_video_path": str(src_path),
        "crop_video_path": str(crop_video_path),
        "crop_filename": crop_filename,
        "preview_box_path": str(preview_box_path) if preview_box_path.is_file() else "",
        "show_paste_box": bool(show_paste_box),
        "orig_width": width,
        "orig_height": height,
        "orig_fps": float(TARGET_FPS),
        "effective_fps": TARGET_FPS,
        "orig_frame_count": frames_read,
        "padded_frame_count": padded_frames,
        "crop_w": crop_w,
        "crop_h": crop_h,
        "crop_boxes": crop_boxes,
        "head_boxes": head_boxes,
    }


def composite_crop_to_full_video(
    crop_info: dict,
    generated_crop_video: str | Path,
    dest_file: str | Path,
    show_paste_box: bool | None = None,
    log_cb=print,
) -> Path:
    """Paste the generated rectangular head+hair patch back onto the 24 FPS full video
    using sub-pixel border phase alignment + 2D spatial illumination field + 2-band Laplacian blending.
    Optionally draws the 'Kotak Tempel' outline when ``show_paste_box=True``.
    """
    import numpy as np
    from mosaicdiff.head_detector import draw_paste_box_outline, seamless_paste_rectangle
    from mosaicdiff.videoio import VideoWriter, copy_audio, open_capture

    orig_path = Path(crop_info["orig_video_path"]).resolve()
    gen_path = Path(generated_crop_video).resolve()
    dst_path = Path(dest_file).resolve()
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    orig_w = int(crop_info["orig_width"])
    orig_h = int(crop_info["orig_height"])
    frame_count = int(crop_info["orig_frame_count"])
    crop_boxes: list[tuple[int, int, int, int]] = crop_info["crop_boxes"]
    head_boxes: list[tuple[int, int, int, int] | None] = crop_info["head_boxes"]
    draw_box = bool(crop_info.get("show_paste_box", False)) if show_paste_box is None else bool(show_paste_box)

    if log_cb:
        outline_note = " + Outline Kotak Tempel AKTIF" if draw_box else ""
        log_cb(
            f"🎬 Menempelkan (Multi-Band Seamless Compositing{outline_note}) kembali ke video utuh 24 FPS ({orig_w}x{orig_h})..."
        )

    cap_orig, _, _, _, _ = open_capture(orig_path)
    cap_gen, _, _, _, _ = open_capture(gen_path)

    silent_tmp = dst_path.with_name(f"{dst_path.stem}.silent_tmp{dst_path.suffix}")
    writer = VideoWriter(silent_tmp, orig_w, orig_h, Fraction(TARGET_FPS, 1))

    last_gen_frame: np.ndarray | None = None
    blend_state: dict = {}
    idx = 0
    try:
        while idx < frame_count:
            ok_orig, orig_frame = cap_orig.read()
            if not ok_orig:
                break
            ok_gen, gen_frame = cap_gen.read()
            if ok_gen and gen_frame is not None:
                last_gen_frame = gen_frame
            elif last_gen_frame is not None:
                gen_frame = last_gen_frame
            else:
                gen_frame = None

            hbox = head_boxes[idx] if idx < len(head_boxes) else None
            cbox = crop_boxes[idx] if idx < len(crop_boxes) else (crop_boxes[-1] if crop_boxes else None)

            if gen_frame is not None and cbox is not None:
                orig_frame = seamless_paste_rectangle(
                    orig_frame=orig_frame,
                    gen_patch=gen_frame,
                    crop_box=cbox,
                    head_box=hbox,
                    state=blend_state,
                )
                if draw_box:
                    orig_frame = draw_paste_box_outline(orig_frame, cbox, hbox)

            writer.write(orig_frame)
            idx += 1
    finally:
        cap_orig.release()
        cap_gen.release()
        writer.close()

    audio_tmp = dst_path.with_name(f"{dst_path.stem}.audio_tmp{dst_path.suffix}")
    try:
        if copy_audio(silent_tmp, orig_path, audio_tmp):
            audio_tmp.replace(dst_path)
            silent_tmp.unlink(missing_ok=True)
        else:
            silent_tmp.replace(dst_path)
    except Exception as exc:
        audio_tmp.unlink(missing_ok=True)
        silent_tmp.replace(dst_path)
        if log_cb:
            log_cb(f"Notice: Audio asli tidak dapat disalin ({exc}), menyimpan video tanpa audio.")

    if log_cb:
        log_cb(f"✅ Berhasil menempelkan rectangle kepala+rambut ke video utuh (24 FPS): {dst_path.name}")

    return dst_path


def prepare_comfy_inputs(
    comfy_root: str | Path,
    video_input_path: str | Path,
    head_image_path: str | Path,
) -> tuple[str, str]:
    """Copy video and head image into ComfyUI/input/ folder and return filenames."""
    c_in = Path(comfy_root).resolve() / "input"
    c_in.mkdir(parents=True, exist_ok=True)

    v_src = Path(video_input_path).resolve()
    h_src = Path(head_image_path).resolve()

    if not v_src.is_file():
        raise FileNotFoundError(f"Input video file tidak ditemukan: {v_src}")
    if not h_src.is_file():
        raise FileNotFoundError(f"Input foto kepala tidak ditemukan: {h_src}")

    v_dst = c_in / v_src.name
    h_dst = c_in / h_src.name

    if v_src != v_dst and (not v_dst.is_file() or v_src.stat().st_mtime > v_dst.stat().st_mtime):
        shutil.copy2(v_src, v_dst)

    if h_src != h_dst and (not h_dst.is_file() or h_src.stat().st_mtime > h_dst.stat().st_mtime):
        shutil.copy2(h_src, h_dst)

    return v_src.name, h_src.name


OFFICIAL_BFS_API_WORKFLOW = (
    Path(__file__).resolve().parent.parent / "workflow_ltx2.5_v1.1_head_swap_drag_and_drop.api.json"
)


def load_bfs_prompt(
    api_path: str | Path | None = None,
    log_cb=print,
) -> dict:
    """Load the official BFS API prompt graph directly from disk."""
    if api_path is None:
        target = OFFICIAL_BFS_API_WORKFLOW
        if not target.is_file():
            target = Path("/content/mosdif/workflow_ltx2.5_v1.1_head_swap_drag_and_drop.api.json")
    else:
        target = Path(api_path).resolve()
        if not target.name.endswith(".api.json"):
            companion = target.with_name(target.stem + ".api.json")
            if companion.is_file():
                target = companion

    if not target.is_file():
        raise FileNotFoundError(
            f"Official BFS API workflow file not found at: {target}\n"
            f"Pastikan repository mosdif ter-clone dengan lengkap."
        )

    with open(target, "r", encoding="utf-8") as f:
        prompt_dict = json.load(f)

    if not isinstance(prompt_dict, dict) or not any("class_type" in v for v in prompt_dict.values()):
        raise ComfyAPIError(f"Format file API prompt tidak valid: {target}")

    if log_cb:
        log_cb(f"[OK] Berhasil memuat workflow resmi BFS API: {target.name} ({len(prompt_dict)} nodes)")

    return prompt_dict


def convert_workflow_to_prompt(
    workflow_path: str | Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8188,
    log_cb=print,
) -> dict:
    """Primary loader alias for backwards compatibility."""
    return load_bfs_prompt(api_path=workflow_path, log_cb=log_cb)


def build_ltx_sigmas(steps: int) -> str:
    """Generate dynamic LTX-2.5 sigmas schedule for the requested step count."""
    if steps == 8:
        return "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
    import numpy as np
    base = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]
    x_base = np.linspace(0, 1, len(base))
    x_target = np.linspace(0, 1, max(2, int(steps)) + 1)
    sigmas = np.interp(x_target, x_base, base)
    sigmas[0] = 1.0
    sigmas[-1] = 0.0
    return ", ".join(f"{s:.6f}".rstrip("0").rstrip(".") for s in sigmas)


def normalize_aspect_ratio(val: str, default: str = "1:1 (Square)") -> str:
    """Map any user aspect ratio label to ComfyUI ResolutionSelector COMBO choices."""
    v = (val or "").lower().strip()
    if "9:16" in v:
        return "9:16 (Portrait Widescreen)"
    if "16:9" in v:
        return "16:9 (Widescreen)"
    if "3:4" in v:
        return "3:4 (Portrait Standard)"
    if "4:3" in v:
        return "4:3 (Standard)"
    if "1:1" in v:
        return "1:1 (Square)"
    if "2:3" in v:
        return "2:3 (Portrait Photo)"
    if "3:2" in v:
        return "3:2 (Photo)"
    if "21:9" in v:
        return "21:9 (Ultrawide)"
    return default


def detect_video_aspect_ratio(video_path: str | Path) -> str:
    """Detect aspect ratio category from video dimensions."""
    try:
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            cap.release()
            if w > 0 and h > 0:
                ratio = w / h
                if ratio >= 1.55:
                    return "16:9 (Widescreen)"
                elif ratio <= 0.65:
                    return "9:16 (Portrait Widescreen)"
                elif ratio <= 0.88:
                    return "3:4 (Portrait Standard)"
                elif 0.88 < ratio < 1.15:
                    return "1:1 (Square)"
                elif 1.15 <= ratio < 1.55:
                    return "4:3 (Standard)"
                else:
                    return "1:1 (Square)"
    except Exception:
        pass
    return "1:1 (Square)"


def _clean_negative_prompt_for_hair(raw_neg: str) -> str:
    """Remove 'hair' from negative prompt so LTX-2.5 can freely regenerate all hair."""
    default_neg = "mouth movement, cartoon, anime, illustration, 3d render, plastic, painting, blurry, deformed"
    text = (raw_neg or "").strip()
    if not text:
        return default_neg
    tokens = [t.strip() for t in text.split(",") if t.strip() and t.strip().lower() != "hair"]
    return ", ".join(tokens) if tokens else default_neg


def patch_bfs_prompt(
    prompt_dict: dict,
    video_input_path: str | Path,
    head_image_path: str | Path,
    prompt_text: str = "",
    negative_prompt_text: str = "",
    lora_strength: float = 0.80,
    steps: int = 8,
    seed: int = 42,
    enable_second_pass: bool = False,
    resolution_mp: float = 1.0,
    aspect_ratio: str = "auto",
    comfy_root: str | Path = "/content/ComfyUI",
    crop_head_only: bool = True,
    show_paste_box: bool = False,
    log_cb=print,
) -> dict:
    """Patch the API prompt graph nodes with user-provided parameters.

    Normalizes video to 24 FPS, crops a dynamic rectangle covering the single actor's
    head and full hair, and injects dynamic Width & Height directly into the graph.
    """
    patched = json.loads(json.dumps(prompt_dict))  # deep copy

    crop_info = None
    guide_video_for_comfy = Path(video_input_path).resolve()
    if crop_head_only:
        crop_info = prepare_head_crop_video(
            video_input_path=video_input_path,
            head_image_path=head_image_path,
            comfy_root=comfy_root,
            resolution_mp=resolution_mp,
            show_paste_box=show_paste_box,
            log_cb=log_cb,
        )
        guide_video_for_comfy = Path(crop_info["crop_video_path"]).resolve()

    v_filename, h_filename = prepare_comfy_inputs(comfy_root, guide_video_for_comfy, head_image_path)

    if crop_info is not None:
        _ACTIVE_CROP_CONTEXT[v_filename] = crop_info
        _ACTIVE_CROP_CONTEXT["__latest__"] = crop_info
        dyn_w, dyn_h = compute_dynamic_generation_dims(
            crop_info["crop_w"],
            crop_info["crop_h"],
            resolution_mp=resolution_mp,
        )
        if log_cb:
            log_cb(
                f"🎯 Resolusi Generasi Rectangle Dinamis: {dyn_w}x{dyn_h} px "
                f"(First Pass: {dyn_w // 2}x{dyn_h // 2} px, Rasio {crop_info['crop_w']}:{crop_info['crop_h']}, {TARGET_FPS} FPS)"
            )
    else:
        _ACTIVE_CROP_CONTEXT.pop(v_filename, None)
        _ACTIVE_CROP_CONTEXT.pop("__latest__", None)
        dyn_w, dyn_h = None, None

    p_text = prompt_text.strip()
    if not p_text.lower().startswith("head_swap:") and not p_text.lower().startswith("head swap"):
        if p_text:
            pos_prompt = f"head_swap:\nFACE:\n{p_text}"
        else:
            pos_prompt = "head_swap:\nFACE:\nPhotorealistic human face and hair, natural skin texture, sharp detailed eyes, realistic lighting"
    else:
        pos_prompt = p_text

    neg_prompt = _clean_negative_prompt_for_hair(negative_prompt_text)

    lora_dir = Path(comfy_root).resolve() / "models" / "loras"
    available_loras = [f.name for f in lora_dir.glob("*.safetensors")] if lora_dir.is_dir() else []

    nodes_to_remove = []

    for nid, node in patched.items():
        ctype = node.get("class_type", "")
        inputs = node.get("inputs", {})
        title = node.get("_meta", {}).get("title", "")

        # 1. Target Guide Video Input (VHS_LoadVideo, Node 460)
        if ctype == "VHS_LoadVideo" or "Guide Reference" in title:
            if "video" in inputs:
                inputs["video"] = v_filename
            if crop_info is not None:
                # Cropped video is already normalized to 24 FPS CFR; force_rate=0 avoids any frame resampling
                inputs["force_rate"] = 0

        # 2. Head Image Input (LoadImage, Node 461)
        if ctype == "LoadImage" or "Face Reference" in title:
            if "image" in inputs:
                inputs["image"] = h_filename

        # 3. LoRA Model Loader (LoraLoaderModelOnly, Node 610)
        if ctype in ("LoraLoaderModelOnly", "LoraLoader") and "head_swap" in str(inputs.get("lora_name", "")).lower():
            if "strength_model" in inputs:
                inputs["strength_model"] = float(lora_strength)
            cur_lora = inputs.get("lora_name", "")
            if cur_lora not in available_loras:
                for cand in available_loras:
                    if "head_swap" in cand.lower():
                        inputs["lora_name"] = cand
                        break

        # 4. Positive Prompt (CLIPTextEncode, Node 448)
        if ctype == "CLIPTextEncode" and ("Positive" in title or nid == "448"):
            inputs["text"] = pos_prompt

        # 5. Negative Prompt (CLIPTextEncode, Node 444)
        if ctype == "CLIPTextEncode" and ("Negative" in title or nid == "444"):
            inputs["text"] = neg_prompt

        # 6. Samplers & Schedulers (Sampling Steps & Sigmas)
        if "seed" in inputs or "noise_seed" in inputs:
            k = "noise_seed" if "noise_seed" in inputs else "seed"
            inputs[k] = int(seed)

        if ctype in ("BasicScheduler", "LTXVScheduler") and "steps" in inputs:
            inputs["steps"] = int(steps)

        if ctype == "ManualSigmas" or nid == "471":
            inputs["sigmas"] = build_ltx_sigmas(int(steps))

        # 7. Dynamic Rectangle Width & Height (Nodes 428 & 429) and ResolutionSelector (Node 403)
        if ctype == "ResolutionSelector" or nid == "403":
            inputs["megapixels"] = float(resolution_mp)
            inputs["multiple"] = 32
            if crop_info is not None or aspect_ratio == "auto" or not aspect_ratio:
                inputs["aspect_ratio"] = detect_video_aspect_ratio(guide_video_for_comfy)
            else:
                inputs["aspect_ratio"] = normalize_aspect_ratio(aspect_ratio)

        if dyn_w is not None and dyn_h is not None:
            if nid == "428" or (ctype == "PrimitiveInt" and title == "Width"):
                inputs["value"] = int(dyn_w)
            if nid == "429" or (ctype == "PrimitiveInt" and title == "Height"):
                inputs["value"] = int(dyn_h)

        # 8. Strictly 24 FPS across all nodes
        if nid == "430" or (ctype == "PrimitiveInt" and "Frame Rate" in title):
            inputs["value"] = TARGET_FPS

        if ctype == "VHS_VideoCombine" and "frame_rate" in inputs:
            inputs["frame_rate"] = TARGET_FPS

        if crop_info is not None:
            if nid == "425" or (ctype == "PrimitiveInt" and "Duration" in title):
                inputs["value"] = max(1, int(math.ceil(crop_info["padded_frame_count"] / float(TARGET_FPS))))
            if nid == "627" or ctype == "EmptyAudio":
                inputs["duration"] = float(max(1.0, crop_info["padded_frame_count"] / float(TARGET_FPS)))

        # 9. UNETLoader (Node 443) required weight_dtype
        if ctype == "UNETLoader" or nid == "443":
            inputs["weight_dtype"] = "default"

        # 10. CLIPLoader (Node 446) required type
        if ctype == "CLIPLoader" or nid == "446":
            inputs.setdefault("type", "ltxv")

        # 11. Second Pass & Comparison Video Handling
        if not enable_second_pass:
            if ctype == "SaveVideo" and ("Second Pass" in title or nid == "75"):
                nodes_to_remove.append(nid)
            elif ctype == "VHS_VideoCombine" and ("comparison" in str(inputs).lower() or nid == "491"):
                nodes_to_remove.append(nid)

        # 12. Math Expression for video length (Node 455) and LTXMultipleControls exact guide alignment
        if ctype == "LTXMultipleControls" and crop_info is not None:
            inputs["guide_ref_resize_mode"] = "match_target"

        if nid == "455" or (ctype == "ComfyMathExpression" and "b" in str(inputs.get("expression", ""))):
            if "values.b" not in inputs and "b" not in inputs:
                inputs["values.b"] = ["430", 0]
            if crop_info is not None:
                padded_n = int(crop_info["padded_frame_count"])
                inputs["expression"] = f"{padded_n} + 0 * a * b"

    for nid in nodes_to_remove:
        patched.pop(nid, None)

    return patched


def run_comfy_workflow(
    prompt_dict: dict,
    dest_file: str | Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8188,
    comfy_root: str | Path = "/content/ComfyUI",
    show_paste_box: bool | None = None,
    log_cb=print,
    progress_cb=None,
    cancel_event=None,
) -> Path:
    """Submit prompt to ComfyUI API, stream progress, and composite generated rectangle back onto the 24 FPS full video."""
    has_second_pass = any(
        n.get("class_type") == "SaveVideo" and "Second Pass" in n.get("_meta", {}).get("title", "")
        for n in prompt_dict.values()
        if isinstance(n, dict)
    )
    needed_restart = ensure_bfs_models(
        comfy_root=comfy_root,
        enable_second_pass=has_second_pass,
        log_cb=log_cb,
    )

    if is_comfy_running(host, port):
        if needed_restart or not is_model_indexed_by_comfy("VAELoader", "vae_name", "ltx-2.5-video-vae-bf16.safetensors", host, port):
            log_cb("Sinkronisasi model ke ComfyUI daemon (restart untuk re-indexing)...")
            stop_comfy_daemon(host=host, port=port, log_cb=log_cb)
            start_comfy_daemon(comfy_root=comfy_root, host=host, port=port, log_cb=log_cb)
    else:
        start_comfy_daemon(comfy_root=comfy_root, host=host, port=port, log_cb=log_cb)

    crop_info = None
    for node in prompt_dict.values():
        if isinstance(node, dict) and node.get("class_type") == "VHS_LoadVideo":
            v_name = node.get("inputs", {}).get("video", "")
            if v_name in _ACTIVE_CROP_CONTEXT:
                crop_info = _ACTIVE_CROP_CONTEXT[v_name]
                break
    if crop_info is None:
        crop_info = _ACTIVE_CROP_CONTEXT.get("__latest__")

    client_id = str(uuid.uuid4())
    url = f"http://{host}:{port}/prompt"

    payload = {
        "prompt": prompt_dict,
        "client_id": client_id,
    }

    req_data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=req_data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as resp:
            res = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as http_err:
        err_body = http_err.read().decode("utf-8", errors="replace")
        try:
            err_json = json.loads(err_body)
            err_details = json.dumps(err_json, indent=2)
        except Exception:
            err_details = err_body
        log_cb(f"❌ ComfyUI /prompt menolak workflow (HTTP {http_err.code}):\n{err_details}")
        raise ComfyAPIError(f"ComfyUI /prompt rejected with HTTP {http_err.code}:\n{err_details}") from http_err

    prompt_id = res.get("prompt_id")
    if not prompt_id:
        raise ComfyAPIError(f"Gagal mengirim workflow ke ComfyUI: {res}")

    log_cb(f"Workflow rectangle kepala+rambut berhasil masuk antrean ComfyUI (Prompt ID: {prompt_id})")

    ws = None
    try:
        import websocket
        ws_url = f"ws://{host}:{port}/ws?clientId={client_id}"
        ws = websocket.create_connection(ws_url, timeout=10)
        log_cb("Terhubung ke WebSocket ComfyUI untuk pemantauan live.")
    except Exception as exc:
        log_cb(f"Notice: WebSocket tidak aktif ({exc}), fallback ke polling HTTP.")

    completed = False

    try:
        while not completed:
            if cancel_event and cancel_event.is_set():
                try:
                    c_req = urllib.request.Request(f"http://{host}:{port}/interrupt", data=b"{}")
                    urllib.request.urlopen(c_req)
                except Exception:
                    pass
                raise InterruptedError("Proses dibatalkan oleh pengguna.")

            if ws is not None:
                try:
                    msg = ws.recv()
                    if isinstance(msg, str):
                        data = json.loads(msg)
                        mtype = data.get("type")
                        mdata = data.get("data", {})

                        if mtype == "status":
                            q_remain = mdata.get("status", {}).get("exec_info", {}).get("queue_remaining", 0)
                            if q_remain == 0 and completed:
                                break

                        elif mtype == "executing":
                            node = mdata.get("node")
                            p_id = mdata.get("prompt_id")
                            if p_id == prompt_id:
                                if node is None:
                                    completed = True
                                    break
                                else:
                                    node_info = prompt_dict.get(str(node), {})
                                    c_name = node_info.get("class_type", f"Node {node}")
                                    title = node_info.get("_meta", {}).get("title", c_name)
                                    log_cb(f"⚙️ Menjalankan: {title} ({c_name})")

                        elif mtype == "progress":
                            val = mdata.get("value", 0)
                            total = mdata.get("max", 1)
                            frac = min(1.0, val / max(1, total))
                            if progress_cb:
                                progress_cb(frac, f"Sampling Head+Hair: {val}/{total}")
                            elif val % 2 == 0 or val == total:
                                log_cb(f"⏳ Sampling Head+Hair: {val}/{total} steps ({frac * 100:.0f}%)")

                        elif mtype == "execution_error":
                            err_node = mdata.get("node_id")
                            err_msg = mdata.get("exception_message", "Unknown error")
                            raise ComfyAPIError(f"Error pada node {err_node}: {err_msg}")

                except websocket.WebSocketTimeoutException:
                    pass

            else:
                time.sleep(2.0)
                h_url = f"http://{host}:{port}/history/{prompt_id}"
                try:
                    with urllib.request.urlopen(h_url) as resp:
                        history = json.loads(resp.read().decode("utf-8"))
                        if prompt_id in history:
                            completed = True
                            break
                except Exception:
                    pass

    finally:
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass

    log_cb("Eksekusi generate rectangle kepala+rambut selesai! Mengambil file hasil...")
    h_url = f"http://{host}:{port}/history/{prompt_id}"
    with urllib.request.urlopen(h_url) as resp:
        history = json.loads(resp.read().decode("utf-8"))

    prompt_res = history.get(prompt_id, {})
    outputs = prompt_res.get("outputs", {})

    c_root = Path(comfy_root).resolve()
    found_videos = []
    for node_out in outputs.values():
        for v_item in node_out.get("videos", node_out.get("gifs", [])):
            fname = v_item.get("filename")
            subf = v_item.get("subfolder", "")
            if fname:
                found_videos.append((subf, fname))

    if not found_videos:
        comfy_out = c_root / "output"
        if comfy_out.is_dir():
            mp4s = sorted(comfy_out.glob("**/*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
            if mp4s:
                found_videos.append(("", mp4s[0].name))

    if not found_videos:
        raise ComfyAPIError("Tidak ditemukan file video hasil dari eksekusi ComfyUI.")

    selected = None
    for pref in ["second_pass", "first_pass", "head_swap", "comparison"]:
        for subf, fname in found_videos:
            if pref in fname.lower():
                selected = (subf, fname)
                break
        if selected:
            break

    if not selected:
        selected = found_videos[0]

    subfolder, filename = selected

    resolved_file = None
    direct_path = c_root / "output" / subfolder / filename
    if direct_path.is_file():
        resolved_file = direct_path
    else:
        matches = [p for p in (c_root / "output").glob(f"**/{filename}") if p.is_file() and p.stat().st_size > 0]
        if matches:
            resolved_file = matches[0]
        else:
            recent_vids = sorted(
                [
                    p for p in (c_root / "output").glob("**/*")
                    if p.suffix.lower() in [".mp4", ".mkv", ".mov", ".webm"] and p.is_file() and p.stat().st_size > 0
                ],
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if recent_vids:
                resolved_file = recent_vids[0]

    if resolved_file is None or not resolved_file.is_file():
        raise FileNotFoundError(f"File video hasil '{filename}' tidak ditemukan di {c_root / 'output'}")

    out_video_path = resolved_file
    log_cb(f"Video rectangle kepala+rambut terpilih: {out_video_path.name} ({out_video_path.stat().st_size / (1024*1024):.2f} MB)")

    target_dst = Path(dest_file).resolve() if dest_file else out_video_path.with_name(f"{out_video_path.stem}_full.mp4")

    if crop_info is not None:
        final_video = composite_crop_to_full_video(
            crop_info=crop_info,
            generated_crop_video=out_video_path,
            dest_file=target_dst,
            show_paste_box=show_paste_box,
            log_cb=log_cb,
        )
        return final_video

    if dest_file:
        target_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out_video_path, target_dst)
        return target_dst

    return out_video_path
