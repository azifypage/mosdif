"""ComfyUI Headless API Runner for BFS Head Swap Workflow.

Executes the official workflow_ltx2.5_v1.1_head_swap_drag_and_drop.json directly
via ComfyUI's REST & WebSocket API without requiring a web browser interface.
Detects and crops only the region around the identified head/face for generation,
then composites the generated head seamlessly back onto the full original video.
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
    """Verify and auto-download any missing official BFS model weights.
    
    Returns True if any model was newly downloaded or linked, indicating that
    the ComfyUI daemon should be restarted to re-index its model catalog.
    """
    c_root = Path(comfy_root).resolve()
    models_dir = c_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    targets = list(REQUIRED_BFS_MODELS)
    if enable_second_pass:
        targets.extend(SECOND_PASS_MODELS)

    # Alternate local locations where weights might already reside
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

        # 1. Check if already present and non-empty (> 1MB)
        if dest_file.is_file() and dest_file.stat().st_size > 1024 * 1024:
            continue

        # 2. Check alternate local directories
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

        # 3. Download directly from high-speed public repository
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

    # Wait until server responds
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


def prepare_head_crop_video(
    video_input_path: str | Path,
    comfy_root: str | Path = "/content/ComfyUI",
    log_cb=print,
) -> dict:
    """Detect head/face across all video frames and write a cropped guide video containing ONLY the head region.

    Returns a dictionary with crop coordinates, per-frame head boxes, and cropped video path so
    the generated head output can be seamlessly pasted back onto the full original video.
    """
    import cv2
    import numpy as np
    from mosaicdiff.head_detector import HeadDetector, smooth_head_trajectory
    from mosaicdiff.videoio import VideoWriter, open_capture

    src_path = Path(video_input_path).resolve()
    if not src_path.is_file():
        raise FileNotFoundError(f"Input video file tidak ditemukan: {src_path}")

    capture, width, height, fps, count = open_capture(src_path)
    detector = HeadDetector()
    detections: list[dict | None] = []
    frames_read = 0

    if log_cb:
        log_cb(f"🔍 Mendeteksi area kepala/wajah pada video '{src_path.name}' ({width}x{height})...")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            det = detector.detect_head(frame, need_mask=False)
            detections.append(det)
            frames_read += 1
    finally:
        capture.release()
        detector.close()

    if frames_read == 0:
        raise ComfyAPIError(f"Video '{src_path.name}' kosong atau tidak dapat dibaca.")

    valid_count = sum(1 for d in detections if d is not None)
    if valid_count == 0:
        raise ComfyAPIError(
            f"Tidak terdeteksi area kepala/wajah pada video '{src_path.name}'! "
            "Pastikan wajah/kepala target terlihat pada video."
        )

    # Smooth trajectory and fill short/edge gaps so the crop window is temporally stable
    smoothed = smooth_head_trajectory(detections, max_gap=120, fill_edges=True)
    head_boxes: list[tuple[int, int, int, int] | None] = [
        d["head_box"] if d is not None else None for d in smoothed
    ]
    present_boxes = [b for b in head_boxes if b is not None]

    # Expand each head box by ~22% context padding around hair/ears/neck so LTX-2.5 has clean context
    # and the feathered mask fades smoothly to 0.0 inside the crop box
    expanded_boxes: list[tuple[int, int, int, int]] = []
    for hx1, hy1, hx2, hy2 in present_boxes:
        hw = max(16, hx2 - hx1)
        hh = max(16, hy2 - hy1)
        pad_x = max(24, int(round(hw * 0.22)))
        pad_y = max(24, int(round(hh * 0.22)))
        ex1 = max(0, hx1 - pad_x)
        ey1 = max(0, hy1 - pad_y)
        ex2 = min(width, hx2 + pad_x)
        ey2 = min(height, hy2 + pad_y)
        expanded_boxes.append((ex1, ey1, ex2, ey2))

    exp_widths = [b[2] - b[0] for b in expanded_boxes]
    exp_heights = [b[3] - b[1] for b in expanded_boxes]
    ref_w = int(np.percentile(exp_widths, 90))
    ref_h = int(np.percentile(exp_heights, 90))

    # Union box of all expanded head positions across the video
    u_x1 = min(b[0] for b in expanded_boxes)
    u_y1 = min(b[1] for b in expanded_boxes)
    u_x2 = max(b[2] for b in expanded_boxes)
    u_y2 = max(b[3] for b in expanded_boxes)
    union_w = u_x2 - u_x1
    union_h = u_y2 - u_y1

    min_side = min(width, height, 256)

    # Decide between a single static crop box (preferred when head movement is localized)
    # vs a smoothly tracked constant-size crop box (when person walks/moves far across the frame)
    if union_w <= ref_w * 1.35 and union_h <= ref_h * 1.35:
        cx1, cy1, cx2, cy2 = u_x1, u_y1, u_x2, u_y2
        # Ensure minimum crop size
        if cx2 - cx1 < min_side:
            cx_mid = (cx1 + cx2) // 2
            cx1 = max(0, cx_mid - min_side // 2)
            cx2 = min(width, cx1 + min_side)
            cx1 = max(0, cx2 - min_side)
        if cy2 - cy1 < min_side:
            cy_mid = (cy1 + cy2) // 2
            cy1 = max(0, cy_mid - min_side // 2)
            cy2 = min(height, cy1 + min_side)
            cy1 = max(0, cy2 - min_side)

        # Snap to even coordinates
        cx1 -= cx1 % 2
        cy1 -= cy1 % 2
        if (cx2 - cx1) % 2 != 0:
            cx2 = cx2 + 1 if cx2 < width else cx2 - 1
        if (cy2 - cy1) % 2 != 0:
            cy2 = cy2 + 1 if cy2 < height else cy2 - 1

        crop_w = max(2, cx2 - cx1)
        crop_h = max(2, cy2 - cy1)
        crop_boxes = [(cx1, cy1, cx2, cy2)] * frames_read
        mode_desc = f"statis pada ({cx1},{cy1})-({cx2},{cy2})"
    else:
        # Smoothly tracked constant-size crop box following the head trajectory
        crop_w = min(width - (width % 2), max(min_side, int(round(ref_w * 1.15 / 8.0) * 8)))
        crop_h = min(height - (height % 2), max(min_side, int(round(ref_h * 1.15 / 8.0) * 8)))
        crop_w -= crop_w % 2
        crop_h -= crop_h % 2

        raw_centers = []
        last_c = (width / 2.0, height / 2.0)
        for b in head_boxes:
            if b is not None:
                last_c = ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)
            raw_centers.append(last_c)

        # Heavy moving-average smoothing (window = 25 frames) for smooth camera tracking
        win = 25
        crop_boxes = []
        for i in range(frames_read):
            sub = raw_centers[max(0, i - win // 2) : min(frames_read, i + win // 2 + 1)]
            scx = float(np.mean([c[0] for c in sub]))
            scy = float(np.mean([c[1] for c in sub]))
            cx1 = int(round(scx - crop_w / 2.0))
            cy1 = int(round(scy - crop_h / 2.0))
            cx1 = max(0, min(width - crop_w, cx1))
            cy1 = max(0, min(height - crop_h, cy1))
            cx1 -= cx1 % 2
            cy1 -= cy1 % 2
            crop_boxes.append((cx1, cy1, cx1 + crop_w, cy1 + crop_h))
        mode_desc = "tracking halus mengikuti pergerakan kepala"

    # Pad frame count to LTX-2.5 requirement: (padded_N - 1) % 8 == 0
    padded_frames = align_up(frames_read)
    eff_fps = max(1, int(round(fps)))

    c_in = Path(comfy_root).resolve() / "input"
    c_in.mkdir(parents=True, exist_ok=True)
    crop_filename = f"head_crop_{src_path.stem}.mp4"
    crop_video_path = c_in / crop_filename

    writer = VideoWriter(crop_video_path, crop_w, crop_h, Fraction(eff_fps, 1))
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
        # Pad remaining frames (if any) so LTX-2.5 VAE processes all frames without truncation
        if last_crop_frame is not None and padded_frames > idx:
            for _ in range(padded_frames - idx):
                writer.write(last_crop_frame)
    finally:
        capture2.release()
        writer.close()

    area_pct = (crop_w * crop_h) / max(1, width * height) * 100.0
    if log_cb:
        log_cb(
            f"✂️ Area kepala berhasil di-crop: {crop_w}x{crop_h} px ({area_pct:.1f}% dari frame utuh {width}x{height} px, {mode_desc})"
        )

    return {
        "orig_video_path": str(src_path),
        "crop_video_path": str(crop_video_path),
        "crop_filename": crop_filename,
        "orig_width": width,
        "orig_height": height,
        "orig_fps": fps,
        "effective_fps": eff_fps,
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
    log_cb=print,
) -> Path:
    """Paste the generated head/face crop back onto the full original video with feathered blending."""
    import cv2
    import numpy as np
    from mosaicdiff.head_detector import create_crop_head_mask
    from mosaicdiff.videoio import VideoWriter, copy_audio, open_capture

    orig_path = Path(crop_info["orig_video_path"]).resolve()
    gen_path = Path(generated_crop_video).resolve()
    dst_path = Path(dest_file).resolve()
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    orig_w = int(crop_info["orig_width"])
    orig_h = int(crop_info["orig_height"])
    orig_fps = float(crop_info["orig_fps"])
    frame_count = int(crop_info["orig_frame_count"])
    crop_boxes: list[tuple[int, int, int, int]] = crop_info["crop_boxes"]
    head_boxes: list[tuple[int, int, int, int] | None] = crop_info["head_boxes"]

    if log_cb:
        log_cb(
            f"🎬 Menempelkan (compositing) hasil generate area kepala kembali ke video utuh ({orig_w}x{orig_h})..."
        )

    cap_orig, _, _, _, _ = open_capture(orig_path)
    cap_gen, _, _, _, _ = open_capture(gen_path)

    silent_tmp = dst_path.with_name(f"{dst_path.stem}.silent_tmp{dst_path.suffix}")
    writer = VideoWriter(silent_tmp, orig_w, orig_h, Fraction(orig_fps).limit_denominator(1000))

    last_gen_frame: np.ndarray | None = None
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
            cbox = crop_boxes[idx] if idx < len(crop_boxes) else None

            if gen_frame is not None and hbox is not None and cbox is not None:
                cx1, cy1, cx2, cy2 = cbox
                cx1 = max(0, cx1)
                cy1 = max(0, cy1)
                cx2 = min(orig_frame.shape[1], cx2)
                cy2 = min(orig_frame.shape[0], cy2)
                cw = cx2 - cx1
                ch = cy2 - cy1

                if cw > 4 and ch > 4:
                    patch = gen_frame
                    if patch.shape[1] != cw or patch.shape[0] != ch:
                        patch = cv2.resize(patch, (cw, ch), interpolation=cv2.INTER_LANCZOS4)

                    alpha = create_crop_head_mask((cx1, cy1, cx2, cy2), hbox)
                    orig_region = orig_frame[cy1:cy2, cx1:cx2].astype(np.float32)
                    patch_f32 = patch.astype(np.float32)

                    # Gentle border-ring luminance/color harmonization so there is zero visible seam
                    ring_mask = (alpha[:, :, 0] > 0.05) & (alpha[:, :, 0] < 0.45)
                    if np.count_nonzero(ring_mask) > 32:
                        diff = np.mean(orig_region[ring_mask] - patch_f32[ring_mask], axis=0)
                        diff = np.clip(diff, -14.0, 14.0)
                        patch_f32 = np.clip(patch_f32 + diff, 0.0, 255.0)

                    blended = orig_region * (1.0 - alpha) + patch_f32 * alpha
                    orig_frame = orig_frame.copy()
                    orig_frame[cy1:cy2, cx1:cx2] = np.clip(np.round(blended), 0, 255).astype(np.uint8)

            writer.write(orig_frame)
            idx += 1
    finally:
        cap_orig.release()
        cap_gen.release()
        writer.close()

    # Copy original audio stream from source video
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
        log_cb(f"✅ Berhasil menempelkan area kepala ke video utuh: {dst_path.name}")

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
    """Load the official BFS API prompt graph directly from disk.
    
    This is the primary deterministic execution path: it directly loads the
    validated 56-node graph without relying on third-party converter servers,
    fragile network calls, or guesswork fallbacks.
    """
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
    log_cb=print,
) -> dict:
    """Patch the API prompt graph nodes with user-provided parameters.

    When crop_head_only=True (default), detects the head/face in video_input_path,
    writes a cropped guide video around the head area, and configures the workflow
    to generate ONLY the head region. run_comfy_workflow() then automatically pastes
    the generated head region back onto the full original video.
    """
    patched = json.loads(json.dumps(prompt_dict))  # deep copy

    crop_info = None
    guide_video_for_comfy = Path(video_input_path).resolve()
    if crop_head_only:
        crop_info = prepare_head_crop_video(
            video_input_path=video_input_path,
            comfy_root=comfy_root,
            log_cb=log_cb,
        )
        guide_video_for_comfy = Path(crop_info["crop_video_path"]).resolve()

    v_filename, h_filename = prepare_comfy_inputs(comfy_root, guide_video_for_comfy, head_image_path)

    if crop_info is not None:
        _ACTIVE_CROP_CONTEXT[v_filename] = crop_info
        _ACTIVE_CROP_CONTEXT["__latest__"] = crop_info
    else:
        _ACTIVE_CROP_CONTEXT.pop(v_filename, None)
        _ACTIVE_CROP_CONTEXT.pop("__latest__", None)

    # Format BFS Head Swap prompt
    p_text = prompt_text.strip()
    if not p_text.lower().startswith("head_swap:") and not p_text.lower().startswith("head swap"):
        if p_text:
            pos_prompt = f"head_swap:\nFACE:\n{p_text}"
        else:
            pos_prompt = "head_swap:\nFACE:\nPhotorealistic human face, natural skin texture, sharp detailed eyes, realistic lighting"
    else:
        pos_prompt = p_text

    neg_prompt = (
        negative_prompt_text.strip()
        or "mouth movement, cartoon, anime, illustration, 3d render, plastic, painting, blurry, deformed"
    )

    # Check available lora files in ComfyUI
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

        # 7. Resolution & Aspect Ratio (Node 403: ResolutionSelector)
        if ctype == "ResolutionSelector" or nid == "403":
            inputs["megapixels"] = float(resolution_mp)
            inputs["multiple"] = 32
            # Always match the aspect ratio of the cropped head video when crop_head_only is active
            if crop_info is not None or aspect_ratio == "auto" or not aspect_ratio:
                inputs["aspect_ratio"] = detect_video_aspect_ratio(guide_video_for_comfy)
            else:
                inputs["aspect_ratio"] = normalize_aspect_ratio(aspect_ratio)

        # 8. Frame Rate & Video Length synchronization with cropped video
        if crop_info is not None:
            if nid == "430" or (ctype == "PrimitiveInt" and "Frame Rate" in title):
                inputs["value"] = int(crop_info["effective_fps"])
            if nid == "425" or (ctype == "PrimitiveInt" and "Duration" in title):
                inputs["value"] = max(1, int(math.ceil(crop_info["padded_frame_count"] / crop_info["effective_fps"])))
            if nid == "627" or ctype == "EmptyAudio":
                inputs["duration"] = float(max(1.0, crop_info["padded_frame_count"] / crop_info["effective_fps"]))

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

        # 12. Math Expression for video length (Node 455: a * b + 1)
        if nid == "455" or (ctype == "ComfyMathExpression" and "b" in str(inputs.get("expression", ""))):
            if "values.b" not in inputs and "b" not in inputs:
                inputs["values.b"] = ["430", 0]
            if crop_info is not None:
                padded_n = int(crop_info["padded_frame_count"])
                inputs["expression"] = f"{padded_n} + 0 * a * b"

    # Prune second pass save nodes if disabled so ComfyUI only generates first pass
    for nid in nodes_to_remove:
        patched.pop(nid, None)

    return patched


def run_comfy_workflow(
    prompt_dict: dict,
    dest_file: str | Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8188,
    comfy_root: str | Path = "/content/ComfyUI",
    log_cb=print,
    progress_cb=None,
    cancel_event=None,
) -> Path:
    """Submit prompt to ComfyUI API, stream progress, and composite generated head crop back onto the full video."""
    # 0. Self-Healing Model Check: Pastikan seluruh file bobot BFS tersedia di ComfyUI/models/
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

    # Verifikasi apakah ComfyUI daemon sudah mengindeks model video VAE
    if is_comfy_running(host, port):
        if needed_restart or not is_model_indexed_by_comfy("VAELoader", "vae_name", "ltx-2.5-video-vae-bf16.safetensors", host, port):
            log_cb("Sinkronisasi model ke ComfyUI daemon (restart untuk re-indexing)...")
            stop_comfy_daemon(host=host, port=port, log_cb=log_cb)
            start_comfy_daemon(comfy_root=comfy_root, host=host, port=port, log_cb=log_cb)
    else:
        start_comfy_daemon(comfy_root=comfy_root, host=host, port=port, log_cb=log_cb)

    # Identify active head-crop metadata if this workflow was patched with crop_head_only=True
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

    log_cb(f"Workflow area kepala berhasil dimasukkan ke antrean ComfyUI (Prompt ID: {prompt_id})")

    # Connect to WebSocket for real-time progress & node execution
    ws = None
    try:
        import websocket
        ws_url = f"ws://{host}:{port}/ws?clientId={client_id}"
        ws = websocket.create_connection(ws_url, timeout=10)
        log_cb("Terhubung ke WebSocket ComfyUI untuk pemantauan live.")
    except Exception as exc:
        log_cb(f"Notice: WebSocket tidak aktif ({exc}), fallback ke polling HTTP.")

    # Poll execution until done
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
                                progress_cb(frac, f"Sampling Head Area: {val}/{total}")
                            elif val % 2 == 0 or val == total:
                                log_cb(f"⏳ Sampling Head Area: {val}/{total} steps ({frac * 100:.0f}%)")

                        elif mtype == "execution_error":
                            err_node = mdata.get("node_id")
                            err_msg = mdata.get("exception_message", "Unknown error")
                            raise ComfyAPIError(f"Error pada node {err_node}: {err_msg}")

                except websocket.WebSocketTimeoutException:
                    pass

            else:
                # HTTP Polling fallback
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

    log_cb("Eksekusi generate area kepala selesai! Mengambil file hasil...")
    # Get history to find written video files
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

    # Select best video: second_pass > first_pass > head_swap > comparison > first item
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
    log_cb(f"Video crop kepala terpilih: {out_video_path.name} ({out_video_path.stat().st_size / (1024*1024):.2f} MB)")

    target_dst = Path(dest_file).resolve() if dest_file else out_video_path.with_name(f"{out_video_path.stem}_full.mp4")

    # Composite the generated head crop back onto the full original video
    if crop_info is not None:
        final_video = composite_crop_to_full_video(
            crop_info=crop_info,
            generated_crop_video=out_video_path,
            dest_file=target_dst,
            log_cb=log_cb,
        )
        return final_video

    if dest_file:
        target_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out_video_path, target_dst)
        return target_dst

    return out_video_path
