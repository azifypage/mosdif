"""Head replacement pipeline: detect head in video, align and composite source head, and refine with LTX-2.5."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from mosaicdiff.geometry import (
    H3_FPS,
    edge_fade,
    expand_box,
    fit_context,
    frames_at_h3_fps,
    generation_size,
    letterbox,
    split_samples,
    stable_crop,
)
from mosaicdiff.head_detector import HeadDetector, create_crop_head_mask, smooth_head_trajectory
from mosaicdiff.align import HeadAligner
from mosaicdiff.paths import nodes_dir, worker_script
from mosaicdiff.settings import DEFAULT_NEGATIVE, DEFAULT_PROMPT, Settings
from mosaicdiff.videoio import VideoWriter, copy_audio, open_capture

PROMPT = DEFAULT_PROMPT


class Cancelled(Exception):
    pass


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    number = 1
    while True:
        candidate = path.with_name(f"{stem} ({number}){suffix}")
        if not candidate.exists():
            return candidate
        number += 1


def process_video(
    source: Path,
    destination: Path,
    settings: Settings,
    log,
    progress,
    cancel,
    head_image: Path | str | None = None,
    prompt: str | None = None,
    seed: int | None = None,
    denoise: float | None = None,
    **kwargs,
) -> Path:
    if head_image is not None:
        settings.head_image = str(head_image).strip()
    if prompt is not None:
        settings.prompt = str(prompt).strip()
    if seed is not None:
        settings.seed = int(seed)
    if denoise is not None:
        settings.denoise = float(denoise)

    # Validate head image
    head_path = Path(settings.head_image) if settings.head_image else None
    if head_path is None or not head_path.is_file():
        # Fallback check common locations
        for cand in [Path("/content/head.jpg"), Path("/content/head.png"), Path("/content/face.jpg"), Path("/content/face.png"), Path("head.jpg"), Path("head.png")]:
            if cand.is_file():
                head_path = cand
                settings.head_image = str(cand)
                break
    if head_path is None or not head_path.is_file():
        raise FileNotFoundError(
            "Foto kepala/wajah sumber ('head_image') tidak ditemukan! "
            "Silakan tentukan path foto wajah pada formulir Cell 5."
        )

    missing = settings.missing()
    if missing:
        listed = ", ".join(label for label, _path in missing)
        raise FileNotFoundError(f"Missing model files: {listed}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination = unique_path(destination)
    import torch

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("MosaicDiff Head Replacement needs an NVIDIA GPU")

    log(f"Memulai Head Replacement:")
    log(f"  Target Video : {source.name}")
    log(f"  Source Head  : {head_path.name}")

    detector = HeadDetector()
    try:
        aligner = HeadAligner(head_path, detector)
    except Exception as exc:
        detector.close()
        raise RuntimeError(f"Gagal memproses foto kepala '{head_path.name}': {exc}")

    comp_path = destination.with_name(destination.stem + ".comp" + destination.suffix)
    try:
        head_boxes, width, height, fps, count = _detect_and_composite(
            source, comp_path, aligner, detector, log, progress, cancel
        )
    finally:
        detector.close()
        torch.cuda.empty_cache()

    present = [box for box in head_boxes if box is not None]
    if not present:
        comp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Tidak ditemukan wajah/kepala pada video {source.name}")

    grown = []
    for box in head_boxes:
        if box is None:
            grown.append(None)
        else:
            grown.append(expand_box(*box, height, width))

    crop = stable_crop([box for box in grown if box is not None], width, height)
    log(f"LTX-2.5 Head crop {crop[2] - crop[0]}x{crop[3] - crop[1]} at {crop[0]},{crop[1]}")

    try:
        _h3(source, comp_path, destination, settings, crop, head_boxes, grown, fps, log, progress, cancel)
    except Cancelled:
        comp_path.unlink(missing_ok=True)
        raise
    except Exception:
        log(f"LTX-2.5 stopped. The intermediate composite video is at {comp_path.name}")
        raise

    comp_path.unlink(missing_ok=True)
    progress(1.0, "Done")
    return destination


def _check(cancel) -> None:
    if cancel.is_set():
        raise Cancelled()


def _detect_and_composite(
    source: Path,
    comp_path: Path,
    aligner: HeadAligner,
    detector: HeadDetector,
    log,
    progress,
    cancel,
) -> tuple[list[tuple[int, int, int, int] | None], int, int, float, int]:
    """Pass 1: Detect head in all frames with trajectory smoothing.
    Pass 2: Align & composite source head into video frames.
    """
    capture, width, height, fps, count = open_capture(source)
    detections = []
    seen = 0
    log(f"Mendeteksi kepala pada seluruh frame video ({count} frames)...")
    try:
        while True:
            _check(cancel)
            ok, frame = capture.read()
            if not ok:
                break
            det = detector.detect_head(frame, need_mask=False)
            detections.append(det)
            seen += 1
            if count and seen % 10 == 0:
                progress(0.18 * min(1.0, seen / count), f"Mendeteksi Kepala ({seen}/{count})")
    finally:
        capture.release()

    if not any(d is not None for d in detections):
        return [], width, height, fps, 0

    log("Menghaluskan lintasan kepala (smoothing trajectory)...")
    smoothed = smooth_head_trajectory(detections)
    head_boxes = [d["head_box"] if d is not None else None for d in smoothed]

    log("Menggabungkan kepala sumber ke frame video...")
    capture, width, height, fps, count = open_capture(source)
    writer = VideoWriter(comp_path, width, height, Fraction(fps).limit_denominator(1000))
    idx = 0
    try:
        while True:
            _check(cancel)
            ok, frame = capture.read()
            if not ok:
                break
            info = smoothed[idx] if idx < len(smoothed) else None
            if info is not None:
                composited, _mask = aligner.composite_head(frame, info)
                writer.write(composited)
            else:
                writer.write(frame)
            idx += 1
            if count and idx % 10 == 0:
                progress(0.18 + 0.22 * min(1.0, idx / count), f"Compositing Head ({idx}/{count})")
    finally:
        capture.release()
        writer.close()

    log(f"Berhasil menggabungkan kepala pada {sum(b is not None for b in head_boxes)} dari {len(head_boxes)} frame")
    return head_boxes, width, height, fps, len(head_boxes)


def _paste(
    frame: np.ndarray,
    patch: np.ndarray,
    crop_box: tuple[int, int, int, int],
    head_box: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """Paste only the feathered head/face area from ``patch`` onto the full video ``frame``."""
    x1, y1, x2, y2 = crop_box
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame.shape[1], x2)
    y2 = min(frame.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return frame
    cw = x2 - x1
    ch = y2 - y1
    fitted = patch
    if fitted.shape[1] != cw or fitted.shape[0] != ch:
        fitted = cv2.resize(fitted, (cw, ch), interpolation=cv2.INTER_LANCZOS4)

    fade = create_crop_head_mask((x1, y1, x2, y2), head_box)
    base = frame.copy()
    region = base[y1:y2, x1:x2].astype(np.float32)
    patch_f32 = fitted.astype(np.float32)

    # Gentle border-ring color/luminance harmonization
    ring = (fade[:, :, 0] > 0.05) & (fade[:, :, 0] < 0.45)
    if np.count_nonzero(ring) > 32:
        diff = np.mean(region[ring] - patch_f32[ring], axis=0)
        diff = np.clip(diff, -14.0, 14.0)
        patch_f32 = np.clip(patch_f32 + diff, 0.0, 255.0)

    mixed = region * (1.0 - fade) + patch_f32 * fade
    base[y1:y2, x1:x2] = np.clip(np.round(mixed), 0, 255).astype(np.uint8)
    return base


def _h3(source, comp_path, destination, settings: Settings, crop, head_boxes, boxes, fps, log, progress, cancel) -> None:
    present = [index for index, box in enumerate(boxes) if box is not None]
    first, last = present[0], present[-1]
    span = frames_at_h3_fps(last - first + 1, fps)
    sampled = [first + index for index in span]
    windows = split_samples(sampled)
    if not windows:
        raise RuntimeError("Bagian video yang terdeteksi kepala terlalu pendek (< 9 frame)")
    gen_w, gen_h = generation_size(crop[2] - crop[0], crop[3] - crop[1], settings.h3_resolution)
    log(f"LTX-2.5 Diffusion pass (Head Area Only), {len(windows)} sample(s), {gen_w}x{gen_h}")
    progress(0.45, "LTX-2.5 Head Diffusion")
    with tempfile.TemporaryDirectory(prefix="mosaicdiff-", dir=str(destination.parent)) as temp_name:
        temp = Path(temp_name)
        specs = []
        for number, indices in enumerate(windows):
            _check(cancel)
            context, overlap = fit_context(settings.h3_seconds, len(indices))
            out_dir = temp / f"window_{number:03d}"
            out_dir.mkdir()

            # Build window-specific inpainting mask covering the head trajectory within crop
            win_boxes = [head_boxes[idx] for idx in indices if idx < len(head_boxes) and head_boxes[idx] is not None]
            mask_path = None
            if win_boxes:
                union_head = (
                    min(b[0] for b in win_boxes),
                    min(b[1] for b in win_boxes),
                    max(b[2] for b in win_boxes),
                    max(b[3] for b in win_boxes),
                )
                mask_arr = create_crop_head_mask(crop, union_head)[:, :, 0]
                mask_u8 = np.clip(np.round(mask_arr * 255.0), 0, 255).astype(np.uint8)
                mask_file = out_dir / "head_mask.png"
                cv2.imwrite(str(mask_file), mask_u8)
                mask_path = str(mask_file)

            specs.append(
                {
                    "frame_indices": indices,
                    "width": gen_w,
                    "height": gen_h,
                    "crop": list(crop),
                    "mask": mask_path,
                    "out_dir": str(out_dir),
                    "context_frames": context,
                    "context_overlap": overlap,
                }
            )
            log(f"Sample {number + 1}: {len(indices)} frames, context {context}")
        comfy_root = settings.resolved("comfy_root")
        c_nodes = comfy_root / "custom_nodes"
        src_nodes = nodes_dir()
        if c_nodes.is_dir() and src_nodes.is_dir():
            import shutil
            for n_dir in src_nodes.iterdir():
                target = c_nodes / n_dir.name
                if n_dir.is_dir() and not target.exists():
                    try:
                        shutil.copytree(str(n_dir), str(target))
                    except Exception:
                        pass

        lora_target = settings.resolved("lora")
        user_prompt = (getattr(settings, "prompt", "") or "").strip()
        if not user_prompt:
            effective_prompt = DEFAULT_PROMPT
        elif not user_prompt.lower().startswith("head_swap:") and not user_prompt.lower().startswith("head swap"):
            effective_prompt = f"head_swap:\nFACE:\n{user_prompt}"
        else:
            effective_prompt = user_prompt

        user_negative = (getattr(settings, "negative_prompt", "") or "").strip()
        effective_negative = user_negative if user_negative else DEFAULT_NEGATIVE

        effective_seed = int(getattr(settings, "seed", 42))
        effective_denoise = float(getattr(settings, "denoise", 0.40))
        effective_lora_strength = float(getattr(settings, "lora_strength", 0.80))
        log(f"LTX-2.5 Refiner: Denoise={effective_denoise:.2f} | LoRA Strength={effective_lora_strength:.2f} | Steps={settings.ltx_steps} | Seed={effective_seed}")
        log(f"LTX-2.5 Positive Prompt:\n{effective_prompt}")
        log(f"LTX-2.5 Negative Prompt:\n{effective_negative}")

        job = {
            "comfy_root": str(comfy_root),
            "nodes_dir": str(nodes_dir()),
            "unet": str(settings.resolved("unet")),
            "lora": str(lora_target) if lora_target.is_file() else None,
            "lora_strength": effective_lora_strength,
            "clip": str(settings.resolved("clip")),
            "video_vae": str(settings.resolved("vae")),
            "prompt": effective_prompt,
            "negative_prompt": effective_negative,
            "seed": effective_seed,
            "steps": int(getattr(settings, "ltx_steps", getattr(settings, "h3_steps", 6))),
            "denoise": effective_denoise,
            "video": str(comp_path),
            "windows": specs,
        }
        job_path = temp / "job.json"
        job_path.write_text(json.dumps(job), encoding="utf-8")
        _run_comfy(settings.resolved("comfy_python"), job_path, log, cancel)
        progress(0.9, "Writing Output Video")
        pastes = {}
        for spec in specs:
            out_dir = Path(spec["out_dir"])
            for order, frame_idx in enumerate(spec["frame_indices"]):
                image = out_dir / f"{order:06d}.png"
                if image.is_file():
                    pastes[int(frame_idx)] = image
        output_fps = H3_FPS if fps > H3_FPS + 0.05 else fps
        # Paste generated head region directly onto the original full video (source)
        _write_output(source, destination, pastes, crop, head_boxes, output_fps, fps, len(boxes))
        if settings.compare:
            compare_path = destination.with_name(destination.stem + "_compare" + destination.suffix)
            try:
                _write_compare(source, destination, compare_path, fps)
                log(f"Comparison {compare_path.name}")
            except Exception as exc:
                compare_path.unlink(missing_ok=True)
                log(f"Comparison was not written: {exc}")
    with_audio = destination.with_name(destination.stem + ".audio" + destination.suffix)
    try:
        if copy_audio(destination, source, with_audio):
            with_audio.replace(destination)
    except Exception as exc:
        with_audio.unlink(missing_ok=True)
        log(f"Audio was not copied: {exc}")
    log(f"Berhasil menulis video hasil: {destination.name}")


def _run_comfy(python: Path, job_path: Path, log, cancel) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = ""
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONUTF8"] = "1"
    process = subprocess.Popen(
        [str(python), "-s", str(worker_script()), "--job", str(job_path)],
        cwd=str(job_path.parent),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert process.stdout is not None
    try:
        for line in process.stdout:
            _check(cancel)
            text = line.rstrip()
            if text and "%|" not in text:
                log(text)
    except Cancelled:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        raise
    code = process.wait()
    if code != 0:
        raise RuntimeError(f"LTX-2.5 diffusion process exited with status {code}")


def _write_output(source_path, destination, pastes, crop, head_boxes, output_fps, source_fps, frame_count: int) -> None:
    capture, width, height, _fps, _count = open_capture(source_path)
    keep = set(frames_at_h3_fps(frame_count, source_fps)) if output_fps == H3_FPS else None
    writer = VideoWriter(destination, width, height, Fraction(output_fps).limit_denominator(1000))
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if keep is not None and index not in keep:
                index += 1
                continue
            image_path = pastes.get(index)
            hbox = head_boxes[index] if index < len(head_boxes) else None
            if image_path is not None and hbox is not None:
                patch = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if patch is not None:
                    frame = _paste(frame, patch, crop, head_box=hbox)
            writer.write(frame)
            index += 1
    finally:
        capture.release()
        writer.close()


def _write_compare(source, restored, destination, source_fps) -> None:
    original, _w, _h, _fps, original_count = open_capture(source)
    result, width, height, _fps, restored_count = open_capture(restored)
    if abs(source_fps - H3_FPS) > 0.05 and original_count != restored_count:
        scale = restored_count / original_count
        frame_map = {index: int(round(index * scale)) for index in range(original_count)}
    else:
        frame_map = None
    writer = VideoWriter(destination, width * 2, height, Fraction(H3_FPS).limit_denominator(1000))
    index = 0
    try:
        while True:
            ok_orig, frame_orig = original.read()
            ok_res, frame_res = result.read()
            if not ok_orig or not ok_res:
                break
            side_by_side = np.hstack([frame_orig, frame_res])
            writer.write(side_by_side)
            index += 1
    finally:
        original.release()
        result.release()
        writer.close()
