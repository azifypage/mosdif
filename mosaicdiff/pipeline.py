"""Head replacement pipeline: normalize to 24 FPS, lock onto 1 actor, dynamically crop head+hair rectangle, and refine with LTX-2.5."""

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
from mosaicdiff.head_detector import (
    HeadDetector,
    create_crop_head_mask,
    draw_paste_box_outline,
    lock_single_actor_trajectory,
    seamless_paste_rectangle,
    smooth_head_trajectory,
)
from mosaicdiff.align import HeadAligner
from mosaicdiff.paths import nodes_dir, worker_script
from mosaicdiff.settings import DEFAULT_NEGATIVE, DEFAULT_PROMPT, Settings
from mosaicdiff.videoio import TARGET_FPS, VideoWriter, copy_audio, normalize_video_to_24fps, open_capture

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

    if not torch.cuda.is_available():
        raise RuntimeError("MosaicDiff Head Replacement needs an NVIDIA GPU")

    log(f"Memulai Head Replacement (24 FPS Standard):")
    log(f"  Target Video : {source.name}")
    log(f"  Source Head  : {head_path.name}")

    norm_source = destination.with_name(destination.stem + ".24fps" + destination.suffix)
    comp_path = destination.with_name(destination.stem + ".comp" + destination.suffix)

    try:
        # 1. Normalize input video to strict 24.0 FPS CFR
        source_24 = normalize_video_to_24fps(source, norm_source, log_cb=log)

        detector = HeadDetector()
        try:
            aligner = HeadAligner(head_path, detector)
            head_boxes, width, height, fps, count = _detect_and_composite(
                source_24, comp_path, aligner, detector, head_path, log, progress, cancel
            )
        finally:
            detector.close()
            torch.cuda.empty_cache()

        present = [box for box in head_boxes if box is not None]
        if not present:
            raise RuntimeError(f"Tidak ditemukan wajah/kepala pada video {source.name}")

        # Expand slightly (+6% margin) around head_boxes for smooth rectangular feathering border
        grown = [expand_box(box, 0.06, width, height) if box is not None else None for box in head_boxes]
        present_grown = [b for b in grown if b is not None]
        x1 = max(0, min(b[0] for b in present_grown))
        y1 = max(0, min(b[1] for b in present_grown))
        x2 = min(width, max(b[2] for b in present_grown))
        y2 = min(height, max(b[3] for b in present_grown))
        x1 -= x1 % 2
        y1 -= y1 % 2
        if (x2 - x1) % 2 != 0:
            x2 = x2 + 1 if x2 < width else x2 - 1
        if (y2 - y1) % 2 != 0:
            y2 = y2 + 1 if y2 < height else y2 - 1
        crop = (x1, y1, x2, y2)
        log(f"LTX-2.5 Full Head+Hair Rectangle {crop[2] - crop[0]}x{crop[3] - crop[1]} at {crop[0]},{crop[1]} (24 FPS)")

        _h3(source_24, comp_path, destination, settings, crop, head_boxes, grown, float(TARGET_FPS), log, progress, cancel)
    finally:
        comp_path.unlink(missing_ok=True)
        norm_source.unlink(missing_ok=True)

    progress(1.0, "Done")
    return destination


def _check(cancel) -> None:
    if cancel.is_set():
        raise Cancelled()


def _detect_and_composite(
    source_24: Path,
    comp_path: Path,
    aligner: HeadAligner,
    detector: HeadDetector,
    head_path: Path,
    log,
    progress,
    cancel,
) -> tuple[list[tuple[int, int, int, int] | None], int, int, float, int]:
    """Pass 1: Detect all faces on 24 FPS video, lock onto 1 primary actor, and estimate full hair rectangle.
    Pass 2: Align & composite source head into video frames.
    """
    capture, width, height, _fps, count = open_capture(source_24)
    all_candidates: list[list[dict]] = []
    sampled_frames: dict[int, np.ndarray] = {}
    sample_stride = max(1, (count or 120) // 16)
    seen = 0

    src_ratios = (0.35, 0.48, 0.35, 0.18)
    src_img = cv2.imread(str(head_path))
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

    log(f"Mendeteksi wajah & mengunci 1 aktor pada seluruh frame video 24 FPS ({count} frames)...")
    try:
        while True:
            _check(cancel)
            ok, frame = capture.read()
            if not ok:
                break
            cands = detector.detect_all_faces(frame)
            all_candidates.append(cands)
            if seen % sample_stride == 0 and len(sampled_frames) < 20:
                sampled_frames[seen] = frame.copy()
            seen += 1
            if count and seen % 10 == 0:
                progress(0.18 * min(1.0, seen / count), f"Mendeteksi Kepala ({seen}/{count})")
    finally:
        capture.release()

    if not any(len(c) > 0 for c in all_candidates):
        return [], width, height, float(TARGET_FPS), 0

    locked = lock_single_actor_trajectory(all_candidates, width, height)

    target_left = [0.35]
    target_top = [0.48]
    target_right = [0.35]
    target_bot = [0.22]
    for f_idx, f_img in sampled_frames.items():
        if f_idx < len(locked) and locked[f_idx] is not None:
            det_kf = locked[f_idx]
            lr, tr, rr, br = detector.estimate_hair_expansion_ratios(
                f_img, det_kf["face_box"], landmarks=det_kf.get("landmarks")
            )
            target_left.append(lr)
            target_top.append(tr)
            target_right.append(rr)
            target_bot.append(br)

    final_ratios = (
        max(float(np.percentile(target_left, 95)), src_ratios[0]),
        max(float(np.percentile(target_top, 95)), src_ratios[1]),
        max(float(np.percentile(target_right, 95)), src_ratios[2]),
        max(float(np.percentile(target_bot, 95)), src_ratios[3]),
    )
    for det in locked:
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

    log("Menghaluskan lintasan 1 aktor (zero-lag 3-frame filter)...")
    smoothed = smooth_head_trajectory(locked, max_gap=90, fill_edges=True, window=3)
    head_boxes = [d["head_box"] if d is not None else None for d in smoothed]

    log("Menggabungkan kepala sumber ke frame video 24 FPS...")
    capture, width, height, _fps, count = open_capture(source_24)
    writer = VideoWriter(comp_path, width, height, Fraction(TARGET_FPS, 1))
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

    log(f"Berhasil mengunci 1 aktor pada {sum(b is not None for b in head_boxes)} dari {len(head_boxes)} frame (24 FPS)")
    return head_boxes, width, height, float(TARGET_FPS), len(head_boxes)


def _paste(
    frame: np.ndarray,
    patch: np.ndarray,
    crop_box: tuple[int, int, int, int],
    head_box: tuple[int, int, int, int] | None = None,
    state: dict | None = None,
    show_paste_box: bool = False,
) -> np.ndarray:
    """Paste the full generated rectangle ``patch`` onto ``frame`` with multi-band seamless blending."""
    out = seamless_paste_rectangle(
        orig_frame=frame,
        gen_patch=patch,
        crop_box=crop_box,
        head_box=head_box,
        state=state,
    )
    if show_paste_box:
        out = draw_paste_box_outline(out, crop_box, head_box)
    return out


def _h3(source_24, comp_path, destination, settings: Settings, crop, head_boxes, boxes, fps, log, progress, cancel) -> None:
    present = [index for index, box in enumerate(boxes) if box is not None]
    first, last = present[0], present[-1]
    sampled = list(range(first, last + 1))
    windows = split_samples(sampled)
    if not windows:
        raise RuntimeError("Bagian video yang terdeteksi kepala terlalu pendek (< 9 frame)")
    gen_w, gen_h = generation_size(crop[2] - crop[0], crop[3] - crop[1], settings.h3_resolution)
    log(f"LTX-2.5 Diffusion pass (Dynamic Rectangle {gen_w}x{gen_h} @ 24 FPS), {len(windows)} sample(s)")
    progress(0.45, "LTX-2.5 Head+Hair Diffusion")
    with tempfile.TemporaryDirectory(prefix="mosaicdiff-", dir=str(destination.parent)) as temp_name:
        temp = Path(temp_name)
        specs = []
        for number, indices in enumerate(windows):
            _check(cancel)
            context, overlap = fit_context(settings.h3_seconds, len(indices))
            out_dir = temp / f"window_{number:03d}"
            out_dir.mkdir()

            # Pure rectangular feathered mask for inpainting the entire head+hair rectangle
            mask_arr = create_crop_head_mask(crop)[:, :, 0]
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
            log(f"Sample {number + 1}: {len(indices)} frames @ 24 FPS, context {context}")
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
        progress(0.9, "Writing Output Video (24 FPS)")
        pastes = {}
        for spec in specs:
            out_dir = Path(spec["out_dir"])
            for order, frame_idx in enumerate(spec["frame_indices"]):
                image = out_dir / f"{order:06d}.png"
                if image.is_file():
                    pastes[int(frame_idx)] = image
        show_paste_box = bool(getattr(settings, "show_paste_box", False))
        _write_output(source_24, destination, pastes, crop, head_boxes, show_paste_box=show_paste_box)
        if settings.compare:
            compare_path = destination.with_name(destination.stem + "_compare" + destination.suffix)
            try:
                _write_compare(source_24, destination, compare_path)
                log(f"Comparison {compare_path.name}")
            except Exception as exc:
                compare_path.unlink(missing_ok=True)
                log(f"Comparison was not written: {exc}")
    with_audio = destination.with_name(destination.stem + ".audio" + destination.suffix)
    try:
        if copy_audio(destination, source_24, with_audio):
            with_audio.replace(destination)
    except Exception as exc:
        with_audio.unlink(missing_ok=True)
        log(f"Audio was not copied: {exc}")
    log(f"Berhasil menulis video hasil (24 FPS): {destination.name}")


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


def _write_output(source_24_path, destination, pastes, crop, head_boxes, show_paste_box: bool = False) -> None:
    capture, width, height, _fps, _count = open_capture(source_24_path)
    writer = VideoWriter(destination, width, height, Fraction(TARGET_FPS, 1))
    blend_state: dict = {}
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            image_path = pastes.get(index)
            hbox = head_boxes[index] if index < len(head_boxes) else None
            if image_path is not None:
                patch = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if patch is not None:
                    frame = _paste(
                        frame,
                        patch,
                        crop,
                        head_box=hbox,
                        state=blend_state,
                        show_paste_box=show_paste_box,
                    )
            writer.write(frame)
            index += 1
    finally:
        capture.release()
        writer.close()


def _write_compare(source_24, restored, destination) -> None:
    original, _w, _h, _fps, _orig_count = open_capture(source_24)
    result, width, height, _fps2, _res_count = open_capture(restored)
    writer = VideoWriter(destination, width * 2, height, Fraction(TARGET_FPS, 1))
    try:
        while True:
            ok_orig, frame_orig = original.read()
            ok_res, frame_res = result.read()
            if not ok_orig or not ok_res:
                break
            side_by_side = np.hstack([frame_orig, frame_res])
            writer.write(side_by_side)
    finally:
        original.release()
        result.release()
        writer.close()
