"""Read frames, normalize video framerate to 24 FPS CFR, and write H.264 files."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
import shutil
import subprocess

import av
import cv2
import numpy as np

TARGET_FPS = 24


def open_capture(path: Path):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0) or float(TARGET_FPS)
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    return capture, width, height, fps, count


def normalize_video_to_24fps(source: Path, destination: Path, log_cb=None) -> Path:
    """Convert any input video to a strict 24.0 FPS Constant Frame Rate (CFR) master video.

    This guarantees 1-to-1 frame synchronization across OpenCV detection, cropped video
    extraction, ComfyUI LTX-2.5 generation, and final full-video compositing with zero lag.
    """
    src = Path(source).resolve()
    dst = Path(destination).resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)

    cap, width, height, orig_fps, orig_count = open_capture(src)
    cap.release()

    if log_cb:
        log_cb(
            f"⏱️ Menormalisasi video input ke {TARGET_FPS} FPS konstan (CFR) "
            f"[asal: {orig_fps:.2f} fps, {width}x{height}]..."
        )

    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin:
        cmd = [
            ffmpeg_bin,
            "-y",
            "-i",
            str(src),
            "-vf",
            f"fps={TARGET_FPS},scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-r",
            str(TARGET_FPS),
            "-vsync",
            "cfr",
            "-c:v",
            "libx264",
            "-crf",
            "17",
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            str(dst),
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res.returncode == 0 and dst.is_file() and dst.stat().st_size > 1024:
            return dst

        # Retry without audio stream in case source has no audio track or incompatible audio
        cmd_no_audio = [
            ffmpeg_bin,
            "-y",
            "-i",
            str(src),
            "-vf",
            f"fps={TARGET_FPS},scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-r",
            str(TARGET_FPS),
            "-vsync",
            "cfr",
            "-c:v",
            "libx264",
            "-crf",
            "17",
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(dst),
        ]
        res2 = subprocess.run(cmd_no_audio, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res2.returncode == 0 and dst.is_file() and dst.stat().st_size > 1024:
            return dst

    # Fallback using PyAV + OpenCV if ffmpeg binary is unavailable
    capture, width, height, fps, _count = open_capture(src)
    frames_raw = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames_raw.append(frame)
    finally:
        capture.release()

    if not frames_raw:
        raise RuntimeError(f"Tidak dapat membaca frame dari video: {src}")

    total_in = len(frames_raw)
    if abs(fps - TARGET_FPS) <= 0.05:
        picked_indices = list(range(total_in))
    else:
        out_count = max(1, int(round(total_in * TARGET_FPS / float(fps))))
        picked_indices = [
            min(total_in - 1, int(round(slot * (total_in - 1) / max(1, out_count - 1))))
            for slot in range(out_count)
        ]

    silent_tmp = dst.with_name(f"{dst.stem}.silent24{dst.suffix}")
    writer = VideoWriter(silent_tmp, width, height, Fraction(TARGET_FPS, 1))
    try:
        for idx in picked_indices:
            writer.write(frames_raw[idx])
    finally:
        writer.close()

    try:
        if copy_audio(silent_tmp, src, dst):
            silent_tmp.unlink(missing_ok=True)
        else:
            silent_tmp.replace(dst)
    except Exception:
        silent_tmp.replace(dst)

    return dst


class VideoWriter:
    def __init__(self, path: Path, width: int, height: int, fps: Fraction = Fraction(TARGET_FPS, 1)) -> None:
        self.path = path
        self.width = width - width % 2
        self.height = height - height % 2
        self.container = av.open(str(path), mode="w")
        self.stream = self.container.add_stream("libx264", rate=fps)
        self.stream.width = self.width
        self.stream.height = self.height
        self.stream.pix_fmt = "yuv420p"
        self.stream.options = {"crf": "18", "preset": "veryfast"}

    def write(self, bgr: np.ndarray) -> None:
        frame = bgr[: self.height, : self.width]
        if frame.shape[0] != self.height or frame.shape[1] != self.width:
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video = av.VideoFrame.from_ndarray(rgb, format="rgb24")
        for packet in self.stream.encode(video):
            self.container.mux(packet)

    def close(self) -> None:
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()


def copy_audio(silent_video: Path, source: Path, destination: Path) -> bool:
    """Copy the source audio onto ``silent_video``. Returns False when there is no audio."""
    video_in = av.open(str(silent_video))
    src = av.open(str(source))
    try:
        if not src.streams.audio:
            return False
        destination.parent.mkdir(parents=True, exist_ok=True)
        out = av.open(str(destination), mode="w")
        try:
            v_in = video_in.streams.video[0]
            v_out = out.add_stream_from_template(v_in)
            a_out = out.add_stream_from_template(src.streams.audio[0])
            for packet in video_in.demux(v_in):
                if packet.dts is None:
                    continue
                packet.stream = v_out
                out.mux(packet)
            for packet in src.demux(src.streams.audio[0]):
                if packet.dts is None:
                    continue
                packet.stream = a_out
                out.mux(packet)
        finally:
            out.close()
    finally:
        video_in.close()
        src.close()
    return True
