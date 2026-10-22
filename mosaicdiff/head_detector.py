"""Head & Face detector and mask generator for video head replacement.

Detects full head (face + hair + forehead + ears + jaw + neck) using MediaPipe
FaceMesh or FaceDetection with fallback to InsightFace and OpenCV Haar cascades.
"""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path

from mosaicdiff.geometry import edge_fade


class HeadDetector:
    def __init__(self, min_detection_confidence: float = 0.5):
        self.conf = float(min_detection_confidence)
        self._mp_face_mesh = None
        self._mp_face_detect = None
        self._insight_app = None
        self._haar = None
        self._backend = "none"

        # 1. Try MediaPipe FaceMesh (468 3D landmarks for full face + jaw contour)
        try:
            import mediapipe as mp
            if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
                self._mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=self.conf,
                    min_tracking_confidence=self.conf,
                )
                self._backend = "mediapipe_mesh"
        except Exception:
            self._mp_face_mesh = None

        # 2. Fallback to MediaPipe FaceDetection (fast bounding boxes + 6 landmarks)
        if self._backend == "none":
            try:
                import mediapipe as mp
                if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_detection"):
                    self._mp_face_detect = mp.solutions.face_detection.FaceDetection(
                        min_detection_confidence=self.conf,
                        model_selection=1,
                    )
                    self._backend = "mediapipe_detect"
            except Exception:
                self._mp_face_detect = None

        # 3. Fallback to InsightFace (e.g. on environments with MediaPipe >= 0.10.21)
        if self._backend == "none":
            try:
                from insightface.app import FaceAnalysis
                app = FaceAnalysis(
                    name="buffalo_l",
                    allowed_modules=["detection"],
                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                )
                app.prepare(ctx_id=0, det_size=(640, 640))
                self._insight_app = app
                self._backend = "insightface"
            except Exception:
                self._insight_app = None

        # 4. Always load OpenCV Haar Cascade as backup detector
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._haar = cv2.CascadeClassifier(cascade_path)
            if self._backend == "none":
                self._backend = "opencv_haar"
        except Exception:
            pass

        # Semantic silhouette segmentation for pixel-perfect hair & head contouring
        self._mp_selfie = None
        try:
            import mediapipe as mp
            if hasattr(mp, "solutions") and hasattr(mp.solutions, "selfie_segmentation"):
                self._mp_selfie = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
        except Exception:
            self._mp_selfie = None

    def close(self):
        if self._mp_selfie is not None:
            try:
                self._mp_selfie.close()
            except Exception:
                pass
            self._mp_selfie = None
        if self._mp_face_mesh is not None:
            try:
                self._mp_face_mesh.close()
            except Exception:
                pass
            self._mp_face_mesh = None
        if self._mp_face_detect is not None:
            try:
                self._mp_face_detect.close()
            except Exception:
                pass
            self._mp_face_detect = None

    def detect_head(self, frame_bgr: np.ndarray, need_mask: bool = True) -> dict | None:
        """Detect full head bounding box, 5 key landmarks, and optional feathered mask.

        Returns:
            dict with:
                'face_box': (fx1, fy1, fx2, fy2)
                'head_box': (hx1, hy1, hx2, hy2) (covers full head & hair)
                'landmarks': array of shape (5, 2) in frame coords
                'mask': 2D float32 array (0.0 to 1.0) of shape (H, W) or None if need_mask=False
            or None if no face/head is found.
        """
        h, w = frame_bgr.shape[:2]

        if self._backend == "mediapipe_mesh" and self._mp_face_mesh is not None:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            results = self._mp_face_mesh.process(rgb)
            if results and results.multi_face_landmarks:
                mesh = results.multi_face_landmarks[0]
                pts = np.array([[lm.x * w, lm.y * h] for lm in mesh.landmark], dtype=np.float32)

                # 5 standard alignment landmarks from FaceMesh indices
                # 33: left eye outer, 263: right eye outer, 1: nose tip, 61: left mouth corner, 291: right mouth corner
                left_eye = pts[33]
                right_eye = pts[263]
                nose = pts[1]
                left_mouth = pts[61]
                right_mouth = pts[291]
                landmarks = np.array([left_eye, right_eye, nose, left_mouth, right_mouth], dtype=np.float32)

                # Face bounding box
                fx1, fy1 = np.min(pts, axis=0)
                fx2, fy2 = np.max(pts, axis=0)

                # Expand to FULL HEAD (covers entire skull, hair volume, ears, and upper neck)
                head_box = self._expand_to_full_head(fx1, fy1, fx2, fy2, w, h)

                mask = None
                if need_mask:
                    seg_mask = None
                    if self._mp_selfie is not None:
                        try:
                            seg_res = self._mp_selfie.process(rgb)
                            if seg_res and seg_res.segmentation_mask is not None:
                                seg_mask = seg_res.segmentation_mask.astype(np.float32)
                        except Exception:
                            seg_mask = None
                    mask = self._generate_head_mask(head_box, pts, w, h, seg_mask=seg_mask)

                return {
                    "face_box": (int(fx1), int(fy1), int(fx2), int(fy2)),
                    "head_box": head_box,
                    "landmarks": landmarks,
                    "mask": mask,
                }

        # MediaPipe FaceDetection fallback
        if self._backend == "mediapipe_detect" and self._mp_face_detect is not None:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            results = self._mp_face_detect.process(rgb)
            if results and results.detections:
                detection = results.detections[0]
                bb = detection.location_data.relative_bounding_box
                fx1 = max(0, int(bb.xmin * w))
                fy1 = max(0, int(bb.ymin * h))
                fx2 = min(w, int((bb.xmin + bb.width) * w))
                fy2 = min(h, int((bb.ymin + bb.height) * h))

                kps = detection.location_data.relative_keypoints
                if len(kps) >= 4:
                    left_eye = np.array([kps[0].x * w, kps[0].y * h])
                    right_eye = np.array([kps[1].x * w, kps[1].y * h])
                    nose = np.array([kps[2].x * w, kps[2].y * h])
                    mouth = np.array([kps[3].x * w, kps[3].y * h])
                    landmarks = np.array([left_eye, right_eye, nose, mouth, mouth], dtype=np.float32)
                else:
                    cx = (fx1 + fx2) / 2
                    cy = (fy1 + fy2) / 2
                    landmarks = np.array([
                        [fx1 + (fx2 - fx1) * 0.3, fy1 + (fy2 - fy1) * 0.35],
                        [fx1 + (fx2 - fx1) * 0.7, fy1 + (fy2 - fy1) * 0.35],
                        [cx, cy],
                        [fx1 + (fx2 - fx1) * 0.35, fy1 + (fy2 - fy1) * 0.75],
                        [fx1 + (fx2 - fx1) * 0.65, fy1 + (fy2 - fy1) * 0.75],
                    ], dtype=np.float32)

                head_box = self._expand_to_full_head(fx1, fy1, fx2, fy2, w, h)

                mask = None
                if need_mask:
                    seg_mask = None
                    if self._mp_selfie is not None:
                        try:
                            seg_res = self._mp_selfie.process(rgb)
                            if seg_res and seg_res.segmentation_mask is not None:
                                seg_mask = seg_res.segmentation_mask.astype(np.float32)
                        except Exception:
                            seg_mask = None
                    mask = self._generate_head_mask(head_box, None, w, h, seg_mask=seg_mask)

                return {
                    "face_box": (fx1, fy1, fx2, fy2),
                    "head_box": head_box,
                    "landmarks": landmarks,
                    "mask": mask,
                }

        # InsightFace fallback
        if self._insight_app is not None:
            try:
                faces = self._insight_app.get(frame_bgr)
                if faces:
                    largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                    fx1, fy1, fx2, fy2 = [int(v) for v in largest.bbox[:4]]
                    fx1, fy1 = max(0, fx1), max(0, fy1)
                    fx2, fy2 = min(w, fx2), min(h, fy2)
                    if getattr(largest, "kps", None) is not None and len(largest.kps) >= 5:
                        landmarks = np.array(largest.kps[:5], dtype=np.float32)
                    else:
                        fw, fh = fx2 - fx1, fy2 - fy1
                        landmarks = np.array([
                            [fx1 + fw * 0.3, fy1 + fh * 0.35],
                            [fx1 + fw * 0.7, fy1 + fh * 0.35],
                            [(fx1 + fx2) / 2, (fy1 + fy2) / 2],
                            [fx1 + fw * 0.35, fy1 + fh * 0.75],
                            [fx1 + fw * 0.65, fy1 + fh * 0.75],
                        ], dtype=np.float32)
                    head_box = self._expand_to_full_head(fx1, fy1, fx2, fy2, w, h)
                    mask = self._generate_head_mask(head_box, None, w, h) if need_mask else None
                    return {
                        "face_box": (fx1, fy1, fx2, fy2),
                        "head_box": head_box,
                        "landmarks": landmarks,
                        "mask": mask,
                    }
            except Exception:
                pass

        # OpenCV Haar cascade fallback
        if self._haar is not None:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            faces = self._haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
            if len(faces) > 0:
                # Pick largest face
                largest = max(faces, key=lambda b: b[2] * b[3])
                fx1, fy1, fw, fh = largest
                fx2, fy2 = fx1 + fw, fy1 + fh

                cx = (fx1 + fx2) / 2
                cy = (fy1 + fy2) / 2
                landmarks = np.array([
                    [fx1 + fw * 0.3, fy1 + fh * 0.35],
                    [fx1 + fw * 0.7, fy1 + fh * 0.35],
                    [cx, cy],
                    [fx1 + fw * 0.35, fy1 + fh * 0.75],
                    [fx1 + fw * 0.65, fy1 + fh * 0.75],
                ], dtype=np.float32)

                head_box = self._expand_to_full_head(fx1, fy1, fx2, fy2, w, h)
                mask = self._generate_head_mask(head_box, None, w, h) if need_mask else None
                return {
                    "face_box": (int(fx1), int(fy1), int(fx2), int(fy2)),
                    "head_box": head_box,
                    "landmarks": landmarks,
                    "mask": mask,
                }

        return None

    def _expand_to_full_head(self, fx1: float, fy1: float, fx2: float, fy2: float, frame_w: int, frame_h: int) -> tuple[int, int, int, int]:
        """Expand face rectangle upwards to cover entire hair/skull, ears, and down to collar."""
        fw = fx2 - fx1
        fh = fy2 - fy1
        cx = (fx1 + fx2) / 2.0

        # Expand top by 55% of face height to include full hair & volume
        hx1 = max(0, int(cx - fw * 0.72))
        hx2 = min(frame_w, int(cx + fw * 0.72))
        hy1 = max(0, int(fy1 - fh * 0.55))
        # Expand bottom by 25% to cover chin and upper neck transition
        hy2 = min(frame_h, int(fy2 + fh * 0.25))

        return (hx1, hy1, hx2, hy2)

    def _generate_head_mask(
        self,
        head_box: tuple[int, int, int, int],
        landmarks_pts: np.ndarray | None,
        frame_w: int,
        frame_h: int,
        seg_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Create a smooth feathered 2D float32 mask (0.0 to 1.0) covering the full head."""
        hx1, hy1, hx2, hy2 = head_box
        hw = hx2 - hx1
        hh = hy2 - hy1

        if seg_mask is not None:
            # High-precision semantic hair & skull silhouette
            box_zone = np.zeros((frame_h, frame_w), dtype=np.float32)
            cv2.rectangle(box_zone, (hx1, hy1), (hx2, hy2), 1.0, -1)
            k_zone = max(11, int(min(hw, hh) * 0.12))
            if k_zone % 2 == 0:
                k_zone += 1
            box_zone = cv2.GaussianBlur(box_zone, (k_zone, k_zone), sigmaX=k_zone / 3.0)

            raw_head = np.clip(seg_mask * box_zone, 0.0, 1.0)
            core = (raw_head > 0.40).astype(np.float32)

            k_size = max(5, int(min(hw, hh) * 0.10))
            if k_size % 2 == 0:
                k_size += 1
            mask = cv2.GaussianBlur(core, (k_size, k_size), sigmaX=k_size / 3.0)
        else:
            mask = np.zeros((frame_h, frame_w), dtype=np.float32)
            cx = int((hx1 + hx2) / 2)
            cy = int((hy1 + hy2) / 2)
            rx = max(4, int(hw * 0.48))
            ry = max(4, int(hh * 0.48))
            cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 1.0, -1)

            k_size = max(5, int(min(hw, hh) * 0.15))
            if k_size % 2 == 0:
                k_size += 1
            mask = cv2.GaussianBlur(mask, (k_size, k_size), sigmaX=k_size / 3.0)

        # Smooth neck fade at the bottom (linear falloff)
        neck_start = int(hy1 + hh * 0.72)
        for y in range(neck_start, min(frame_h, hy2)):
            factor = float(1.0 - (y - neck_start) / max(1, hy2 - neck_start))
            mask[y, hx1:hx2] *= factor

        return np.clip(mask, 0.0, 1.0)


def create_crop_head_mask(
    crop_box: tuple[int, int, int, int],
    head_box: tuple[int, int, int, int] | None,
) -> np.ndarray:
    """Create a feathered 3D float32 mask (H_crop, W_crop, 1) in [0.0, 1.0] focused only on the head area.

    Ensures only the region around the detected head/face is blended into the full video,
    while smoothly fading to 0.0 before reaching the crop edges.
    """
    cx1, cy1, cx2, cy2 = crop_box
    cw = max(2, cx2 - cx1)
    ch = max(2, cy2 - cy1)
    short = min(cw, ch)
    falloff = max(16, min(round(short * 0.10), short // 4))
    border_fade = edge_fade(ch, cw, falloff)  # shape (ch, cw, 1)

    if head_box is None:
        return border_fade

    hx1, hy1, hx2, hy2 = head_box
    # Head coordinates relative to crop_box
    lhx1 = hx1 - cx1
    lhy1 = hy1 - cy1
    lhx2 = hx2 - cx1
    lhy2 = hy2 - cy1

    lhw = max(8, lhx2 - lhx1)
    lhh = max(8, lhy2 - lhy1)
    lcx = int(round((lhx1 + lhx2) / 2.0))
    lcy = int(round((lhy1 + lhy2) / 2.0))

    # Slightly expand ellipse around head_box (+12%) so new hair/jaw contour is never clipped,
    # then feather smoothly outward to 0.0
    rx = max(8, int(round(lhw * 0.56)))
    ry = max(8, int(round(lhh * 0.56)))

    head_plane = np.zeros((ch, cw), dtype=np.float32)
    cv2.ellipse(head_plane, (lcx, lcy), (rx, ry), 0, 0, 360, 1.0, -1)

    k_blur = max(15, int(round(min(lhw, lhh) * 0.28)))
    if k_blur % 2 == 0:
        k_blur += 1
    head_plane = cv2.GaussianBlur(head_plane, (k_blur, k_blur), sigmaX=k_blur / 2.8)

    combined = np.clip(head_plane[:, :, None] * border_fade, 0.0, 1.0)
    return combined


def smooth_head_trajectory(
    detections: list[dict | None],
    max_gap: int = 90,
    fill_edges: bool = False,
) -> list[dict | None]:
    """Smooth head bounding boxes across video frames to prevent any jitter or flickering."""
    n = len(detections)
    if n == 0:
        return []

    work = [dict(d) if d is not None else None for d in detections]

    # 1. Fill missing gaps with linear interpolation if surrounded by valid detections
    last_valid = None
    for i in range(n):
        if work[i] is not None:
            last_valid = i
        elif last_valid is not None:
            next_valid = None
            for j in range(i + 1, min(n, i + max_gap + 1)):
                if work[j] is not None:
                    next_valid = j
                    break
            if next_valid is not None:
                alpha = (i - last_valid) / (next_valid - last_valid)
                b1 = work[last_valid]["head_box"]
                b2 = work[next_valid]["head_box"]
                interp_box = (
                    int(b1[0] * (1 - alpha) + b2[0] * alpha),
                    int(b1[1] * (1 - alpha) + b2[1] * alpha),
                    int(b1[2] * (1 - alpha) + b2[2] * alpha),
                    int(b1[3] * (1 - alpha) + b2[3] * alpha),
                )
                interp_lm = (
                    work[last_valid]["landmarks"] * (1 - alpha)
                    + work[next_valid]["landmarks"] * alpha
                )
                work[i] = {
                    "face_box": interp_box,
                    "head_box": interp_box,
                    "landmarks": interp_lm,
                    "mask": None,
                }

    # Optional: fill leading/trailing frames with nearest valid detection
    if fill_edges and any(d is not None for d in work):
        first_valid = next(i for i, d in enumerate(work) if d is not None)
        last_valid_idx = max(i for i, d in enumerate(work) if d is not None)
        for i in range(0, first_valid):
            work[i] = dict(work[first_valid])
        for i in range(last_valid_idx + 1, n):
            work[i] = dict(work[last_valid_idx])

    # 2. Moving average box smoothing (window = 7 frames)
    smoothed = list(work)
    window = 7
    for i in range(n):
        if work[i] is None:
            continue
        valid_win = [
            work[j]["head_box"]
            for j in range(max(0, i - window // 2), min(n, i + window // 2 + 1))
            if work[j] is not None
        ]
        if valid_win:
            avg_box = tuple(int(round(np.mean([b[k] for b in valid_win]))) for k in range(4))
            smoothed[i] = dict(work[i])
            smoothed[i]["head_box"] = avg_box

    return smoothed
