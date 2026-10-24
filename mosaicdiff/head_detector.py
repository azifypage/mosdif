"""Head, Face & Full-Hair detector, single-actor tracker, and rectangular mask generator.

Detects full head + full hair extent (even long hair reaching shoulders, chest, or stomach),
locks onto a single primary actor across the video, and generates rectangular masks with
smooth feathered borders for seamless full-video compositing.
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

        # 1. Try MediaPipe FaceMesh (detect up to 5 faces per frame so we can lock onto 1 target actor)
        try:
            import mediapipe as mp
            if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
                self._mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=5,
                    refine_landmarks=True,
                    min_detection_confidence=self.conf,
                    min_tracking_confidence=self.conf,
                )
                self._backend = "mediapipe_mesh"
        except Exception:
            self._mp_face_mesh = None

        # 2. Fallback to MediaPipe FaceDetection
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

        # 3. Try InsightFace (provides both detection and identity embeddings if available)
        if self._backend == "none":
            try:
                from insightface.app import FaceAnalysis
                app = FaceAnalysis(
                    name="buffalo_l",
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

        # Semantic silhouette segmentation for hair & upper-body contouring
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

    def _compute_face_signature(self, frame_bgr: np.ndarray, face_box: tuple[int, int, int, int]) -> np.ndarray:
        """Compute a fast normalized color+luminance signature of a face crop for single-actor locking."""
        h, w = frame_bgr.shape[:2]
        fx1, fy1, fx2, fy2 = face_box
        fx1 = max(0, min(w - 1, fx1))
        fy1 = max(0, min(h - 1, fy1))
        fx2 = max(fx1 + 1, min(w, fx2))
        fy2 = max(fy1 + 1, min(h, fy2))
        crop = frame_bgr[fy1:fy2, fx1:fx2]
        if crop.size == 0:
            return np.zeros(24, dtype=np.float32)
        small = cv2.resize(crop, (32, 32), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv], [0], None, [12], [0, 180]).flatten()
        hist_s = cv2.calcHist([hsv], [1], None, [6], [0, 256]).flatten()
        hist_v = cv2.calcHist([hsv], [2], None, [6], [0, 256]).flatten()
        vec = np.concatenate([hist_h, hist_s, hist_v]).astype(np.float32)
        norm = np.linalg.norm(vec) + 1e-6
        return vec / norm

    def detect_all_faces(
        self,
        frame_bgr: np.ndarray,
        expansion_ratios: tuple[float, float, float] | None = None,
    ) -> list[dict]:
        """Detect all candidate faces in a frame (for single-actor selection and tracking)."""
        h, w = frame_bgr.shape[:2]
        candidates: list[dict] = []

        if self._backend == "mediapipe_mesh" and self._mp_face_mesh is not None:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            results = self._mp_face_mesh.process(rgb)
            if results and results.multi_face_landmarks:
                for mesh in results.multi_face_landmarks:
                    pts = np.array([[lm.x * w, lm.y * h] for lm in mesh.landmark], dtype=np.float32)
                    landmarks = np.array([pts[33], pts[263], pts[1], pts[61], pts[291]], dtype=np.float32)
                    fx1, fy1 = np.min(pts, axis=0)
                    fx2, fy2 = np.max(pts, axis=0)
                    fbox = (max(0, int(fx1)), max(0, int(fy1)), min(w, int(fx2)), min(h, int(fy2)))
                    hbox = self._expand_to_full_head(fx1, fy1, fx2, fy2, w, h, ratios=expansion_ratios)
                    sig = self._compute_face_signature(frame_bgr, fbox)
                    candidates.append(
                        {
                            "face_box": fbox,
                            "head_box": hbox,
                            "landmarks": landmarks,
                            "signature": sig,
                            "mask": None,
                        }
                    )
            if candidates:
                return candidates

        if self._backend == "mediapipe_detect" and self._mp_face_detect is not None:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            results = self._mp_face_detect.process(rgb)
            if results and results.detections:
                for detection in results.detections:
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
                        cx, cy = (fx1 + fx2) / 2, (fy1 + fy2) / 2
                        landmarks = np.array(
                            [
                                [fx1 + (fx2 - fx1) * 0.3, fy1 + (fy2 - fy1) * 0.35],
                                [fx1 + (fx2 - fx1) * 0.7, fy1 + (fy2 - fy1) * 0.35],
                                [cx, cy],
                                [fx1 + (fx2 - fx1) * 0.35, fy1 + (fy2 - fy1) * 0.75],
                                [fx1 + (fx2 - fx1) * 0.65, fy1 + (fy2 - fy1) * 0.75],
                            ],
                            dtype=np.float32,
                        )
                    fbox = (fx1, fy1, fx2, fy2)
                    hbox = self._expand_to_full_head(fx1, fy1, fx2, fy2, w, h, ratios=expansion_ratios)
                    sig = self._compute_face_signature(frame_bgr, fbox)
                    candidates.append(
                        {
                            "face_box": fbox,
                            "head_box": hbox,
                            "landmarks": landmarks,
                            "signature": sig,
                            "mask": None,
                        }
                    )
            if candidates:
                return candidates

        if self._insight_app is not None:
            try:
                faces = self._insight_app.get(frame_bgr)
                for face in faces:
                    fx1, fy1, fx2, fy2 = [int(v) for v in face.bbox[:4]]
                    fx1, fy1 = max(0, fx1), max(0, fy1)
                    fx2, fy2 = min(w, fx2), min(h, fy2)
                    if getattr(face, "kps", None) is not None and len(face.kps) >= 5:
                        landmarks = np.array(face.kps[:5], dtype=np.float32)
                    else:
                        fw, fh = fx2 - fx1, fy2 - fy1
                        landmarks = np.array(
                            [
                                [fx1 + fw * 0.3, fy1 + fh * 0.35],
                                [fx1 + fw * 0.7, fy1 + fh * 0.35],
                                [(fx1 + fx2) / 2, (fy1 + fy2) / 2],
                                [fx1 + fw * 0.35, fy1 + fh * 0.75],
                                [fx1 + fw * 0.65, fy1 + fh * 0.75],
                            ],
                            dtype=np.float32,
                        )
                    fbox = (fx1, fy1, fx2, fy2)
                    hbox = self._expand_to_full_head(fx1, fy1, fx2, fy2, w, h, ratios=expansion_ratios)
                    emb = getattr(face, "normed_embedding", None)
                    if emb is not None:
                        sig = np.array(emb, dtype=np.float32)
                    else:
                        sig = self._compute_face_signature(frame_bgr, fbox)
                    candidates.append(
                        {
                            "face_box": fbox,
                            "head_box": hbox,
                            "landmarks": landmarks,
                            "signature": sig,
                            "mask": None,
                        }
                    )
                if candidates:
                    return candidates
            except Exception:
                pass

        if self._haar is not None:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            faces = self._haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
            for fx1, fy1, fw, fh in faces:
                fx2, fy2 = int(fx1 + fw), int(fy1 + fh)
                fx1, fy1 = int(fx1), int(fy1)
                cx, cy = (fx1 + fx2) / 2, (fy1 + fy2) / 2
                landmarks = np.array(
                    [
                        [fx1 + fw * 0.3, fy1 + fh * 0.35],
                        [fx1 + fw * 0.7, fy1 + fh * 0.35],
                        [cx, cy],
                        [fx1 + fw * 0.35, fy1 + fh * 0.75],
                        [fx1 + fw * 0.65, fy1 + fh * 0.75],
                    ],
                    dtype=np.float32,
                )
                fbox = (fx1, fy1, fx2, fy2)
                hbox = self._expand_to_full_head(fx1, fy1, fx2, fy2, w, h, ratios=expansion_ratios)
                sig = self._compute_face_signature(frame_bgr, fbox)
                candidates.append(
                    {
                        "face_box": fbox,
                        "head_box": hbox,
                        "landmarks": landmarks,
                        "signature": sig,
                        "mask": None,
                    }
                )

        return candidates

    def detect_head(
        self,
        frame_bgr: np.ndarray,
        need_mask: bool = True,
        expansion_ratios: tuple[float, float, float] | None = None,
    ) -> dict | None:
        """Detect the primary head in a single frame."""
        h, w = frame_bgr.shape[:2]
        candidates = self.detect_all_faces(frame_bgr, expansion_ratios=expansion_ratios)
        if not candidates:
            return None

        # Pick largest face in single-image mode
        best = max(
            candidates,
            key=lambda c: (c["face_box"][2] - c["face_box"][0]) * (c["face_box"][3] - c["face_box"][1]),
        )
        if need_mask:
            seg_mask = self.get_person_segmentation(frame_bgr)
            best["mask"] = self._generate_head_mask(best["head_box"], None, w, h, seg_mask=seg_mask)
        return best

    def get_person_segmentation(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        """Return 2D float32 person silhouette mask in [0.0, 1.0] if SelfieSegmentation is available."""
        if self._mp_selfie is None:
            return None
        try:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            seg_res = self._mp_selfie.process(rgb)
            if seg_res and seg_res.segmentation_mask is not None:
                return seg_res.segmentation_mask.astype(np.float32)
        except Exception:
            pass
        return None

    def estimate_hair_expansion_ratios(
        self,
        frame_bgr: np.ndarray,
        face_box: tuple[int, int, int, int],
    ) -> tuple[float, float, float]:
        """Estimate (top_ratio, bottom_ratio, side_ratio) needed to cover all hair attached to the head.

        Handles short hair, wide/voluminous hair, hijabs, and long hair reaching down to the
        shoulders, chest, or stomach.
        """
        h, w = frame_bgr.shape[:2]
        fx1, fy1, fx2, fy2 = face_box
        fw = max(12.0, float(fx2 - fx1))
        fh = max(12.0, float(fy2 - fy1))
        cx = (fx1 + fx2) / 2.0

        # Base minimum ratios (standard head + short/medium hair + neck)
        top_ratio = 0.65
        bottom_ratio = 0.45
        side_ratio = 0.85

        # Search window extending up above skull, wide to shoulders, and down to stomach (fy2 + 2.8 * fh)
        sx1 = max(0, int(round(cx - fw * 1.65)))
        sx2 = min(w, int(round(cx + fw * 1.65)))
        sy1 = max(0, int(round(fy1 - fh * 0.95)))
        sy2 = min(h, int(round(fy2 + fh * 2.80)))

        if sx2 - sx1 < 16 or sy2 - sy1 < 16:
            return (top_ratio, bottom_ratio, side_ratio)

        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

        # Sample skin color from central face
        skin_x1 = max(0, int(fx1 + fw * 0.22))
        skin_x2 = min(w, int(fx2 - fw * 0.22))
        skin_y1 = max(0, int(fy1 + fh * 0.20))
        skin_y2 = min(h, int(fy2 - fh * 0.20))
        skin_patch = lab[skin_y1:skin_y2, skin_x1:skin_x2]
        skin_mean = np.mean(skin_patch, axis=(0, 1)) if skin_patch.size > 0 else np.array([150.0, 135.0, 140.0])

        # Sample hair color from crown (above forehead) and left/right temples
        crown_y1 = max(0, int(fy1 - fh * 0.42))
        crown_y2 = max(crown_y1 + 2, int(fy1 + fh * 0.08))
        crown_x1 = max(0, int(fx1 - fw * 0.28))
        crown_x2 = min(w, int(fx2 + fw * 0.28))
        crown_patch = lab[crown_y1:crown_y2, crown_x1:crown_x2]

        if crown_patch.size == 0:
            return (top_ratio, bottom_ratio, side_ratio)

        # Exclude skin-like pixels from the crown/temple sample to isolate hair/hijab color
        crown_flat = crown_patch.reshape(-1, 3)
        dist_to_skin = np.linalg.norm(crown_flat - skin_mean, axis=1)
        non_skin = crown_flat[dist_to_skin > 14.0]
        if len(non_skin) < 16:
            non_skin = crown_flat

        hair_median = np.median(non_skin, axis=0)
        hair_std = np.clip(np.std(non_skin, axis=0), 10.0, 38.0)

        # Sample outer background from top-left and top-right corners of search box
        bg_samples = []
        bw_strip = max(4, (sx2 - sx1) // 10)
        bh_strip = max(4, (sy2 - sy1) // 10)
        bg_samples.append(lab[sy1 : sy1 + bh_strip, sx1 : sx1 + bw_strip].reshape(-1, 3))
        bg_samples.append(lab[sy1 : sy1 + bh_strip, sx2 - bw_strip : sx2].reshape(-1, 3))
        bg_all = np.concatenate([b for b in bg_samples if b.size > 0], axis=0)
        bg_median = np.median(bg_all, axis=0) if len(bg_all) > 0 else np.array([0.0, 128.0, 128.0])

        roi_lab = lab[sy1:sy2, sx1:sx2]
        dist_hair = np.linalg.norm((roi_lab - hair_median) / (hair_std * 1.65), axis=2)
        dist_bg = np.linalg.norm(roi_lab - bg_median, axis=2)
        dist_skin_roi = np.linalg.norm(roi_lab - skin_mean, axis=2)

        # Candidate hair mask: matches hair distribution, not skin, and distinguishable from corner background
        hair_mask = (dist_hair < 2.2) & (dist_skin_roi > 12.0) & (dist_bg > 16.0)

        # Constrain with person silhouette segmentation if available
        seg = self.get_person_segmentation(frame_bgr)
        if seg is not None:
            seg_roi = seg[sy1:sy2, sx1:sx2]
            hair_mask = hair_mask & (seg_roi > 0.35)

        hair_u8 = hair_mask.astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        hair_u8 = cv2.morphologyEx(hair_u8, cv2.MORPH_CLOSE, kernel)
        hair_u8 = cv2.morphologyEx(hair_u8, cv2.MORPH_OPEN, kernel)

        # Keep only connected components that overlap the head/scalp seed zone
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(hair_u8, connectivity=8)
        seed_x1 = max(0, int(fx1 - fw * 0.45) - sx1)
        seed_x2 = min(sx2 - sx1, int(fx2 + fw * 0.45) - sx1)
        seed_y1 = max(0, int(fy1 - fh * 0.55) - sy1)
        seed_y2 = min(sy2 - sy1, int(fy2 + fh * 0.30) - sy1)

        connected = np.zeros_like(hair_u8)
        if seed_x2 > seed_x1 and seed_y2 > seed_y1:
            seed_labels = set(np.unique(labels[seed_y1:seed_y2, seed_x1:seed_x2])) - {0}
            for lbl in seed_labels:
                if stats[lbl, cv2.CC_STAT_AREA] >= 36:
                    connected[labels == lbl] = 255

        ys, xs = np.where(connected > 0)
        if len(ys) > 32:
            # Use 3rd and 97th percentiles to reject isolated noise pixels
            min_y = float(np.percentile(ys, 3)) + sy1
            max_y = float(np.percentile(ys, 97)) + sy1
            min_x = float(np.percentile(xs, 3)) + sx1
            max_x = float(np.percentile(xs, 97)) + sx1

            det_top = (fy1 - min_y) / fh + 0.12
            det_bot = (max_y - fy2) / fh + 0.15
            det_side = max(cx - min_x, max_x - cx) / fw + 0.12

            top_ratio = float(np.clip(max(top_ratio, det_top), 0.65, 1.25))
            bottom_ratio = float(np.clip(max(bottom_ratio, det_bot), 0.45, 2.80))
            side_ratio = float(np.clip(max(side_ratio, det_side), 0.85, 1.75))

        return (top_ratio, bottom_ratio, side_ratio)

    def _expand_to_full_head(
        self,
        fx1: float,
        fy1: float,
        fx2: float,
        fy2: float,
        frame_w: int,
        frame_h: int,
        ratios: tuple[float, float, float] | None = None,
    ) -> tuple[int, int, int, int]:
        """Expand face rectangle to cover full skull, hair (including long hair), ears, and neck/chest."""
        fw = fx2 - fx1
        fh = fy2 - fy1
        cx = (fx1 + fx2) / 2.0

        top_r, bot_r, side_r = ratios if ratios is not None else (0.65, 0.45, 0.85)

        hx1 = max(0, int(round(cx - fw * side_r)))
        hx2 = min(frame_w, int(round(cx + fw * side_r)))
        hy1 = max(0, int(round(fy1 - fh * top_r)))
        hy2 = min(frame_h, int(round(fy2 + fh * bot_r)))

        return (hx1, hy1, hx2, hy2)

    def _generate_head_mask(
        self,
        head_box: tuple[int, int, int, int],
        landmarks_pts: np.ndarray | None,
        frame_w: int,
        frame_h: int,
        seg_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Create a smooth feathered 2D float32 rectangular mask (0.0 to 1.0) covering the full head + hair box."""
        hx1, hy1, hx2, hy2 = head_box
        hx1, hy1 = max(0, hx1), max(0, hy1)
        hx2, hy2 = min(frame_w, hx2), min(frame_h, hy2)
        hw = max(2, hx2 - hx1)
        hh = max(2, hy2 - hy1)

        mask = np.zeros((frame_h, frame_w), dtype=np.float32)
        short = min(hw, hh)
        falloff = max(12, min(round(short * 0.08), short // 5))
        rect_fade = edge_fade(hh, hw, falloff)[:, :, 0]
        mask[hy1:hy2, hx1:hx2] = rect_fade
        return np.clip(mask, 0.0, 1.0)


def create_crop_head_mask(
    crop_box: tuple[int, int, int, int],
    head_box: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """Create a pure rectangular feathered 3D float32 mask (H_crop, W_crop, 1) in [0.0, 1.0].

    Preserves the full rectangular area (including all hair at corners and lower region)
    while applying a smooth feather only along the outer edges of the rectangle.
    """
    cx1, cy1, cx2, cy2 = crop_box
    cw = max(2, cx2 - cx1)
    ch = max(2, cy2 - cy1)
    short = min(cw, ch)
    falloff = max(12, min(round(short * 0.07), short // 6))
    return edge_fade(ch, cw, falloff)


def lock_single_actor_trajectory(
    all_frame_candidates: list[list[dict]],
    frame_w: int,
    frame_h: int,
) -> list[dict | None]:
    """Select the single primary target actor across all frames and reject all other actors.

    Uses spatial continuity (center distance + box overlap) and face appearance/embedding
    similarity from the dominant anchor actor, ensuring ONLY 1 actor is tracked.
    """
    n = len(all_frame_candidates)
    if n == 0:
        return []

    # 1. Score every candidate across all frames to find the primary actor's anchor frame
    best_score = -1.0
    anchor_frame = -1
    anchor_cand: dict | None = None
    frame_diag = float(np.hypot(frame_w, frame_h))

    for idx, cands in enumerate(all_frame_candidates):
        for c in cands:
            fx1, fy1, fx2, fy2 = c["face_box"]
            area = float(max(1, (fx2 - fx1) * (fy2 - fy1)))
            cx = (fx1 + fx2) / 2.0
            cy = (fy1 + fy2) / 2.0
            # Centrality bonus (0.6 to 1.0) so foreground central subject wins over edge bystanders
            dist_center = np.hypot(cx - frame_w / 2.0, cy - frame_h / 2.0) / max(1.0, frame_diag * 0.5)
            centrality = max(0.6, 1.0 - 0.35 * dist_center)
            score = area * centrality
            if score > best_score:
                best_score = score
                anchor_frame = idx
                anchor_cand = c

    if anchor_cand is None or anchor_frame < 0:
        return [None] * n

    locked: list[dict | None] = [None] * n
    locked[anchor_frame] = dict(anchor_cand)
    anchor_sig = anchor_cand.get("signature")

    def _pick_matching_actor(cands: list[dict], prev: dict, gap_frames: int) -> dict | None:
        if not cands:
            return None
        px1, py1, px2, py2 = prev["face_box"]
        pw = max(16.0, float(px2 - px1))
        ph = max(16.0, float(py2 - py1))
        pcx = (px1 + px2) / 2.0
        pcy = (py1 + py2) / 2.0
        prev_sig = prev.get("signature")

        # Max allowed spatial jump grows gently with gap_frames, bounded to reject other actors
        max_jump = max(pw, ph) * (1.35 + 0.15 * min(gap_frames, 15))

        best_c = None
        best_cost = 1e9

        for c in cands:
            fx1, fy1, fx2, fy2 = c["face_box"]
            cw = max(16.0, float(fx2 - fx1))
            ch = max(16.0, float(fy2 - fy1))
            ccx = (fx1 + fx2) / 2.0
            ccy = (fy1 + fy2) / 2.0

            dist = float(np.hypot(ccx - pcx, ccy - pcy))
            size_ratio = max(cw / pw, pw / cw, ch / ph, ph / ch)

            # Reject candidates that jump too far or suddenly change face scale > 2.4x (another person)
            if dist > max_jump or size_ratio > 2.4:
                continue

            # Appearance / identity similarity check
            sim_penalty = 0.0
            csig = c.get("signature")
            if csig is not None and prev_sig is not None and len(csig) == len(prev_sig):
                sim_prev = float(np.dot(csig, prev_sig))
                sim_anch = float(np.dot(csig, anchor_sig)) if (anchor_sig is not None and len(anchor_sig) == len(csig)) else sim_prev
                sim = 0.6 * sim_prev + 0.4 * sim_anch
                # If InsightFace 512-d embedding is used, reject if cosine similarity is clearly another person
                if len(csig) >= 128 and sim < 0.20:
                    continue
                sim_penalty = max(0.0, 1.0 - sim) * max(pw, ph) * 0.8

            cost = dist + (size_ratio - 1.0) * max(pw, ph) * 0.5 + sim_penalty
            if cost < best_cost:
                best_cost = cost
                best_c = c

        return dict(best_c) if best_c is not None else None

    # 2. Track forward from anchor_frame to end
    last_det = locked[anchor_frame]
    last_idx = anchor_frame
    for i in range(anchor_frame + 1, n):
        chosen = _pick_matching_actor(all_frame_candidates[i], last_det, i - last_idx)
        if chosen is not None:
            locked[i] = chosen
            last_det = chosen
            last_idx = i

    # 3. Track backward from anchor_frame to 0
    last_det = locked[anchor_frame]
    last_idx = anchor_frame
    for i in range(anchor_frame - 1, -1, -1):
        chosen = _pick_matching_actor(all_frame_candidates[i], last_det, last_idx - i)
        if chosen is not None:
            locked[i] = chosen
            last_det = chosen
            last_idx = i

    return locked


def smooth_head_trajectory(
    detections: list[dict | None],
    max_gap: int = 90,
    fill_edges: bool = False,
    window: int = 3,
) -> list[dict | None]:
    """Smooth head bounding boxes with a tight zero-phase centered filter (window=3) for zero lag."""
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
                f1 = work[last_valid].get("face_box", b1)
                f2 = work[next_valid].get("face_box", b2)
                interp_box = (
                    int(round(b1[0] * (1 - alpha) + b2[0] * alpha)),
                    int(round(b1[1] * (1 - alpha) + b2[1] * alpha)),
                    int(round(b1[2] * (1 - alpha) + b2[2] * alpha)),
                    int(round(b1[3] * (1 - alpha) + b2[3] * alpha)),
                )
                interp_fbox = (
                    int(round(f1[0] * (1 - alpha) + f2[0] * alpha)),
                    int(round(f1[1] * (1 - alpha) + f2[1] * alpha)),
                    int(round(f1[2] * (1 - alpha) + f2[2] * alpha)),
                    int(round(f1[3] * (1 - alpha) + f2[3] * alpha)),
                )
                interp_lm = (
                    work[last_valid]["landmarks"] * (1 - alpha)
                    + work[next_valid]["landmarks"] * alpha
                )
                work[i] = {
                    "face_box": interp_fbox,
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

    # 2. Zero-phase centered smoothing (small window=3 prevents temporal lag while removing 1-frame jitter)
    smoothed = list(work)
    half = max(1, window // 2)
    for i in range(n):
        if work[i] is None:
            continue
        valid_win = [
            work[j]["head_box"]
            for j in range(max(0, i - half), min(n, i + half + 1))
            if work[j] is not None
        ]
        if valid_win:
            avg_box = tuple(int(round(np.mean([b[k] for b in valid_win]))) for k in range(4))
            smoothed[i] = dict(work[i])
            smoothed[i]["head_box"] = avg_box

    return smoothed
