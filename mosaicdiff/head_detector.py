"""Head, Face & Hair detector, single-actor tracker, and tight rectangular mask generator.

Computes a tight bounding rectangle strictly around (face + actual attached hair) with minimal
pixel tolerance (~6 px), locks onto a single primary actor across the video, and generates
rectangular masks with a thin feathered edge (4-8 px).
"""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path

from mosaicdiff.geometry import edge_fade

# Maximum pixel tolerance outside the detected (face + hair) boundary
TIGHT_TOLERANCE_PX = 6


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
        expansion_ratios: tuple[float, float, float, float] | None = None,
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
        expansion_ratios: tuple[float, float, float, float] | None = None,
    ) -> dict | None:
        """Detect the primary head in a single frame with tight (face + hair) bounds."""
        h, w = frame_bgr.shape[:2]
        candidates = self.detect_all_faces(frame_bgr, expansion_ratios=expansion_ratios)
        if not candidates:
            return None

        best = max(
            candidates,
            key=lambda c: (c["face_box"][2] - c["face_box"][0]) * (c["face_box"][3] - c["face_box"][1]),
        )
        if expansion_ratios is None:
            ratios = self.estimate_hair_expansion_ratios(frame_bgr, best["face_box"])
            fx1, fy1, fx2, fy2 = best["face_box"]
            best["head_box"] = self._expand_to_full_head(fx1, fy1, fx2, fy2, w, h, ratios=ratios)
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
    ) -> tuple[float, float, float, float]:
        """Estimate tight 4-side ratios (left_r, top_r, right_r, bottom_r) beyond face_box.

        Starts at (0.0, 0.08, 0.0, 0.0) — i.e., exact face_box edges when no hair extends
        past the face on that side! Only expands a side when true hair pixels connected to
        the scalp are detected on that specific side, stopping before torso clothing.
        """
        h, w = frame_bgr.shape[:2]
        fx1, fy1, fx2, fy2 = face_box
        fw = max(12.0, float(fx2 - fx1))
        fh = max(12.0, float(fy2 - fy1))
        cx = (fx1 + fx2) / 2.0

        # Tight defaults: 0.0 on left, right, and bottom (exact face edge), 0.08 on top
        left_r = 0.0
        top_r = 0.08
        right_r = 0.0
        bot_r = 0.0

        sx1 = max(0, int(round(cx - fw * 1.45)))
        sx2 = min(w, int(round(cx + fw * 1.45)))
        sy1 = max(0, int(round(fy1 - fh * 0.75)))
        sy2 = min(h, int(round(fy2 + fh * 2.50)))

        if sx2 - sx1 < 16 or sy2 - sy1 < 16:
            return (left_r, top_r, right_r, bot_r)

        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

        # 1. Sample central face skin color
        skin_x1 = max(0, int(fx1 + fw * 0.22))
        skin_x2 = min(w, int(fx2 - fw * 0.22))
        skin_y1 = max(0, int(fy1 + fh * 0.20))
        skin_y2 = min(h, int(fy2 - fh * 0.20))
        skin_patch = lab[skin_y1:skin_y2, skin_x1:skin_x2]
        skin_mean = np.mean(skin_patch, axis=(0, 1)) if skin_patch.size > 0 else np.array([150.0, 135.0, 140.0])

        # 2. Sample scalp/crown hair strip right above forehead
        crown_y1 = max(0, int(fy1 - fh * 0.28))
        crown_y2 = max(crown_y1 + 2, int(fy1 - fh * 0.02))
        crown_x1 = max(0, int(fx1 + fw * 0.12))
        crown_x2 = min(w, int(fx2 - fw * 0.12))
        crown_patch = lab[crown_y1:crown_y2, crown_x1:crown_x2]
        if crown_patch.size == 0:
            return (left_r, top_r, right_r, bot_r)

        # 3. Sample outer background from top corners of search ROI
        bw_strip = max(4, (sx2 - sx1) // 10)
        bh_strip = max(4, (sy2 - sy1) // 12)
        bg_left = lab[sy1 : sy1 + bh_strip, sx1 : sx1 + bw_strip].reshape(-1, 3)
        bg_right = lab[sy1 : sy1 + bh_strip, sx2 - bw_strip : sx2].reshape(-1, 3)
        bg_all = np.concatenate([b for b in (bg_left, bg_right) if b.size > 0], axis=0)
        bg_median = np.median(bg_all, axis=0) if len(bg_all) > 0 else np.array([0.0, 128.0, 128.0])

        crown_flat = crown_patch.reshape(-1, 3)
        dist_crown_skin = np.linalg.norm(crown_flat - skin_mean, axis=1)
        dist_crown_bg = np.linalg.norm(crown_flat - bg_median, axis=1)
        valid_hair_samples = crown_flat[(dist_crown_skin > 18.0) & (dist_crown_bg > 22.0)]

        # If no distinct hair pixels exist above the forehead, return tight face box ratios
        if len(valid_hair_samples) < max(12, int(crown_flat.shape[0] * 0.15)):
            return (left_r, top_r, right_r, bot_r)

        hair_median = np.median(valid_hair_samples, axis=0)
        hair_std = np.clip(np.std(valid_hair_samples, axis=0), 8.0, 24.0)

        roi_lab = lab[sy1:sy2, sx1:sx2]
        dist_hair = np.linalg.norm((roi_lab - hair_median) / (hair_std * 1.35), axis=2)
        dist_bg = np.linalg.norm(roi_lab - bg_median, axis=2)
        dist_skin = np.linalg.norm(roi_lab - skin_mean, axis=2)

        hair_mask = (dist_hair < 1.45) & (dist_skin > 18.0) & (dist_bg > 24.0)

        # Intersect with person silhouette if available
        seg = self.get_person_segmentation(frame_bgr)
        if seg is not None:
            seg_roi = seg[sy1:sy2, sx1:sx2]
            hair_mask = hair_mask & (seg_roi > 0.55)

        # Exclude central neck/chest zone directly under the chin so dark shirts are not merged
        neck_x1 = max(0, int(fx1 + fw * 0.18) - sx1)
        neck_x2 = min(sx2 - sx1, int(fx2 - fw * 0.18) - sx1)
        neck_y1 = max(0, int(fy2 + fh * 0.05) - sy1)
        if neck_x2 > neck_x1 and neck_y1 < (sy2 - sy1):
            hair_mask[neck_y1:, neck_x1:neck_x2] = False

        hair_u8 = hair_mask.astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        hair_u8 = cv2.morphologyEx(hair_u8, cv2.MORPH_OPEN, kernel)
        hair_u8 = cv2.morphologyEx(hair_u8, cv2.MORPH_CLOSE, kernel)

        # Keep only components connected to the scalp seed right around the upper head
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(hair_u8, connectivity=8)
        seed_x1 = max(0, int(fx1 - fw * 0.20) - sx1)
        seed_x2 = min(sx2 - sx1, int(fx2 + fw * 0.20) - sx1)
        seed_y1 = max(0, int(fy1 - fh * 0.35) - sy1)
        seed_y2 = min(sy2 - sy1, int(fy1 + fh * 0.35) - sy1)

        connected = np.zeros_like(hair_u8)
        if seed_x2 > seed_x1 and seed_y2 > seed_y1:
            seed_labels = set(np.unique(labels[seed_y1:seed_y2, seed_x1:seed_x2])) - {0}
            for lbl in seed_labels:
                if stats[lbl, cv2.CC_STAT_AREA] >= 30:
                    connected[labels == lbl] = 255

        # Prevent downward bleeding into dark torso clothing: check row width below chin
        chin_rel_y = max(0, int(fy2 - sy1))
        for r_y in range(chin_rel_y, connected.shape[0]):
            row_xs = np.where(connected[r_y, :] > 0)[0]
            if len(row_xs) == 0:
                connected[r_y:, :] = 0
                break
            row_span = float(row_xs[-1] - row_xs[0])
            # If width suddenly jumps to torso/shoulder width (> 1.45 * fw), cut off at collar
            if row_span > fw * 1.45:
                connected[r_y:, :] = 0
                break

        ys, xs = np.where(connected > 0)
        if len(ys) > 24:
            # 5th and 95th percentiles for tight, outlier-free hair boundary (0% extra padding)
            min_y = float(np.percentile(ys, 5)) + sy1
            max_y = float(np.percentile(ys, 95)) + sy1
            min_x = float(np.percentile(xs, 5)) + sx1
            max_x = float(np.percentile(xs, 95)) + sx1

            left_r = float(np.clip(max(0.0, (fx1 - min_x) / fw), 0.0, 0.90))
            top_r = float(np.clip(max(0.08, (fy1 - min_y) / fh), 0.08, 0.85))
            right_r = float(np.clip(max(0.0, (max_x - fx2) / fw), 0.0, 0.90))
            bot_r = float(np.clip(max(0.0, (max_y - fy2) / fh), 0.0, 2.20))

        return (left_r, top_r, right_r, bot_r)

    def _expand_to_full_head(
        self,
        fx1: float,
        fy1: float,
        fx2: float,
        fy2: float,
        frame_w: int,
        frame_h: int,
        ratios: tuple[float, ...] | None = None,
        tolerance_px: int = TIGHT_TOLERANCE_PX,
    ) -> tuple[int, int, int, int]:
        """Expand face box strictly by the 4-side hair ratios plus a tiny pixel tolerance (~6 px)."""
        fw = fx2 - fx1
        fh = fy2 - fy1

        if ratios is None:
            left_r, top_r, right_r, bot_r = (0.0, 0.08, 0.0, 0.0)
        elif len(ratios) == 4:
            left_r, top_r, right_r, bot_r = ratios
        else:
            # Backwards-compatible fallback if 3-tuple (top, bot, side_from_center) was passed
            top_r, bot_r, side_c = ratios[0], ratios[1], ratios[2]
            left_r = max(0.0, side_c - 0.5)
            right_r = max(0.0, side_c - 0.5)

        hx1 = max(0, int(round(fx1 - fw * left_r)) - tolerance_px)
        hy1 = max(0, int(round(fy1 - fh * top_r)) - tolerance_px)
        hx2 = min(frame_w, int(round(fx2 + fw * right_r)) + tolerance_px)
        hy2 = min(frame_h, int(round(fy2 + fh * bot_r)) + tolerance_px)

        return (hx1, hy1, hx2, hy2)

    def _generate_head_mask(
        self,
        head_box: tuple[int, int, int, int],
        landmarks_pts: np.ndarray | None,
        frame_w: int,
        frame_h: int,
        seg_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Create a tight feathered 2D float32 rectangular mask (0.0 to 1.0) covering head_box."""
        hx1, hy1, hx2, hy2 = head_box
        hx1, hy1 = max(0, hx1), max(0, hy1)
        hx2, hy2 = min(frame_w, hx2), min(frame_h, hy2)
        hw = max(2, hx2 - hx1)
        hh = max(2, hy2 - hy1)

        mask = np.zeros((frame_h, frame_w), dtype=np.float32)
        short = min(hw, hh)
        falloff = max(4, min(8, short // 12))
        rect_fade = edge_fade(hh, hw, falloff)[:, :, 0]
        mask[hy1:hy2, hx1:hx2] = rect_fade
        return np.clip(mask, 0.0, 1.0)


def create_crop_head_mask(
    crop_box: tuple[int, int, int, int],
    head_box: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """Create a tight rectangular feathered 3D float32 mask (H_crop, W_crop, 1) with 4-8 px border fade."""
    cx1, cy1, cx2, cy2 = crop_box
    cw = max(2, cx2 - cx1)
    ch = max(2, cy2 - cy1)
    short = min(cw, ch)
    falloff = max(4, min(8, short // 12))
    return edge_fade(ch, cw, falloff)


def lock_single_actor_trajectory(
    all_frame_candidates: list[list[dict]],
    frame_w: int,
    frame_h: int,
) -> list[dict | None]:
    """Select the single primary target actor across all frames and reject all other actors."""
    n = len(all_frame_candidates)
    if n == 0:
        return []

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

            if dist > max_jump or size_ratio > 2.4:
                continue

            sim_penalty = 0.0
            csig = c.get("signature")
            if csig is not None and prev_sig is not None and len(csig) == len(prev_sig):
                sim_prev = float(np.dot(csig, prev_sig))
                sim_anch = float(np.dot(csig, anchor_sig)) if (anchor_sig is not None and len(anchor_sig) == len(csig)) else sim_prev
                sim = 0.6 * sim_prev + 0.4 * sim_anch
                if len(csig) >= 128 and sim < 0.20:
                    continue
                sim_penalty = max(0.0, 1.0 - sim) * max(pw, ph) * 0.8

            cost = dist + (size_ratio - 1.0) * max(pw, ph) * 0.5 + sim_penalty
            if cost < best_cost:
                best_cost = cost
                best_c = c

        return dict(best_c) if best_c is not None else None

    last_det = locked[anchor_frame]
    last_idx = anchor_frame
    for i in range(anchor_frame + 1, n):
        chosen = _pick_matching_actor(all_frame_candidates[i], last_det, i - last_idx)
        if chosen is not None:
            locked[i] = chosen
            last_det = chosen
            last_idx = i

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

    if fill_edges and any(d is not None for d in work):
        first_valid = next(i for i, d in enumerate(work) if d is not None)
        last_valid_idx = max(i for i, d in enumerate(work) if d is not None)
        for i in range(0, first_valid):
            work[i] = dict(work[first_valid])
        for i in range(last_valid_idx + 1, n):
            work[i] = dict(work[last_valid_idx])

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
