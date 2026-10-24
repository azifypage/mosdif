"""Head, Face & Full-Hair detector, single-actor tracker, and feathered rectangular mask generator.

Ensures the entire head (face, skull dome, ears, back of the head/cranium, and all hair)
is comfortably inside the rectangle with proper breathing room and smooth edge feathering.
"""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path

from mosaicdiff.geometry import edge_fade

# Comfortable pixel margin outside the detected (head + hair) boundary for feathering & context
HEAD_PADDING_PX = 20


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

        # Semantic silhouette segmentation for hair & head contouring
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

    def _estimate_yaw_back_of_head(self, landmarks: np.ndarray | None, face_box: tuple[int, int, int, int]) -> tuple[float, float]:
        """Estimate extra (left_extra, right_extra) ratios to cover the back of the head when turned."""
        if landmarks is None or len(landmarks) < 3:
            return (0.08, 0.08)
        left_eye = landmarks[0]
        right_eye = landmarks[1]
        nose = landmarks[2]
        eye_span = max(8.0, float( abs(right_eye[0] - left_eye[0]) ))
        eye_mid_x = float((left_eye[0] + right_eye[0]) / 2.0)
        # Positive offset -> nose shifted right in image -> back of head is on the LEFT
        # Negative offset -> nose shifted left in image -> back of head is on the RIGHT
        yaw_shift = ( float(nose[0]) - eye_mid_x ) / eye_span
        left_extra = float(np.clip(max(0.05, yaw_shift * 0.65), 0.05, 0.45))
        right_extra = float(np.clip(max(0.05, -yaw_shift * 0.65), 0.05, 0.45))
        return (left_extra, right_extra)

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
                    hbox = self._expand_to_full_head(
                        fx1, fy1, fx2, fy2, w, h, ratios=expansion_ratios, landmarks=landmarks
                    )
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
                    hbox = self._expand_to_full_head(
                        fx1, fy1, fx2, fy2, w, h, ratios=expansion_ratios, landmarks=landmarks
                    )
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
                    hbox = self._expand_to_full_head(
                        fx1, fy1, fx2, fy2, w, h, ratios=expansion_ratios, landmarks=landmarks
                    )
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
                hbox = self._expand_to_full_head(
                    fx1, fy1, fx2, fy2, w, h, ratios=expansion_ratios, landmarks=landmarks
                )
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
        """Detect the primary head in a single frame covering face, cranium, back of head, and hair."""
        h, w = frame_bgr.shape[:2]
        candidates = self.detect_all_faces(frame_bgr, expansion_ratios=expansion_ratios)
        if not candidates:
            return None

        best = max(
            candidates,
            key=lambda c: (c["face_box"][2] - c["face_box"][0]) * (c["face_box"][3] - c["face_box"][1]),
        )
        if expansion_ratios is None:
            ratios = self.estimate_hair_expansion_ratios(
                frame_bgr, best["face_box"], landmarks=best.get("landmarks")
            )
            fx1, fy1, fx2, fy2 = best["face_box"]
            best["head_box"] = self._expand_to_full_head(
                fx1, fy1, fx2, fy2, w, h, ratios=ratios, landmarks=best.get("landmarks")
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
        landmarks: np.ndarray | None = None,
    ) -> tuple[float, float, float, float]:
        """Estimate 4-side ratios (left_r, top_r, right_r, bottom_r) beyond face_box to cover the ENTIRE head & hair.

        1. Starts with a full anatomical cranium base (top=0.48, sides=0.28 + yaw back-of-head offset, bottom=0.18)
           so the skull dome, ears, and back of the head are ALWAYS inside the rectangle even when turned.
        2. Above the chin/neck (y <= fy2 + 0.15*fh), uses the full person foreground silhouette
           (SelfieSegmentation + background contrast) so any hair volume, bun, ponytail, or back-of-head hair
           is 100% captured.
        3. Below the chin (y > fy2 + 0.15*fh), tracks hair color downward if long hair hangs over shoulders/chest.
        """
        h, w = frame_bgr.shape[:2]
        fx1, fy1, fx2, fy2 = face_box
        fw = max(12.0, float(fx2 - fx1))
        fh = max(12.0, float(fy2 - fy1))
        cx = (fx1 + fx2) / 2.0

        # Yaw-aware anatomical base: ensures skull dome, ears, and back of head are always included
        yaw_left, yaw_right = self._estimate_yaw_back_of_head(landmarks, face_box)
        left_r = 0.28 + yaw_left
        top_r = 0.48
        right_r = 0.28 + yaw_right
        bot_r = 0.22

        # Search window reaches 3.8x face height below chin to cover chest and stomach hair
        sx1 = max(0, int(round(cx - fw * 1.85)))
        sx2 = min(w, int(round(cx + fw * 1.85)))
        sy1 = max(0, int(round(fy1 - fh * 0.95)))
        sy2 = min(h, int(round(fy2 + fh * 3.80)))

        if sx2 - sx1 < 16 or sy2 - sy1 < 16:
            return (left_r, top_r, right_r, bot_r)

        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

        # Sample background from top-left and top-right corners of search window
        bw_strip = max(4, (sx2 - sx1) // 10)
        bh_strip = max(4, (sy2 - sy1) // 14)
        bg_left = lab[sy1 : sy1 + bh_strip, sx1 : sx1 + bw_strip].reshape(-1, 3)
        bg_right = lab[sy1 : sy1 + bh_strip, sx2 - bw_strip : sx2].reshape(-1, 3)
        bg_all = np.concatenate([b for b in (bg_left, bg_right) if b.size > 0], axis=0)
        bg_median = np.median(bg_all, axis=0) if len(bg_all) > 0 else np.array([0.0, 128.0, 128.0])
        bg_std = np.clip(np.std(bg_all, axis=0), 8.0, 28.0) if len(bg_all) > 0 else np.array([15.0, 15.0, 15.0])

        roi_lab = lab[sy1:sy2, sx1:sx2]
        dist_bg = np.linalg.norm((roi_lab - bg_median) / (bg_std * 1.3), axis=2)

        seg = self.get_person_segmentation(frame_bgr)
        seg_roi = seg[sy1:sy2, sx1:sx2] if seg is not None else None

        # PART A: Upper head & back-of-head zone (from top of search box down to chin level fy2 + 0.12*fh)
        chin_y_rel = min(sy2 - sy1, max(1, int(round(fy2 + fh * 0.12)) - sy1))
        if seg_roi is not None:
            upper_fg = (seg_roi[:chin_y_rel, :] > 0.35) & (dist_bg[:chin_y_rel, :] > 1.20)
        else:
            upper_fg = dist_bg[:chin_y_rel, :] > 1.80

        upper_u8 = upper_fg.astype(np.uint8) * 255
        k_upper = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        upper_u8 = cv2.morphologyEx(upper_u8, cv2.MORPH_CLOSE, k_upper)
        upper_u8 = cv2.morphologyEx(upper_u8, cv2.MORPH_OPEN, k_upper)

        # Keep connected component overlapping the face/cranium
        n_lbl, lbls, stats, _ = cv2.connectedComponentsWithStats(upper_u8, connectivity=8)
        f_rx1 = max(0, int(fx1 - fw * 0.38) - sx1)
        f_rx2 = min(sx2 - sx1, int(fx2 + fw * 0.38) - sx1)
        f_ry1 = max(0, int(fy1 - fh * 0.48) - sy1)
        f_ry2 = min(chin_y_rel, int(fy2) - sy1)

        head_upper = np.zeros_like(upper_u8)
        if f_rx2 > f_rx1 and f_ry2 > f_ry1:
            seed_ids = set(np.unique(lbls[f_ry1:f_ry2, f_rx1:f_rx2])) - {0}
            for lid in seed_ids:
                if stats[lid, cv2.CC_STAT_AREA] >= 36:
                    head_upper[lbls == lid] = 255

        u_ys, u_xs = np.where(head_upper > 0)
        if len(u_ys) > 30:
            u_min_y = float(np.percentile(u_ys, 2)) + sy1
            u_min_x = float(np.percentile(u_xs, 2)) + sx1
            u_max_x = float(np.percentile(u_xs, 98)) + sx1

            left_r = float(np.clip(max(left_r, (fx1 - u_min_x) / fw + 0.08), 0.28, 1.35))
            top_r = float(np.clip(max(top_r, (fy1 - u_min_y) / fh + 0.10), 0.48, 1.20))
            right_r = float(np.clip(max(right_r, (u_max_x - fx2) / fw + 0.08), 0.28, 1.35))

        # PART B: Long hair extending down over neck, shoulders, chest, and stomach (up to fy2 + 3.8*fh)
        skin_x1 = max(0, int(fx1 + fw * 0.22))
        skin_x2 = min(w, int(fx2 - fw * 0.22))
        skin_y1 = max(0, int(fy1 + fh * 0.20))
        skin_y2 = min(h, int(fy2 - fh * 0.20))
        skin_patch = lab[skin_y1:skin_y2, skin_x1:skin_x2]
        skin_mean = np.mean(skin_patch, axis=(0, 1)) if skin_patch.size > 0 else np.array([150.0, 135.0, 140.0])

        # Sample hair reference pixels from BOTH:
        # 1) Upper head outside inner face box (crown/temples)
        # 2) Side-lock zones beside cheeks/jaw/neck (where long hair drapes down onto the chest)
        sample_y2 = min(sy2 - sy1, max(chin_y_rel, int(round(fy2 + fh * 0.65)) - sy1))
        hair_sample_mask = np.zeros((sample_y2, sx2 - sx1), dtype=bool)
        hair_sample_mask[:chin_y_rel, :] = head_upper > 0

        side_lx1 = max(0, int(round(fx1 - fw * 0.75)) - sx1)
        side_lx2 = min(sx2 - sx1, max(0, int(round(fx1 + fw * 0.05)) - sx1))
        side_rx1 = max(0, int(round(fx2 - fw * 0.05)) - sx1)
        side_rx2 = min(sx2 - sx1, int(round(fx2 + fw * 0.75)) - sx1)
        side_y1 = max(0, int(round(fy1)) - sy1)
        if sample_y2 > side_y1:
            if side_lx2 > side_lx1:
                hair_sample_mask[side_y1:sample_y2, side_lx1:side_lx2] = True
            if side_rx2 > side_rx1:
                hair_sample_mask[side_y1:sample_y2, side_rx1:side_rx2] = True

        # Exclude inner face box from hair reference sampling
        inner_fx1 = max(0, int(fx1) - sx1)
        inner_fx2 = min(sx2 - sx1, int(fx2) - sx1)
        inner_fy1 = max(0, int(fy1) - sy1)
        inner_fy2 = min(sample_y2, int(fy2) - sy1)
        if inner_fx2 > inner_fx1 and inner_fy2 > inner_fy1:
            hair_sample_mask[inner_fy1:inner_fy2, inner_fx1:inner_fx2] = False

        if seg_roi is not None:
            hair_sample_mask = hair_sample_mask & (seg_roi[:sample_y2, :] > 0.40) & (dist_bg[:sample_y2, :] > 1.25)
        else:
            hair_sample_mask = hair_sample_mask & (dist_bg[:sample_y2, :] > 1.75)

        hair_pixels = roi_lab[:sample_y2, :][hair_sample_mask]
        if len(hair_pixels) >= 24:
            dist_hp_skin = np.linalg.norm(hair_pixels - skin_mean, axis=1)
            valid_hp = hair_pixels[dist_hp_skin > 13.0]
            if len(valid_hp) >= 16:
                hair_median = np.median(valid_hp, axis=0)
                hair_p20 = np.percentile(valid_hp, 20, axis=0)
                hair_p80 = np.percentile(valid_hp, 80, axis=0)
                hair_std = np.clip(np.std(valid_hp, axis=0), 10.0, 28.0)

                # Chroma-weighted scale: downweight L (channel 0) so shadowed/highlighted hair on chest & stomach matches crown hair
                chroma_scale = hair_std * np.array([2.25, 1.55, 1.55], dtype=np.float32)
                dist_med = np.linalg.norm((roi_lab - hair_median) / chroma_scale, axis=2)
                dist_p20 = np.linalg.norm((roi_lab - hair_p20) / chroma_scale, axis=2)
                dist_p80 = np.linalg.norm((roi_lab - hair_p80) / chroma_scale, axis=2)
                dist_hair = np.minimum(dist_med, np.minimum(dist_p20, dist_p80))

                # Also sample specifically from jaw/neck side-locks (closest in lighting to chest & stomach hair)
                jaw_y1 = min(sample_y2, max(0, int(round(fy2 - fh * 0.25)) - sy1))
                if sample_y2 > jaw_y1:
                    jaw_mask = hair_sample_mask[jaw_y1:sample_y2, :]
                    jaw_hp = roi_lab[jaw_y1:sample_y2, :][jaw_mask]
                    if len(jaw_hp) >= 16:
                        jaw_valid = jaw_hp[np.linalg.norm(jaw_hp - skin_mean, axis=1) > 13.0]
                        if len(jaw_valid) >= 12:
                            jaw_med = np.median(jaw_valid, axis=0)
                            dist_jaw = np.linalg.norm((roi_lab - jaw_med) / chroma_scale, axis=2)
                            dist_hair = np.minimum(dist_hair, dist_jaw)

                dist_skin = np.linalg.norm(roi_lab - skin_mean, axis=2)

                full_hair = (dist_hair < 1.85) & (dist_skin > 13.0) & (dist_bg > 1.20)
                if seg_roi is not None:
                    full_hair = full_hair & (seg_roi > 0.35)

                # Exclude only the inner face & upper throat skin (NEVER erase chest or stomach!)
                face_throat_y2 = min(sy2 - sy1, max(0, int(round(fy2 + fh * 0.14)) - sy1))
                ft_x1 = max(0, int(round(fx1 + fw * 0.14)) - sx1)
                ft_x2 = min(sx2 - sx1, int(round(fx2 - fw * 0.14)) - sx1)
                if ft_x2 > ft_x1 and face_throat_y2 > inner_fy1:
                    full_hair[inner_fy1:face_throat_y2, ft_x1:ft_x2] = False

                # Restrict lower chest/stomach region (below shoulders) to the natural chest/stomach width corridor
                chest_y1 = min(sy2 - sy1, max(0, int(round(fy2 + fh * 0.45)) - sy1))
                corridor_x1 = max(0, int(round(cx - fw * 1.40)) - sx1)
                corridor_x2 = min(sx2 - sx1, int(round(cx + fw * 1.40)) - sx1)
                if chest_y1 < (sy2 - sy1):
                    if corridor_x1 > 0:
                        full_hair[chest_y1:, :corridor_x1] = False
                    if corridor_x2 < (sx2 - sx1):
                        full_hair[chest_y1:, corridor_x2:] = False

                # Check if BOTH outer arm flanks simultaneously match hair color (e.g. solid shirt of identical color)
                roi_l = roi_lab[:, :, 0]
                sobel_x = np.abs(cv2.Sobel(roi_l, cv2.CV_32F, 1, 0, ksize=3))
                strand_tex = cv2.GaussianBlur(sobel_x, (9, 9), 0)

                sh_y1 = min(sy2 - sy1, max(0, int(round(fy2 + fh * 0.90)) - sy1))
                sh_y2 = min(sy2 - sy1, max(sh_y1 + 1, int(round(fy2 + fh * 2.20)) - sy1))
                sl_x1 = max(0, int(round(cx - fw * 1.80)) - sx1)
                sl_x2 = max(sl_x1, int(round(cx - fw * 1.30)) - sx1)
                sr_x1 = min(sx2 - sx1, int(round(cx + fw * 1.30)) - sx1)
                sr_x2 = min(sx2 - sx1, int(round(cx + fw * 1.80)) - sx1)
                left_arm_match = False
                right_arm_match = False
                if sh_y2 > sh_y1:
                    if sl_x2 > sl_x1:
                        l_patch = dist_hair[sh_y1:sh_y2, sl_x1:sl_x2]
                        left_arm_match = bool(l_patch.size > 20 and np.mean(l_patch < 1.55) > 0.68)
                    if sr_x2 > sr_x1:
                        r_patch = dist_hair[sh_y1:sh_y2, sr_x1:sr_x2]
                        right_arm_match = bool(r_patch.size > 20 and np.mean(r_patch < 1.55) > 0.68)

                if left_arm_match and right_arm_match and sh_y1 < (sy2 - sy1):
                    # Only when BOTH outer arms are solidly filled with hair color (identical-color wide shirt),
                    # require either closer hair color match or vertical hair strand texture on the lower torso
                    chest_mask = full_hair[sh_y1:, :] & ((dist_hair[sh_y1:, :] < 1.35) | (strand_tex[sh_y1:, :] > 4.2))
                    full_hair[sh_y1:, :] = chest_mask

                # Bridge gaps across ears/jaw/collar/chest with a large vertical closing kernel so chest & stomach hair connects
                fh_u8 = full_hair.astype(np.uint8) * 255
                k_vert = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 35))
                fh_u8 = cv2.morphologyEx(fh_u8, cv2.MORPH_CLOSE, k_vert)
                num_l, l_map, l_stats, _ = cv2.connectedComponentsWithStats(fh_u8, connectivity=8)

                # Seed hair components from head, behind ears, neck, shoulders, and upper chest (up to fy2 + 1.45*fh)
                seed_y2 = min(sy2 - sy1, max(chin_y_rel, int(round(fy2 + fh * 1.45)) - sy1))
                seed_x1 = max(0, int(round(cx - fw * 1.30)) - sx1)
                seed_x2 = min(sx2 - sx1, int(round(cx + fw * 1.30)) - sx1)
                conn_hair = np.zeros_like(fh_u8)
                seed_hair_ids = set(np.unique(l_map[:seed_y2, seed_x1:seed_x2])) - {0}
                min_comp_area = max(40, int(fw * fh * 0.02))
                for lid in seed_hair_ids:
                    if l_stats[lid, cv2.CC_STAT_AREA] >= min_comp_area:
                        conn_hair[l_map == lid] = 255

                # If hair already reaches the neck/shoulders (y > fy2 + 0.35*fh), also include any lower hair locks
                # on the chest & stomach (up to fy2 + 3.60*fh) that were separated by a collar or necklace
                h_ys_init, _ = np.where(conn_hair > 0)
                if len(h_ys_init) > 30 and (float(np.percentile(h_ys_init, 95)) + sy1) > (fy2 + fh * 0.35):
                    for lid in range(1, num_l):
                        if lid in seed_hair_ids:
                            continue
                        area = l_stats[lid, cv2.CC_STAT_AREA]
                        comp_x = l_stats[lid, cv2.CC_STAT_LEFT] + l_stats[lid, cv2.CC_STAT_WIDTH] * 0.5
                        comp_y = l_stats[lid, cv2.CC_STAT_TOP]
                        if area >= max(60, int(fw * fh * 0.035)) and seed_x1 <= comp_x <= seed_x2 and comp_y >= chin_y_rel:
                            conn_hair[l_map == lid] = 255

                h_ys, h_xs = np.where(conn_hair > 0)
                if len(h_ys) > 30:
                    max_y = float(np.percentile(h_ys, 98.5)) + sy1
                    min_x = float(np.percentile(h_xs, 2)) + sx1
                    max_x = float(np.percentile(h_xs, 98)) + sx1
                    bot_r = float(np.clip(max(bot_r, (max_y - fy2) / fh + 0.22), 0.22, 3.65))
                    left_r = float(np.clip(max(left_r, (fx1 - min_x) / fw + 0.10), 0.28, 1.40))
                    right_r = float(np.clip(max(right_r, (max_x - fx2) / fw + 0.10), 0.28, 1.40))

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
        landmarks: np.ndarray | None = None,
        padding_px: int = HEAD_PADDING_PX,
    ) -> tuple[int, int, int, int]:
        """Expand face box to cover full skull, back of head, and hair with comfortable feathering margin."""
        fw = fx2 - fx1
        fh = fy2 - fy1
        yaw_left, yaw_right = self._estimate_yaw_back_of_head(landmarks, (int(fx1), int(fy1), int(fx2), int(fy2)))

        if ratios is None:
            left_r, top_r, right_r, bot_r = (0.28 + yaw_left, 0.48, 0.28 + yaw_right, 0.18)
        elif len(ratios) == 4:
            left_r = max(ratios[0], 0.28 + yaw_left)
            top_r = max(ratios[1], 0.48)
            right_r = max(ratios[2], 0.28 + yaw_right)
            bot_r = max(ratios[3], 0.18)
        else:
            top_r = max(ratios[0], 0.48)
            bot_r = max(ratios[1], 0.18)
            side_c = max(ratios[2] - 0.5, 0.28)
            left_r = side_c + yaw_left
            right_r = side_c + yaw_right

        hx1 = max(0, int(round(fx1 - fw * left_r)) - padding_px)
        hy1 = max(0, int(round(fy1 - fh * top_r)) - padding_px)
        hx2 = min(frame_w, int(round(fx2 + fw * right_r)) + padding_px)
        hy2 = min(frame_h, int(round(fy2 + fh * bot_r)) + padding_px)

        return (hx1, hy1, hx2, hy2)

    def _generate_head_mask(
        self,
        head_box: tuple[int, int, int, int],
        landmarks_pts: np.ndarray | None,
        frame_w: int,
        frame_h: int,
        seg_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Create a smooth feathered 2D float32 rectangular mask (0.0 to 1.0) covering head_box."""
        hx1, hy1, hx2, hy2 = head_box
        hx1, hy1 = max(0, hx1), max(0, hy1)
        hx2, hy2 = min(frame_w, hx2), min(frame_h, hy2)
        hw = max(2, hx2 - hx1)
        hh = max(2, hy2 - hy1)

        mask = np.zeros((frame_h, frame_w), dtype=np.float32)
        short = min(hw, hh)
        falloff = max(18, min(round(short * 0.10), short // 5))
        rect_fade = edge_fade(hh, hw, falloff)[:, :, 0]
        mask[hy1:hy2, hx1:hx2] = rect_fade
        return np.clip(mask, 0.0, 1.0)


def create_crop_head_mask(
    crop_box: tuple[int, int, int, int],
    head_box: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """Create a rectangular feathered 3D float32 mask (H_crop, W_crop, 1) with smooth edge feathering."""
    cx1, cy1, cx2, cy2 = crop_box
    cw = max(2, cx2 - cx1)
    ch = max(2, cy2 - cy1)
    short = min(cw, ch)
    falloff = max(18, min(round(short * 0.10), short // 5))
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
