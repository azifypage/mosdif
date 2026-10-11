"""Head alignment, color harmonization, and compositing for video head replacement."""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path

from mosaicdiff.head_detector import HeadDetector


class HeadAligner:
    def __init__(self, source_image_path: str | Path, detector: HeadDetector):
        path = Path(source_image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Source head image not found: {path}")

        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Could not decode image at {path}")
        self.source_bgr = image
        self.sh, self.sw = image.shape[:2]

        # Detect source head
        detected = detector.detect_head(image)
        if detected is not None:
            self.source_box = detected["head_box"]
            self.source_landmarks = detected["landmarks"]
            self.source_mask = detected["mask"]
        else:
            # Fallback to central crop
            cx, cy = self.sw // 2, self.sh // 2
            hw, hh = int(self.sw * 0.35), int(self.sh * 0.45)
            self.source_box = (max(0, cx - hw), max(0, cy - hh), min(self.sw, cx + hw), min(self.sh, cy + hh))
            self.source_landmarks = np.array([
                [cx - hw * 0.3, cy - hh * 0.2],
                [cx + hw * 0.3, cy - hh * 0.2],
                [cx, cy],
                [cx - hw * 0.2, cy + hh * 0.3],
                [cx + hw * 0.2, cy + hh * 0.3],
            ], dtype=np.float32)
            self.source_mask = np.ones((self.sh, self.sw), dtype=np.float32)

    def composite_head(
        self,
        target_frame: np.ndarray,
        target_info: dict,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Align source head to target head in target_frame and alpha-blend with feathered mask.

        Returns:
            (composited_frame_bgr, final_head_mask)
        """
        th, tw = target_frame.shape[:2]
        target_box = target_info["head_box"]
        dst_landmarks = target_info.get("landmarks")
        target_mask = target_info.get("mask")

        if target_mask is None:
            # Generate mask from target_box
            from mosaicdiff.head_detector import HeadDetector
            target_mask = HeadDetector()._generate_head_mask(target_box, None, tw, th)

        # 1. Compute Affine Transformation Matrix
        warped_source = None
        warped_mask = None

        if dst_landmarks is not None and self.source_landmarks is not None:
            try:
                M, inliers = cv2.estimateAffinePartial2D(self.source_landmarks, dst_landmarks)
                if M is not None:
                    # Check scale validity (reject crazy transforms)
                    scale = np.sqrt(M[0, 0] ** 2 + M[0, 1] ** 2)
                    if 0.15 <= scale <= 6.0:
                        warped_source = cv2.warpAffine(
                            self.source_bgr,
                            M,
                            (tw, th),
                            flags=cv2.INTER_LANCZOS4,
                            borderMode=cv2.BORDER_REFLECT_101,
                        )
                        warped_mask = cv2.warpAffine(
                            self.source_mask,
                            M,
                            (tw, th),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=0.0,
                        )
            except Exception:
                pass

        # 2. Fallback to bounding-box proportional resize & placement
        if warped_source is None:
            sx1, sy1, sx2, sy2 = self.source_box
            source_crop = self.source_bgr[sy1:sy2, sx1:sx2]
            source_m_crop = self.source_mask[sy1:sy2, sx1:sx2]

            tx1, ty1, tx2, ty2 = target_box
            tw_box = max(4, tx2 - tx1)
            th_box = max(4, ty2 - ty1)

            resized_crop = cv2.resize(source_crop, (tw_box, th_box), interpolation=cv2.INTER_LANCZOS4)
            resized_mask = cv2.resize(source_m_crop, (tw_box, th_box), interpolation=cv2.INTER_LINEAR)

            warped_source = target_frame.copy()
            warped_mask = np.zeros((th, tw), dtype=np.float32)

            warped_source[ty1:ty2, tx1:tx2] = resized_crop
            warped_mask[ty1:ty2, tx1:tx2] = resized_mask

        # 3. Tone & Luminance Harmonization
        warped_source = self._match_skin_tone(warped_source, target_frame, target_box)

        # 4. Multiply masks to restrict composite within target head boundary
        blend_mask = np.clip(target_mask * warped_mask, 0.0, 1.0)
        blend_3ch = blend_mask[:, :, np.newaxis]

        # 5. Composite Frame
        composited = (
            target_frame.astype(np.float32) * (1.0 - blend_3ch) +
            warped_source.astype(np.float32) * blend_3ch
        )
        composited = np.clip(composited, 0, 255).astype(np.uint8)

        return composited, blend_mask

    def _match_skin_tone(
        self,
        source_frame: np.ndarray,
        target_frame: np.ndarray,
        target_box: tuple[int, int, int, int],
    ) -> np.ndarray:
        """Transfer average color & brightness from target neck/skin to source head (Reinhard color transfer)."""
        tx1, ty1, tx2, ty2 = target_box
        neck_y1 = max(0, ty2 - int((ty2 - ty1) * 0.35))
        neck_y2 = min(target_frame.shape[0], ty2 + int((ty2 - ty1) * 0.2))
        neck_x1 = max(0, tx1 + int((tx2 - tx1) * 0.2))
        neck_x2 = min(target_frame.shape[1], tx2 - int((tx2 - tx1) * 0.2))

        if neck_y2 <= neck_y1 or neck_x2 <= neck_x1:
            return source_frame

        target_neck = target_frame[neck_y1:neck_y2, neck_x1:neck_x2]
        source_head = source_frame[ty1:ty2, tx1:tx2]

        if target_neck.size == 0 or source_head.size == 0:
            return source_frame

        try:
            # LAB color space transfer
            t_lab = cv2.cvtColor(target_neck, cv2.COLOR_BGR2LAB).astype(np.float32)
            s_lab = cv2.cvtColor(source_frame, cv2.COLOR_BGR2LAB).astype(np.float32)

            t_mean = np.mean(t_lab, axis=(0, 1))
            t_std = np.std(t_lab, axis=(0, 1)) + 1e-5

            s_crop_lab = s_lab[ty1:ty2, tx1:tx2]
            s_mean = np.mean(s_crop_lab, axis=(0, 1))
            s_std = np.std(s_crop_lab, axis=(0, 1)) + 1e-5

            # Blend color stats gently (50% transfer to preserve source identity colors while harmonizing with scene)
            factor = 0.5
            adj_mean = s_mean * (1 - factor) + t_mean * factor
            adj_std = s_std * (1 - factor) + t_std * factor

            adjusted = s_lab.copy()
            for ch in range(3):
                adjusted[:, :, ch] = (adjusted[:, :, ch] - s_mean[ch]) * (adj_std[ch] / s_std[ch]) + adj_mean[ch]

            adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
            return cv2.cvtColor(adjusted, cv2.COLOR_LAB2BGR)
        except Exception:
            return source_frame
