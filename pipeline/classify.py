"""
Classification module.
  - Room type: HuggingFace CLIP zero-shot (5 classes)
  - Camera move: optical flow (static / pan / walk)
  - Quality: Laplacian variance + brightness (sharp / blurry / dark)
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from PIL import Image

import torch
from transformers import CLIPProcessor, CLIPModel

from pipeline.scene_split import SubShot

# ── Constants ──────────────────────────────────────────────────────────────

ROOM_LABELS = ["bedroom", "kitchen", "bathroom", "living room", "exterior"]

ROOM_PROMPTS = [
    "a photo of a bedroom with a bed",
    "a photo of a kitchen with counters and appliances",
    "a photo of a bathroom with a sink or toilet or shower",
    "a photo of a living room with sofas and furniture",
    "a photo of the exterior of a house or building",
]

CAMERA_STATIC = "static"
CAMERA_PAN = "pan"
CAMERA_WALK = "walk"

QUALITY_SHARP = "sharp"
QUALITY_BLURRY = "blurry"
QUALITY_DARK = "dark"

BLUR_THRESHOLD = 80.0      # Laplacian variance below this → blurry
DARK_THRESHOLD = 45.0      # mean brightness below this → dark
MOTION_PAN_MIN = 1.2       # mean optical flow magnitude for pan
MOTION_WALK_MIN = 3.5      # mean optical flow magnitude for walk/dolly


class Classifier:
    """
    Wraps CLIP for room classification and OpenCV for quality/motion.
    Load once; call classify_subshot() per sub-shot.
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

    # ── Public API ──────────────────────────────────────────────────────────

    def classify_subshot(self, shot: SubShot) -> SubShot:
        """
        Fill shot.room_type, room_confidence, camera_move, quality,
        blur_score, brightness_score, motion_score in-place.
        """
        if not shot.frames:
            shot.room_type = "living room"
            shot.room_confidence = 0.0
            shot.camera_move = CAMERA_STATIC
            shot.quality = QUALITY_BLURRY
            return shot

        # ── Quality: use middle frame ───────────────────────────────────────
        mid = shot.frames[len(shot.frames) // 2]
        shot.blur_score, shot.brightness_score = self._quality_scores(mid)

        if shot.brightness_score < DARK_THRESHOLD:
            shot.quality = QUALITY_DARK
        elif shot.blur_score < BLUR_THRESHOLD:
            shot.quality = QUALITY_BLURRY
        else:
            shot.quality = QUALITY_SHARP

        # ── Camera move: optical flow on consecutive sampled frames ─────────
        shot.motion_score = self._mean_optical_flow(shot.frames)

        if shot.motion_score < MOTION_PAN_MIN:
            shot.camera_move = CAMERA_STATIC
        elif shot.motion_score < MOTION_WALK_MIN:
            shot.camera_move = CAMERA_PAN
        else:
            shot.camera_move = CAMERA_WALK

        # ── Room type: CLIP on 3 best-quality frames ────────────────────────
        best_frames = self._pick_best_frames(shot.frames, n=3)
        room_idx, confidence = self._clip_room(best_frames)
        shot.room_type = ROOM_LABELS[room_idx]
        shot.room_confidence = confidence

        return shot

    def classify_batch(
        self,
        shots: List[SubShot],
        progress_callback=None,
    ) -> List[SubShot]:
        """Classify a list of sub-shots, calling progress_callback(i, total) each step."""
        for i, shot in enumerate(shots):
            self.classify_subshot(shot)
            if progress_callback:
                progress_callback(i + 1, len(shots))
        return shots

    # ── CLIP room classification ─────────────────────────────────────────────

    def _clip_room(self, frames: List[np.ndarray]) -> Tuple[int, float]:
        """Return (room_label_index, max_confidence) averaged over frames."""
        if not frames:
            return 0, 0.0

        pil_images = [
            Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            for f in frames
        ]

        with torch.no_grad():
            inputs = self.processor(
                text=ROOM_PROMPTS,
                images=pil_images,
                return_tensors="pt",
                padding=True,
            ).to(self.device)

            outputs = self.model(**inputs)
            # logits_per_image: shape (n_images, n_labels)
            probs = outputs.logits_per_image.softmax(dim=-1)  # (n_images, 5)
            avg_probs = probs.mean(dim=0)                      # (5,)

        best_idx = int(avg_probs.argmax().item())
        confidence = float(avg_probs[best_idx].item())
        return best_idx, confidence

    # ── Quality helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _quality_scores(frame: np.ndarray) -> Tuple[float, float]:
        """Return (laplacian_variance, mean_brightness)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(np.mean(gray))
        return lap_var, brightness

    # ── Optical flow ─────────────────────────────────────────────────────────

    @staticmethod
    def _mean_optical_flow(frames: List[np.ndarray]) -> float:
        """
        Compute mean magnitude of dense optical flow between consecutive frames.
        Returns 0.0 for single-frame shots.
        """
        if len(frames) < 2:
            return 0.0

        magnitudes = []
        # Use up to 5 pairs evenly spaced to stay fast
        indices = _evenly_spaced_pairs(len(frames), max_pairs=5)

        for i, j in indices:
            gray1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frames[j], cv2.COLOR_BGR2GRAY)

            # Resize to 128×128 for speed
            gray1 = cv2.resize(gray1, (128, 128))
            gray2 = cv2.resize(gray2, (128, 128))

            flow = cv2.calcOpticalFlowFarneback(
                gray1, gray2, None,
                pyr_scale=0.5, levels=3, winsize=13,
                iterations=3, poly_n=5, poly_sigma=1.1,
                flags=0,
            )
            mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            magnitudes.append(float(np.mean(mag)))

        return float(np.mean(magnitudes)) if magnitudes else 0.0

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _pick_best_frames(
        frames: List[np.ndarray], n: int = 3
    ) -> List[np.ndarray]:
        """
        Pick up to n frames with the highest sharpness (Laplacian variance).
        Falls back to evenly-spaced if fewer than n frames available.
        """
        if len(frames) <= n:
            return frames

        scored = []
        for f in frames:
            gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            scored.append((lap_var, f))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [f for _, f in scored[:n]]


# ── Standalone helper ────────────────────────────────────────────────────────

def _evenly_spaced_pairs(n: int, max_pairs: int) -> List[Tuple[int, int]]:
    """Return up to max_pairs (i, i+1) index pairs evenly spaced over n frames."""
    if n < 2:
        return []
    step = max(1, (n - 1) // max_pairs)
    pairs = []
    i = 0
    while i < n - 1 and len(pairs) < max_pairs:
        pairs.append((i, i + 1))
        i += step
    return pairs
