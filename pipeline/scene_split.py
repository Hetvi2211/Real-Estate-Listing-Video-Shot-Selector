"""
Sub-shot splitter using PySceneDetect.
Splits each MP4 clip into scene-level sub-shots and samples frames at 1 fps.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, field

try:
    from scenedetect import open_video, SceneManager
    from scenedetect.detectors import ContentDetector
    SCENEDETECT_AVAILABLE = True
except ImportError:
    SCENEDETECT_AVAILABLE = False


@dataclass
class SubShot:
    """Represents a single sub-shot extracted from a clip."""
    shot_id: str
    source_clip: str
    start_time: float       # seconds
    end_time: float         # seconds
    duration: float         # seconds
    frames: List[np.ndarray] = field(default_factory=list)  # sampled frames (1 fps)
    frame_timestamps: List[float] = field(default_factory=list)
    # Classification results (filled later)
    room_type: str = ""
    room_confidence: float = 0.0
    camera_move: str = ""
    quality: str = ""
    blur_score: float = 0.0
    brightness_score: float = 0.0
    motion_score: float = 0.0
    final_score: float = 0.0
    thumbnail: np.ndarray = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "shot_id": self.shot_id,
            "source_clip": self.source_clip,
            "start_time": round(self.start_time, 2),
            "end_time": round(self.end_time, 2),
            "duration": round(self.duration, 2),
            "num_frames": len(self.frames),
            "room_type": self.room_type,
            "room_confidence": round(self.room_confidence, 3),
            "camera_move": self.camera_move,
            "quality": self.quality,
            "blur_score": round(self.blur_score, 1),
            "brightness_score": round(self.brightness_score, 1),
            "motion_score": round(self.motion_score, 3),
            "final_score": round(self.final_score, 3),
        }


def split_clip_into_subshots(
    clip_path: str,
    threshold: float = 27.0,
    min_scene_len: int = 15,
) -> List[SubShot]:
    """
    Split a single MP4 into sub-shots using PySceneDetect ContentDetector.
    Falls back to treating the whole clip as one shot if detection fails.

    Args:
        clip_path: Path to the MP4 file.
        threshold: ContentDetector sensitivity (lower = more cuts detected).
        min_scene_len: Minimum scene length in frames.

    Returns:
        List of SubShot objects with sampled frames (1 fps).
    """
    clip_name = Path(clip_path).stem
    subshots: List[SubShot] = []

    # ── Get video metadata ──────────────────────────────────────────────────
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps
    cap.release()

    if duration_sec < 0.5:
        return []

    # ── Detect scene boundaries ─────────────────────────────────────────────
    scene_list = []

    if SCENEDETECT_AVAILABLE:
        try:
            video = open_video(clip_path)
            scene_manager = SceneManager()
            scene_manager.add_detector(
                ContentDetector(threshold=threshold, min_scene_len=min_scene_len)
            )
            scene_manager.detect_scenes(video, show_progress=False)
            scene_list = scene_manager.get_scene_list()
        except Exception:
            scene_list = []

    # Fallback: treat entire clip as one scene
    if not scene_list:
        scene_list = [
            (
                _make_timecode(0, fps),
                _make_timecode(total_frames - 1, fps),
            )
        ]

    # ── Extract 1-fps frames per scene ──────────────────────────────────────
    cap = cv2.VideoCapture(clip_path)

    for idx, (start_tc, end_tc) in enumerate(scene_list):
        try:
            start_s = float(start_tc.get_seconds())
            end_s = float(end_tc.get_seconds())
        except Exception:
            # Fallback for simple (start_frame, end_frame) tuples
            start_s = 0.0
            end_s = duration_sec

        end_s = min(end_s, duration_sec)
        shot_duration = end_s - start_s

        if shot_duration < 0.25:
            continue

        frames, timestamps = _sample_frames_1fps(cap, start_s, end_s, fps)

        if not frames:
            continue

        shot_id = f"{clip_name}_shot{idx:03d}"
        thumbnail = _pick_middle_frame(frames)

        subshot = SubShot(
            shot_id=shot_id,
            source_clip=clip_path,
            start_time=start_s,
            end_time=end_s,
            duration=shot_duration,
            frames=frames,
            frame_timestamps=timestamps,
            thumbnail=thumbnail,
        )
        subshots.append(subshot)

    cap.release()
    return subshots


def process_folder(
    folder_path: str,
    threshold: float = 27.0,
    min_scene_len: int = 15,
    progress_callback=None,
) -> List[SubShot]:
    """
    Process all MP4 files in a folder and return flat list of sub-shots.

    Args:
        folder_path: Directory containing .mp4 files.
        threshold: Scene detection sensitivity.
        min_scene_len: Minimum frames per scene.
        progress_callback: Optional callable(clip_path, current, total).

    Returns:
        Flat list of SubShot objects.
    """
    folder = Path(folder_path)
    mp4_files = sorted(folder.glob("*.mp4")) + sorted(folder.glob("*.MP4"))
    mp4_files = list(dict.fromkeys(mp4_files))  # deduplicate

    all_subshots: List[SubShot] = []

    for i, clip_path in enumerate(mp4_files):
        if progress_callback:
            progress_callback(str(clip_path), i + 1, len(mp4_files))
        shots = split_clip_into_subshots(
            str(clip_path), threshold=threshold, min_scene_len=min_scene_len
        )
        all_subshots.extend(shots)

    return all_subshots


# ── Helpers ──────────────────────────────────────────────────────────────────

def _sample_frames_1fps(
    cap: cv2.VideoCapture,
    start_s: float,
    end_s: float,
    fps: float,
) -> tuple:
    """Sample one frame per second from [start_s, end_s)."""
    frames = []
    timestamps = []

    t = start_s
    step = 1.0  # 1 fps

    while t < end_s:
        frame_no = int(t * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = cap.read()
        if ret and frame is not None:
            # Resize to 224×224 for CLIP compatibility (keep original for display)
            frames.append(frame)
            timestamps.append(round(t, 2))
        t += step

    return frames, timestamps


def _pick_middle_frame(frames: List[np.ndarray]) -> np.ndarray:
    """Return the middle frame as a thumbnail."""
    if not frames:
        return None
    return frames[len(frames) // 2]


class _SimpleTimecode:
    """Minimal timecode shim when PySceneDetect is unavailable."""
    def __init__(self, seconds: float):
        self._seconds = seconds

    def get_seconds(self) -> float:
        return self._seconds


def _make_timecode(frame: int, fps: float) -> "_SimpleTimecode":
    return _SimpleTimecode(frame / max(fps, 1))
