"""
Ranking module.
Scores each sub-shot and selects the best 12 with room-type diversity.

Scoring heuristic (higher = better):
  +4  sharp quality
  -6  blurry quality
  -4  dark quality
  +2  pan camera move (smooth, looks cinematic)
  +1  static camera move (clean)
  -1  walk/dolly (can look shaky; still allowed)
  +2  room_confidence bonus (scaled 0-2)
  -0.5 per second under 1.5s (very short sub-shots are less usable)
  +0.3 per second of duration up to 8s cap

Diversity constraint:
  Ensure every room type present in the input appears at least once in
  the picks. Fill remaining slots with highest-scored remaining shots.
"""

from typing import List, Dict, Tuple
from pipeline.scene_split import SubShot
from pipeline.classify import ROOM_LABELS, QUALITY_SHARP, QUALITY_BLURRY, QUALITY_DARK
from pipeline.classify import CAMERA_PAN, CAMERA_STATIC, CAMERA_WALK

NUM_PICKS = 12
TARGET_DURATION_SEC = 60.0  # target output video length


def score_shot(shot: SubShot) -> float:
    """Compute numeric score for a single sub-shot."""
    score = 0.0

    # Quality
    if shot.quality == QUALITY_SHARP:
        score += 4.0
    elif shot.quality == QUALITY_BLURRY:
        score -= 6.0
    elif shot.quality == QUALITY_DARK:
        score -= 4.0

    # Camera move
    if shot.camera_move == CAMERA_PAN:
        score += 2.0
    elif shot.camera_move == CAMERA_STATIC:
        score += 1.0
    elif shot.camera_move == CAMERA_WALK:
        score -= 1.0

    # Classification confidence bonus (0 – 2)
    score += shot.room_confidence * 2.0

    # Duration bonus (reward clips 1.5–8s, penalise very short ones)
    dur = shot.duration
    if dur < 1.5:
        score -= 0.5 * (1.5 - dur)
    score += min(dur, 8.0) * 0.3

    return round(score, 4)


def score_and_rank(shots: List[SubShot]) -> List[SubShot]:
    """Compute final_score for every shot and sort descending."""
    for shot in shots:
        shot.final_score = score_shot(shot)
    return sorted(shots, key=lambda s: s.final_score, reverse=True)


def pick_top_shots(
    shots: List[SubShot],
    n: int = NUM_PICKS,
) -> List[SubShot]:
    """
    Select up to n shots with room-type diversity.

    Strategy:
      1. Score and sort all shots.
      2. Build a diversity reserve: pick the best shot for each room type
         that is present in the input.
      3. Fill remaining slots from the sorted list (skipping already-chosen).
      4. Sort final picks by start_time to maintain a natural walkthrough order.

    Returns list of SubShot, ≤ n items.
    """
    if not shots:
        return []

    ranked = score_and_rank(shots)

    # Which room types exist in this batch?
    present_rooms = set(s.room_type for s in shots)

    picks: List[SubShot] = []
    picked_ids = set()

    # ── Step 1: Diversity reserve ────────────────────────────────────────────
    for room in present_rooms:
        best_for_room = next(
            (s for s in ranked if s.room_type == room and s.shot_id not in picked_ids),
            None,
        )
        if best_for_room:
            picks.append(best_for_room)
            picked_ids.add(best_for_room.shot_id)

    # ── Step 2: Fill remaining slots ─────────────────────────────────────────
    for shot in ranked:
        if len(picks) >= n:
            break
        if shot.shot_id not in picked_ids:
            picks.append(shot)
            picked_ids.add(shot.shot_id)

    # ── Step 3: Sort by original clip order (source + start_time) ───────────
    picks.sort(key=lambda s: (s.source_clip, s.start_time))

    return picks[:n]


def build_diversity_summary(picks: List[SubShot]) -> Dict[str, int]:
    """Return {room_type: count} for reporting."""
    summary: Dict[str, int] = {}
    for shot in picks:
        summary[shot.room_type] = summary.get(shot.room_type, 0) + 1
    return summary


def compute_target_per_clip_duration(
    picks: List[SubShot],
    target_total: float = TARGET_DURATION_SEC,
) -> Dict[str, float]:
    """
    Calculate how many seconds to trim/use from each pick so the
    final stitched video is ≈ target_total seconds.

    Returns {shot_id: trim_duration}.
    """
    if not picks:
        return {}

    total_available = sum(s.duration for s in picks)

    if total_available <= target_total:
        # Use each clip as-is
        return {s.shot_id: s.duration for s in picks}

    # Proportional trim
    ratio = target_total / total_available
    return {s.shot_id: round(s.duration * ratio, 3) for s in picks}
