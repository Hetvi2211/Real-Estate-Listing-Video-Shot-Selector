pip uninstall pyarrow streamlit -y"""
FFmpeg stitch module.
Builds a concat list and calls FFmpeg to produce a 60-second preview MP4.
"""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Optional

from pipeline.scene_split import SubShot
from pipeline.rank import compute_target_per_clip_duration, TARGET_DURATION_SEC


def stitch_preview(
    picks: List[SubShot],
    output_path: str,
    target_duration: float = TARGET_DURATION_SEC,
    add_fade: bool = True,
) -> str:
    """
    Stitch the selected sub-shots into a single MP4 preview.

    Strategy:
      - Trim each clip to its allotted duration (proportional to fill ≈60s).
      - Use FFmpeg's concat demuxer for speed (stream copy when possible).
      - Falls back to re-encode with libx264 if stream copy fails.

    Args:
        picks: Ordered list of selected SubShot objects.
        output_path: Where to write the preview MP4.
        target_duration: Target total duration in seconds.
        add_fade: Add simple fade-in/out at start and end.

    Returns:
        Absolute path to the output file.

    Raises:
        RuntimeError: If FFmpeg is not installed or the stitch fails.
    """
    _check_ffmpeg()

    trim_map = compute_target_per_clip_duration(picks, target_duration)

    # ── Build per-shot trimmed clips in temp dir ──────────────────────────
    with tempfile.TemporaryDirectory() as tmpdir:
        trimmed_paths: List[str] = []

        for i, shot in enumerate(picks):
            trim_dur = trim_map.get(shot.shot_id, shot.duration)
            trimmed = os.path.join(tmpdir, f"seg_{i:03d}.mp4")
            _trim_clip(
                src=shot.source_clip,
                dst=trimmed,
                start=shot.start_time,
                duration=trim_dur,
            )
            if os.path.exists(trimmed) and os.path.getsize(trimmed) > 0:
                trimmed_paths.append(trimmed)

        if not trimmed_paths:
            raise RuntimeError("No trimmed segments were produced.")

        # ── Write concat list ───────────────────────────────────────────────
        concat_file = os.path.join(tmpdir, "concat.txt")
        with open(concat_file, "w") as f:
            for p in trimmed_paths:
                f.write(f"file '{p}'\n")

        # ── Run FFmpeg concat ───────────────────────────────────────────────
        _run_concat(concat_file, output_path, add_fade=add_fade)

    return os.path.abspath(output_path)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _trim_clip(src: str, dst: str, start: float, duration: float) -> None:
    """Trim a clip segment using stream copy (fast, no re-encode)."""
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(round(start, 3)),
        "-i", src,
        "-t", str(round(max(duration, 0.5), 3)),
        "-c", "copy",
        "-avoid_negative_ts", "1",
        dst,
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=60
    )
    # If stream copy fails, re-encode
    if result.returncode != 0 or not os.path.exists(dst):
        cmd_re = [
            "ffmpeg", "-y",
            "-ss", str(round(start, 3)),
            "-i", src,
            "-t", str(round(max(duration, 0.5), 3)),
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            dst,
        ]
        subprocess.run(cmd_re, capture_output=True, text=True, timeout=120)


def _run_concat(
    concat_file: str,
    output_path: str,
    add_fade: bool = True,
) -> None:
    """Run FFmpeg concat demuxer to join trimmed segments."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    base_cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", concat_file,
    ]

    if add_fade:
        # Simple fade-in 0.5s at start, fade-out 0.5s at end using vf
        # We compute total duration to place fade-out correctly
        base_cmd += [
            "-vf", "fade=t=in:st=0:d=0.5,fade=t=out:st=55:d=0.5",
            "-c:v", "libx264", "-preset", "fast", "-crf", "22",
            "-c:a", "aac", "-b:a", "128k",
        ]
    else:
        base_cmd += ["-c", "copy"]

    base_cmd.append(output_path)

    result = subprocess.run(
        base_cmd, capture_output=True, text=True, timeout=180
    )

    if result.returncode != 0:
        # Fallback without fade
        fallback = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", concat_file,
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            output_path,
        ]
        result2 = subprocess.run(
            fallback, capture_output=True, text=True, timeout=180
        )
        if result2.returncode != 0:
            raise RuntimeError(
                f"FFmpeg concat failed:\n{result2.stderr[-2000:]}"
            )


def _check_ffmpeg() -> None:
    """Raise RuntimeError if ffmpeg is not on PATH."""
    result = subprocess.run(
        ["ffmpeg", "-version"], capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(
            "FFmpeg is not installed or not on PATH. "
            "Install it with: sudo apt-get install ffmpeg"
        )
