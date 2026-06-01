"""
Real-Estate Listing-Video Shot Selector
Streamlit UI — drop a folder of MP4 clips, get a ranked picks table
and a 60-second auto-edited preview MP4.
"""

import os
import sys
import time
import json
import tempfile
import shutil
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score

from pipeline.scene_split import process_folder, SubShot
from pipeline.classify import Classifier, ROOM_LABELS
from pipeline.rank import pick_top_shots, score_and_rank, build_diversity_summary
from pipeline.stitch import stitch_preview

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RE Shot Selector · Groovy Web",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .stProgress > div > div { background-color: #143cba; }
    .metric-card {
        background: #f7f9fc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 8px;
    }
    .shot-pill {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
    }
    .pill-sharp       { background: #d1fae5; color: #065f46; }
    .pill-blurry      { background: #fee2e2; color: #991b1b; }
    .pill-dark        { background: #fef3c7; color: #92400e; }
    .pill-bedroom     { background: #dbeafe; color: #1e40af; }
    .pill-kitchen     { background: #fce7f3; color: #9d174d; }
    .pill-bathroom    { background: #cffafe; color: #155e75; }
    .pill-living-room { background: #d1fae5; color: #065f46; }
    .pill-exterior    { background: #e0e7ff; color: #3730a3; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Constants ─────────────────────────────────────────────────────────────────
ROOM_COLORS = {
    "bedroom":     "#dbeafe",
    "kitchen":     "#fce7f3",
    "bathroom":    "#cffafe",
    "living room": "#d1fae5",
    "exterior":    "#e0e7ff",
}
QUALITY_COLORS = {
    "sharp":  "🟢",
    "blurry": "🔴",
    "dark":   "🟡",
}
CAMERA_ICONS = {
    "static": "📷",
    "pan":    "🎥",
    "walk":   "🚶",
}

NUM_PICKS = 12


# ── Safe temp-dir helpers ─────────────────────────────────────────────────────
def _safe_copy_file(src_bytes: bytes, dest: str, retries: int = 5) -> bool:
    """Write bytes to dest, retrying on Windows PermissionError (WinError 32)."""
    for attempt in range(retries):
        try:
            with open(dest, "wb") as f:
                f.write(src_bytes)
            return True
        except PermissionError:
            time.sleep(0.4 * (attempt + 1))
    return False


def _make_work_dir() -> str:
    """Always create a brand-new temp directory so stale locks never apply."""
    return tempfile.mkdtemp(prefix="re_shots_")


# ── Cached model loader ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading CLIP model…")
def load_classifier() -> Classifier:
    return Classifier()


# ── Helpers ───────────────────────────────────────────────────────────────────
def bgr_to_pil(frame: np.ndarray, max_size: int = 320) -> Image.Image:
    """Convert OpenCV BGR frame to resized PIL image."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    img.thumbnail((max_size, max_size))
    return img


def shots_to_dataframe(shots: List[SubShot]) -> pd.DataFrame:
    rows = [s.to_dict() for s in shots]
    df = pd.DataFrame(rows)
    col_order = [
        "shot_id", "source_clip", "start_time", "end_time", "duration",
        "room_type", "room_confidence", "camera_move", "quality",
        "blur_score", "brightness_score", "motion_score", "final_score",
    ]
    df = df[[c for c in col_order if c in df.columns]]
    if "source_clip" in df.columns:
        df["source_clip"] = df["source_clip"].apply(lambda p: Path(p).name)
    return df


def render_dataframe(df: pd.DataFrame, *, height: Optional[int] = None) -> None:
    """Render a dataframe; falls back to HTML table if pyarrow is broken."""
    try:
        if height:
            st.dataframe(df, use_container_width=True, height=height)
        else:
            st.dataframe(df, use_container_width=True)
    except Exception:
        max_h = height or 500
        html = df.to_html(index=False, escape=False)
        st.markdown(
            f"<div style='max-height:{max_h}px;overflow:auto'>{html}</div>",
            unsafe_allow_html=True,
        )


def confusion_heatmap(y_true, y_pred, labels) -> plt.Figure:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        ax=ax, linewidths=0.5, linecolor="white",
    )
    acc = accuracy_score(y_true, y_pred)
    ax.set_title(f"Room-type confusion matrix   accuracy={acc:.2%}", fontsize=11)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("Ground Truth", fontsize=10)
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.image(
        "https://raw.githubusercontent.com/streamlit/streamlit/develop/frontend/app/src/assets/streamlit-logo-primary-colormark-darktext.png",
        width=160,
    )
    st.markdown("## 🏠 RE Shot Selector")
    st.caption("Groovy Web · CHARUSAT 2026 · Hetvi Rabari")
    st.divider()

    st.markdown("### ⚙️ Settings")
    scene_threshold = st.slider(
        "Scene-detect threshold", min_value=15, max_value=50, value=27,
        help="Lower = detect more cuts; higher = fewer, longer sub-shots",
    )
    min_scene_len = st.slider(
        "Min scene length (frames)", 5, 60, 15,
        help="Discard sub-shots shorter than this many frames",
    )
    num_picks = st.slider("Number of picks", 6, 20, NUM_PICKS)
    target_duration = st.slider("Target preview duration (s)", 30, 120, 60)

    st.divider()
    st.markdown("### 📋 Quality gates")
    st.markdown("- Top-1 accuracy ≥ **0.70**")
    st.markdown("- Preview ≤ **90 s** for 20 clips")
    st.markdown("- **All 5** room types in picks (if present)")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ═══════════════════════════════════════════════════════════════════════════════
st.title("🏠 Real-Estate Listing-Video Shot Selector")
st.markdown(
    "_Drop a folder of raw MP4 clips → get ranked picks → auto-edited 60-second preview._"
)

tabs = st.tabs(["📁 Process Clips", "📊 Picks & Preview", "🧪 Evaluation", "📖 How It Works"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Process Clips
# ─────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    st.markdown("### Upload MP4 clips")

    col_a, col_b = st.columns([2, 1])

    with col_a:
        uploaded_files = st.file_uploader(
            "Upload one or more MP4 clips",
            type=["mp4", "MP4"],
            accept_multiple_files=True,
            help="Upload your raw listing clips. At least 1 required; 5–20 recommended.",
        )

    with col_b:
        st.markdown("**Or specify a local folder path:**")
        folder_path = st.text_input(
            "Local folder path",
            placeholder="C:\\path\\to\\clips",
            help="Absolute path to a folder of .mp4 files on this machine.",
        )

    run_button = st.button(
        "🚀 Run Pipeline",
        type="primary",
        disabled=(not uploaded_files and not folder_path),
    )

    if run_button:
        # ── Always create a fresh temp dir — avoids ALL WinError 32 issues ──
        work_dir = _make_work_dir()
        st.session_state["work_dir"] = work_dir
        clip_dir = work_dir

        # ── Save uploaded files ────────────────────────────────────────────
        if uploaded_files:
            failed = []
            for uf in uploaded_files:
                dest = os.path.join(work_dir, uf.name)
                ok = _safe_copy_file(uf.getbuffer().tobytes(), dest)
                if not ok:
                    failed.append(uf.name)
            if failed:
                st.error(
                    f"Could not write these files after retries: {failed}\n"
                    "Close any program that might have them open, then try again."
                )
                st.stop()
            clip_dir = work_dir

        elif folder_path:
            folder_path = folder_path.strip().strip('"').strip("'")
            if os.path.isdir(folder_path):
                clip_dir = folder_path
            else:
                st.error(f"Folder not found: `{folder_path}`")
                st.stop()
        else:
            st.error("Please upload files or provide a valid folder path.")
            st.stop()

        # ── Verify there are MP4s ──────────────────────────────────────────
        all_clips = (
            list(Path(clip_dir).glob("*.mp4")) +
            list(Path(clip_dir).glob("*.MP4"))
        )
        if not all_clips:
            st.error(f"No .mp4 files found in `{clip_dir}`. Check the path or upload files.")
            st.stop()

        total_clips = len(all_clips)

        # ── Step 1: Scene split ────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### Step 1 of 3 — Splitting clips into sub-shots")
        progress_bar = st.progress(0.0)
        status_text = st.empty()

        t_start = time.time()

        def split_progress(clip_path, current, total):
            progress_bar.progress(current / total)
            status_text.text(
                f"Splitting {Path(clip_path).name} … ({current}/{total} clips)"
            )

        try:
            all_subshots: List[SubShot] = process_folder(
                clip_dir,
                threshold=scene_threshold,
                min_scene_len=min_scene_len,
                progress_callback=split_progress,
            )
        except Exception as e:
            st.error(f"Scene splitting failed: {e}")
            st.stop()

        progress_bar.progress(1.0)
        st.success(
            f"✅ Found **{len(all_subshots)} sub-shots** from {total_clips} clip(s) "
            f"in {time.time() - t_start:.1f}s"
        )

        if not all_subshots:
            st.error("No sub-shots detected. Check that your MP4 files are valid.")
            st.stop()

        # ── Step 2: Classify ───────────────────────────────────────────────
        st.markdown("#### Step 2 of 3 — Classifying sub-shots")
        progress_bar2 = st.progress(0.0)
        status2 = st.empty()
        clf = load_classifier()

        def classify_progress(i, total):
            progress_bar2.progress(i / total)
            status2.text(f"Classifying {i}/{total} sub-shots…")

        t2 = time.time()
        try:
            clf.classify_batch(all_subshots, progress_callback=classify_progress)
        except Exception as e:
            st.error(f"Classification failed: {e}")
            st.stop()

        progress_bar2.progress(1.0)
        st.success(f"✅ Classification done in {time.time() - t2:.1f}s")

        # ── Step 3: Rank & pick ────────────────────────────────────────────
        st.markdown("#### Step 3 of 3 — Ranking and selecting top picks")
        picks = pick_top_shots(all_subshots, n=num_picks)
        st.success(f"✅ Selected **{len(picks)} picks** for the preview")

        # ── Store in session state ─────────────────────────────────────────
        st.session_state["all_subshots"] = all_subshots
        st.session_state["picks"] = picks
        st.session_state["work_dir"] = work_dir
        st.session_state["clip_dir"] = clip_dir

        total_wall = time.time() - t_start
        st.metric("⏱ Total wall-clock time", f"{total_wall:.1f}s")

        if total_clips <= 20 and total_wall <= 90:
            st.success("✅ Quality gate: pipeline ran in ≤ 90s for ≤ 20 clips")
        elif total_clips > 20:
            st.info(f"ℹ️ {total_clips} clips processed (gate applies to ≤ 20 clips)")
        else:
            st.warning(f"⚠️ Took {total_wall:.1f}s (gate: ≤ 90s). Try on faster hardware/GPU.")

        st.markdown("---")
        st.markdown("👉 **Switch to the 'Picks & Preview' tab to see results.**")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Picks & Preview
# ─────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    if "picks" not in st.session_state:
        st.info("Run the pipeline in the **Process Clips** tab first.")
        st.stop()

    picks: List[SubShot] = st.session_state["picks"]
    all_subshots: List[SubShot] = st.session_state["all_subshots"]
    work_dir: str = st.session_state["work_dir"]

    # ── Summary metrics ────────────────────────────────────────────────────
    st.markdown("### 📊 Summary")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total sub-shots", len(all_subshots))
    m2.metric("Picks selected", len(picks))
    div_summary = build_diversity_summary(picks)
    m3.metric("Room types", len(div_summary))
    sharp_count = sum(1 for s in picks if s.quality == "sharp")
    m4.metric("Sharp picks", f"{sharp_count}/{len(picks)}")
    avg_score = sum(s.final_score for s in picks) / max(len(picks), 1)
    m5.metric("Avg score", f"{avg_score:.2f}")

    # ── Diversity check ────────────────────────────────────────────────────
    all_rooms_in_input = set(s.room_type for s in all_subshots)
    all_rooms_in_picks = set(s.room_type for s in picks)
    missing = all_rooms_in_input - all_rooms_in_picks

    if not missing:
        st.success(
            f"✅ Quality gate: all {len(all_rooms_in_input)} room type(s) in picks — "
            + ", ".join(sorted(all_rooms_in_input))
        )
    else:
        st.warning(
            f"⚠️ Missing rooms in picks: {', '.join(missing)}. "
            "Try lowering the scene threshold or uploading more clips."
        )

    # ── Room distribution bar chart ────────────────────────────────────────
    st.markdown("### 🗂️ Room distribution in picks")
    room_counts = {r: 0 for r in ROOM_LABELS}
    for s in picks:
        room_counts[s.room_type] = room_counts.get(s.room_type, 0) + 1

    fig_room, ax_room = plt.subplots(figsize=(7, 2.5))
    bars = ax_room.barh(
        list(room_counts.keys()),
        list(room_counts.values()),
        color=["#dbeafe", "#fce7f3", "#cffafe", "#d1fae5", "#e0e7ff"],
    )
    ax_room.set_xlabel("Count")
    ax_room.set_title("Picks per room type")
    ax_room.bar_label(bars, padding=3)
    plt.tight_layout()
    st.pyplot(fig_room, use_container_width=False)

    # ── Picks gallery ──────────────────────────────────────────────────────
    st.markdown("### 🎬 Top Picks (thumbnails)")
    cols = st.columns(4)
    for i, shot in enumerate(picks):
        col = cols[i % 4]
        with col:
            if shot.thumbnail is not None:
                col.image(bgr_to_pil(shot.thumbnail), use_container_width=True)
            q_icon = QUALITY_COLORS.get(shot.quality, "⬜")
            c_icon = CAMERA_ICONS.get(shot.camera_move, "📹")
            col.caption(
                f"#{i+1} {q_icon} {shot.quality} · {c_icon} {shot.camera_move}\n"
                f"🏠 **{shot.room_type}** ({shot.room_confidence:.0%})\n"
                f"⏱ {shot.duration:.1f}s · score {shot.final_score:.2f}"
            )

    # ── Full ranked table ──────────────────────────────────────────────────
    st.markdown("### 📋 Ranked picks table")
    df_picks = shots_to_dataframe(picks)
    render_dataframe(df_picks, height=400)

    st.download_button(
        "⬇️ Download picks table (CSV)",
        data=df_picks.to_csv(index=False).encode(),
        file_name="picks.csv",
        mime="text/csv",
    )

    # ── All sub-shots table (collapsible) ──────────────────────────────────
    with st.expander("🔍 All sub-shots (ranked)"):
        all_ranked = score_and_rank(all_subshots)
        df_all = shots_to_dataframe(all_ranked)
        render_dataframe(df_all, height=500)
        st.download_button(
            "⬇️ Download all sub-shots (CSV)",
            data=df_all.to_csv(index=False).encode(),
            file_name="all_subshots.csv",
            mime="text/csv",
        )

    # ── FFmpeg stitch ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🎞️ Generate Preview MP4")

    stitch_col1, _ = st.columns([2, 1])
    with stitch_col1:
        add_fade = st.checkbox("Add fade-in/fade-out", value=True)

    stitch_button = st.button("🎬 Stitch Preview MP4", type="primary")

    if stitch_button:
        out_path = os.path.join(work_dir, "preview.mp4")
        with st.spinner("Stitching with FFmpeg…"):
            t_s = time.time()
            try:
                stitch_preview(
                    picks,
                    output_path=out_path,
                    target_duration=target_duration,
                    add_fade=add_fade,
                )
                t_elapsed = time.time() - t_s
                st.success(f"✅ Preview generated in {t_elapsed:.1f}s → `preview.mp4`")
                st.session_state["preview_path"] = out_path

                if os.path.exists(out_path):
                    with open(out_path, "rb") as vf:
                        st.download_button(
                            "⬇️ Download preview.mp4",
                            data=vf.read(),
                            file_name="preview.mp4",
                            mime="video/mp4",
                        )
                    st.video(out_path)

            except RuntimeError as e:
                st.error(f"FFmpeg error: {e}")
                st.info("Make sure FFmpeg is on PATH. Run: winget install ffmpeg")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Evaluation
# ─────────────────────────────────────────────────────────────────────────────
with tabs[2]:
    st.markdown("### 🧪 Room-type classification evaluation")
    st.markdown(
        "Upload a **labels CSV** (columns: `shot_id`, `room_type`) containing "
        "≥ 30 hand-annotated sub-shots to produce the confusion matrix and report."
    )

    labels_file = st.file_uploader("Upload labels.csv", type=["csv"])

    if labels_file:
        gt_df = pd.read_csv(labels_file)

        if "shot_id" not in gt_df.columns or "room_type" not in gt_df.columns:
            st.error("CSV must have columns: `shot_id`, `room_type`")
        elif "all_subshots" not in st.session_state:
            st.warning("Run the pipeline first (Process Clips tab).")
        else:
            all_shots_session: List[SubShot] = st.session_state["all_subshots"]
            gt_map = dict(zip(gt_df["shot_id"], gt_df["room_type"]))

            y_true, y_pred = [], []
            for shot in all_shots_session:
                if shot.shot_id in gt_map:
                    y_true.append(gt_map[shot.shot_id])
                    y_pred.append(shot.room_type)

            if len(y_true) < 5:
                st.warning(
                    f"Only {len(y_true)} matching shots found. "
                    "Check that shot_ids in the CSV match generated shot_ids."
                )
                st.markdown("**Generated shot IDs (first 20):**")
                st.write([s.shot_id for s in all_shots_session[:20]])
            else:
                acc = accuracy_score(y_true, y_pred)
                label_set = sorted(set(y_true) | set(y_pred))

                c1, c2, c3 = st.columns(3)
                c1.metric("Top-1 accuracy", f"{acc:.2%}")
                c2.metric("Samples evaluated", len(y_true))
                c3.metric(
                    "Gate (≥ 0.70)",
                    "✅ PASS" if acc >= 0.70 else "❌ FAIL",
                )

                fig = confusion_heatmap(y_true, y_pred, label_set)
                st.pyplot(fig, use_container_width=False)

                os.makedirs("eval", exist_ok=True)

                from sklearn.metrics import classification_report
                report = classification_report(
                    y_true, y_pred, labels=label_set, output_dict=True
                )
                report["top1_accuracy"] = float(acc)
                report["n_samples"] = len(y_true)

                with open("eval/report.json", "w") as f:
                    json.dump(report, f, indent=2)
                fig.savefig("eval/confusion.png", dpi=150, bbox_inches="tight")

                st.success("Artefacts saved: `eval/confusion.png`, `eval/report.json`")

                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    with open("eval/report.json") as f:
                        st.download_button(
                            "⬇️ Download report.json",
                            data=f.read(),
                            file_name="report.json",
                            mime="application/json",
                        )
                with col_d2:
                    with open("eval/confusion.png", "rb") as f:
                        st.download_button(
                            "⬇️ Download confusion.png",
                            data=f.read(),
                            file_name="confusion.png",
                            mime="image/png",
                        )

                st.markdown("#### Per-class metrics")
                rows = []
                for label in label_set:
                    if label in report and isinstance(report[label], dict):
                        r = report[label]
                        rows.append({
                            "Room type": label,
                            "Precision": round(r["precision"], 3),
                            "Recall": round(r["recall"], 3),
                            "F1": round(r["f1-score"], 3),
                            "Support": int(r["support"]),
                        })
                render_dataframe(pd.DataFrame(rows))

    # ── Template generator ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📝 Generate annotation template")
    st.markdown(
        "No labels CSV yet? Download an empty template pre-filled with shot IDs."
    )
    if "all_subshots" in st.session_state:
        template_df = pd.DataFrame(
            [{"shot_id": s.shot_id, "room_type": ""} for s in st.session_state["all_subshots"]]
        )
        st.download_button(
            "⬇️ Download annotation template",
            data=template_df.to_csv(index=False).encode(),
            file_name="labels_template.csv",
            mime="text/csv",
        )
        st.caption(
            f"{len(template_df)} rows. Fill the `room_type` column with: "
            "bedroom / kitchen / bathroom / living room / exterior"
        )
    else:
        st.info("Run the pipeline first to generate the template.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — How It Works
# ─────────────────────────────────────────────────────────────────────────────
with tabs[3]:
    st.markdown(
        """
### Pipeline overview

```
MP4 clips folder
    │
    ▼
[1] PySceneDetect  ─── ContentDetector(threshold=27)
    │                   Splits each clip at cut points
    │                   Samples 1 frame/second per sub-shot
    ▼
[2] Classification (per sub-shot)
    ├── Room type ─── HuggingFace CLIP zero-shot
    │                 "a photo of a bedroom with a bed" × 5 prompts
    │                 → argmax probability → bedroom / kitchen /
    │                   bathroom / living room / exterior
    │
    ├── Camera move ── Dense optical flow (Farneback)
    │                  mean |flow| < 1.2 → static
    │                  mean |flow| < 3.5 → pan
    │                  else             → walk
    │
    └── Quality ─────── Laplacian variance  < 80  → blurry
                        Mean brightness    < 45  → dark
                        else                     → sharp
    │
    ▼
[3] Ranking heuristic
    score  = +4 sharp / -6 blurry / -4 dark
           + +2 pan / +1 static / -1 walk
           + room_confidence × 2
           + duration bonus (capped at 8s)
    pick_top_shots: diversity-first (1 per room type), then fill by score
    │
    ▼
[4] FFmpeg concat
    Trim each pick proportionally to fill ≈60s
    ffmpeg -f concat -safe 0 -i concat.txt -c copy preview.mp4
```

### Quality gates
| Gate | Requirement |
|---|---|
| Top-1 room accuracy | ≥ 0.70 on held-out ≥ 30 sub-shots |
| Preview generation time | ≤ 90s for 20 clips |
| Room diversity in picks | All input room types represented |

### Tools used
- **Streamlit** — UI
- **PySceneDetect** — sub-shot splitting
- **HuggingFace CLIP** (openai/clip-vit-base-patch32) — zero-shot room classification
- **OpenCV** — frame sampling, optical flow, blur/brightness detection
- **FFmpeg** — final MP4 stitching
- **scikit-learn** — confusion matrix, accuracy
- **matplotlib / seaborn** — charts

### Reproducing the evaluation
```bash
pip install -r requirements.txt
python eval/run_eval.py --folder /path/to/clips --gen-template --labels eval/labels.csv
# Fill eval/labels.csv, then:
python eval/run_eval.py --folder /path/to/clips --labels eval/labels.csv
# Outputs: eval/confusion.png, eval/report.json, eval/eval_results.csv
```
        """
    )