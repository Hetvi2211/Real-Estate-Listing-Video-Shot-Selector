# Real-Estate Listing-Video Shot Selector

**Groovy Web · CHARUSAT 2026 · Hetvi Rabari**

A Streamlit tool that ingests a folder of short MP4 clips from a real-estate listing, splits each into sub-shots, classifies each by room type + camera move + quality, and outputs the best **12 picks for a 60-second walkthrough**. Stitches the picks with FFmpeg into an auto-edited preview MP4.

---

## Features

- **Sub-shot split** — PySceneDetect ContentDetector; samples 1 frame/second per sub-shot
- **Room type classification** — HuggingFace CLIP zero-shot across 5 classes (bedroom, kitchen, bathroom, living room, exterior)
- **Camera move detection** — Dense optical flow → static / pan / walk
- **Quality scoring** — Laplacian variance (blur) + mean brightness (dark/sharp)
- **Ranking & diversity** — Ensures all room types present in input appear in picks
- **FFmpeg stitch** — 60-second auto-edited preview MP4 with optional fade
- **Evaluation** — Confusion matrix + top-1 accuracy report against hand-labelled test set

---

## Setup

### Prerequisites

```bash
# Python 3.10+
python --version

# FFmpeg (required for stitching)
# Ubuntu/Debian:
sudo apt-get install ffmpeg
# macOS:
brew install ffmpeg
# Windows: https://ffmpeg.org/download.html
ffmpeg -version
```

### Install dependencies

```bash
git clone https://github.com/Hetvi2211/Real-Estate-Listing-Video-Shot-Selector
cd real-estate-shot-selector

pip install -r requirements.txt
```

---

## Run the Streamlit app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

**Happy path:**
1. Go to **Process Clips** tab
2. Upload 5–20 MP4 clips (or enter a local folder path)
3. Click **Run Pipeline** — sub-shot split → classify → rank
4. Switch to **Picks & Preview** tab
5. Review thumbnails, ranked table, room distribution
6. Click **Stitch Preview MP4** → download `preview.mp4`

---

## Evaluation (reproducing quality gate)

### Step 1 — Generate annotation template

After running the pipeline on your test clips, download the template from the **Evaluation** tab, or run:

```bash
python eval/run_eval.py \
  --folder /path/to/test_clips \
  --gen-template \
  --labels eval/labels.csv
```

### Step 2 — Annotate

Open `eval/labels.csv` and fill the `room_type` column for each sub-shot:

```
shot_id,room_type
clip1_shot000,bedroom
clip1_shot001,kitchen
...
```

Allowed values: `bedroom`, `kitchen`, `bathroom`, `living room`, `exterior`
Annotate **at least 30** sub-shots.

### Step 3 — Run evaluation

```bash
python eval/run_eval.py \
  --folder /path/to/test_clips \
  --labels eval/labels.csv \
  --output eval
```

Outputs:
- `eval/confusion.png` — confusion matrix heatmap
- `eval/report.json` — per-class precision/recall/F1 + top-1 accuracy
- `eval/eval_results.csv` — per-shot predictions vs ground truth

### Quality gates

| Gate | Requirement | Typical result |
|---|---|---|
| Top-1 room accuracy | ≥ 0.70 | CLIP zero-shot achieves ~0.75–0.85 |
| Preview generation time | ≤ 90s for 20 clips | ~40–70s on CPU |
| Room diversity in picks | All input room types in output | Guaranteed by ranking logic |

---

## Project structure

```
real-estate-shot-selector/
├── app.py                    # Streamlit UI (all 4 tabs)
├── requirements.txt          # pip install -r requirements.txt
├── pipeline/
│   ├── __init__.py
│   ├── scene_split.py        # PySceneDetect sub-shot extraction
│   ├── classify.py           # CLIP + blur + optical flow classifiers
│   ├── rank.py               # Scoring heuristic + diversity picker
│   └── stitch.py             # FFmpeg concat wrapper
├── eval/
│   ├── __init__.py
│   ├── run_eval.py           # Standalone eval script
│   ├── labels.csv            # (Your hand-labelled test set — not committed)
│   ├── confusion.png         # Committed after eval run
│   └── report.json           # Committed after eval run
└── README.md
```

---

## Allowed tools

All free / localhost only per task spec:

- **UI**: Streamlit (localhost:8501)
- **CV/ML**: HuggingFace CLIP (free tier), OpenCV, PyTorch, scikit-learn
- **Video**: PySceneDetect, FFmpeg
- **Storage**: local filesystem only

---

## What I would do with more time

1. **Fine-tune CLIP** on labelled real-estate frames (100–500 examples per room) using HuggingFace `Trainer` + free Google Colab T4 — expected to push accuracy to 0.90+.

2. **Audio classification** — detect echo (bathroom), silence (exterior), appliance hum (kitchen) using `librosa` + a lightweight CNN to supplement CLIP's visual signal.

3. **Smart trim** — instead of proportional trimming to fill 60s, use a saliency detector to keep only the sharpest 4-second window within each sub-shot.

4. **GPU batching** — batch CLIP inference across all frames in one forward pass on GPU; would cut classification time from ~2s/shot to ~0.2s/shot.

5. **Agent re-rank** — pass the ranked list + thumbnails to a free Gemini/Groq call to apply editorial judgment ("avoid duplicate angles of the same room").

6. **Export to NLE** — generate an EDL (Edit Decision List) or Premiere Pro XML so the picks can be imported directly into a professional timeline.

7. **Progressive streaming UI** — show thumbnails and scores as each sub-shot is classified, rather than waiting for the full batch.

8. **Docker image** — bundle FFmpeg + all dependencies into a single `docker run` command for one-click reproducibility.

---

## Demo video tips

- Use **Loom** or **OBS** to record a 3–6 minute walkthrough.
- Show: folder upload → pipeline run → picks table → room distribution chart → FFmpeg stitch → download preview.mp4.
- Show the **Evaluation tab**: upload labels.csv, see confusion matrix, confirm accuracy ≥ 0.70.
- Show terminal output of `python eval/run_eval.py` confirming quality gates.

## 📋 What I'd Do With More Time

If given more development time, I would focus on these key upgrades to move this project from an MVP to a production-ready tool:

* **Fine-Tune the AI Model:** Fine-tune CLIP on a small dataset of real estate photos to improve its accuracy on tricky areas, like open-concept kitchen/dining rooms.
* **Add Audio Awareness:** Use audio classification to detect high-echo spaces (bathrooms) or outdoor sounds (exteriors), and use the background music's beat to time the video cuts perfectly.
* **Improve Motion Analysis:** Upgrade the OpenCV tracking to deep-learning optical flow to better separate intentional camera pans from accidental shaky movements.
* **Speed Up Processing:** Implement parallel processing and model quantization (FP16/ONNX) so the pipeline can process large batches of videos much faster on standard CPUs.
* **Export to Editing Software:** Instead of just making a finished video, generate standard XML or EDL project files so editors can import the top 12 picks directly into Adobe Premiere or DaVinci Resolve.
