"""
Evaluation script — run from project root:

    python eval/run_eval.py --folder /path/to/clips --labels eval/labels.csv

Labels CSV format (hand-annotated):
    shot_id,room_type
    clip1_shot000,bedroom
    clip1_shot001,kitchen
    ...

Outputs:
    eval/confusion.png   — confusion matrix heatmap
    eval/report.json     — classification report dict
    eval/eval_results.csv — per-shot predictions vs ground truth
"""

import argparse
import json
import os
import sys
import time

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pipeline.scene_split import process_folder
from pipeline.classify import Classifier, ROOM_LABELS
from pipeline.rank import score_and_rank


EVAL_DIR = os.path.join(os.path.dirname(__file__))


def run_eval(folder: str, labels_csv: str, output_dir: str = EVAL_DIR):
    os.makedirs(output_dir, exist_ok=True)

    # ── Load ground-truth labels ──────────────────────────────────────────
    gt_df = pd.read_csv(labels_csv)
    gt_map = dict(zip(gt_df["shot_id"], gt_df["room_type"]))
    print(f"Loaded {len(gt_map)} ground-truth labels.")

    # ── Process clips ──────────────────────────────────────────────────────
    print("Splitting clips into sub-shots…")
    t0 = time.time()
    subshots = process_folder(folder)
    print(f"  {len(subshots)} sub-shots in {time.time()-t0:.1f}s")

    # ── Classify ───────────────────────────────────────────────────────────
    print("Loading CLIP model…")
    clf = Classifier()
    print("Classifying sub-shots…")
    t1 = time.time()

    def progress(i, total):
        if i % 5 == 0 or i == total:
            print(f"  {i}/{total} classified", end="\r")

    clf.classify_batch(subshots, progress_callback=progress)
    print(f"\n  Done in {time.time()-t1:.1f}s")

    # ── Match predictions to ground truth ──────────────────────────────────
    y_true, y_pred, matched_ids = [], [], []
    unmatched = []

    for shot in subshots:
        if shot.shot_id in gt_map:
            y_true.append(gt_map[shot.shot_id])
            y_pred.append(shot.room_type)
            matched_ids.append(shot.shot_id)
        else:
            unmatched.append(shot.shot_id)

    if not y_true:
        print("ERROR: No sub-shots matched ground-truth labels.")
        print("Check that shot_ids in labels.csv match the generated shot_ids.")
        sys.exit(1)

    print(f"\n  Matched {len(y_true)}/{len(gt_map)} labeled shots.")
    if unmatched:
        print(f"  {len(unmatched)} unmatched shot IDs (first 5): {unmatched[:5]}")

    # ── Metrics ────────────────────────────────────────────────────────────
    acc = accuracy_score(y_true, y_pred)
    print(f"\n🎯 Top-1 accuracy: {acc:.4f} ({acc*100:.1f}%)")

    label_set = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=label_set)
    report = classification_report(y_true, y_pred, labels=label_set, output_dict=True)
    report["top1_accuracy"] = float(acc)
    report["n_samples"] = len(y_true)
    report["label_set"] = label_set

    gate_pass = acc >= 0.70
    print(f"{'✅' if gate_pass else '❌'} Quality gate (≥0.70): {'PASS' if gate_pass else 'FAIL'}")

    # ── Save report.json ──────────────────────────────────────────────────
    report_path = os.path.join(output_dir, "report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved {report_path}")

    # ── Save confusion.png ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_set,
        yticklabels=label_set,
        ax=ax,
        linewidths=0.5,
        linecolor="white",
    )
    ax.set_title(
        f"Room-type confusion matrix\nTop-1 accuracy: {acc:.2%}  (n={len(y_true)})",
        fontsize=12,
        pad=12,
    )
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Ground Truth", fontsize=11)
    plt.tight_layout()
    confusion_path = os.path.join(output_dir, "confusion.png")
    plt.savefig(confusion_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {confusion_path}")

    # ── Save per-shot CSV ──────────────────────────────────────────────────
    results_df = pd.DataFrame(
        {
            "shot_id": matched_ids,
            "true_room": y_true,
            "pred_room": y_pred,
            "correct": [t == p for t, p in zip(y_true, y_pred)],
        }
    )
    csv_path = os.path.join(output_dir, "eval_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")

    print("\n=== Per-class summary ===")
    for label in label_set:
        if label in report and isinstance(report[label], dict):
            r = report[label]
            print(
                f"  {label:<12} precision={r['precision']:.2f}  "
                f"recall={r['recall']:.2f}  f1={r['f1-score']:.2f}  "
                f"support={int(r['support'])}"
            )

    return acc, report


def generate_sample_labels_csv(folder: str, output: str):
    """
    Helper: generate a template labels.csv from a clip folder.
    Writes shot_id,room_type rows with empty room_type for manual annotation.
    """
    subshots = process_folder(folder)
    rows = [{"shot_id": s.shot_id, "room_type": ""} for s in subshots]
    df = pd.DataFrame(rows)
    df.to_csv(output, index=False)
    print(f"Template labels CSV written to {output} ({len(rows)} rows). Fill in room_type column.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate room-type classification accuracy.")
    parser.add_argument("--folder", required=True, help="Folder of MP4 clips")
    parser.add_argument("--labels", default="eval/labels.csv", help="CSV with shot_id,room_type")
    parser.add_argument("--output", default="eval", help="Output directory for artefacts")
    parser.add_argument("--gen-template", action="store_true",
                        help="Generate empty labels.csv template and exit")
    args = parser.parse_args()

    if args.gen_template:
        generate_sample_labels_csv(args.folder, args.labels)
    else:
        run_eval(args.folder, args.labels, args.output)
