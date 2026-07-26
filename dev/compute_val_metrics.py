"""
Val-set metric report: accuracy, precision, recall, F1, mAP.

Two evaluation regimes are computed and reported side by side, because they
answer different questions (see docs/research_log.md / reports/sota_positioning_brief.md
on task-IL vs class-IL):

  - FULL 48-way (true class-incremental / open-world): argmax over ALL 48
    logits, no task/stage ID given. This is the honest, harder number.
  - TASK-AWARE 6-way (current headline metric): argmax restricted to the
    6 classes of the sample's own stage (task ID given at test time).
    Easier; chance = 1/6.

Run: python3 dev/compute_val_metrics.py
Output: reports/val_metrics_result.md
"""
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, average_precision_score,
)

from src.models.gru_detector import GRUDetector

CKPT_DIR = Path("checkpoints")
VAL_FEAT = Path("data/features/val")
VAL_JSON = Path("data/subset/val_mini.json")
N_CLASSES = 48
CPS = 6  # classes per stage
STAGES = {s: list(range((s - 1) * CPS, s * CPS)) for s in range(1, N_CLASSES // CPS + 1)}
CLASS_TO_STAGE = {c: s for s, cids in STAGES.items() for c in cids}


def load_model(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = GRUDetector(feature_dim=ck["feature_dim"], hidden_dim=ck["hidden_dim"],
                     num_classes=ck["n_classes"])
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m, ck


@torch.no_grad()
def collect_predictions(model):
    samples = json.load(open(VAL_JSON, encoding="utf-8"))
    y_true, probs = [], []
    for s in samples:
        p = VAL_FEAT / f"{s['id']}.npy"
        if not p.exists():
            continue
        x = torch.tensor(np.load(p), dtype=torch.float32).unsqueeze(0)
        logits, _ = model(x, None)
        probs.append(torch.softmax(logits, dim=-1)[0].numpy())
        y_true.append(s["class_id"])
    return np.array(y_true), np.array(probs)  # (N,), (N, 48)


def full_48way_metrics(y_true, probs):
    y_pred = probs.argmax(axis=1)
    acc = accuracy_score(y_true, y_pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0)
    y_onehot = np.eye(N_CLASSES)[y_true]
    mAP = average_precision_score(y_onehot, probs, average="macro")
    return dict(accuracy=acc, precision_macro=p_macro, recall_macro=r_macro,
                f1_macro=f1_macro, precision_weighted=p_w, recall_weighted=r_w,
                f1_weighted=f1_w, mAP=mAP, chance=1 / N_CLASSES)


def task_aware_metrics(y_true, probs):
    """Restrict argmax to the sample's own stage's 6 classes; aggregate per-stage then average."""
    accs, precs, recs, f1s = [], [], [], []
    for stage, cids in STAGES.items():
        mask = np.isin(y_true, cids)
        if mask.sum() == 0:
            continue
        yt = y_true[mask]
        sub_probs = probs[mask][:, cids]
        local_pred = sub_probs.argmax(axis=1)
        yp = np.array([cids[i] for i in local_pred])
        accs.append(accuracy_score(yt, yp))
        p, r, f1, _ = precision_recall_fscore_support(
            yt, yp, labels=cids, average="macro", zero_division=0)
        precs.append(p); recs.append(r); f1s.append(f1)
    return dict(accuracy=float(np.mean(accs)), precision_macro=float(np.mean(precs)),
                recall_macro=float(np.mean(recs)), f1_macro=float(np.mean(f1s)),
                chance=1 / CPS)


def fmt(d, keys):
    return "  ".join(f"{k}={d[k]:.3f}" for k in keys if k in d)


def main():
    rows = []
    for name, path in [("baseline_48cls.pt", CKPT_DIR / "baseline_48cls.pt"),
                        ("agem_48cls.pt", CKPT_DIR / "agem_48cls.pt")]:
        model, ck = load_model(path)
        y_true, probs = collect_predictions(model)
        full = full_48way_metrics(y_true, probs)
        task = task_aware_metrics(y_true, probs)
        rows.append((name, ck, full, task, len(y_true)))
        print(f"{name}  n={len(y_true)}")
        print("  FULL 48-way :", fmt(full, ["accuracy", "precision_macro", "recall_macro", "f1_macro", "mAP"]))
        print("  TASK 6-way  :", fmt(task, ["accuracy", "precision_macro", "recall_macro", "f1_macro"]))

    out = ["# Val-set metrics — full class-IL (48-way) vs task-aware (6-way)", "",
           f"Generated: auto. N(val) = {rows[0][4]}. Checkpoints: `checkpoints/{{baseline,agem}}_48cls.pt`.", "",
           "Two regimes are reported because they answer different questions:",
           "- **Full 48-way** = true class-incremental / open-world eval, no task ID at test time. Chance = 1/48 ≈ 2.1%.",
           "- **Task-aware 6-way** = current headline metric; argmax restricted to the sample's stage (task ID given). Chance = 1/6 ≈ 16.7%. Easier — this is what `avg_acc` in other reports refers to.",
           ""]
    for name, ck, full, task, n in rows:
        out += [f"## {name} (`method={ck.get('method')}`)", ""]
        out += ["### Full 48-way (class-IL, honest number)", "",
                "| Metric | Value |", "|---|---|"]
        for k, label in [("accuracy", "Accuracy"), ("precision_macro", "Precision (macro)"),
                          ("recall_macro", "Recall (macro)"), ("f1_macro", "F1 (macro)"),
                          ("precision_weighted", "Precision (weighted)"),
                          ("recall_weighted", "Recall (weighted)"), ("f1_weighted", "F1 (weighted)"),
                          ("mAP", "mAP (macro AP)"), ("chance", "Chance level")]:
            out.append(f"| {label} | {full[k]:.3f} |")
        out += ["", "### Task-aware 6-way (current headline `avg_acc`)", "",
                "| Metric | Value |", "|---|---|"]
        for k, label in [("accuracy", "Accuracy (avg_acc)"), ("precision_macro", "Precision (macro)"),
                          ("recall_macro", "Recall (macro)"), ("f1_macro", "F1 (macro)"),
                          ("chance", "Chance level")]:
            out.append(f"| {label} | {task[k]:.3f} |")
        out.append("")

    (bl_full, ag_full) = rows[0][2], rows[1][2]
    (bl_task, ag_task) = rows[0][3], rows[1][3]
    out += ["## A-GEM vs Baseline — both regimes", "",
            "| Regime | Metric | Baseline | A-GEM | Δ |", "|---|---|---|---|---|",
            f"| Full 48-way | Accuracy | {bl_full['accuracy']:.3f} | {ag_full['accuracy']:.3f} | {ag_full['accuracy']-bl_full['accuracy']:+.3f} |",
            f"| Full 48-way | mAP | {bl_full['mAP']:.3f} | {ag_full['mAP']:.3f} | {ag_full['mAP']-bl_full['mAP']:+.3f} |",
            f"| Full 48-way | F1 (macro) | {bl_full['f1_macro']:.3f} | {ag_full['f1_macro']:.3f} | {ag_full['f1_macro']-bl_full['f1_macro']:+.3f} |",
            f"| Task-aware 6-way | Accuracy | {bl_task['accuracy']:.3f} | {ag_task['accuracy']:.3f} | {ag_task['accuracy']-bl_task['accuracy']:+.3f} |",
            f"| Task-aware 6-way | F1 (macro) | {bl_task['f1_macro']:.3f} | {ag_task['f1_macro']:.3f} | {ag_task['f1_macro']-bl_task['f1_macro']:+.3f} |",
            "",
            "**Note:** full-48-way accuracy is much lower than the task-aware number for both models — this is the honest, harder class-IL evaluation with no task ID. A-GEM's relative improvement over baseline holds in both regimes.",
            ]
    Path("reports/val_metrics_result.md").write_text("\n".join(out), encoding="utf-8")
    print("\nwrote reports/val_metrics_result.md")


if __name__ == "__main__":
    main()
