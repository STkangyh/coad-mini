"""
Measure how different neighbouring segment means are at k=3 vs k=4.

The paper explains why the adjacent-difference term helps at k=3 but barely at
k=4 by saying finer segments leave neighbouring means already close, so c_{i+1}-c_i
carries little. Until now that was only drawn (arrow thickness, "not to scale").
This measures it on the same SSv2 48-class features and the same _segment_means
the diff ablation (dev/run_diff_ablation.py) uses.

Run: python3 dev/measure_segment_neighbor_gap.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.fecam_head import _segment_means  # noqa: E402
from src.trainer import load_samples  # noqa: E402
from src.utils.provenance import save_results  # noqa: E402

FEATURE_DIR = ROOT / "data/features"
KS = [3, 4]


def neighbour_stats(windows, k):
    l2, rel, cos = [], [], []
    for w in windows:
        c = _segment_means(w, k).reshape(k, -1)
        for a, b in zip(c[:-1], c[1:]):
            d = np.linalg.norm(b - a)
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            l2.append(d)
            rel.append(d / ((na + nb) / 2))
            cos.append(float(a @ b / (na * nb)))
    l2, rel, cos = map(np.asarray, (l2, rel, cos))
    return {"pairs": int(len(l2)),
            "l2_mean": float(l2.mean()), "l2_std": float(l2.std()),
            "rel_mean": float(rel.mean()), "rel_std": float(rel.std()),
            "cos_mean": float(cos.mean()), "cos_std": float(cos.std())}


def load_windows(split, samples):
    d = FEATURE_DIR / split
    return [np.load(d / f"{s['id']}.npy") for s in samples if (d / f"{s['id']}.npy").exists()]


def main():
    splits = {"train": load_samples(ROOT / "data/subset/train_mini.json"),
              "val": load_samples(ROOT / "data/subset/val_mini.json")}
    results = {}
    for split, samples in splits.items():
        windows = load_windows(split, samples)
        print(f"[{split}] {len(windows)} clips, window shape {windows[0].shape}")
        for k in KS:
            r = neighbour_stats(windows, k)
            results[f"{split}|k{k}"] = r
            print(f"  k={k}: pairs={r['pairs']:5d}  L2 {r['l2_mean']:.4f}±{r['l2_std']:.4f}  "
                  f"relative {r['rel_mean']:.4f}±{r['rel_std']:.4f}  "
                  f"cosine {r['cos_mean']:.4f}±{r['cos_std']:.4f}")
        a, b = results[f"{split}|k3"], results[f"{split}|k4"]
        print(f"  k=4 / k=3 ratio: L2 {b['l2_mean']/a['l2_mean']:.3f}  "
              f"relative {b['rel_mean']/a['rel_mean']:.3f}")

    out = ROOT / "reports/segment_neighbor_gap_raw.json"
    save_results(out, {"features": str(FEATURE_DIR.relative_to(ROOT)), "ks": KS, "results": results})
    print(f"raw -> {out}")


if __name__ == "__main__":
    main()
