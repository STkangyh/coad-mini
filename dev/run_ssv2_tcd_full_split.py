"""
Literal TCD-style SSv2 split: 84 base classes + 90 incremental (9 sessions of
10), on the FULL 174-class subset -- what run_base_heavy_split.py's docstring
flagged as unreachable ("no raw-video access to the 126 classes outside our
curated 48-class subset"). That blocker is gone (ssv2_video_access_result.md):
raw video was downloaded and data/features_full/ has all 174 classes extracted.

This mirrors run_base_heavy_split.py's method (FeCAM vs GRU+A-GEM, true
class-IL eval, per-class exemplar memory) but at TCD's actual scale instead of
the 48-class approximation, and uses TCD's own reported exemplar rate
(20/class, from sota_positioning_brief.md's literature table) instead of this
repo's derived 8.33/class.

HONEST LIMIT: TCD's paper does not appear to publish which specific 84 of the
174 classes are "base" (or if it does, we have not located the exact list) --
so the base/incremental assignment here is a random 84/90 split, repeated
across seeds (same spirit as run_live_query_sim.py's class-order seeding) to
avoid the result depending on one arbitrary partition. This reproduces TCD's
STRUCTURE and SCALE (84 base, 90 incremental in 10-class sessions, exact
class count) -- not necessarily its literal class assignment.

Run: python3 dev/run_ssv2_tcd_full_split.py [--seeds 0 1 2] [--skip-gru]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn

from src.models.gru_detector import GRUDetector
from src.models.fecam_head import POOLINGS, FeCAMHead
from src.trainer import load_samples, set_seed, train_epoch
from src.utils.gem import AGEM
from src.utils.provenance import save_results

FEATURE_DIR = Path("data/features_full")
FEATURE_DIM = 512
N_CLASSES = 174
N_BASE = 84
INC_SESSION_SIZE = 10        # TCD's "10x9" variant (the other is 5x18)
EXEMPLARS_PER_CLASS = 20      # TCD's own reported rate, not our derived 8.33
EPOCHS = 15
HIDDEN_DIM = 256
REPLAY_RATIO = 0.25
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

train_all = load_samples("data/subset/train_full.json")
val_all = load_samples("data/subset/val_full.json")


def filter_by(samples, class_ids):
    return [s for s in samples if s["class_id"] in class_ids]


def make_sessions(seed):
    order = np.random.RandomState(seed).permutation(N_CLASSES)
    base = list(order[:N_BASE])
    rest = order[N_BASE:]
    incs = [list(rest[i:i + INC_SESSION_SIZE]) for i in range(0, len(rest), INC_SESSION_SIZE)]
    return [base] + incs


def load_val_features(pooling="mean"):
    pool_fn = POOLINGS[pooling]
    X, y = [], []
    for s in val_all:
        p = FEATURE_DIR / "val" / f"{s['id']}.npy"
        if p.exists():
            X.append(pool_fn(np.load(p).astype(np.float64)))
            y.append(s["class_id"])
    return np.stack(X).astype(np.float64), np.array(y)


def load_train_by_class(pooling="mean"):
    pool_fn = POOLINGS[pooling]
    by_class = {c: [] for c in range(N_CLASSES)}
    for s in train_all:
        p = FEATURE_DIR / "train" / f"{s['id']}.npy"
        if p.exists():
            by_class[s["class_id"]].append(pool_fn(np.load(p).astype(np.float64)))
    return {c: np.stack(v).astype(np.float64) for c, v in by_class.items() if v}


def eval_fecam_classIL(head, seen_classes, Xva, yva):
    seen = set(seen_classes)
    mask = np.isin(yva, list(seen))
    if mask.sum() == 0:
        return 0.0
    s = head.scores(Xva[mask])
    not_seen = [c for c in range(N_CLASSES) if c not in seen]
    s[:, not_seen] = -1e18
    return float((s.argmax(axis=1) == yva[mask]).mean())


def run_fecam(sessions, Xtr_by_class, Xva, yva, pooling="mean"):
    dim = next(iter(Xtr_by_class.values())).shape[1]
    head = FeCAMHead(feature_dim=dim, max_classes=N_CLASSES, pooling=pooling,
                     few_shot_correction=True)   # matches the served config
    seen, accs = [], []
    t0 = time.perf_counter()
    for cids in sessions:
        cids = [c for c in cids if c in Xtr_by_class]
        X = np.concatenate([Xtr_by_class[c] for c in cids])
        y = np.concatenate([np.full(len(Xtr_by_class[c]), c) for c in cids])
        head.observe(X, y)
        seen.extend(cids)
        accs.append(eval_fecam_classIL(head, seen, Xva, yva))
    return {"avg_inc": float(np.mean(accs)), "last": accs[-1],
            "per_session": accs, "train_s": time.perf_counter() - t0}


@torch.no_grad()
def eval_gru_classIL(model, seen_classes):
    model.eval()
    correct = total = 0
    seen = set(seen_classes)
    for s in val_all:
        if s["class_id"] not in seen:
            continue
        p = FEATURE_DIR / "val" / f"{s['id']}.npy"
        if not p.exists():
            continue
        x = torch.tensor(np.load(p), dtype=torch.float32).unsqueeze(0).to(device)
        logits, _ = model(x, None)
        mask = torch.full((N_CLASSES,), -1e18, device=device)
        mask[list(seen)] = 0.0
        pred = int((logits[0] + mask).argmax().item())
        total += 1
        correct += int(pred == s["class_id"])
    return correct / total if total else 0.0


def run_gru_agem(sessions, seed):
    set_seed(seed)
    model = GRUDetector(feature_dim=FEATURE_DIM, hidden_dim=HIDDEN_DIM,
                        num_classes=N_CLASSES).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    gem = AGEM(mem_per_stage=EXEMPLARS_PER_CLASS * len(sessions[0]),
               selection="balanced", replay_ratio=REPLAY_RATIO)

    seen, accs = [], []
    t0 = time.perf_counter()
    for cids in sessions:
        session_train = filter_by(train_all, cids)
        for _ in range(EPOCHS):
            gem.precompute_ref(model, device)
            train_epoch(model, session_train, FEATURE_DIR / "train",
                        criterion, optimizer, device, orth=gem)
        seen.extend(cids)
        accs.append(eval_gru_classIL(model, seen))
        gem.mem_per_stage = round(EXEMPLARS_PER_CLASS * len(cids))
        gem.add_stage(session_train, FEATURE_DIR / "train", model=model, device=device)
        print(f"    session done ({len(seen)}/{N_CLASSES} classes seen, "
              f"acc={accs[-1]:.3f}, {time.perf_counter()-t0:.0f}s)", flush=True)
    return {"avg_inc": float(np.mean(accs)), "last": accs[-1],
            "per_session": accs, "train_s": time.perf_counter() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--skip-gru", action="store_true",
                    help="FeCAM only -- GRU+A-GEM at this scale is far slower")
    ap.add_argument("--pooling", choices=list(POOLINGS), default="mean",
                    help="FeCAM pooling -- chunks4 is the served config, mean was the "
                         "original GRU-comparison run")
    args = ap.parse_args()

    print(f"device: {device} | {N_BASE} base + {(N_CLASSES-N_BASE)//INC_SESSION_SIZE}"
          f"x{INC_SESSION_SIZE} incremental, exemplars/class={EXEMPLARS_PER_CLASS} (TCD's own rate) "
          f"| FeCAM pooling={args.pooling}")
    Xtr_by_class = load_train_by_class(args.pooling)
    Xva, yva = load_val_features(args.pooling)
    print(f"train classes with features: {len(Xtr_by_class)}/{N_CLASSES}, "
          f"val samples: {len(yva)}\n")

    results = {"fecam": [], "gru_agem": []}
    for seed in args.seeds:
        sessions = make_sessions(seed)
        sizes = [len(s) for s in sessions]
        print(f"=== seed {seed} | session sizes: {sizes} ===")

        r_fecam = run_fecam(sessions, Xtr_by_class, Xva, yva, args.pooling)
        curve = " ".join(f"{a:.3f}" for a in r_fecam["per_session"])
        print(f"  FeCAM (no replay)      avg_inc={r_fecam['avg_inc']:.3f} "
              f"last={r_fecam['last']:.3f} ({r_fecam['train_s']:.1f}s)  {curve}", flush=True)
        results["fecam"].append(r_fecam)

        if not args.skip_gru:
            r_gru = run_gru_agem(sessions, seed)
            curve = " ".join(f"{a:.3f}" for a in r_gru["per_session"])
            print(f"  GRU+A-GEM              avg_inc={r_gru['avg_inc']:.3f} "
                  f"last={r_gru['last']:.3f} ({r_gru['train_s']:.1f}s)  {curve}")
            print(f"  -> FeCAM lead          avg_inc={r_fecam['avg_inc']-r_gru['avg_inc']:+.3f} "
                  f"last={r_fecam['last']-r_gru['last']:+.3f}\n", flush=True)
            results["gru_agem"].append(r_gru)

    summary = {
        "fecam_avg_inc": float(np.mean([r["avg_inc"] for r in results["fecam"]])),
        "fecam_last": float(np.mean([r["last"] for r in results["fecam"]])),
    }
    if not args.skip_gru:
        summary["gru_avg_inc"] = float(np.mean([r["avg_inc"] for r in results["gru_agem"]]))
        summary["gru_last"] = float(np.mean([r["last"] for r in results["gru_agem"]]))
        summary["fecam_lead_avg_inc"] = summary["fecam_avg_inc"] - summary["gru_avg_inc"]
        summary["fecam_lead_last"] = summary["fecam_last"] - summary["gru_last"]
    print("SUMMARY", summary)

    out = Path(f"reports/ssv2_tcd_full_split_{args.pooling}_raw.json")
    save_results(out, {"seeds": args.seeds, "pooling": args.pooling,
                       "summary": summary, "per_seed": results})
    print(f"raw -> {out}")


if __name__ == "__main__":
    main()
