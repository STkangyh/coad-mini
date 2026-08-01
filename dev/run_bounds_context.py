"""
What do our accuracy numbers mean? Bracket them with a floor and a ceiling.

Audit item 5. We report 88.84 on UCF101 and 23.6 on SSv2 without anything to
compare against except other papers, whose protocols and backbones differ. Two
reference points computed on OUR features settle what those numbers are worth:

  FLOOR   CLIP zero-shot. Frozen CLIP already classifies these videos with no
          training at all, via its text tower. Anything at or below this means
          our head contributed nothing.

  CEILING A linear probe trained jointly on every class at once with backprop.
          This is the same frozen features, no continual constraint, no
          closed-form restriction -- the best a linear decision rule can do here.

Between them sit two versions of our own head, and the gap between them is the
price of the incremental protocol itself:

  FeCAM joint     all classes observed at once
  FeCAM class-IL  the deployed setting -- sessions, no task ID at test

Zero-shot needs CLIP's joint image-text space, so it uses the 512-d mean-pooled
embedding (the standard way to zero-shot a video). Everything else is reported
on both mean-512 and the deployed chunks4-2048, which also shows whether the
pooling helps a *trained* classifier or only helps FeCAM.

The probe's regularisation strength is chosen on a held-out slice of train, never
on test -- the same discipline the pooling study needed (see
reports/ssv2_temporal_pooling_result.md §10).

Run: python3 dev/run_bounds_context.py [--bench ucf101 ssv2]
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, ".")

from src.models.fecam_head import POOLINGS, FeCAMHead  # noqa: E402
from src.trainer import load_samples  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/bounds_context_raw.json"
PROBE_C = [0.01, 0.1, 1.0, 10.0]


# ── prompts ──────────────────────────────────────────────────────────────────
def ucf101_labels():
    """101 CamelCase names -> readable phrases ('ApplyEyeMakeup' -> 'apply eye makeup')."""
    lines = (ROOT / "data/ucf101/ucfTrainTestlist/classInd.txt").read_text().split()
    names = [lines[i] for i in range(1, len(lines), 2)]
    return [re.sub(r"(?<!^)(?=[A-Z])", " ", n).lower() for n in names]


def ssv2_labels():
    """Template strings with the placeholder filled in."""
    d = json.loads((ROOT / "checkpoints/class_labels.json").read_text())
    out = []
    for c in range(48):
        e = d[str(c)]
        t = e["template"] if isinstance(e, dict) else str(e)
        out.append(t.replace("[something]", "something").replace("[", "").replace("]", "").lower())
    return out


PROMPTS = ["a photo of a person {}.", "a video of {}.", "{}"]


@torch.no_grad()
def zero_shot_weights(labels):
    """Text embeddings averaged over prompt templates -> (C, 512), L2-normalised."""
    from transformers import AutoModel, AutoTokenizer
    model = AutoModel.from_pretrained("openai/clip-vit-base-patch32").eval()
    tok = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    W = []
    for lab in labels:
        texts = [p.format(lab) for p in PROMPTS]
        inp = tok(texts, return_tensors="pt", padding=True)
        # text_model + text_projection, mirroring exactly how the cached image
        # features were produced (vision_model + visual_projection). This
        # transformers version's get_text_features returns an output object, not
        # a tensor, so going through the two modules is also the reliable path.
        f = model.text_projection(model.text_model(**inp).pooler_output)
        f = f / f.norm(dim=-1, keepdim=True)
        m = f.mean(0)
        W.append((m / m.norm()).numpy())
    return np.stack(W).astype(np.float64)


# ── data ─────────────────────────────────────────────────────────────────────
def load_ucf101():
    d = ROOT / "data/features_ucf101_b32"
    man = json.loads((d / "manifest.json").read_text())
    out = {}
    for split in ("train", "test"):
        W, y = [], []
        for s in man["splits"][split]["samples"]:
            p = d / split / f"{s['id']}.npy"
            if p.exists():
                W.append(np.load(p).astype(np.float32))
                y.append(s["class_id"])
        out[split] = (np.stack(W), np.array(y))
    return out, ucf101_labels(), 101


def load_ssv2():
    d = ROOT / "data/features"
    out = {}
    for split, sub in (("train", "train"), ("test", "val")):
        W, y = [], []
        for s in load_samples(f"data/subset/{'train' if split=='train' else 'val'}_mini.json"):
            p = d / sub / f"{s['id']}.npy"
            if p.exists():
                W.append(np.load(p).astype(np.float32))
                y.append(s["class_id"])
        out[split] = (np.stack(W), np.array(y))
    return out, ssv2_labels(), 48


BENCHES = {"ucf101": load_ucf101, "ssv2": load_ssv2}


def pool(W, name):
    fn = POOLINGS[name]
    return np.stack([fn(w.astype(np.float64)) for w in W])


# ── the four reference points ────────────────────────────────────────────────
def acc_zero_shot(Xte_512, yte, Wtxt):
    x = Xte_512 / np.linalg.norm(Xte_512, axis=1, keepdims=True)
    return float(((x @ Wtxt.T).argmax(axis=1) == yte).mean())


def acc_fecam(Xtr, ytr, Xte, yte, n_classes, sessions=None):
    head = FeCAMHead(feature_dim=Xtr.shape[1], max_classes=n_classes)
    if sessions is None:                       # joint: everything at once
        head.observe(Xtr, ytr)
    else:
        for cids in sessions:
            m = np.isin(ytr, cids)
            head.observe(Xtr[m], ytr[m])
    return float((head.scores(Xte).argmax(axis=1) == yte).mean())


def acc_linear_probe(Xtr, ytr, Xte, yte, seed=0):
    """Joint logistic regression; C picked on a held-out slice of TRAIN."""
    from sklearn.linear_model import LogisticRegression
    rng = np.random.RandomState(seed)
    hold = np.zeros(len(ytr), dtype=bool)
    for c in np.unique(ytr):
        idx = np.where(ytr == c)[0]
        hold[rng.choice(idx, max(1, len(idx) // 5), replace=False)] = True

    best, best_c = -1.0, None
    for C in PROBE_C:
        clf = LogisticRegression(C=C, max_iter=2000, n_jobs=-1)
        clf.fit(Xtr[~hold], ytr[~hold])
        a = clf.score(Xtr[hold], ytr[hold])
        if a > best:
            best, best_c = a, C
    clf = LogisticRegression(C=best_c, max_iter=2000, n_jobs=-1).fit(Xtr, ytr)
    return float(clf.score(Xte, yte)), best_c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", nargs="+", default=["ucf101", "ssv2"], choices=list(BENCHES))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                    help="class orders for the incremental run")
    args = ap.parse_args()

    results = {}
    for bench in args.bench:
        t0 = time.perf_counter()
        data, labels, n_classes = BENCHES[bench]()
        (Wtr, ytr), (Wte, yte) = data["train"], data["test"]
        print(f"\n{'='*76}\n{bench}: train {Wtr.shape}, test {Wte.shape}, {n_classes} classes")

        feats = {name: (pool(Wtr, name), pool(Wte, name)) for name in ("mean", "chunks4")}
        # TCD-style sessions for UCF101 (51 base + 10), uniform 6-class for SSv2
        base, inc = (51, 10) if bench == "ucf101" else (6, 6)

        Wtxt = zero_shot_weights(labels)
        zs = acc_zero_shot(feats["mean"][1], yte, Wtxt)
        print(f"\n{'reference point':34s} {'features':14s} {'training':22s} {'acc':>7s}")
        print("-" * 82)
        print(f"{'CLIP zero-shot (FLOOR)':34s} {'mean 512':14s} {'none':22s} {100*zs:6.2f}")

        row = {"zero_shot": zs}
        for fname, (Xtr, Xte) in feats.items():
            dim = Xtr.shape[1]
            joint = acc_fecam(Xtr, ytr, Xte, yte, n_classes)
            il = [acc_fecam(Xtr, ytr, Xte, yte, n_classes, sessions=(
                lambda o: [list(o[:base])] + [list(o[i:i + inc]) for i in range(base, n_classes, inc)]
            )(np.random.RandomState(s).permutation(n_classes))) for s in args.seeds]
            probe, C = acc_linear_probe(Xtr, ytr, Xte, yte)
            print(f"{'FeCAM class-IL' + (' (DEPLOYED)' if fname == 'chunks4' else ''):34s} "
                  f"{fname + ' ' + str(dim):14s} {'closed-form, sessions':22s} "
                  f"{100*np.mean(il):6.2f}")
            print(f"{'FeCAM joint':34s} {fname + ' ' + str(dim):14s} "
                  f"{'closed-form, all at once':22s} {100*joint:6.2f}")
            print(f"{'linear probe (CEILING)':34s} {fname + ' ' + str(dim):14s} "
                  f"{'backprop, all at once':22s} {100*probe:6.2f}   (C={C})")
            row[fname] = {"dim": dim, "fecam_classIL": float(np.mean(il)),
                          "fecam_classIL_seeds": il, "fecam_joint": joint,
                          "linear_probe": probe, "probe_C": C}

        d = row["chunks4"]
        print(f"\n  floor -> deployed : {100*(d['fecam_classIL'] - zs):+.2f} pp above zero-shot")
        print(f"  deployed -> joint : {100*(d['fecam_joint'] - d['fecam_classIL']):+.2f} pp "
              f"(cost of the incremental protocol)")
        print(f"  deployed -> ceiling: {100*(d['linear_probe'] - d['fecam_classIL']):+.2f} pp "
              f"(headroom left in these features)")
        print(f"  [{time.perf_counter()-t0:.0f}s]")
        results[bench] = row

    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nraw -> {OUT}")


if __name__ == "__main__":
    main()
