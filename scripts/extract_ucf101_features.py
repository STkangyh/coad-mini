"""
CLIP feature extraction for UCF101, on the official recognition split.

Why UCF101: it is the one benchmark every paper in the SSv2-CIL lineage we
audited reports under the same TCD protocol (TCD ICCV'21, STSP ECCV'24, CSTA,
ESSENTIAL ICCV'25), so a number here can be placed directly beside theirs --
unlike our 48-class SSv2 subset, where we can only argue about orderings.
It is also *static-biased* (ESSENTIAL's term), the regime a frozen-CLIP
mean-pool head should suit, whereas SSv2 is temporal-biased.

Reuses the exact frame-sampling and encoder path as the SSv2 pipeline
(scripts/extract_clip_features.py) so the two feature sets are comparable:
16 uniformly sampled frames -> CLIP ViT-B/32 vision tower -> L2-normalized
512-d per frame.

Output
  data/features_ucf101_b32/{train,test}/<video_id>.npy   (16, 512) float32
  data/features_ucf101_b32/manifest.json                 ids, labels, splits

Usage
  python3 scripts/extract_ucf101_features.py                 # both splits
  python3 scripts/extract_ucf101_features.py --split test    # one split
  python3 scripts/extract_ucf101_features.py --limit 50      # quick smoke
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoProcessor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.extract_clip_features import extract, sample_frames  # noqa: E402

UCF_DIR = ROOT / "data/ucf101"
VIDEO_DIR = UCF_DIR / "UCF-101"
LIST_DIR = UCF_DIR / "ucfTrainTestlist"
OUT_DIR = ROOT / "data/features_ucf101_b32"
MODEL_NAME = "openai/clip-vit-base-patch32"
N_FRAMES = 16


def load_class_index() -> dict[str, int]:
    """classInd.txt is 1-based; we store 0-based ids."""
    idx = {}
    for line in (LIST_DIR / "classInd.txt").read_text().splitlines():
        if not line.strip():
            continue
        num, name = line.split()
        idx[name] = int(num) - 1
    return idx


def load_split(split: str, class_index: dict[str, int]) -> list[dict]:
    """Official split 1. trainlist carries labels; testlist does not, so the
    label comes from the directory component (which is the class name)."""
    fname = "trainlist01.txt" if split == "train" else "testlist01.txt"
    samples = []
    for line in (LIST_DIR / fname).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rel = line.split()[0]                      # "Class/v_xxx.avi"
        cls_name = rel.split("/")[0]
        samples.append({
            "id": Path(rel).stem,
            "rel": rel,
            "class_id": class_index[cls_name],
            "class_name": cls_name,
        })
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", nargs="+", default=["train", "test"])
    ap.add_argument("--num-frames", type=int, default=N_FRAMES)
    ap.add_argument("--limit", type=int, default=None, help="smoke-test only")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if not VIDEO_DIR.exists():
        raise SystemExit(f"missing {VIDEO_DIR} — run scripts/get_ucf101.sh --extract")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device} | model: {MODEL_NAME} | frames/video: {args.num_frames}")
    model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()
    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    class_index = load_class_index()
    manifest = {"model": MODEL_NAME, "num_frames": args.num_frames,
                "n_classes": len(class_index), "splits": {}}

    for split in args.split:
        samples = load_split(split, class_index)
        if args.limit:
            samples = samples[:args.limit]
        out = OUT_DIR / split
        out.mkdir(parents=True, exist_ok=True)

        done = failed = 0
        t0 = time.perf_counter()
        for i, s in enumerate(samples):
            dst = out / f"{s['id']}.npy"
            if dst.exists() and not args.overwrite:
                done += 1
                continue
            try:
                frames = sample_frames(VIDEO_DIR / s["rel"], args.num_frames)
                np.save(dst, extract(frames, model, processor, device))
                done += 1
            except Exception as e:                 # keep going; report at the end
                failed += 1
                if failed <= 5:
                    print(f"  FAIL {s['rel']}: {e}")
            if (i + 1) % 200 == 0:
                el = time.perf_counter() - t0
                rate = (i + 1) / el
                print(f"  [{split}] {i+1}/{len(samples)}  {el:6.0f}s  "
                      f"{rate:4.1f} vid/s  eta {(len(samples)-i-1)/rate/60:5.1f}m",
                      flush=True)

        manifest["splits"][split] = {
            "n": len(samples), "written": done, "failed": failed,
            "samples": [{"id": s["id"], "class_id": s["class_id"]} for s in samples],
        }
        print(f"[{split}] done={done} failed={failed} "
              f"({time.perf_counter()-t0:.0f}s)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest))
    print(f"manifest -> {OUT_DIR/'manifest.json'}")


if __name__ == "__main__":
    main()
