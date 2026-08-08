"""
Full 174-class SSv2 subset builder (vs. create_mini_subset.py's 48-class subset).

Now that raw video (COAD_VIDEO_DIR) and the full label package (train.json,
validation.json, labels.json -- all 174 classes) are available, this builds
train_full.json / val_full.json spanning ALL 174 classes, enabling a literal
TCD-style base-heavy reproduction (84 base + 90 incremental) instead of the
48-class approximation run_ssv2_pooling_protocol.py's HONEST LIMIT flagged.

Source: the labels zip's train.json/validation.json give {id, label, template,
placeholders} with no numeric class_id -- labels.json maps template -> id.
Unlike create_mini_subset.py's 48-class *_subset.json (which already carried
class_id/stage), this derives class_id here and assigns "stage" as a stable
sort key only (1..174 by class_id) -- actual base/incremental grouping for a
TCD-style split happens downstream, not in this subset.

Run: python3 scripts/create_ssv2_full_subset.py [--train-per-class 100] [--val-per-class 50]
"""
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import SUBSET_DIR, VIDEO_DIR  # noqa: E402

LABELS_DIR = Path("/tmp/ssv2_labels/labels")
SEED = 42


def build(split, src_name, per_class, template_to_id):
    with open(LABELS_DIR / src_name, encoding="utf-8") as f:
        raw = json.load(f)

    by_class = defaultdict(list)
    missing_video, missing_template = 0, 0
    for item in raw:
        # labels.json keys have no brackets ("Approaching something..."),
        # train/validation.json templates do ("Approaching [something]...").
        bare_template = item["template"].replace("[", "").replace("]", "")
        cid = template_to_id.get(bare_template)
        if cid is not None:
            cid = int(cid)   # labels.json values are strings; keep class_id an int like *_mini.json
        if cid is None:
            missing_template += 1
            continue
        if not (VIDEO_DIR / f"{item['id']}.webm").exists():
            missing_video += 1
            continue
        by_class[cid].append({
            "id": item["id"],
            "video_path": f"videos/{item['id']}.webm",
            "label": item["label"],
            "template": item["template"],
            "class_id": cid,
            "class_name": item["template"],
            "placeholders": item.get("placeholders", []),
        })

    rng = random.Random(SEED)
    out = []
    for cid in sorted(by_class, key=int):
        pool = by_class[cid]
        n = min(per_class, len(pool))
        out.extend(rng.sample(pool, n))

    out.sort(key=lambda x: (int(x["class_id"]), x["id"]))
    dst = SUBSET_DIR / f"{split}_full.json"
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"{split}: {len(out)} samples across {len(by_class)}/174 classes "
          f"(missing_video={missing_video}, missing_template={missing_template}) -> {dst}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-per-class", type=int, default=100)
    ap.add_argument("--val-per-class", type=int, default=50)
    args = ap.parse_args()

    if VIDEO_DIR is None:
        raise RuntimeError("COAD_VIDEO_DIR is not set.")
    if not LABELS_DIR.exists():
        raise RuntimeError(f"{LABELS_DIR} not found -- unzip the labels package there first.")

    with open(LABELS_DIR / "labels.json", encoding="utf-8") as f:
        template_to_id = json.load(f)
    print(f"labels.json: {len(template_to_id)} classes")

    build("train", "train.json", args.train_per_class, template_to_id)
    build("val", "validation.json", args.val_per_class, template_to_id)


if __name__ == "__main__":
    main()
