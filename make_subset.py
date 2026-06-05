import json
import csv
from collections import defaultdict
from pathlib import Path

from config import LABELS_DIR, SUBSET_DIR

TARGET_TEMPLATES = [
    # Stage 1 – Open / Close / Handle (6)
    "Opening [something]",
    "Closing [something]",
    "Turning [something] upside down",
    "Putting [something] onto [something]",
    "Folding [something]",
    "Picking [something] up",

    # Stage 2 – Container actions (6)
    "Putting [something] into [something]",
    "Taking [something] out of [something]",
    "Putting [something] on a surface",
    "Taking [something] from [somewhere]",
    "Stuffing [something] into [something]",
    "Putting [something] next to [something]",

    # Stage 3 – Directional move vertical/depth (6)
    "Moving [something] up",
    "Moving [something] down",
    "Moving [something] towards the camera",
    "Moving [something] away from the camera",
    "Moving [something] and [something] closer to each other",
    "Moving [something] and [something] away from each other",

    # Stage 4 – Directional move horizontal / push-pull (6)
    "Pushing [something] from left to right",
    "Pushing [something] from right to left",
    "Pulling [something] from left to right",
    "Pulling [something] from right to left",
    "Pushing [something] with [something]",
    "Poking [something] so lightly that it doesn't or almost doesn't move",

    # Stage 5 – Cover / Throw / Drop (6)
    "Covering [something] with [something]",
    "Uncovering [something]",
    "Throwing [something]",
    "Squeezing [something]",
    "Throwing [something] against [something]",
    "Dropping [something] onto [something]",

    # Stage 6 – Push force / Hit / Tear (6)
    "Pushing [something] so that it slightly moves",
    "Pushing [something] so that it falls off the table",
    "Hitting [something] with [something]",
    "Tearing [something] into two pieces",
    "Poking [something] so it slightly moves",
    "Tearing [something] just a little bit",

    # Stage 7 – Lift / Drop / Fall (6)
    "Lifting [something] up completely without letting it drop down",
    "Lifting up one end of [something], then letting it drop down",
    "Lifting [something] up completely, then letting it drop down",
    "Lifting up one end of [something] without letting it drop down",
    "Lifting [something] with [something] on it",
    "[Something] falling like a rock",

    # Stage 8 – Pretend / Show (6)
    "Pretending to open [something] without actually opening it",
    "Pretending to pick [something] up",
    "Pretending to put [something] on a surface",
    "Showing that [something] is empty",
    "Pretending to take [something] from [somewhere]",
    "Showing that [something] is inside [something]",
]

# class_id 고정
TEMPLATE_TO_CLASS = {t: i for i, t in enumerate(TARGET_TEMPLATES)}

# Continual Learning stage 매핑 (8 stages × 6 classes = 48)
STAGE_MAP = {
    "Opening [something]":                    1,
    "Closing [something]":                    1,
    "Turning [something] upside down":        1,
    "Putting [something] onto [something]":   1,
    "Folding [something]":                    1,
    "Picking [something] up":                 1,

    "Putting [something] into [something]":   2,
    "Taking [something] out of [something]":  2,
    "Putting [something] on a surface":       2,
    "Taking [something] from [somewhere]":    2,
    "Stuffing [something] into [something]":  2,
    "Putting [something] next to [something]":2,

    "Moving [something] up":                              3,
    "Moving [something] down":                            3,
    "Moving [something] towards the camera":              3,
    "Moving [something] away from the camera":            3,
    "Moving [something] and [something] closer to each other": 3,
    "Moving [something] and [something] away from each other": 3,

    "Pushing [something] from left to right": 4,
    "Pushing [something] from right to left": 4,
    "Pulling [something] from left to right": 4,
    "Pulling [something] from right to left": 4,
    "Pushing [something] with [something]":   4,
    "Poking [something] so lightly that it doesn't or almost doesn't move": 4,

    "Covering [something] with [something]":  5,
    "Uncovering [something]":                 5,
    "Throwing [something]":                   5,
    "Squeezing [something]":                  5,
    "Throwing [something] against [something]":5,
    "Dropping [something] onto [something]":  5,

    "Pushing [something] so that it slightly moves":       6,
    "Pushing [something] so that it falls off the table":  6,
    "Hitting [something] with [something]":                6,
    "Tearing [something] into two pieces":                 6,
    "Poking [something] so it slightly moves":             6,
    "Tearing [something] just a little bit":               6,

    "Lifting [something] up completely without letting it drop down":   7,
    "Lifting up one end of [something], then letting it drop down":     7,
    "Lifting [something] up completely, then letting it drop down":     7,
    "Lifting up one end of [something] without letting it drop down":   7,
    "Lifting [something] with [something] on it":                       7,
    "[Something] falling like a rock":                                  7,

    "Pretending to open [something] without actually opening it":  8,
    "Pretending to pick [something] up":                           8,
    "Pretending to put [something] on a surface":                  8,
    "Showing that [something] is empty":                           8,
    "Pretending to take [something] from [somewhere]":             8,
    "Showing that [something] is inside [something]":              8,
}

SPLITS = ["train", "validation"]
OUT_DIR = SUBSET_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)


def match_template(template):
    for t in TARGET_TEMPLATES:
        if template == t:
            return t
    return None


def process_split(split):
    if LABELS_DIR is None:
        raise RuntimeError(
            "COAD_LABELS_DIR is not set. Point it to the directory containing train.json and validation.json."
        )
    path = LABELS_DIR / f"{split}.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    counts = defaultdict(int)
    subset = []

    for item in data:
        tmpl = item["template"]
        matched = match_template(tmpl)
        if matched:
            counts[matched] += 1
            subset.append({
                "id": item["id"],
                "video_path": f"videos/{item['id']}.webm",
                "label": item["label"],
                "template": tmpl,
                "class_id": TEMPLATE_TO_CLASS[matched],
                "class_name": matched,
                "stage": STAGE_MAP[matched],
                "placeholders": item.get("placeholders", []),
            })

    return counts, subset


# ── 카운트 출력 및 subset 저장 ──────────────────────────────────────────────
for split in SPLITS:
    counts, subset = process_split(split)

    out_json = OUT_DIR / f"{split}_subset.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(subset, f, ensure_ascii=False, indent=2)

    out_csv = OUT_DIR / f"{split}_subset.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "video_path", "label", "template", "class_id", "class_name", "stage"]
        )
        writer.writeheader()
        for item in subset:
            writer.writerow({k: item[k] for k in writer.fieldnames})

    print(f"\n=== {split} ===")
    total = 0
    for t in TARGET_TEMPLATES:
        c = counts.get(t, 0)
        total += c
        print(f"  {c:5d}  {t}")
    print(f"  -----")
    print(f"  {total:5d}  TOTAL")
    print(f"  → saved: {out_json.name}, {out_csv.name}")
