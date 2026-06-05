"""
각 클래스당 N개 랜덤 샘플 → train_mini.json / val_mini.json 생성
"""

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import SUBSET_DIR, VIDEO_DIR

SAMPLES_PER_CLASS = 100
SEED = 42

random.seed(SEED)


def create_mini(split: str):
    if VIDEO_DIR is None:
        raise RuntimeError(
            "COAD_VIDEO_DIR is not set. Point it to the Something-Something V2 video directory."
        )
    src = SUBSET_DIR / f"{split}_subset.json"
    with open(src, encoding="utf-8") as f:
        data = json.load(f)

    # video 존재 여부 필터
    data = [
        d for d in data
        if (VIDEO_DIR / f"{d['id']}.webm").exists()
    ]

    # 클래스별 그룹화
    by_class = defaultdict(list)
    for item in data:
        by_class[item["class_id"]].append(item)

    mini = []
    for class_id in sorted(by_class):
        pool = by_class[class_id]
        n = min(SAMPLES_PER_CLASS, len(pool))
        sampled = random.sample(pool, n)
        mini.extend(sampled)
        print(f"  class {class_id} ({pool[0]['class_name']:<45s}): {n:3d} / {len(pool)}")

    # class_id でソート → stage 順 보장
    mini.sort(key=lambda x: (x["stage"], x["class_id"]))

    out = SUBSET_DIR / f"{split}_mini.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(mini, f, ensure_ascii=False, indent=2)

    print(f"  → {out}  ({len(mini)} samples)\n")
    return mini


if __name__ == "__main__":
    for split in ["train", "val"]:
        # val_subset.json 이 없으면 validation_subset.json 참조
        src = SUBSET_DIR / f"{split}_subset.json"
        if not src.exists():
            alt = SUBSET_DIR / "validation_subset.json"
            if split == "val" and alt.exists():
                import shutil
                shutil.copy(alt, src)

        print(f"=== {split} ===")
        create_mini(split)
