"""
Build an ImageNet-100 subset directory (symlinks) from a full ImageNet root.

The CIL literature's "ImageNet-100" is a fixed 100-class subset. Different code
bases ship slightly different lists; to stay explicit and reproducible we take
the class list from a text file (one WNID per line, e.g. n01440764) that you
provide — use the list from the paper/codebase you are comparing against
(e.g. the split shipped with UCIR/PODNet: https://github.com/hshustc/CVPR19_Incremental_Learning).

Usage:
  python3 benchmarks/pycil/make_imagenet100.py \
      --imagenet /path/to/imagenet --classes imagenet100_classes.txt \
      --out /path/to/imagenet100

Then: COAD_IMAGENET_DIR=/path/to/imagenet100 ./benchmarks/pycil/setup_imagenet.sh
"""
import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imagenet", required=True, help="full ImageNet root (train/ + val/)")
    ap.add_argument("--classes", required=True, help="text file: one WNID per line (100 lines)")
    ap.add_argument("--out", required=True, help="output subset root")
    args = ap.parse_args()

    src = Path(args.imagenet)
    out = Path(args.out)
    wnids = [l.strip() for l in open(args.classes) if l.strip()]
    assert len(wnids) == 100, f"expected 100 classes, got {len(wnids)}"

    for split in ("train", "val"):
        (out / split).mkdir(parents=True, exist_ok=True)
        missing = []
        for w in wnids:
            s, d = src / split / w, out / split / w
            if not s.is_dir():
                missing.append(w)
                continue
            if not d.exists():
                d.symlink_to(s.resolve())
        if missing:
            raise SystemExit(f"[{split}] missing class dirs: {missing[:5]}{'...' if len(missing) > 5 else ''}")
        print(f"[{split}] linked {len(wnids)} classes")
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
