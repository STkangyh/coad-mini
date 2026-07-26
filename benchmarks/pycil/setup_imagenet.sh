#!/bin/bash
# Wire an existing ImageNet directory into PyCIL (which ships with an
# `assert 0, "You should specify the folder of your dataset"` placeholder).
#
# Usage:
#   COAD_IMAGENET_DIR=/path/to/imagenet ./benchmarks/pycil/setup_imagenet.sh
#
# Expected layout (standard torchvision ImageFolder):
#   $COAD_IMAGENET_DIR/train/n01440764/*.JPEG ...
#   $COAD_IMAGENET_DIR/val/n01440764/*.JPEG ...
#
# For ImageNet-100 (the common CIL subset), build a subset dir first with
# make_imagenet100.py in this folder, then point COAD_IMAGENET_DIR at it.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PYCIL="$HERE/../../external/PyCIL"
DATA_PY="$PYCIL/utils/data.py"

if [ -z "${COAD_IMAGENET_DIR:-}" ]; then
  echo "ERROR: set COAD_IMAGENET_DIR to your ImageNet root (with train/ and val/)"; exit 1
fi
if [ ! -d "$COAD_IMAGENET_DIR/train" ] || [ ! -d "$COAD_IMAGENET_DIR/val" ]; then
  echo "ERROR: $COAD_IMAGENET_DIR must contain train/ and val/ subdirs"; exit 1
fi
if [ ! -f "$DATA_PY" ]; then
  echo "ERROR: PyCIL checkout not found at $PYCIL (clone it first — see README.md)"; exit 1
fi

python3 - "$DATA_PY" "$COAD_IMAGENET_DIR" <<'EOF'
import sys
data_py, root = sys.argv[1], sys.argv[2].rstrip("/")
src = open(data_py).read()
src = src.replace('assert 0, "You should specify the folder of your dataset"\n        ', "")
src = src.replace("[DATA-PATH]/train/", f"{root}/train/")
src = src.replace("[DATA-PATH]/val/", f"{root}/val/")
open(data_py, "w").write(src)
print(f"patched {data_py} -> {root}")
EOF

echo "Done. Run e.g.:"
echo "  cd external/PyCIL && python3 main.py --config ../../benchmarks/pycil/exps/simplecil_imagenet100.json"
