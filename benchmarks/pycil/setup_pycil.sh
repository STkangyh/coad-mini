#!/bin/bash
# Reproducible PyCIL setup for this repo (clone + local patches for non-CUDA Macs).
# Usage: ./benchmarks/pycil/setup_pycil.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$HERE/../.."
PYCIL="$ROOT/external/PyCIL"
PIN="f3509b8"   # commit this setup was verified against

if [ ! -d "$PYCIL" ]; then
  git clone https://github.com/G-U-N/PyCIL "$PYCIL"
fi
git -C "$PYCIL" checkout -q "$PIN" 2>/dev/null || echo "(pin $PIN not found; using current HEAD)"

python3 - "$PYCIL" <<'EOF'
import sys, shutil, pathlib
pycil = pathlib.Path(sys.argv[1])
here = pathlib.Path(__file__).resolve()  # unused; patches below

# 1) MPS device support in trainer._set_device
t = pycil / "trainer.py"
src = t.read_text()
if 'elif device == "mps"' not in src:
    old = '        if device == -1:\n            device = torch.device("cpu")\n        else:'
    new = '        if device == -1:\n            device = torch.device("cpu")\n        elif device == "mps":\n            device = torch.device("mps")\n        else:'
    assert old in src, "trainer.py anchor not found"
    t.write_text(src.replace(old, new, 1))
    print("patched trainer.py (mps device)")

# 2) simplecil hardcoded .cuda() -> self._device
s = pycil / "models/simplecil.py"
src = s.read_text()
if "data = data.cuda()" in src:
    src = src.replace("data = data.cuda()\n                label = label.cuda()",
                      "data = data.to(self._device)\n                label = label.to(self._device)")
    s.write_text(src)
    print("patched models/simplecil.py (.cuda -> _device)")

# 3) global cuda fallback shim (covers utils/inc_net.py etc.)
m = pycil / "main.py"
src = m.read_text()
if "import cuda_shim" not in src:
    m.write_text("import cuda_shim  # non-CUDA fallback (see benchmarks/pycil/)\n" + src)
    print("patched main.py (cuda_shim import)")
EOF
cp "$HERE/cuda_shim.py" "$PYCIL/cuda_shim.py"

echo ""
echo "Done. Smoke test:"
echo "  cd external/PyCIL && PYTORCH_ENABLE_MPS_FALLBACK=1 \\"
echo "    python3 main.py --config ../../benchmarks/pycil/exps/simplecil_cifar100_smoke.json"
