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

# 4) upstream bug: the end-of-run accuracy matrix assumes #class-groups == #tasks,
#    which only holds when init_cls == increment. With b0=50 inc=10 there are 6
#    tasks but 10 class groups, so the write goes out of bounds and the run dies
#    with "could not broadcast input array from shape (7,) into shape (6,)" AFTER
#    all training/eval finished. Size the table by the real group count instead.
t = pycil / "trainer.py"
src = t.read_text()

OLD = """        np_acctable = np.zeros([task + 1, task + 1])
        for idxx, line in enumerate({v}):
            idxy = len(line)
            np_acctable[idxx, :idxy] = np.array(line)
        np_acctable = np_acctable.T
        forgetting = np.mean((np.max(np_acctable, axis=1) - np_acctable[:, task])[:task])"""

NEW = """        n_tasks = len({v})
        n_groups = max(len(line) for line in {v})
        np_acctable = np.zeros([n_tasks, n_groups])
        for idxx, line in enumerate({v}):
            np_acctable[idxx, :len(line)] = np.array(line)
        np_acctable = np_acctable.T
        forgetting = np.mean(
            (np.max(np_acctable, axis=1) - np_acctable[:, n_tasks - 1])[:n_groups - 1])"""

n = 0
for var in ("cnn_matrix", "nme_matrix"):
    old, new = OLD.format(v=var), NEW.format(v=var)
    if old in src:
        src = src.replace(old, new, 1)
        n += 1
if n:
    t.write_text(src)
    print(f"patched trainer.py (accuracy-matrix sizing, {n} block(s))")
elif "n_groups = max(len(line) for line in cnn_matrix)" in src:
    print("trainer.py accuracy-matrix already patched")
else:
    print("NOTE: accuracy-matrix anchor not found; upstream may have changed")
EOF
cp "$HERE/cuda_shim.py" "$PYCIL/cuda_shim.py"

echo ""
echo "Done. Smoke test:"
echo "  cd external/PyCIL && PYTORCH_ENABLE_MPS_FALLBACK=1 \\"
echo "    python3 main.py --config ../../benchmarks/pycil/exps/simplecil_cifar100_smoke.json"
