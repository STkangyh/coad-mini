# PyCIL benchmark setup (CIFAR-100 & ImageNet)

Standard-benchmark comparison infrastructure for the paper: run [PyCIL](https://github.com/G-U-N/PyCIL)'s
classic CIL baselines AND our edge-friendly analytic heads on the **same splits**,
so results are directly comparable.

## Layout

```
benchmarks/pycil/
├── setup_pycil.sh              # clone external/PyCIL (pinned) + apply non-CUDA patches
├── cuda_shim.py                # .cuda() -> MPS/CPU fallback (injected into PyCIL main.py)
├── mps_device.patch            # reference: trainer._set_device "mps" support
├── exps/                       # our experiment configs (PyCIL-format JSON)
│   ├── simplecil_cifar100_smoke.json   # 2-epoch smoke, verified end-to-end on this Mac
│   └── simplecil_imagenet100.json      # ready once ImageNet data is wired
├── run_analytic_cifar100.py    # OUR heads (NCM/SLDA/FeCAM) on PyCIL's exact splits
├── setup_imagenet.sh           # wire an existing ImageNet dir into PyCIL
└── make_imagenet100.py         # build the 100-class subset via symlinks
```

`external/PyCIL` is a vendored checkout (gitignored); `setup_pycil.sh` recreates it
with all patches from scratch.

## Quick start

```bash
./benchmarks/pycil/setup_pycil.sh

# PyCIL baseline (smoke): SimpleCIL on CIFAR-100, b0=50 inc=10, 2 epochs, MPS
cd external/PyCIL && PYTORCH_ENABLE_MPS_FALLBACK=1 \
  python3 main.py --config ../../benchmarks/pycil/exps/simplecil_cifar100_smoke.json

# Our analytic heads on the SAME splits (frozen CLIP B/32, ~3 min incl. extraction)
python3 benchmarks/pycil/run_analytic_cifar100.py             # b0=50 inc=10
python3 benchmarks/pycil/run_analytic_cifar100.py --init 10 --inc 10
```

CIFAR-100 downloads automatically. If the U-Toronto server is slow (~38 kB/s
observed), fetch the identical tarball (md5 `eb9058c3a382ffc7106e4002c42a8d85`)
from a mirror and drop it at `external/PyCIL/data/cifar-100-python.tar.gz`:

```bash
curl -L -o external/PyCIL/data/cifar-100-python.tar.gz \
  https://data.brainchip.com/dataset-mirror/cifar100/cifar-100-python.tar.gz
```

## ImageNet

Data is NOT downloadable automatically (license). Once you have ImageNet locally:

```bash
# full ImageNet-1000: point PyCIL at it
COAD_IMAGENET_DIR=/path/to/imagenet ./benchmarks/pycil/setup_imagenet.sh

# ImageNet-100 subset: build symlink dir from a 100-class WNID list first
python3 benchmarks/pycil/make_imagenet100.py \
  --imagenet /path/to/imagenet --classes imagenet100_classes.txt --out /path/to/imagenet100
COAD_IMAGENET_DIR=/path/to/imagenet100 ./benchmarks/pycil/setup_imagenet.sh
```

Use the WNID list from the codebase you compare against (e.g. UCIR/PODNet's split)
and record which one in the paper.

## Non-CUDA (this Mac) notes

- PyCIL supports CPU natively via `"device": [-1]`; our patch adds `"device": ["mps"]`.
- Many PyCIL internals hardcode `.cuda()` — `cuda_shim.py` redirects those to
  MPS/CPU globally. No effect on CUDA machines.
- **Upstream bug we patch (not Mac-specific).** PyCIL's end-of-run accuracy matrix
  allocates `[task+1, task+1]`, assuming the number of class groups equals the
  number of tasks. That only holds when `init_cls == increment`. With our
  `b0=50 inc=10` config there are **6 tasks but 10 class groups**, so the run dies
  with `could not broadcast input array from shape (7,) into shape (6,)` —
  *after* training and evaluation have finished and the top1/top5 curves and
  Average Accuracy have already printed. `setup_pycil.sh` sizes the table by the
  real group count instead, so the forgetting matrix prints too. Reduces to the
  original behaviour when `init_cls == increment`.
- Classic baselines (iCaRL/DER/FOSTER: 160–200 epochs of ResNet training per run)
  are **impractical on this laptop**; run them on a CUDA box with the same configs.
  The smoke config proves the pipeline; our analytic heads run in seconds anywhere.

## Fair-comparison caveat (for the paper)

Our heads use a **frozen web-pretrained CLIP encoder** — that's the
pretrained-model-based CIL track (SimpleCIL/RanPAC/FeCAM), not the
train-ResNet-from-scratch track most PyCIL classics belong to. Compare within
the PTM track, or report both tracks explicitly labeled; never present a frozen-CLIP
number as beating a from-scratch method "like for like".
