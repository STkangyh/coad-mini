#!/usr/bin/env bash
# Build a self-contained payload to carry to the Jetson, so board time is spent
# measuring rather than downloading.
#
# The loaner board may have no network, a captive portal, or a blocked HF; the
# bundle assumes none of them work. Verified locally with HF_HUB_OFFLINE=1.
#
# Usage:  ./scripts/jetson_bundle.sh [outdir]        # default: ./jetson_payload
set -euo pipefail

OUT="${1:-jetson_payload}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HF="${HF_HOME:-$HOME/.cache/huggingface}/hub"

mkdir -p "$OUT/hf_cache/hub"
cd "$ROOT"

echo "==> code + checkpoints"
# data/ and external/ are excluded on purpose; features are copied separately
# below because only --stream-sim needs them.
tar --exclude='.git' --exclude='data' --exclude='external' \
    --exclude='__pycache__' --exclude='*.pyc' --exclude="$OUT" \
    -czf "$OUT/coad-mini.tar.gz" .

echo "==> model weights (offline cache)"
for m in models--openai--clip-vit-base-patch32 models--apple--mobileclip_s0_timm; do
    if [ -d "$HF/$m" ]; then
        cp -RL "$HF/$m" "$OUT/hf_cache/hub/"   # -L: resolve symlinks into blobs
        echo "    $m"
    else
        echo "    MISSING $m -- run once online first, or --stream-sim/MobileCLIP will fail"
    fi
done

echo "==> UCF101 features (only needed for --stream-sim)"
if [ -d data/features_ucf101_b32 ]; then
    tar -czf "$OUT/features_ucf101_b32.tar.gz" data/features_ucf101_b32
else
    echo "    absent, skipping"
fi

cat > "$OUT/SETUP.md" <<'EOF'
# On the Jetson

    tar xzf coad-mini.tar.gz -C ~/coad-mini && cd ~/coad-mini
    export HF_HOME=$PWD/../hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    tar xzf ../features_ucf101_b32.tar.gz      # only for --stream-sim

CPU-only torch is enough for the headline number and installs from PyPI on
aarch64. Do NOT start with the CUDA wheel -- that is the slow, failure-prone path
and nothing in phases 1-3 needs it.

    pip install numpy pillow transformers torch --index-url https://pypi.org/simple

Then follow dev/bench_jetson.py --help. Fix the power mode FIRST; timings taken
with DVFS floating are not reproducible and not reportable.
EOF

du -sh "$OUT"
echo "==> $OUT ready"
