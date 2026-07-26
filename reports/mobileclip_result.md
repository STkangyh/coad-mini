# MobileCLIP-S0 (edge vision encoder) — CPU inference latency finding

Generated: 2026-07 (auto). Motivated by `docs/project_focus.md` gap #2: our CLIP
ViT-B/32 encoder is the inference bottleneck (156ms/16-frame window on CPU), and
the edge-CL framing calls for testing a purpose-built lightweight encoder.

## Status

- ✅ **Encoder benchmarking** (params, CPU/MPS latency) — done, no video data needed.
- ✅ **Pipeline integration** — `scripts/extract_clip_features.py` now supports
  `--backbone mobileclip_s0`, ready to run.
- ⛔ **Downstream classification accuracy (GRU/A-GEM/FeCAM on MobileCLIP features)** —
  **RESOLVED (2026-07-27)** — the downstream accuracy comparison was blocked on raw
  video access; UCF101's 13,320 videos unblocked it. MobileCLIP-S0 matches CLIP B/32
  to within 0.21pp on UCF101 (88.63 vs 88.84 with FeCAM) while being 20x slower on
  CPU -- see reports/encoder_swap_result.md. Original blocker note below.

  **blocked**: this session's environment has no access to the raw Something-Something V2
  `.webm` videos (`COAD_VIDEO_DIR` from `docs/`-referenced setup is unavailable here).
  The already-extracted CLIP B/32 `.npy` features remain in `data/features/` (unaffected),
  but MobileCLIP requires a fresh extraction pass over the raw videos. **Run when video
  access is restored** — see §4 for the exact command.

## 1. Model

**apple/mobileclip_s0_timm** (Hugging Face, `TimmWrapperModel`) — the image-tower-only
release of Apple's MobileCLIP-S0 (Vasu et al., CVPR 2024, "MobileCLIP: Fast Image-Text
Models through Multi-Modal Reinforced Training"), designed for on-device (CoreML/ANE)
inference. No text tower or CLIP projection head in this checkpoint — used as a plain
image feature extractor, same role as CLIP B/32's `vision_model` in our pipeline.

| | CLIP ViT-B/32 (current) | MobileCLIP-S0 |
|---|---|---|
| Vision encoder params | 87.5M | **10.9M** (8× smaller) |
| Output feature dim | 512 | 1024 (pre-projection backbone feature) |
| Target runtime | generic (any) | CoreML / Apple Neural Engine |

## 2. CPU inference latency (16-frame window, this Mac, warmed up, n=8)

| Backbone | CPU (4 threads) | CPU (10 threads) | MPS |
|---|---|---|---|
| CLIP ViT-B/32 | 161ms | 158ms | — |
| **MobileCLIP-S0** | 2,092ms | 2,814ms (more threads = *worse*) | **125ms** |

- MobileCLIP-S0 is **13–18× slower than CLIP B/32 on generic CPU** despite having
  8× fewer parameters.
- On MPS it drops to 125ms — comparable to or faster than CLIP B/32's CPU number.
- Raising CPU thread count from 4→10 made MobileCLIP *slower* (2.09s→2.81s), consistent
  with thread-oversubscription overhead on ops (depthwise/grouped convolutions,
  reparameterized branches) that don't parallelize the way transformer matmuls do.

## 3. Interpretation — an honest, edge-relevant finding

**Parameter count / FLOPs do not predict real edge latency — the deployment runtime
has to have optimized kernels for the architecture.** MobileCLIP's efficiency claims
in Apple's paper are measured on CoreML/ANE, the hardware it was co-designed with.
Naive PyTorch CPU eager execution of its depthwise-conv/reparameterized blocks gets
none of that benefit and is instead slower than a well-optimized ViT (CLIP B/32's
transformer matmuls hit fast BLAS paths on generic CPUs).

This directly qualifies our project's edge-CL claims: **"CPU-only" is not one thing** —
an architecture chosen for FLOP-efficiency can still be a bad fit for a specific CPU
inference stack, while a "bigger" transformer can be the pragmatically faster choice
absent specialized kernels. For real embedded deployment (smart glasses / robot SoCs
with an NPU), MobileCLIP would need to run through CoreML (Apple silicon) or an
equivalent NPU-targeted runtime (e.g. ONNX Runtime with a vendor NPU execution
provider) to realize its design intent — plain `transformers`/PyTorch CPU inference,
as used elsewhere in this repo, is the wrong benchmark for it.

## 4. Downstream accuracy — how to complete this study

Once `COAD_VIDEO_DIR` points at the real SSv2 `.webm` files again:

```bash
# 1. Extract MobileCLIP-S0 features (pipeline already supports this backbone)
COAD_VIDEO_DIR=/path/to/20bn-something-something-v2 \
  python3 scripts/extract_clip_features.py --backbone mobileclip_s0
# writes data/features_mobileclip_s0/{train,val}/*.npy (1024-d), ~9500 videos.
# NOTE: at 2.1-2.8s/window on CPU this would take many hours; extracting via
# MPS (as this repo has done before for other backbones, see
# feature-extraction-env memory note) is strongly recommended for this step —
# the CPU number above is the deployment-relevant one, not the offline
# extraction one.

# 2. Compare against the CLIP B/32 main model on the same 8-stage protocol
python3 experiments/run_capacity_ablation.py \
  --configs best_current_h256 \
  --feature-dir data/features_mobileclip_s0 --feature-dim 1024 \
  --seeds 0 1 2 --epochs 15

# 3. Also worth running the backprop-free heads (dev/run_cpu_friendly_methods.py /
#    dev/run_modern_cpu_methods.py) on the new feature dir — FeCAM's fit is fast
#    enough (seconds) that re-running there is cheap once features exist.
```

Expected reference point: CLIP B/32 + GRU + A-GEM gives task-aware 0.387 /
full-48-way 0.105 (`reports/measured_evidence.md`); FeCAM on B/32 gives task-aware
0.410 / full-48-way 0.157 (`reports/cpu_friendly_methods_result.md`).

## 5. Code changes (done, committed)

- `scripts/extract_clip_features.py`: new `mobileclip_s0` `BackboneConfig` entry;
  `extract()` gained a `TimmWrapperModel` branch (uses `pooler_output` directly,
  since this checkpoint has no `vision_model`/`visual_projection` attributes like
  the CLIP-family models already handled).
- No changes to training/CL code — a new feature directory is a drop-in
  `--feature-dir`/`--feature-dim` swap for the existing harness, same as
  OpenCLIP L/14 and SigLIP were.

## References

- MobileCLIP: Vasu et al., CVPR 2024 — https://arxiv.org/abs/2311.17049
- HF checkpoint used: https://huggingface.co/apple/mobileclip_s0_timm
