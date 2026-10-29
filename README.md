# coad-mini

[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-md.svg)](https://huggingface.co/spaces/Yhoon-3/coad-mini)
[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-live-2ea44f?logo=github)](https://stkangyh.github.io/coad-mini/)

**▶ [Try the live demo](https://yhoon-3-coad-mini.hf.space)** · **[Project page](https://stkangyh.github.io/coad-mini/)** · **[Research log](reports/RESEARCH_LOG.md)**

Class-incremental video action recognition that runs on a CPU without backpropagation. A frozen CLIP ViT-B/32
encoder, an **order-preserving temporal pooling** and a closed-form **FeCAM** covariance head learn new action
classes by updating running statistics: no gradient steps, no stored exemplars, real-time on a laptop CPU.

```
video -> 16 frames -> frozen CLIP ViT-B/32 (512-d per frame)
      -> order-preserving pooling: k segment means (+ adjacent differences)
      -> FeCAM head: per-class running mean + shared covariance, Mahalanobis scoring
```

## Results

**UCF101, TCD protocol** (51 base classes, average incremental accuracy %, 3 seeds). Competitor numbers are each
paper's own reported results.

| Method | Exemplars | Trained on UCF101 | 10×5 | 5×10 | 2×25 |
|---|---|---|--:|--:|--:|
| TCD (ICCV'21) | 5 videos / class | TSM backbone, fine-tuned | 74.89 | 73.43 | 72.19 |
| FrameMaker (NeurIPS'22) | 40 condensed frames / class | TSM backbone, fine-tuned | 78.13 | 76.38 | 75.77 |
| STSP (ECCV'24) | 0 | TSM backbone, null-space projected | 81.15 | 82.84 | 79.25 |
| ST-Prompt (ICCV'23) | 0 | space-time prompts, CLIP frozen | 84.75 | 85.54 | 85.67 |
| **coad-mini** | **0** | **nothing (closed-form head)** | **90.00** | **89.99** | **89.97** |
| ESSENTIAL (ICCV'25) | sparse frame features | temporal encoder + retrieval, 8×RTX 3090 | 95.1 | 93.9 | 93.3 |

- **Invariant to increment size.** From 10×5 to 2×25 the other methods move by −2.70 to +0.92pp; this one moves
  −0.03pp. At every class count the schedules share, accuracy is bit-identical across schedules and seeds (final
  accuracy 88.61 in all cases); the residual comes only from the extra evaluation points a finer schedule averages over.
- **Order-preserving pooling closes much of the motion gap.** On Something-Something V2, segment pooling raises final
  accuracy from 15.74 to 24.71 (48-class subset) and from 11.74 to 18.50 (standard 174-class split). Selecting the
  variant on held-out data gives +7.83pp; any order-preserving variant adds 7–9pp over mean pooling.
- **Real time on a CPU.** Enrolling and predicting together runs at 39.4 fps on an Apple M4 CPU (10 fps budget).
  A closed-form fit takes 42.4 ms per session against 270 s for a GRU fine-tune.
- **Known limit.** On SSv2's 174-class split ESSENTIAL reaches 48.9% against our 21.17%: a trained temporal encoder
  still extracts motion that closed-form pooling does not.

Every number links back to a script in `dev/` and a raw JSON result in `reports/`, stamped with the commit that
produced it. Start with [reports/RESEARCH_LOG.md](reports/RESEARCH_LOG.md), which also lists every claim we later
corrected.

## Demo app

`uvicorn app:app --port 8000` (or the included `Dockerfile`, Hugging Face Spaces-ready; see [docs/DEPLOY.md](docs/DEPLOY.md)).
The served head is FeCAM with `chunks4` pooling (`checkpoints/fecam_head.npz`, rebuilt by `dev/build_fecam_head.py`).

- **Realtime captioning** on an uploaded video or a live webcam (`POST /predict_rt`).
- **Few-shot enrollment**: add a new action from a few clips; it appears in the live stream immediately and persists
  across restarts (`POST /enroll`, `GET /classes`, `POST /reset_classes`). Details: [docs/few_shot_enrollment.md](docs/few_shot_enrollment.md).
- **Out-of-distribution score** on the stream with a calibrated threshold (`POST /predict_anomaly`). Details:
  [docs/anomaly_detection.md](docs/anomaly_detection.md).

The app still serves the earlier GRU baseline and GRU + A-GEM models side by side for comparison.

## Reproducing the main numbers

| Result | Script |
|---|---|
| UCF101 TCD protocol, pooling sweep | `dev/run_ucf101_pooling_sweep.py` |
| SSv2 174-class split | `dev/run_ssv2_tcd_174_pooling.py` |
| SSv2 pooling selection on held-out data | `dev/run_ssv2_pooling_protocol.py` |
| Difference-term ablation | `dev/run_diff_ablation.py` |
| Realtime enrol + predict throughput | `dev/bench_realtime_incremental.py` |
| Pooling × head accuracy / fps sweep | `dev/run_ap_fps_sweep.py` |
| FLOPs accounting | `dev/compute_flops.py` |

## History

The project began as GRU + A-GEM continual learning on a 48-class SSv2 subset (June 2026). Capacity, a larger
backbone, attention and state-space models did not improve it; only more data did
([data_scale](reports/data_scale_result.md), [gru_attention](reports/gru_attention_result.md),
[docs/research_log.md](docs/research_log.md)). Removing backprop entirely then beat it on every metric, which is
where the current method comes from.

## Repository layout

```text
coad-mini/
├── app.py                  # FastAPI demo server (FeCAM head + GRU/A-GEM for comparison)
├── config.py               # Shared constants and path configuration
├── src/
│   ├── models/
│   │   ├── fecam_head.py   # Closed-form FeCAM head + temporal pooling variants
│   │   ├── gru_detector.py # Earlier GRU classifier
│   │   └── gru_attention.py, ssm_detector.py
│   ├── enroll/few_shot.py  # Few-shot enrollment
│   ├── anomaly/detector.py # OOD scoring
│   ├── utils/              # A-GEM, provenance stamping of result files
│   └── trainer.py
├── dev/                    # Benchmark and analysis scripts behind the results
├── benchmarks/pycil/       # CIFAR-100 cross-check on PyCIL splits
├── experiments/            # Earlier GRU / A-GEM training scripts
├── static/index.html       # Demo UI
├── tests/                  # pytest suites (run in CI)
├── scripts/                # Data preparation + feature extraction
├── docs/                   # GitHub Pages site and module docs
├── reports/                # Result reports, raw JSON, RESEARCH_LOG.md
└── checkpoints/            # fecam_head.npz, baseline_48cls.pt, agem_48cls.pt, class_labels.json
```

## Requirements

- Python 3.10+
- `ffmpeg` and `ffprobe` available on `PATH` for the FastAPI upload demo
- Access to the Something-Something V2 videos if you want to regenerate features

Install Python dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Path configuration

The data-prep scripts no longer depend on hardcoded local paths. Set these environment variables when regenerating data:

```bash
export COAD_VIDEO_DIR=/path/to/20bn-something-something-v2
export COAD_LABELS_DIR=/path/to/labels
```

- `COAD_VIDEO_DIR` should contain the raw `.webm` videos
- `COAD_LABELS_DIR` should contain `train.json` and `validation.json`

## Quick start

### 1. Run the demo server with the included checkpoints

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Endpoints:

- `GET /` - demo UI
- `GET /health` - checkpoint/server status
- `GET /forgetting` - forgetting curves from saved checkpoints
- `POST /predict` - upload a video file
- `POST /predict_rt` - realtime frame-based inference
- `POST /predict_anomaly` - realtime inference + OOD/anomaly score
- `POST /enroll` - few-shot register a new action class (multipart: `label` + `files`); hot-swaps the live model
- `GET /classes` - list base + enrolled classes
- `POST /reset_classes` - remove enrolled actions, revert to the base 48

Responses include the FeCAM head's prediction under `fecam` next to the GRU models' predictions.

### 2. Recreate the mini subset manifests

```bash
python3 scripts/make_subset.py
python3 scripts/create_mini_subset.py
```

### 3. Extract CLIP features

```bash
python3 scripts/extract_clip_features.py
```

This writes per-video features under `data/features/{train,val}`.

### 4. Build the FeCAM head and run benchmarks

```bash
# Rebuild the served FeCAM head (pooling is stored in the checkpoint)
python3 dev/build_fecam_head.py chunks4

# Earlier GRU / A-GEM experiments
# Simple baseline run
python3 experiments/train_baseline.py

# Stage-based continual learning
python3 experiments/train_stage.py

# Multi-seed comparison
python3 experiments/run_stage_seeds.py

# Save checkpoints for demo
python3 experiments/save_checkpoints.py
```

## GitHub Pages release

This repository includes a GitHub Pages site under `docs/` and an automatic deployment workflow:

- Workflow: `.github/workflows/pages.yml`
- Published URL: `https://stkangyh.github.io/coad-mini/`

If this is your first Pages deploy, open **Settings → Pages** in the repository and ensure:

- **Source** is set to **GitHub Actions**

## Included artifacts vs ignored artifacts

This repo is set up to be GitHub-friendly:

- **Included:** source code, mini subset manifests, small checkpoints, and project docs
- **Ignored:** local virtual environments, extracted feature tensors, cache files, and large regenerated subset exports

If you plan to publish the repository, commit the code and lightweight metadata, but avoid committing the full extracted feature directory.

## Notes

- The served model is the FeCAM head. The GRU baseline, GRU + A-GEM and the older OrthGrad / COAD variant remain in the repository as the project's earlier stages.
- Training and benchmark scripts expect the mini subset JSON files and extracted features to exist under `data/`.
