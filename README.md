# coad-mini

Compact continual-learning experiments for video action recognition on a 48-class subset of Something-Something V2. The project compares a plain sequential baseline against replay-based mitigation methods, mainly **A-GEM**, using **CLIP ViT-B/32 frame features** and a **GRU classifier**.

> **📓 연구 일지 (Research Log)** — 실험 설계부터 최종 결과까지, 코드에 담기지 않은 의사결정 과정을 기록했습니다.  
> OrthGrad 가설 → 실패 → A-GEM 전환 → Scaling → Ablation → Demo 까지의 흐름을 담고 있습니다.  
> → **[docs/research_log.md](docs/research_log.md)**

## What this repository contains

- **Training scripts** for baseline sequential learning, stage-wise continual learning, and multi-seed analysis
- **A FastAPI demo app** for comparing Baseline vs A-GEM predictions on uploaded videos
- **Few-shot enrollment** of brand-new action classes (48 → 48+K) without catastrophic forgetting — see [docs/few_shot_enrollment.md](docs/few_shot_enrollment.md)
- **Anomaly / novel-action (OOD) detection** on the prediction stream — see [docs/anomaly_detection.md](docs/anomaly_detection.md)
- **Saved checkpoints** for the 48-class setup
- **Mini subset manifests** used by the experiments

## Anomaly / OOD detection

Flag when the current action is out-of-distribution (an action the model was
never trained on, or a low-confidence / novel one). Scoring functions (max
softmax prob, predictive entropy, energy) follow a single convention —
**higher score == more OOD** — and `AnomalyScorer` adds a calibrated threshold
plus temporal smoothing. Calibrate with
`python3 experiments/calibrate_anomaly.py --metric energy --target-fpr 0.05`,
and query `POST /predict_anomaly` (base64 frames, same payload as
`/predict_rt`) for `{top1, top1_prob, anomaly_score, is_anomaly}`. Full
details: [docs/anomaly_detection.md](docs/anomaly_detection.md).

## Method overview

The pipeline is:

`video -> 16 uniformly sampled frames -> CLIP ViT-B/32 embeddings -> GRU classifier -> staged continual-learning evaluation`

Current default setup:

- **48 classes**
- **8 stages x 6 classes**
- **16 sampled frames per video**
- **512-dim CLIP visual features**

## Repository layout

```text
coad-mini/
├── app.py                      # FastAPI demo server
├── config.py                   # Shared constants and path configuration
├── experiments/                # Training and analysis scripts
│   ├── train_baseline.py       # Single-pass baseline training
│   ├── train_coad.py           # Older orthogonal-gradient experiment
│   ├── train_stage.py          # Stage-based continual learning comparison
│   ├── run_seeds.py            # Multi-seed baseline vs OrthGrad
│   ├── run_stage_seeds.py      # Multi-seed baseline vs A-GEM
│   ├── run_analysis.py         # A-GEM replay/memory ablations
│   └── save_checkpoints.py     # Train and save Baseline/A-GEM checkpoints
├── src/                        # Core library code
│   ├── models/
│   │   └── gru_detector.py
│   ├── utils/
│   │   ├── gem.py
│   │   └── orthogonal_grad.py
│   └── trainer.py              # Shared train/eval loops
├── scripts/                    # Data preparation
│   ├── make_subset.py
│   ├── create_mini_subset.py
│   └── extract_clip_features.py
├── dev/                        # Development utilities
│   ├── make_dummy_data.py
│   └── make_pattern_data.py
├── checkpoints/
│   ├── baseline_48cls.pt
│   ├── agem_48cls.pt
│   └── class_labels.json
└── data/
    └── subset/
        ├── train_mini.json
        └── val_mini.json
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

### 4. Run experiments

```bash
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

- The project currently contains both **A-GEM** and an older **OrthGrad / COAD** variant. The demo app and saved checkpoints focus on **Baseline vs A-GEM**.
- Training scripts expect the mini subset JSON files and extracted features to exist under `data/`.
