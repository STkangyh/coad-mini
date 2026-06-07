---
title: coad-mini
emoji: ⚡
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# coad-mini — live demo

Continual-learning action recognition on a 48-class subset of **Something-Something V2**
(A-GEM + GRU on CLIP ViT-B/32 features). One FastAPI app, three apps on the same pipeline:

- **Realtime captioning** — Baseline vs A-GEM, over an uploaded video or your **webcam**
- **Few-shot enrollment** — teach a brand-new action from a few clips; it appears in the
  live caption instantly and isn't forgotten (the continual-learning loop)
- **Anomaly / OOD detection** — flags out-of-distribution / low-confidence actions

> This Space runs the FastAPI server from the repo's `Dockerfile` (CPU, port 7860).
> Source code, research log, and experiments: **https://github.com/STkangyh/coad-mini**

<!--
Deploy: this file is the Hugging Face Space's README (the YAML front-matter above
configures the Space). On GitHub it lives at deploy/space-README.md so the project
README stays clean. Copy it to the Space repo root as README.md — see docs/DEPLOY.md.
-->
