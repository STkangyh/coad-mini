# OpenCLIP ViT-L/14 결과

Generated: 2026-06-06 21:12 KST (auto, pipeline_driver)

## 설정

```text
A-GEM (balanced, mem=50, replay=0.25)
GRU hidden=256, layers=1
48 classes / 8 stages / 15 epochs / seeds 0 1 2
backbone = openclip_l14
feature_dim = 768
```

## 결과

- **Avg Acc = 0.401 ± 0.033**  (seeds: 0.357, 0.410, 0.436)
- S1 Drop = +0.047 ± 0.044
- S2 Drop = +0.124
- seeds 수집됨: 3

## Baseline(CLIP B/32) 대비

| | Avg Acc | S1 Drop |
|---|---|---|
| CLIP B/32 (512d) | 0.388 | -0.078 |
| openclip_l14 (768d) | 0.401 | +0.047 |
| Δ | **+0.013** | +0.125 |

**판정: 소폭 향상 — 방향성은 맞으나 성공 기준엔 약간 못 미침.**
