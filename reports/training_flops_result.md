# Training FLOPs 회계 — 하드웨어 무관 효율 지표 병기

Generated: 2026-07 (auto). 스크립트: [`dev/compute_training_flops.py`](../dev/compute_training_flops.py)

**왜 했나:** 우리 효율 주장이 지금까지 wall-clock뿐이었다. wall-clock은 하드웨어·구현에
의존해서 edge-CL 문헌 옆에 놓을 수 없다. 서베이가 이 방향 선행연구로 지목한 두 편은 모두
FLOPs를 쓴다 — **SparCL**(NeurIPS'22, "DER++ 대비 23× 적은 training FLOPs"),
**BudgetCL**(CVPR'23, 하드웨어를 아예 추상화). FLOPs를 병기하면 우리 숫자가 같은 표에
들어간다. 배경: `reports/pycil_survey_edge_gap.md`.

**왜 분석적으로 계산했나:** analytic head(FeCAM/SLDA/NCM/Ridge/RanDumb)는 `nn.Module`이
아니라 numpy 통계 누적이라 어떤 FLOP 카운터(fvcore/thop/ptflops)로도 추적이 안 된다.
전 방법을 손으로 유도해야 backprop 계열과 backprop-free 계열에 **같은 규약**을 적용할 수 있다.

## 규약 (논문마다 다르므로 명시)

- **1 MAC = 2 FLOPs** (곱 1 + 합 1). MAC을 1 FLOP으로 세는 논문은 아래 수치의 정확히 절반.
- **training FLOPs = forward + backward**, backward = forward의 2배(입력 grad 1회 + 가중치 grad 1회)
- elementwise 연산(활성함수·정규화)은 원소당 1 FLOP
- 프로토콜: 8 stage × 600 샘플 × 15 epoch, D=512, T=16, H=256, C=48

**검증**: GRU 파라미터 수를 손으로 유도한 값(603,696)이 실제 모델 인스턴스와 **정확히 일치**
(스크립트에 `assert`로 박아둠). 즉 레이어 shape 이해가 맞다는 sanity check를 통과했다.

## 결과

| 방법 | training FLOPs | vs GRU | wall-clock(fit only) | 달성 FLOPs/s |
|---|---|---|---|---|
| GRU + A-GEM | **4.64 T** | 1× | 270.0 s | 17.2 G/s |
| **FeCAM (shared cov)** | **2.85 G** | **1,628×** | **42.4 ms** | 67.2 G/s |
| Deep SLDA | 6.61 G | 702× | 1.9 s | 3.6 G/s |
| NCM prototype | 44.3 M | **104,791×** | 2.4 ms | 18.4 G/s |
| Ridge RLS (closed-form) | 3.06 G | 1,514× | 21.3 ms | 143.8 G/s |
| Ridge + RP 2000 | 65.2 G | 71× | — | — |
| RanDumb (RFF 2000) | 64.3 G | 72× | 532 ms | 120.9 G/s |

### GRU + A-GEM 내역
| 항목 | FLOPs | 비중 |
|---|---|---|
| 메인 학습 업데이트 | 4.09 T | 88.2% |
| A-GEM 참조 gradient(`precompute_ref`) | 286 G | 6.2% |
| A-GEM 투영(`apply`, dot+axpy) | 261 G | 5.6% |

→ **A-GEM의 부가 비용은 전체의 11.8%**뿐. 즉 GRU 계열이 비싼 건 A-GEM 때문이 아니라
**backprop 자체와 15 epoch 반복** 때문이다.

### FeCAM 내역
| 항목 | FLOPs | 비중 |
|---|---|---|
| 공분산 `D.T @ D` | 2.52 G | 89.6% |
| 상관행렬 역행렬(1회, 캐시) | 268 M | 9.6% |
| Tukey 변환 + 정규화 | 19.7 M | 0.7% |
| 클래스 평균 누적 | 2.5 M | 0.1% |

## ⭐ 부수 발견 1 — 기존 리포트의 학습시간이 잘못 귀속돼 있었음 (수정)

`reports/cpu_friendly_methods_result.md` 등에 적힌 `train_s`는 **순수 학습 시간이 아니다.**
러너의 타이머(`dev/run_modern_cpu_methods.py:186-194`)가 stage 1에서 `s1_drop` 지표를 재려고
호출하는 **`model.scores(Xva)` 평가 1회를 타이머 안에 포함**하고 있다.

실제 데이터로 분해 측정한 결과:

| | 기존 보고 `train_s` | 실제 fit only | 평가가 차지한 비율 |
|---|---|---|---|
| **FeCAM** | 8.7 s | **42.4 ms** | **99.5%** |
| Deep SLDA | 2.1 s | 1.86 s | 11% |
| RanDumb | 0.9 s | 532 ms | 41% |
| NCM | 0.0 s (타이머 해상도 이하) | 2.4 ms | — |

→ **FeCAM의 실제 학습은 8.7초가 아니라 42밀리초.** 기존에 "GRU 대비 30배 빠름"이라고
쓴 건 **크게 과소평가**였고, 정확히는 **약 6,400배**(270 s / 42.4 ms)다. 다만 이 비교는
평가 비용이 섞여 있던 값이므로, 앞으로는 **fit-only wall-clock + FLOPs를 함께** 보고한다.
(FeCAM의 `scores()`가 느린 건 별개 이슈 — 클래스별 einsum 루프라 배치화로 최적화 가능,
`reports/pycil_bridge_result.md`에 이미 기록됨.)

## ⭐ 부수 발견 2 — FLOPs와 wall-clock은 서로 다른 걸 잡아낸다 (병기의 근거)

**Deep SLDA vs FeCAM**: FLOPs는 6.61 G vs 2.85 G로 2.3배 차이인데, wall-clock은
1.9 s vs 42 ms로 **44배** 차이다. 원인은 알고리즘이 아니라 **구현**:

- 우리 SLDA는 **진짜 per-sample 스트리밍** 구현 — 샘플마다 rank-1 외적(`d @ d.T`)과
  전체 D×D 읽기-수정-쓰기를 한다 → 4,800회의 작은 연산, BLAS 배칭 없음 → **3.6 GFLOP/s**
- FeCAM은 stage마다 배치 `D.T @ D` 한 번 → BLAS 최적 경로 → **67 GFLOP/s**

같은 하드웨어에서 달성 처리량이 19배 차이 난다. **FLOPs만 보면 SLDA가 2.3배 비싼 방법으로
보이지만, 실제로는 구현 방식이 44배를 만든다** — 그래서 둘 다 보고해야 한다.
(이건 SparCL/BudgetCL이 FLOPs를 택한 이유이기도 하고, 동시에 FLOPs만으로는
부족하다는 반례이기도 하다.)

## ⭐ 부수 발견 3 — 인코더가 모든 걸 압도한다 (edge 서사에 정직하게 반영해야 함)

frozen CLIP ViT-B/32 forward는 **16프레임 윈도우 1개당 141 GFLOPs**.
학습셋 4,800개 전체를 인코딩하면 **677 TFLOPs**이고, 이는:

- **GRU+A-GEM 전체 학습의 146배**
- **FeCAM 전체 학습의 241,240배**

즉 **head 학습 비용은 인코더 앞에서 반올림 오차**다. 이건 우리가 이미 실측으로 본
현상과 일치한다 — 데모에서 CLIP 인코딩 156 ms vs head 예측 밀리초 단위
(`reports/mobileclip_result.md`).

**정직한 해석:**
- 인코더 비용은 **모든 head에 동일**하고 우리 파이프라인에선 영상당 1회 지불 후 `.npy`로
  캐시된다. 따라서 **방법 간 비교에서는 상수로 상쇄**되고, "증분 학습 비용"을 논할 땐
  head FLOPs가 올바른 비교 대상이 맞다.
- 그러나 **"기기에서 새 클래스를 배우는 총비용"**을 주장할 때 head FLOPs만 내세우면
  오도가 된다. 총비용은 인코더가 지배한다 → 논문에선 **인코더 비용을 명시적으로 분리 표기**할 것.
- 이는 우리 edge 로드맵에서 **경량 인코더 탐색이 head 최적화보다 훨씬 중요**하다는
  근거이기도 하다(MobileCLIP 시도가 CPU에서 실패한 건 별개 문제 — `mobileclip_result.md`).

## 문헌과의 비교 가능성

SparCL은 Split CIFAR-10/Tiny-ImageNet에서 DER++ 대비 **최대 23× 적은 training FLOPs**를
보고한다(ResNet-18 from scratch, backprop 유지). 우리는 같은 단위로:

| | 레버 | 감소 배수 |
|---|---|---|
| SparCL | backprop을 **희소화** | 최대 23× |
| **우리(FeCAM)** | backprop을 **제거** | **1,628×** |

**단, 트랙이 다르므로 직접 우열 비교는 금지**: SparCL은 from-scratch ResNet-18 전체를
학습하고(백본 포함), 우리는 frozen 인코더 위 head만 학습한다. 위 표는 "같은 지표로
표현했을 때의 자릿수"를 보이는 것이지 동일 조건 비교가 아니다. 논문에 실을 땐
**트랙 라벨(from-scratch vs PTM)을 반드시 병기**할 것 — `reports/pycil_bridge_result.md`
§4의 공정성 주석과 같은 원칙.

## 재현

```bash
python3 dev/compute_training_flops.py
```

## 후속 (미착수)

- **inference FLOPs**도 같은 규약으로 산출 — 현재 학습만 회계함. 데모의 실시간 예측
  비용을 문헌과 비교하려면 필요.
- **BudgetCL 식 iteration-budget 프로토콜**로도 보고(`docs/project_focus.md` 갭 9번).
- FeCAM `scores()` 배치화 — 위 부수 발견 1에서 드러난 평가 병목.
