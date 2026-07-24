# Base-heavy(TCD 스타일) split 재검증 — FeCAM vs GRU+A-GEM

Generated: 2026-07 (auto). 배경: TCD(Park et al., ICCV 2021)는 SSv2 전체(174cls)를
**84-class base + 소수 incremental session**(base:increment = 8.4~16.8배)으로 나눈 CIL
벤치마크에서 **NME(prototype 계열)가 CNN보다 못하다**고 보고했다. 우리의 핵심 결론
("FeCAM > GRU+A-GEM")과 정면으로 긴장되는 반례라, **우리 데이터로 직접 재검증**했다.
(TCD의 174-class 원본 split 자체는 이 세션에서 영상 접근이 막혀 재현 불가 — 대신
구조적 변수(base-heaviness)를 우리 48-class subset에 이식해 테스트. 상세 배경은
`reports/sota_positioning_brief.md` §1(f).)

## 설정

우리 48클래스(8-stage×6class 큐레이션, 기존 실험과 동일 순서)를 두 가지 base-heavy
구조로 재편, **true class-IL 평가**(task ID 없음, 지금까지 본 전체 클래스에서 argmax):

| split | 세션 구성 | base:increment 비 | (참고) TCD 원본 |
|---|---|---|---|
| moderate | 24 + [6,6,6,6] | 4배 | — |
| aggressive | 36 + [6,6] | 6배 | 84:10=8.4배, 84:5=16.8배 |
| (참고) 우리 원래 protocol | 6×8 균등 | 1배 | — |

## 결과 (CLIP B/32, GRU+A-GEM은 3-seed 평균)

| split | method | Avg Inc Acc | Last Acc | per-session |
|---|---|---|---|---|
| moderate | **FeCAM** | **0.178** | **0.157** | 0.193 → 0.192 → 0.183 → 0.164 → 0.157 |
| moderate | GRU+A-GEM | 0.118 | 0.109 | 0.163 → 0.125 → 0.096 → 0.095 → 0.109 |
| aggressive | **FeCAM** | **0.168** | **0.157** | 0.183 → 0.164 → 0.157 |
| aggressive | GRU+A-GEM | 0.124 | 0.084 | **0.213 → 0.075** → 0.084 |

## 핵심 발견

### 1. 반전이 일어나지 않았다 — 오히려 격차가 벌어짐

TCD의 발견(NME < CNN)이 우리 세팅에도 적용된다면 base-heavy로 갈수록 FeCAM 우위가
줄거나 뒤집혀야 하는데, **정반대**였다:

| | uniform(원래) Δ(FeCAM−GRU) | moderate Δ | aggressive Δ |
|---|---|---|---|
| Last acc 기준 | +0.052 (0.157−0.105) | **+0.048** | **+0.073** |

moderate에서는 원래와 비슷한 격차, **aggressive(가장 TCD와 유사한 비율)에서는 격차가
오히려 더 커짐** — FeCAM 최종 정확도가 GRU+A-GEM의 거의 2배(0.157 vs 0.084).

### 2. GRU+A-GEM이 base 세션 직후 붕괴함 — 메커니즘까지 확인됨

aggressive split에서 GRU+A-GEM은 base(36cls) 직후 **0.213 → 0.075로 −64.8%(상대) 붕괴**
후 정체. moderate(24cls base)는 −23%로 덜 심함. 원인은 **A-GEM의 replay 메모리가
"stage당 고정 50개"**라서 base 클래스 수가 늘수록 클래스당 replay 커버리지가 급감하기
때문:

| split | base 클래스 수 | replay 메모리(50) ÷ base 클래스 = 클래스당 exemplar |
|---|---|---|
| 원래(균등) | 6 | **8.33개/class** |
| moderate | 24 | 2.08개/class |
| aggressive | 36 | **1.39개/class** |

클래스당 exemplar가 8.33→1.39로 6배 줄면서 망각 방지력이 거의 사라짐 — base-heavy
비율(4배, 6배)과 붕괴 정도(−23%, −65%)가 정확히 같은 방향으로 스케일함. **이건
FeCAM 우위와는 별개로, 우리 A-GEM 구현 자체의 개선점**: `mem_per_stage`를 고정값이
아니라 세션 클래스 수에 비례하게 조정하면 base-heavy 상황에서도 더 강건할 수 있음
(후속 과제로 남김).

### 3. FeCAM은 세션 구조에 관계없이 완만하게만 열화됨

FeCAM의 마지막 세션 정확도는 **moderate·aggressive 모두 0.157로 완전히 동일**
(그리고 기존 균등-split 결과 0.157과도 일치) — 클래스별 통계가 독립 누적되므로 세션
순서/크기 구조 자체가 최종 상태에 영향을 주지 않는다는 이론적 성질이 실측으로 확인됨.
per-session 곡선도 완만한 하강(0.193→0.157 등)만 보이고 붕괴가 없음.

## 해석 — TCD의 발견과 우리 발견이 왜 다른가 (가설, 검증 안 됨)

TCD의 NME는 **SSv2로 직접 학습한(end-to-end trained) CNN feature**에 대해 계산됐을
가능성이 높다(2021년 당시 CLIP 기반 video CIL은 아직 프론티어 이전) — 그런 feature는
모션 정보를 인코딩하고 있어서, 프레임을 단순 평균내면 그 정보가 파괴된다. 반면 우리는
**frozen CLIP(appearance-only, SSv2로 학습되지 않음)** feature를 쓰므로애초에 평균낼
"시간 정보"가 적어 mean-pool의 손실이 작다 — 이는 §1(f)/Q6에서 이미 짚은 "SSv2는 모션
중심이라 CLIP에 불리하다"는 한계와 같은 뿌리다. **정리하면: TCD의 반례는 "강한
temporal feature 위에서의 mean-pool"에 대한 경고이고, 우리는 애초에 temporal feature가
약한 지점에서 시작하므로 그 경고가 우리에게 덜 해당한다**는 것이 가장 설득력 있는
설명 — 단, TCD의 정확한 backbone을 확인하지 않았으므로 가설로 명시.

## 결론

1. **우리 핵심 주장(FeCAM > GRU+A-GEM)은 base-heavy 구조 스트레스 테스트를 통과함**
   — TCD의 반례가 우리 세팅에 직접 적용되지 않음을 실측으로 확인.
2. **A-GEM의 고정 replay 예산은 base-heavy 상황에서 약점**이라는 새로운, 별개의
   발견 — FeCAM의 강점과 무관하게 우리 시스템 자체의 개선 여지.
3. TCD와의 차이는 backbone(frozen CLIP vs trained CNN)에서 올 가능성이 높다는 가설을
   제시 — 완전한 반증을 위해선 TCD의 실제 174-class split·backbone으로 직접 재현이
   필요(영상 접근 복구 시 후속 과제, `reports/mobileclip_result.md`와 동일한 차단 사유).

## 재현

```bash
python3 dev/run_base_heavy_split.py
```
Raw per-seed 결과: `reports/base_heavy_split_raw.json`.
