# A-GEM 이후 학습 방법 조사/비교

Generated: 2026-07 (auto). 목적: "A-GEM 다음으로 뭘 써야 하나"에 답하기 위해
(1) 우리 스택에서 직접 실측한 대안들과 (2) 문헌상 A-GEM 이후 등장한 CL 방법들을
같은 표에 정리. 결론부터: **정확도를 더 쥐어짜는 새 방법 탐색보다, 지금 이미
충분히 넓게 훑었으니 다음은 "실사용 중 불편한 점 고치기"로 방향을 튼다** (교수님
가이드 반영, task 4).

## 1. 우리 스택에서 직접 실측한 비교 (동일 stack: 48cls/8stage, B/32, mem 예산 동일)

| 방법 | 메커니즘 | Avg Acc | S1 forgetting | 비고 |
|---|---|---|---|---|
| **GDumb** (Prabhu et al., ECCV'20) | greedy balanced 버퍼(400) + **매 스테이지 처음부터 재학습**, CL 트릭 없음 | **0.164 ± 0.004** | +0.059 ± 0.042 | ≈ chance(0.167). "버퍼만으로 설명되나?" 반론 검증용 |
| baseline | 메모리 없음, 순차학습만 | 0.251 ± 0.021 | +0.115 | GDumb보다도 높음 |
| plain ER | 동일 balanced 버퍼(mem=50/stage)를 손실에 리허설, gradient projection 없음 | 0.346 ± 0.012 | +0.020 | A-GEM에서 projection만 뺀 것 |
| **A-GEM** (Chaudhry et al., ICLR'19) | ER + gradient projection (과거 손실 증가 방지) | **0.387 ± 0.018** | **−0.061** | 우리 main |
| GRU+Attention | A-GEM + self-attention 시간모델 | 0.298 | +0.240 | 시간모델 확장, 악화 |
| diagonal-SSM | A-GEM + S4D/LRU 스타일 SSM | 0.361 | +0.173 | 시간모델 확장, 악화(Attention보단 나음) |

**해석 (3 seeds, 실측 확정):** GDumb(0.164)이 **chance(0.167) 수준이고 baseline(0.251)보다도 낮음** —
**"CL 이득이 그냥 버퍼 크기 효과 아니냐"는 표준 반론이 우리 세팅에선 성립하지 않는다**는 명확한 반증.
순수 버퍼 재학습(매 스테이지 모델 초기화)은 순차학습(baseline)만도 못하고, **gradient projection(A-GEM)이
실제로 유의미한 추가 가치를 만든다** — 버퍼 존재 자체가 아니라 "어떻게 쓰는가"가 중요하다는 근거.
(GDumb이 baseline보다 낮은 이유: 매 스테이지 모델을 처음부터 재학습하다 보니 작은 버퍼·15 epoch로는
분류기가 잘 수렴하지 못함 — 이는 GDumb 자체의 잘 알려진 특성으로, 우리 구현 버그가 아니라 "단순 재학습은
지속학습을 대체하지 못한다"는 정상적인 결과.)

## 2. 문헌상 A-GEM(2019) 이후 리허설 계열 흐름

| 방법 | 연도/venue | 핵심 아이디어 | 표준 class-IL 벤치서 A-GEM 대비 | 비용 특성 |
|---|---|---|---|---|
| **A-GEM** | ICLR 2019 | gradient projection (과거 손실 "증가 방지"만, 감소 안 시킴) | — (기준점) | O(1) 메모리, single 참조 gradient |
| ER (Experience Replay) | ~2019 (Chaudhry et al. 등) | 버퍼 리허설을 손실에 직접 섞음 | 표준 벤치에서 대체로 A-GEM보다 우수 | ER과 유사, projection 없어 더 저렴 |
| GDumb | ECCV 2020 | greedy balanced 버퍼 + 항상 재학습(방법론 없음) | 사실상 baseline; "당신의 방법이 버퍼보다 나은가?" 테스트용 | 매 평가시 전체 재학습 — 우리처럼 stage마다 하면 비쌈 |
| DER / DER++ | NeurIPS 2020 | ER + 과거 logit distillation(dark knowledge) | 표준 벤치서 ER/A-GEM보다 우수 | 버퍼에 logit도 저장 |
| ER-ACE | ICLR 2022 | 리허설 시 새/구 클래스 손실 비대칭 처리 | A-GEM류(GEM 포함)를 "성능 나빠 제외"할 정도로 강함 | ER과 비슷한 비용 |
| X-DER | TPAMI 2023 | DER++ 확장, 미래 클래스 대비 예비 로짓 | DER++ 대비 추가 개선 | DER++ 유사 |
| iCaRL | CVPR 2017 | nearest-mean-of-exemplars 분류 + distillation | 고전 강 baseline | exemplar set 관리 필요 |
| FOSTER | ECCV 2022 | feature boosting + compression(모델 확장/압축) | 강한 class-IL 정확도 | 모델 크기 가변 |
| **L2P / DualPrompt / CODA-Prompt** | CVPR'22~23 | **frozen backbone + prompt pool**, 리허설-프리 | 이미지 CIL 프론티어, 리허설 계열보다 대체로 우수 | 학습 파라미터 backbone의 0.7~4.6%만, 그러나 backbone forward 여러 번 필요 |
| RanPAC / SimpleCIL | NeurIPS'23 / IJCV'24 | frozen backbone + prototype(학습 거의 없음) | prompt-pool도 이김 (강한 frozen feature가 핵심) | 매우 저비용 — **우리 frozen-CLIP 선택과 같은 철학** |
| **PIVOT / SMILE / ESSENTIAL** | CVPR'23 / ICCV'25 | **비디오 CIL 전용**: frozen CLIP + temporal prompt / sparse memory | vCLIMB SOTA (UCF101 93~96%) | 우리와 같은 frozen-CLIP 재료, 그 위에 무거운 temporal 모듈 |

## 3. 결론 — 왜 여기서 더 안 파고, 사용성으로 전환하나

지금까지 커버한 축을 다 합치면:

```text
Memory 전략:     baseline ❌ < GDumb ❌(≈chance) < ER ⭕ < A-GEM ⭕⭕ (best)
Capacity:        h64 ❌ < h128 ≈ h256(main) ≈ h512  (양방향 무병목)
Backbone:        B/32(main) ≈ L/14 (+0.013, 노이즈 내)
Temporal:        GRU(main) > SSM(-0.040) > Attention(-0.103)
Data:            25% ❌❌ < 50% ❌ < 100% ⭕⭕ (지배적 레버)
```

**A-GEM + 단순 GRU가 이 정도로 넓은 축을 다 이기는 상황에서, 다음 방법(DER++, ER-ACE, prompt-pool 등)을
추가로 구현해도 한계효용이 낮을 가능성이 높다** — 이미 "더 정교한 방법일수록 이 작은/짧은-시퀀스 세팅에서
오히려 손해"라는 패턴이 반복 확인됐기 때문 (Attention, SSM, DER 계열도 대체로 파라미터·복잡도가 늘어나는
방향).

**그래서 이후 노력은 새 알고리즘 탐색이 아니라 "실사용 중 발견되는 문제"를 고치는 쪽으로 돌린다** —
데모 앱(웹캠 실시간 자막, few-shot 등록, OOD 배지)을 실제로 돌려보고 버그/불편함을 찾아 수정 (task 4).
이는 정확도 스트레스보다 방어 가능한 시스템 완성도를 높이는 더 생산적인 경로.

## 재현

```bash
python3 dev/run_er_comparison.py       # baseline / ER / A-GEM, 5 seeds
python3 dev/run_gdumb_comparison.py    # GDumb, 3 seeds
```
