# CPU-only 학습 방법 조사 — backprop-free 스트리밍/닫힌해 계열 실측

Generated: 2026-07 (auto). 질문: "CPU로만 학습하는 다른 방법이 더 있나?"
→ 문헌 조사 후, frozen-feature CL의 표준 계열(**backprop조차 없는** 스트리밍 통계/닫힌해 방법)을
우리 스택에서 직접 실측. **결과: Deep SLDA가 A-GEM GRU와 task-aware 동률, 진짜 class-IL에선 +34% 우위,
학습 130배 빠름, replay 버퍼 불필요.**

## 1. 조사한 방법 계열 (문헌)

frozen backbone 위에서는 gradient 학습 자체를 없앨 수 있다 — 클래스 통계만 스트리밍으로 누적:

| 방법 | 출처 | 메커니즘 | 저장량 | CL 성질 |
|---|---|---|---|---|
| **NCM 프로토타입** | SimpleCIL (IJCV'24) | 클래스별 feature 평균 1개, cosine 최근접 | C×D | 클래스별 독립 → 망각 원천 불가 |
| **Deep SLDA** | Hayes & Kanan, CVPR-W 2020 | 클래스 평균 + **공유 공분산 1개**(스트리밍), LDA 판별 | C×D + D² | iCaRL 대비 100× 빠름·1000× 저메모리(원논문) |
| **Ridge RLS / ACIL** | ACIL (NeurIPS'22), GACL | Gram 통계 누적(G+=HᵀH, C+=HᵀY) → 닫힌해 W=(G+λI)⁻¹C | D² + D×C | recursive least squares, exemplar-free |
| **RanPAC** | NeurIPS 2023 | 위 + 고정 랜덤 ReLU 투영으로 차원 확장(~10k) | RP-dim² | 클래스 분리도 향상 |

공통 성질: **단일 패스(single-pass)·replay 버퍼 불필요(통계만 저장)·backprop 없음** → CPU에 이상적.

## 2. 우리 스택 실측 (동일 8-stage 프로토콜, B/32, mean-pool 512-d)

`dev/run_cpu_friendly_methods.py` — 16프레임 feature를 mean-pool한 512-d 비디오 임베딩 사용.
train 4800 / val 4702 전량.

| 방법 | task-aware | S1 drop | **full-48 acc** | full F1 | full mAP | 학습(초) |
|---|---|---|---|---|---|---|
| NCM 프로토타입 | 0.378 | +0.000 | 0.124 | 0.105 | 0.085 | **0.0** |
| **Deep SLDA** | **0.390** | +0.005 | **0.141** | **0.123** | 0.102 | 2.1 |
| Ridge RLS (닫힌해) | 0.373 | +0.013 | **0.143** | 0.122 | 0.092 | 0.0 |
| Ridge + RP(2000, 3 seeds) | 0.332±0.008 | +0.076 | 0.127 | 0.116 | 0.080 | 0.4 |
| **(참고) GRU + A-GEM** | 0.387±0.018 | **−0.061** | 0.105 | 0.075 | **0.105** | ~270 |

(GRU full-48-way 수치는 `data_scale_full48way_result.md`의 100% 데이터 기준.)

## 3. 핵심 발견

1. **Deep SLDA ≈ A-GEM (task-aware), SLDA ≫ A-GEM (진짜 class-IL).**
   task-aware 0.390 vs 0.387로 동률인데, task ID 없는 full-48-way에선 **0.141 vs 0.105 (+34%)**,
   F1(macro) **0.123 vs 0.075 (+64%)**. 게다가 학습 2.1초(GRU+A-GEM ~270초의 **1/130**),
   replay 버퍼 불필요(공분산+평균 = 상수 메모리), 하이퍼파라미터 사실상 없음.

2. **학습을 아예 안 하는 NCM(0.0초)조차 full-48-way에서 GRU를 이김** (0.124 vs 0.105).
   SimpleCIL 논문의 주장("frozen feature 위에선 프로토타입이 복잡한 방법을 이긴다")이
   우리 세팅에서 그대로 재현됨.

3. **왜 이기나 — 망각이 구조적으로 거의 없기 때문.** 이 방법들은 클래스별 통계가 독립적으로
   누적되므로 gradient 간섭에 의한 망각이 원천적으로 없다(S1 drop ≈ 0; SLDA/ridge의 공유
   통계 드리프트만 미세하게 존재). GRU+A-GEM의 낮은 full-48-way F1(0.075)이 보여주듯,
   backprop 계열의 진짜 적은 로짓 쏠림(망각)이었고, 통계 계열은 그 문제 자체가 없다.

4. **시간 모델링의 기여가 사실상 0임을 재확인.** mean-pool(시간 순서 완전 폐기) 기반 선형
   판별이 시퀀스를 읽는 GRU와 동률/우위 — "CLIP 프레임 feature에는 GRU가 활용할 시간 정보가
   거의 없다"는 기존 결론(Attention/SSM 실패)과 정확히 일치하는 독립 증거.

5. **랜덤 투영(RanPAC-lite)은 우리 규모에선 역효과.** RP 2000-d가 오히려 하락 — RanPAC의
   이득은 강한 ViT feature + 10k-d + 대규모 데이터에서 나오는 것으로, 소규모(4800개)에서는
   과적합/분산만 증가.

## 4. 한계·공정성 주석

- SLDA/NCM/ridge는 **단일 패스에 전체 스테이지 데이터의 통계**를 쓴다(샘플 저장은 안 함).
  A-GEM은 mem=50/stage 버퍼 + 15 epoch. "저장량" 기준으론 통계 계열이 오히려 가볍다
  (D²=512² 공분산 ≈ 1MB vs replay 버퍼 350개 feature 참조).
- mean-pool은 시간 정보를 버림 — 그럼에도 이겼다는 것이 포인트이지만, 진짜 모션 feature
  (비디오 백본)에서는 결론이 달라질 수 있음.
- task-aware S1 backward-transfer(−0.061)는 A-GEM만의 강점으로 남음(리허설이 과거 stage
  성능을 오히려 올림). 통계 계열은 ±0 근처.

## 5. 시사점 (미팅용)

- **"CPU 학습"을 넘어 "학습 자체가 초 단위"인 극한 지점이 존재**하고, 우리 세팅에선 그게
  이미 backprop 계열과 대등하거나 낫다. 효율 축 주장이 한층 강해짐:
  GRU+A-GEM 4.5분 → **SLDA 2초** (모두 CPU).
- **few-shot enrollment에 직접 응용 가능**: 새 동작 등록을 gradient 학습(현행 FewShotEnroller)
  대신 **프로토타입 평균 1줄 계산**으로 대체하면 등록이 즉시(밀리초) 끝나고 기존 클래스
  망각이 구조적으로 0 — 데모 사용성의 다음 업그레이드 후보.
- 논문 서사 강화: "frozen feature 위에서는 (a) 아키텍처 복잡도도 (b) gradient 학습 자체도
  필수가 아니다 — 병목은 표현(CLIP)과 데이터"로 결론이 한 단계 더 일반화됨.

## 참고 문헌

- Deep SLDA: Hayes & Kanan, CVPR-W 2020 — https://arxiv.org/abs/1909.01520
- SimpleCIL/APER: Zhou et al., IJCV 2024 — https://arxiv.org/abs/2303.07338
- RanPAC: McDonnell et al., NeurIPS 2023 — https://arxiv.org/abs/2307.02251
- ACIL (analytic CIL, recursive least squares): https://arxiv.org/abs/2205.14922
- GACL (exemplar-free generalized analytic CL): https://arxiv.org/abs/2403.15706
- XLDA (edge-scale LDA CL): https://arxiv.org/abs/2307.11317

## 재현

```bash
python3 dev/run_cpu_friendly_methods.py   # 전체 ~10초 (feature 로딩 포함)
```
