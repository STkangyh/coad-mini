# HDC(초고차원 컴퓨팅)를 4번째 head로 시도 — SLDA·FeCAM을 못 넘었다

Generated: 2026-08-08. 스크립트: [`dev/run_hdc_comparison.py`](../dev/run_hdc_comparison.py)
(신규) · 원시결과: `reports/hdc_comparison_raw.json`
배경: [`recent_cpu_only_cl_survey_2026_result.md §1`](recent_cpu_only_cl_survey_2026_result.md)

**질문:** ImageHD가 시사한 "공분산 없이도 되는" HDC 방식을, 우리 SSv2 8-stage 프로토콜에서
NCM/SLDA/FeCAM과 나란히 재면 실제로 경쟁력이 있는가?

**ImageHD 그대로가 아니라 고전적 지도학습 HDC로 구현한 이유:** ImageHD 원문 Table I를 다시
확인하니 **비지도(Unsup.) 클러스터링 방식**이고, 보고된 40.4배/383배는 **FPGA 대 CPU/GPU**
하드웨어 비교였다 — 우리 세팅(라벨 있음, CPU만)과 안 맞아 그대로 못 옮긴다는 걸
`recent_cpu_only_cl_survey_2026_result.md`에서 이미 짚었다. 대신 HDC 문헌의 표준
지도학습 분류기(랜덤 투영 인코딩 + 클래스별 hypervector 번들링 + 코사인 유사도 분류)를
구현했다 — NCM의 "클래스 평균"을 초고차원 이진 공간에서 하는 것과 같은 정신이다.

## 1. 방법

- **인코딩**: `h(x) = sign(R @ x)`, R은 고정 랜덤 가우시안 투영 행렬(hdc_dim × 512).
  연속값 특징을 이진 hypervector({-1,+1}^hdc_dim)로 바꾸는 표준 방식(locality-sensitive
  hashing과 동일한 원리).
- **번들링**: 클래스별로 소속 샘플들의 인코딩된 hypervector를 **누적합**만 한다(재이진화
  안 함, OnlineHD 등 실무 구현과 동일) — 새 클래스 추가는 그냥 덧셈이라 공분산 역행렬이
  전혀 필요 없다.
- **분류**: 쿼리의 인코딩과 각 클래스 hypervector 사이 코사인 유사도, argmax.
- 나머지(데이터·8-stage 프로토콜·평가)는 `ssv2_head_curves_result.md`와 완전히 동일 —
  스크립트 자체가 NCM/SLDA/FeCAM 재실행분이 기존 리포트 수치(20.89/22.77/24.19)와
  정확히 일치하는지 자동 검증한다(통과 확인).

## 2. 결과

| head | avg_inc | last | fit(s) |
|---|---|---|---|
| NCM | 20.89% | 12.38% | 0.08 |
| **HDC(D=10000)** | **21.47%** | **12.93%** | **3.58** |
| SLDA | 22.77% | 14.14% | 0.13 |
| **FeCAM** | **24.19%** | **15.74%** | **0.40** |

**차원을 바꿔도 정확도는 안 올라간다:**

| hdc_dim | avg_inc | fit(s) |
|---|---|---|
| 2000 | 19.96% | 0.83 |
| 5000 | 21.08% | 1.88 |
| **10000**(표준값) | **21.47%** | 3.58 |
| 20000 | 21.09% | 6.59 |

## 3. 해석 — 정확도도 속도도 못 이겼다

**정확도: NCM보다는 낫지만 SLDA·FeCAM에는 못 미친다.** HDC는 "공분산을 안 쓴다"는 점에서
구조적으로 NCM에 가깝고(공유 공분산 자체가 없음), 실제 순위도 그 자리에 온다 — 2000~20000
차원을 다 스윕해도 22% 벽을 못 넘는다.

**속도도 더 느리다 — FeCAM보다 9배.** `recent_cpu_only_cl_survey_2026_result.md`가 이미
경고했듯, ImageHD의 속도 우위는 **FPGA의 word-packed 이진 hypervector + bitwise 연산**
(XOR+popcount를 하드웨어로 병렬 처리)에서 나온다. 이 구현은 순수 numpy로 `sign()`
이후에도 **float64 배열**로 코사인 유사도를 계산해 — 이진 벡터의 이점을 전혀 못 살리고,
그냥 D_hdc×512 크기의 랜덤 투영 행렬 곱셈 비용만 고스란히 낸다(D_hdc=10000이 FeCAM의
D=512 공분산 연산보다 훨씬 큰 행렬). **"gradient 없음"이 "빠름"을 보장하지 않는다**는
FoRo 쪽에서 이미 짚었던 교훈이 여기서도 그대로 나타난다.

## 4. 판정 — 이 구현으로는 채택 안 함

이 형태(numpy, float 코사인 유사도)의 HDC는 우리 head 후보로 넣을 이유가 없다 — 정확도도
낮고 속도도 느리다. HDC가 실제로 이점을 내려면 **비트 패킹(uint64 워드에 64비트씩 압축) +
popcount 기반 Hamming distance**로 다시 구현해야 하는데, 이건 순수 CPU에서도 일부 이득이
있을 수 있지만(FPGA만큼은 아니어도), 이번 구현 범위를 벗어난다.

## 5. 정직한 한계

- **비트 패킹 구현을 안 했다.** "진짜" HDC의 속도 이점은 이진 연산 최적화에서 나오는데,
  이 스크립트는 그 최적화를 전혀 안 했다 — 지금 결과는 "이 구현 방식"의 한계지 "HDC라는
  아이디어 자체"의 한계라고 단정할 수는 없다.
- **인코딩 방식은 하나만 시도했다.** 랜덤 투영+sign 말고 다른 HDC 인코딩(레벨
  hypervector, N-gram 등)은 안 봤다.
- **단일 랜덤 투영(seed=0)만 썼다** — R 행렬을 바꿔가며 분산을 재지 않았다.
- **fit(s)는 이 스크립트의 numpy 구현 기준**이라, 최적화된 구현이면 절대 시간은 달라질 수
  있다(다만 상대적으로 FeCAM보다 느린 이유 자체는 §3에서 설명한 구조적 문제라 유지될
  가능성이 높다).

## 재현

```bash
python3 dev/run_hdc_comparison.py                    # hdc_dim=10000(기본)
python3 dev/run_hdc_comparison.py --hdc-dim 5000
```
