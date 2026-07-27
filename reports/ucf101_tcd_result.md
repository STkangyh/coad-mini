# UCF101 TCD 프로토콜 결과 — 표준 벤치마크 첫 진입

Generated: 2026-07-26. 스크립트: [`dev/run_ucf101_tcd_protocol.py`](../dev/run_ucf101_tcd_protocol.py)
· 원시결과: `reports/ucf101_tcd_raw.json` · 데이터: [`scripts/get_ucf101.sh`](../scripts/get_ucf101.sh)

**왜 했나:** 지금까지 우리 수치는 전부 48-class SSv2 자체 subset이라, 남의 논문과
**서열만** 논할 수 있었고 숫자를 나란히 놓을 수 없었다. UCF101은 우리가 원문을 읽은
SSv2-CIL 계보 4편(TCD·STSP·CSTA·ESSENTIAL)이 **전부 같은 프로토콜로 보고**하는
벤치마크라, 여기 숫자는 직접 비교 가능하다.

## 설정 (TCD 원문 §4.2 그대로)

| 항목 | 값 |
|---|---|
| 데이터 | UCF101 공식 recognition split 1 — train 9,537 / test 3,783 / 101 클래스 |
| 특징 | frozen **CLIP ViT-B/32**, 16프레임 균등 샘플 → mean-pool → 512-d |
| base | **51 클래스** |
| 증분 | 나머지 50개를 **10 / 5 / 2**개씩 (= TCD 표기 10×5, 5×10, 2×25 stages) |
| 클래스 순서 | **TCD의 실제 순서와 비트 단위 동일** (아래 검증) — 3개 평균 |
| 지표 | average incremental accuracy (매 스텝 후 지금까지 본 전체 클래스 top-1, **task ID 없음**) |
| exemplar | **0** (통계만 누적) |

### 프로토콜 일치 검증 (추정이 아니라 확인함)

TCD 공개 저장소([bellos1203/TCD](https://github.com/bellos1203/TCD))에서 두 가지를 확인했다:

1. **클래스 순서** — 저장소의 `class_list.pkl`이 `np.random.RandomState(1000).permutation(101)`과
   **101개 전부 일치**(불일치 0). 즉 우리 시드 기반 순서가 TCD가 실제로 쓴 순서와 같다.
   "비슷하게 랜덤"이 아니라 **문자 그대로 같은 순서**다.
2. **실행 설정** — `scripts/ucf101/ucf101_51_10.sh`가
   `--seed 1000/1993/2021 --init_task 51 --nb_class 10 --K 5 --budget_type class
   --store_frames uniform`을 사용. 우리가 맞춘 값과 전부 동일하며,
   특히 **exemplar 예산이 "클래스당 5개"**임을 확인(아래 GRU 비교에 반영).

## 결과

| head | 10×5 | 5×10 | 2×25 | fit 시간 |
|---|---|---|---|---|
| NCM prototype | 83.28±1.05 | 83.23±1.01 | 83.19±0.98 | 0.1~0.2 s |
| Deep SLDA | 85.12±0.81 | 85.10±0.78 | 85.08±0.76 | 0.1~0.5 s |
| **FeCAM (shared cov)** | **88.84±0.52** | **88.86±0.54** | **88.84±0.50** | 0.3~1.2 s |

## 문헌 대비 위치 (UCF101, 각 논문 자체 보고치)

| 방법 | 백본 | exemplar | 10×5 | 5×10 | 2×25 |
|---|---|---|---|---|---|
| iCaRL | ViT | ✔ | 70.6 | 69.5 | 67.3 |
| PODNet | TSM | ✔ | 73.26 | 71.58 | 70.28 |
| TCD (ICCV'21) | ResNet-34+TSM | ✔ | 74.89 (NME 77.16) | 73.43 | 72.19 |
| UCIR | ViT | ✔ | 77.6 | 74.6 | 71.8 |
| FrameMaker (NeurIPS'22) | TSM | ✔ | 78.13 | 76.38 | 75.77 |
| L2P | ViT | ✕ | 81.2 | 80.1 | 78.6 |
| STSP (ECCV'24) | TSM | **0** | 81.15 | 82.84 | 79.25 |
| ST-prompt | CLIP | ✕ | 84.8 | 85.5 | 85.7 |
| **우리 (frozen CLIP + FeCAM)** | **frozen CLIP** | **0** | **88.84** | **88.86** | **88.84** |
| ESSENTIAL (ICCV'25 Highlight) | frozen CLIP | sparse+prompt | **95.1** | 93.9 | 93.3 |

**→ ESSENTIAL 다음 2위.** TCD(+14.0), FrameMaker(+10.7), STSP(+7.7), ST-prompt(+4.0)를
모두 상회한다. 그것도 **backprop 0회, exemplar 0개, 백본 학습 0회, fit 0.3초**로.

## 우리 자체 baseline(GRU+A-GEM)과의 직접 비교

스크립트: [`dev/run_ucf101_gru_agem.py`](../dev/run_ucf101_gru_agem.py) · 원시결과
`reports/ucf101_gru_agem_raw.json`. 같은 프로토콜(10×5, 동일 클래스 순서 3개),
15 epoch/세션, exemplar는 **TCD와 동일한 클래스당 5개**(저장소 스크립트에서 확인한 값).

| 세션 | 0 (51cls) | 1 (61) | 2 (71) | 3 (81) | 4 (91) | 5 (101) |
|---|---|---|---|---|---|---|
| GRU + A-GEM | 84.6 | **60.1** | 49.0 | 49.3 | 49.0 | 52.3 |
| **FeCAM** | 90.1 | **90.1** | 88.9 | 88.5 | 88.1 | 87.4 |

| | avg inc acc | last | 학습시간/seed |
|---|---|---|---|
| GRU + A-GEM | 57.39±0.16 | 52.30 | **480 s** |
| **FeCAM** | **88.84±0.52** | **87.44** | **0.3 s** |
| 차이 | **+31.45 %p** | **+35.14 %p** | **1,600배 빠름** |

**핵심은 base 세션이 아니라 그 직후다.** GRU+A-GEM은 base 51클래스를 84.6%로 잘 학습한다
(FeCAM 90.1과 5.6%p 차이). 그런데 **첫 증분에서 −24.5%p(−29%) 붕괴**하고 이후 50% 근처에
정체한다. FeCAM은 같은 지점에서 **−0.08%p**다.

즉 이 벤치마크에서 GRU+A-GEM의 문제는 **표현력이 아니라 망각**임이 분명해졌다.
`reports/base_heavy_split_result.md`에서 SSv2로 관찰한 "base가 클수록 A-GEM이 무너진다"는
현상이, 51클래스 base라는 훨씬 극단적인 조건에서 그대로 재현된 것이다 —
이번엔 exemplar 예산을 클래스당 5개로 **늘려줬는데도** 그렇다.

> **정직하게 병기할 것:** 우리 GRU+A-GEM 57.39는 위 문헌 표의 **모든 방법보다 낮다**
> (최하위 iCaRL 70.6). 당연한 결과다 — 저들은 비디오 CIL 전용으로 설계된 방법(distillation,
> temporal 모듈 등)이고 우리 baseline은 단순 GRU + A-GEM이다. 우리 주장은
> "우리 baseline이 강하다"가 아니라 **"같은 frozen feature 위에서 backprop을 없애는 쪽이
> 훨씬 낫다"**이며, 이 비교는 그것을 보여준다.

## 발견 1 — 증분 크기에 완전히 불변 (구조적 성질)

| 방법 | 10×5 → 2×25 변화 |
|---|---|
| TCD | 74.89 → 72.19 (**−2.70**) |
| FrameMaker | 78.13 → 75.77 (**−2.36**) |
| ESSENTIAL | 95.1 → 93.3 (**−1.80**) |
| STSP | 81.15 → 79.25 (−1.90, 비단조) |
| **우리 (FeCAM)** | 88.84 → 88.84 (**0.00**) |

세션을 잘게 쪼갤수록 다른 방법은 전부 열화하는데 **우리는 변화가 0.02%p 이내**다.
NCM·SLDA도 마찬가지(0.09%p, 0.04%p).

이유는 명확하다 — **클래스별 통계가 독립적으로 누적**되므로 "언제 몇 개씩 들어오는지"가
최종 상태에 영향을 주지 않는다. gradient 간섭이 없으니 세션 경계 자체가 무의미하다.

**edge 서사에 직결:** 스마트글래스에서 사용자가 **한 번에 한 클래스씩** 등록해도
페널티가 없다는 뜻이다. 다른 방법은 잘게 쪼갤수록 손해를 본다.

## 발견 2 — SSv2와의 대비가 방법의 적용 범위를 정확히 규정한다

같은 head, 같은 인코더인데 결과가 극단적으로 다르다:

| 벤치마크 | 성격 | 우리 FeCAM |
|---|---|---|
| **UCF101** | **static-biased** (appearance로 구별 가능) | **88.84** |
| SSv2 (48-class subset) | **temporal-biased** (모션으로만 구별) | 15.7 (full-way) |

"static-biased / temporal-biased" 분류는 우리가 만든 게 아니라 **ESSENTIAL 논문이
자기 §4.1에서 쓰는 용어**다(UCF101·HMDB51·ActivityNet·Kinetics = static, SSv2만 temporal).

### FeCAM만의 현상이 아님 — 세 head 전부 같은 폭으로 떨어진다

SSv2에서도 UCF101과 똑같은 프로토콜(8-stage 곡선, 세 head 전부)로 재현했다.
스크립트: [`dev/run_ssv2_head_curves.py`](../dev/run_ssv2_head_curves.py), 원시결과
`reports/ssv2_head_curves_raw.json`.

| head | UCF101 (static) | SSv2 (temporal) | 격차 |
|---|---|---|---|
| NCM | 81.4 | 12.4 | **−69** |
| Deep SLDA | 83.5 | 14.1 | **−69** |
| **FeCAM** | **87.4** | **15.7** | **−72** |

세 head 모두 **69~72%p라는 거의 동일한 폭**으로 떨어진다. 즉 이 격차는 FeCAM의
공분산 정규화 유무와 무관하게 **"frozen CLIP + mean-pool" 계열 전체에 적용되는
구조적 성질** — head를 뭘 쓰든 인코더가 appearance만 보므로 생기는 한계라는 뜻이다.
(→ 그림: `figures/ssv2_head_curves.png`, `figures/static_vs_temporal.png`)

즉 **frozen CLIP + mean-pool의 성공/실패가 데이터셋 성격으로 정확히 예측된다.**
이건 약점 고백이 아니라 **적용 조건을 명시하는 것**이고, TCD가 SSv2에서 NME 열세를
보고한 것과도 완전히 일관된다(`reports/base_heavy_split_result.md`).

논문에서는 이렇게 쓰면 된다 — *"appearance-discriminative한 행동에 대해서는
학습 없는 통계 head가 무거운 학습 방법을 능가하며, motion-discriminative한 경우에는
그렇지 않다. 우리는 그 경계를 정량화한다."*

## ⚠️ 정직한 caveat (논문에 반드시 병기)

1. **트랙이 다르다.** 우리는 백본을 **전혀 학습하지 않는다**(frozen CLIP). TCD·STSP·
   FrameMaker는 백본을 UCF101로 학습한다. 위 표는 **위치 파악용이지 동일 조건 순위가
   아니다.** `reports/pycil_bridge_result.md` §4와 같은 원칙.
2. **CLIP 사전학습 데이터 오염 가능성.** CLIP은 웹 스케일로 학습됐고 UCF101은 YouTube
   기반이라 노출 가능성을 배제할 수 없다. PTM 트랙 공통의 알려진 caveat이며 문헌 관례대로 명기.
3. **ESSENTIAL도 같은 frozen CLIP을 쓰고 95.1이다.** 즉 우리와의 6.3%p 격차는
   백본이 아니라 **그들의 학습형 temporal encoder + memory retrieval 모듈**에서 온다.
   "frozen CLIP이라서 잘 나왔다"로 퉁칠 수 없고, **동시에 개선 여지가 그만큼 있다는 뜻**이다.
4. GRU+A-GEM 비교는 **10×5 열에서만** 수행했다(5×10, 2×25는 미실행 — seed당 480초라
   전 조합은 비용이 큼). 증분 크기를 줄이면 세션이 늘어 GRU 쪽이 더 불리해질 것으로
   예상되므로, 10×5만 보고하는 것이 baseline에 유리한 쪽(보수적)이다.

## 재현

```bash
./scripts/get_ucf101.sh --extract                    # 6.5GB
python3 scripts/extract_ucf101_features.py           # ~56분 (MPS)
python3 dev/run_ucf101_tcd_protocol.py               # 초 단위
python3 dev/run_ucf101_gru_agem.py                   # ~24분 (3 seed, CPU)
```

## 후속

- ~~GRU+A-GEM을 같은 프로토콜로~~ — **완료**(위 절). 격차 +31.45%p.
- **HMDB51 — 착수했으나 보류.** 영상은 구할 수 있으나 **공식 train/test split을 구할 수 없음**(serre-lab 공식 URL이 http/https 모두 HTML만 반환, HF 미러들은 영상만 포함, TCD 저장소에도 없음). 자체 분할을 만들면 "비교 가능한 숫자"라는 목적 자체가 사라지므로 중단. split 확보 시 재개.
- FLOPs·wall-clock을 이 벤치마크 기준으로도 병기(`reports/flops_result.md` 규약 사용)
