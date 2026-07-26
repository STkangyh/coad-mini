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
| 클래스 순서 | 랜덤 셔플 **3개 평균** (TCD 자체 시드 1000 / 1993 / 2021) |
| 지표 | average incremental accuracy (매 스텝 후 지금까지 본 전체 클래스 top-1, **task ID 없음**) |
| exemplar | **0** (통계만 누적) |

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
4. GRU+A-GEM은 아직 이 프로토콜에서 안 돌렸다 — 우리 자체 baseline과의 비교는 후속.

## 재현

```bash
./scripts/get_ucf101.sh --extract                    # 6.5GB
python3 scripts/extract_ucf101_features.py           # ~56분 (MPS)
python3 dev/run_ucf101_tcd_protocol.py               # 초 단위
```

## 후속

- **GRU+A-GEM을 같은 프로토콜로** — 우리 자체 baseline 대비 우위를 표준 벤치에서도 확인
- **HMDB51** 추가(2GB, 같은 계보가 전부 보고) — TCD 프로토콜은 26 base + 5/25
- FLOPs·wall-clock을 이 벤치마크 기준으로도 병기(`reports/flops_result.md` 규약 사용)
