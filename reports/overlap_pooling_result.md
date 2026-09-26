# Overlapping 구간 pooling — 도움 안 됨, 오히려 겹칠수록 단조적으로 나빠짐

Generated: 2026-08-21. 스크립트: [`dev/run_overlap_pooling.py`](../dev/run_overlap_pooling.py)
(신규) · 원시결과: `reports/overlap_pooling_raw.json`
배경: [`paper_outline_draft.md`](paper_outline_draft.md) 상단 "⚠️ 먼저 결정해야 할 것 — overlap"
(Notion 개요에 "overlap이 중요"·"overlap 유무" ablation이 있었으나 구현이 없었음)

**질문:** `src/models/fecam_head.py`의 `_segment_means(w, k)`는 항상 **비중첩** 구간으로만
윈도우를 나눈다. 구간을 넓혀서 이웃 구간과 겹치게(overlap) 만들면, 구간 경계에서 잘리는
동작을 완화해 정확도가 더 오를까?

## 1. 방법 — 차원을 고정한 채 overlap만 바꾼다

기존 `run_ssv2_pooling_protocol.py`가 이미 감사받은 방법론(선택은 TRAIN 내부
fit/select 분할로만, val은 절대 안 봄 — `ssv2_temporal_pooling_result.md §10(a)`가
찾아낸 1.14%p 선택 편향을 피하기 위함)을 **그대로 재사용**했다.

**overlap 정의:** k구간 pooling에서 각 구간의 원래 길이(`seg_len = T/k`, T=16)에
`overlap` 배수를 곱한 만큼을 **양쪽으로** 넓힌다.

```python
pad = seg_len * overlap / 2.0
lo = round(구간 시작 - pad); hi = round(구간 끝 + pad)   # [0, T]로 클리핑
```

`overlap=0.0`은 기존 `_segment_means`와 (반올림 방식만 다르고) 사실상 같은 경계가
된다 — 이걸 "같은 코드 경로의 zero-overlap 기준점"으로 스윕에 같이 넣어서, overlap
효과가 "구현이 다른 두 함수를 비교해서 생긴 차이"가 아님을 보장했다.

**출력 차원은 그대로 k×D다** — `mean+std`(같은 차원, 순서 정보 없음)가 겨우 +1.04pp였던
`ssv2_temporal_pooling_result.md §4(b)`의 통제와 같은 논리로, overlap의 효과가 있다면
그건 "차원이 늘어서"가 아니라 **경계를 완화한 것 자체**여야 한다.

**스윕 범위:** k∈{3,4}(기존 스윕이 찾은 정점 부근), overlap∈{0.0, 0.25, 0.5, 1.0, 2.0}.
adjdiff와의 결합은 이번 스코프에서 제외(§5).

## 2. 결과 — 5시드, TRAIN 내부 select 세트 (avg_inc %)

| pooling | k=3 | k=4 |
|---|---|---|
| overlap 0.0(zero-overlap 기준점) | 44.13 ± 3.29 | 44.87 ± 3.40 |
| overlap 0.25 | 44.09 ± 3.16 | 44.87 ± 3.40 |
| overlap 0.5 | 43.41 ± 3.29 | 44.69 ± 3.48 |
| overlap 1.0 | 42.57 ± 3.14 | 43.45 ± 3.18 |
| overlap 2.0 | 40.12 ± 3.06 | 42.07 ± 3.07 |
| (참고) 기존 `chunks3`/`chunks4` 구현 | 43.92 ± 3.23 | 44.87 ± 3.40 |
| (참고) `mean` | 35.02 ± 3.01 | — |

**overlap=0.25는 k=4에서 문자 그대로 무효과다** — T=16, k=4면 구간 길이가 4프레임이고
`pad = 4×0.25/2 = 0.5프레임`, 반올림하면 0이 돼서 경계가 전혀 안 움직인다(44.87로 완전
동일). 이건 "약간의 overlap이 효과 없다"가 아니라 **프레임 해상도가 너무 낮아 이
overlap 값이 아예 표현이 안 된 것** — 순수한 양자화 artifact다.

**overlap이 커질수록(0.5 이상) 단조적으로 나빠진다** — k=3, k=4 둘 다에서 일관되게.
extremes(0.0→2.0) 비교:

| k | Δ(0.0→2.0) | pooled sd | 효과크기 |
|---|---|---|---|
| 3 | −4.01pp | 3.18 | **1.26× sd** |
| 4 | −2.80pp | 3.24 | **0.86× sd** |

## 3. val 확정 수치 — 선택된 config는 기존 배포본과 동일

TRAIN 내부 선택에서 뽑힌 건 이번에도 **`chunks4`** (동률로 `chunks4_ov0.25`도 함께,
하지만 위에서 확인했듯 이건 사실상 같은 구성). val(5시드, 2 프로토콜) 재확인:

| 프로토콜 | chunks4 vs mean | chunks4_ov0.25 vs chunks4_ov0.0(overlap만의 차이) |
|---|---|---|
| uniform-8×6 | **+6.91pp** avg_inc / +7.83pp last (3.1× sd) | +0.00pp (0.0× sd) |
| base-heavy-24+4×6 | **+7.66pp** avg_inc / +7.83pp last (8.7× sd) | +0.00pp (0.0× sd) |

`chunks4 vs mean`의 +6.91/+7.66/+7.83 수치가 **기존 `ssv2_temporal_pooling_result.md §10`의
23.56%(+7.83pp) 헤드라인과 정확히 일치** — 이번 스크립트가 기존 방법론을 올바르게
재현했다는 자체 검증이기도 하다.

## 4. 해석 — 왜 겹칠수록 나빠지는가

`ssv2_temporal_pooling_result.md §4`가 이미 밝힌 chunks-pooling의 작동 원리를 그대로
적용하면 설명된다: **순서가 살아남는 이유는 각 구간이 출력 벡터의 "자기 자리"를
독점하기 때문**이다(영상을 뒤집으면 구간 1과 k가 자리를 바꿔서 다른 벡터가 됨). overlap을
키우면 이웃 구간들이 점점 더 많은 프레임을 공유하게 되고, 그 구간들의 평균 벡터끼리
점점 더 비슷해진다(상관된다) — **자리는 여전히 다르지만 담긴 내용이 서로 닮아간다.**
극단적으로 overlap이 무한히 커지면 모든 구간이 전체 윈도우 평균에 수렴해 `mean`과
동일해질 것이다(차원만 k배로 낭비).

실제로 `overlap=2.0`(k=4에서 구간 폭이 원래의 3배, 16프레임 중 12프레임)에서도
42.07%로 `mean`의 35.02%보다는 한참 높다 — **완전히 붕괴하지는 않았지만, 겹칠수록
그 방향(mean 쪽)으로 단조 이동**한다는 게 이 결과가 보여주는 것이다.

## 5. 판정 — 채택 안 함

이 스코프(k=3,4 · overlap 0~2.0 · adjdiff 미결합)에서 overlap은:
- 작은 값에서는 **양자화 때문에 아예 효과가 없거나**(k=4, ov=0.25),
- 큰 값에서는 **일관되게 손해**(k=3에서 1.26×sd, k=4에서 0.86×sd — 노이즈로 완전히
  치부할 수도 없지만 `ssv2_temporal_pooling_result.md §10(b)`가 "상위 변형은 구분 안 됨"
  판정 기준으로 썼던 것과 비슷한 애매한 중간 지대다. 다만 **방향이 5개 overlap 값
  전부에서 두 k 모두 일관되게 하락**이라, 우연한 노이즈라기엔 패턴이 너무 깔끔하다).

**paper_outline_draft.md 상단의 미해결 항목에 대한 답: overlap을 도입할 근거가 없다.**
Introduction·Ablation에서 overlap 문장은 빼거나, "시도했으나 개선 없음"으로 명시하는 쪽을
권한다 — HDC 때와 같은 원칙(음성 결과도 정직하게 보고).

## 6. 정직한 한계

- **adjdiff와의 결합은 안 봤다.** 기존 배포본은 `chunks3_adjdiff`처럼 구간 평균 +
  인접 차분을 함께 쓴다 — overlap이 "차분" 항에는 다르게 작용할 수 있어(겹치는 구간의
  차분은 신호가 더 흐려질 수도, 노이즈에 덜 민감해질 수도 있음) 이 조합은 미검증.
- **T=16이 고정이라 overlap 해상도가 거칠다.** k=4, ov=0.25가 양자화로 무효과가 된 것처럼,
  프레임 수가 적어 overlap의 미세한 값들을 제대로 표현하지 못한다 — 더 긴 윈도우(더 많은
  프레임)라면 다른 결과가 나올 수 있다.
- **큰 overlap의 효과크기(0.86~1.26× sd)가 확정적이지 않다.** "5개 값 모두 일관되게
  하락"이라는 패턴 자체는 근거가 되지만, 개별 지점 하나하나는 이 프로젝트의 다른 곳(예:
  `ssv2_temporal_pooling_result.md §10(b)`, 1위-2위 차 0.43pp/sd 3.4pp)이 "구분 안 됨"으로
  판정했던 것과 비슷한 크기다.
- **k=2 이하, overlap>2.0은 안 봤다.** 스윕 범위를 기존 최적 구간 수(3~4) 근방으로
  좁혔다 — 다른 k에서는 overlap이 다르게 작동할 여지가 이론적으로 남아있다.

## 재현

```bash
python3 dev/run_overlap_pooling.py                    # 기본 5시드
python3 dev/run_overlap_pooling.py --seeds 0 1         # 빠른 확인용 2시드
```
