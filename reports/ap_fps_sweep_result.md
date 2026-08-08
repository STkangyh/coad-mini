# 정확도(mAP)와 속도(FPS)를 한 평면에 — 24개 조합 중 무엇이 Pareto인가

Generated: 2026-08-06. 스크립트: [`dev/run_ap_fps_sweep.py`](../dev/run_ap_fps_sweep.py)
원시결과: `reports/ap_fps_sweep_raw.json`

**질문:** 지금 배포 중인 조합(CLIP B/32 + chunks4 + FeCAM)은 정말 Pareto-optimal인가?

**왜 이 실험이 필요했나:** 이 프로젝트는 지금까지 **축을 하나씩 따로** 최적화해왔다 —
pooling은 SSv2 정확도로 골랐고([`ssv2_pooling_protocol`](RESEARCH_LOG.md)), head는 pooling을
고정한 채 비교했고([`ssv2_head_curves`](ssv2_head_curves_result.md)), 속도는 배포 조합
하나만 쟀다([`realtime_incremental_result`](realtime_incremental_result.md)). 세 축을 한
평면에 올린 적이 없어서, 현재 배포 조합은 **세 번의 독립적인 1축 결정을 조립한 결과**일 뿐
"아무것도 이걸 동시에 이기지 못한다"가 검증된 적은 없었다.

**축:** backbone(clip_b32 512-d / openclip_l14 768-d) × pooling(mean / chunks4 /
chunks3_adjdiff) × head(NCM / SLDA / Ridge-RLS / FeCAM) = **24조합**. 전부 backprop-free.

**FPS 모델:** [`realtime_incremental_result.md`](realtime_incremental_result.md)가 확립한
배포 경로 그대로 — ring buffer라 매 스텝 **새 프레임 1장만** 인코딩하고, 16프레임 버퍼를
pooling해서 채점한다. `fps = 1000 / (encode_1frame_ms + pool_ms + head_scores_ms)`,
**CPU·batch=1**(엣지 조건이자 스트리밍의 실제 조건 — 배치 처리량을 쓰면 인코더가 실제보다
2배 빨라 보인다).

---

## 1. ⚠️ 먼저 — 하마터면 틀린 결론을 낼 뻔했다 (측정 아티팩트)

첫 실행 결과는 **Ridge-RLS가 mAP에서 전 구간 압도, FeCAM이 최하위**였다:

| head | mAP(원시) | 정확도 |
|---|---|---|
| Ridge-RLS | **0.1549** | 0.1838 |
| FeCAM | 0.0278 | **0.2356** |

*(clip_b32 + chunks4 기준)*

**수상한 지점:** FeCAM의 mAP가 pooling·backbone을 뭘 바꿔도 0.028~0.034에서 **거의 안
움직였다** — 같은 조건에서 정확도는 0.157 → 0.275로 올랐는데. 이 불일치가 힌트였다.

**원인:** macro AP는 **각 클래스 열 안에서 샘플들을 줄세운다**. 그런데 FeCAM의 점수는 음의
마할라노비스 거리라 **샘플마다 절대 스케일이 다르다** — 실측해보니 행 평균이 −1455 ~ −278로
**오프셋 편차가 1176**인데 행 내부 편차는 4.5~14.3뿐이었다. 즉 원시 AP는
*"이 샘플이 클래스 c인가"*가 아니라 ***"이 샘플이 전반적으로 모든 클래스에서 먼가"*** 를
재고 있었다. top-1 정확도는 행 내부 argmax라 이 오프셋에 **완전히 면역**이라, 두 지표가
정반대로 나온 것.

**수정:** 채점 행렬을 **행 z-score**(샘플별로 클래스 축을 표준화)한 뒤 AP를 계산한다. argmax가
안 바뀌므로 **정확도는 그대로**고, 모든 head에 **동일하게** 적용하므로 "각 head가 우연히 가진
점수 스케일"이 아니라 **랭킹 품질 자체**를 비교하게 된다. FeCAM mAP 0.0278 → **0.2152**(7.7배).

`mAP_raw`도 raw JSON에 함께 남겼다 — 아티팩트가 보이도록.

**교훈:** 서로 다른 head를 AP로 비교할 때 **점수 스케일 정규화는 선택이 아니라 필수**다.
정확도만 봤으면 이 함정 자체를 못 봤을 것이고, AP만 봤으면 **정반대 결론**을 냈을 것이다.

## 2. 결과 — FeCAM이 12/12 전부 승리

수정 후, **모든 backbone×pooling 조합에서 FeCAM이 mAP·정확도 둘 다 1위**다.

### CLIP B/32 (38.5 fps)

| pooling | NCM | SLDA | Ridge-RLS | **FeCAM** |
|---|---|---|---|---|
| mean | 0.0877 | 0.1008 | 0.0970 | **0.1248** |
| chunks4 | 0.1199 | 0.1748 | 0.1687 | **0.2152** |
| chunks3+adjdiff | 0.1395 | 0.1871 | 0.1792 | **0.2220** |

### OpenCLIP L/14 (4.8 fps)

| pooling | NCM | SLDA | Ridge-RLS | **FeCAM** |
|---|---|---|---|---|
| mean | 0.1060 | 0.1166 | 0.1135 | **0.1425** |
| chunks4 | 0.1467 | 0.2063 | 0.1946 | **0.2481** |
| chunks3+adjdiff | 0.1722 | 0.2227 | 0.2104 | **0.2543** |

## 3. ⭐ 핵심 — FPS를 결정하는 건 오직 backbone이다

| 항목 | 프레임당 비용 | 비중 |
|---|---|---|
| CLIP B/32 인코딩 | 25.9 ms | **99%** |
| OpenCLIP L/14 인코딩 | 208.5 ms | **99.9%** |
| pooling | 0.01~0.1 ms | 무시 가능 |
| head 채점(FeCAM, 최대 D=3840) | 0.1~3.5 ms | 최대 1% |

**pooling과 head를 아무리 바꿔도 fps는 안 움직인다**(clip_b32 안에서 34.3~38.6, 대부분 38.5).
`realtime_incremental_result.md`의 "인코더가 91%"가 이 스윕에서 **99%로 더 극단화**돼 재확인됐다
— 그때는 D=512 mean pooling 기준이었고, 여기선 D=3840까지 키워도 head 비용이 여전히 1% 안쪽이다.

→ **분류기 쪽을 최적화해서 얻을 속도는 없다.** 속도 결정은 전적으로 backbone 선택이다.

## 4. Pareto — 10 fps 예산이 절반을 잘라낸다

프로젝트의 실시간 목표는 **10 fps**(100 ms/frame)다. OpenCLIP L/14는 **4.8 fps로 예산의 절반도
못 낸다** — 즉 mAP가 아무리 높아도 **배포 후보에서 구조적으로 탈락**이다.

| | 최고 mAP | fps | 예산 |
|---|---|---|---|
| CLIP B/32 + chunks3+adjdiff + FeCAM | 0.2220 | 36.2 | ✅ 3.6배 여유 |
| OpenCLIP L/14 + chunks3+adjdiff + FeCAM | **0.2543** | 4.7 | ❌ 절반 미달 |

**L/14가 사는 mAP는 +0.032(+14% 상대)인데 대가는 8배 느림이다.** 예산을 지키는 조합 중에서는
**CLIP B/32 + FeCAM**이 pooling과 무관하게 최선이다 — 이건 현재 배포 조합을 **정당화한다**.

## 5. ⚠️ chunks3+adjdiff가 chunks4보다 높게 나왔지만 — 바꾸면 안 된다

두 backbone 모두에서 chunks3+adjdiff가 chunks4를 근소하게 앞선다(B/32: mAP +0.0068,
정확도 +1.15pp). **그런데 이걸 근거로 배포 pooling을 바꾸는 건 정확히 자체 감사 2번이
경계했던 선택 편향이다.**

- 이 스윕은 **val에서 평가**했다. 반면 pooling 선택은 `run_ssv2_pooling_protocol.py`가
  **train을 fit/select로 쪼개 held-out으로** 골랐고(val은 건드리지 않음), 거기서 두 후보는
  **통계적으로 동률**이었다(44.87±3.40 vs 44.43±3.21 — 표준편차가 차이의 8배).
- 지금 val에서 보이는 +0.7~1.2pp는 그 동률 범위 **안**이다. 여기서 val 기준으로 갈아타면
  **val을 선택에 쓴 것**이 되어, 이후 val 수치의 의미가 사라진다.
- 게다가 이번 결과는 **단일 실행·시드 없음**이다 — pooling 프로토콜이 3시드로 오차막대를
  뽑았던 것과 비교 불가.

**결론: 현재 chunks4 유지가 옳다.** chunks3+adjdiff로 바꿀 근거가 되려면 **held-out
프로토콜을 다시 돌려서** 이겨야 한다.

## 6. 부수적 관찰 — head 간 순위가 pooling에 따라 안 바뀐다

FeCAM > SLDA > Ridge-RLS > NCM 순서가 **6개 backbone×pooling 조합 전부에서 동일**하다
(단 하나의 예외: clip_b32+mean에서 SLDA 0.1008 vs Ridge 0.0970으로 거의 붙음). head의
상대적 우열은 **표현이 좋아져도 유지되는 안정적 성질**로 보인다 — pooling·backbone을 바꿔가며
head를 재선택할 필요는 없다는 뜻.

## 7. 정직한 한계

- **단일 실행, 시드 없음.** head는 전부 결정론적(닫힌 형태)이라 fit 자체는 재현되지만,
  클래스 순서를 바꿨을 때의 분산은 안 쟀다. 다만 FeCAM은 순서 무관 항등식이 이미
  확립돼 있어([`bounds_context_result.md §2`](bounds_context_result.md)) FeCAM 행에 대해서는
  이 한계가 적용되지 않는다.
- **SSv2 48클래스 subset만.** 174클래스 전체(이번에 확보)로는 안 돌렸다 — 기존 pooling·head
  결과들과 비교 가능하도록 48클래스를 유지했다.
- **인코더 타이밍은 이 맥(Apple Silicon) CPU 기준.** 절대 fps는 하드웨어에 따라 달라진다.
  다만 backbone 간 **비율**(B/32가 L/14보다 8배 빠름)은 이식성이 있다.
- **SLDA는 배치 fit으로 구현**했다(원본 스트리밍 구현은 D=3840에서 O(N·D²)로 수분 소요).
  running mean/covariance라 최종 통계는 동일하고, 이 스크립트가 재는 건 **predict 시간**이라
  결론에 영향 없다.
- **openclip_l14 특징은 48클래스 subset만** 추출돼 있다 — 이번 세션에 재추출하지 않았으므로
  provenance가 없다(clip_b32는 재추출·스탬핑 완료, [`ssv2_video_access_result.md §3`](ssv2_video_access_result.md)).

## 재현

```bash
python3 dev/run_ap_fps_sweep.py
```

```bash
python3 dev/run_ap_fps_sweep.py --backbones clip_b32 --skip-encoder-timing
```
