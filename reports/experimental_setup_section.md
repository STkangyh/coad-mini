# 실험 환경/세팅 (논문 초안용)

Generated: 2026-08-20. 새 실험 없음 — 기존에 검증된 사실만 정리한 작성용 섹션.
근거: [`ap_fps_sweep_result.md`](ap_fps_sweep_result.md),
[`realtime_incremental_result.md`](realtime_incremental_result.md),
[`ssv2_head_curves_result.md`](ssv2_head_curves_result.md),
[`hdc_comparison_result.md`](hdc_comparison_result.md),
`dev/compute_val_metrics.py`, `src/models/fecam_head.py`, arXiv:1212.0402(UCF101),
Goyal et al. ICCV 2017(SSv2), Qualcomm 공식 SSv2 배포 문서.

## 1. 데이터셋

### 1.1 UCF101

Soomro, Zamir, Shah. "UCF101: A Dataset of 101 Human Action Classes From Videos
in The Wild." [arXiv:1212.0402](https://arxiv.org/abs/1212.0402) (2012). 원 논문
Table 직접 확인(PDF 다운로드 + `pdftotext -layout` + grep).

| 항목 | 값 |
|---|---|
| 클래스 수 | 101 |
| 총 클립 수 | 13,320 |
| 클래스당 그룹 수 | 25 |
| 그룹당 클립 수 | 4–7 |
| 총 재생시간 | 1,600분 |
| 클립 길이 범위 | 1.06초 ~ 71.04초 |
| 프레임레이트 | 25 fps |
| 해상도 | 320×240 |
| 출처 | YouTube (2012) |

프로젝트 자체 실측(참고): 평균 클립 길이 약 7.2초(n=15 샘플, `live_query_sim_result.md`) —
위 1.06~71.04초 범위 안에 있어 일관됨.

### 1.2 Something-Something v2 (SSv2)

Goyal et al. "The 'Something Something' Video Database for Learning and
Evaluating Visual Common Sense." ICCV 2017, pp. 5843-5851,
DOI 10.1109/ICCV.2017.622. 배포 스펙은 Qualcomm 공식
[다운로드 안내 문서](https://www.qualcomm.com/developer/software/something-something-v-2-dataset)
및 프로젝트 자체 ffprobe 실측 기준(이번 세션에 직접 확인).

| 항목 | 값 |
|---|---|
| 클래스 수 | 174 |
| 총 비디오 수 | 220,847 |
| train / val / test | 168,913 / 24,777 / 27,157(test는 라벨 없음) |
| 크라우드 작업자 | 5,400명 이상(공식 문서 기준) |
| 프레임레이트 | 12 fps(실측) |
| 코덱/포맷 | VP9, webm |
| 해상도(높이) | 240px |
| 클립 길이 | 약 2~6초(문헌); 실측 샘플 2개 — 3.08초(37프레임), 4.67초(56프레임) |
| 총 다운로드 용량 | 19.4GB |

**⚠️ 미해결 불일치:** ICCV 논문 관련 웹서치 결과에는 "크라우드 작업자 1,300명"이라는
수치가 나오는데, 공식 배포 문서의 "5,400명 이상"과 다르다. 이 차이는 원 논문 PDF를
직접 대조해 확인하지 못한 상태다 — 논문에 넣기 전 원문 재확인 필요.

### 1.3 이 프로젝트에서 실제로 쓰는 서브셋

`data/subset/{train,val}_mini.json` 기준 **48클래스** 서브셋(SSv2 174클래스 중 부분
집합), CIL 프로토콜은 **스테이지당 6클래스 × 8스테이지**로 순차 노출한다
(`CPS=6`, `N_STAGES=8`, `dev/run_hdc_comparison.py` / `dev/compute_val_metrics.py`
공통 정의).

## 2. 하이퍼파라미터

### 2.1 Backbone (특징 추출)

| 이름 | 모델 | 특징 차원 | 비고 |
|---|---|---|---|
| **clip_b32**(배포 중) | `openai/clip-vit-base-patch32` | 512 | 10fps 예산 안 |
| openclip_l14(비교용, 미채택) | `laion/CLIP-ViT-L-14-laion2B-s32B-b82K` | 768 | 인코더만 196ms/frame, 예산 밖 |

두 backbone 모두 **frozen** — 파인튜닝 없음, 파라미터 갱신 0회.

### 2.2 Pooling (프레임 → 윈도우 임베딩)

| 이름 | 배수 | 설명 |
|---|---|---|
| mean | 1× | 프레임 평균 |
| chunks4(배포 중) | 4× | 윈도우를 4구간으로 나눠 구간별 평균 concat |
| chunks3_adjdiff | 5× | 3구간 평균 + 인접 구간 차분 concat |

pooling은 SSv2 정확도 기준으로 선정(`ssv2_pooling_protocol`, `RESEARCH_LOG.md`).

### 2.3 Head (분류기)

| 이름 | 정규화/shrinkage | 비고 |
|---|---|---|
| NCM | 없음(코사인 유사도) | 공분산 미사용, 최하위 baseline |
| SLDA | 등방 shrinkage `1e-2 × I` | 공유 공분산 1개 |
| **FeCAM**(배포 중) | `Σs = Σ + γ1·V1·I + γ2·V2·(1−I)`, γ1=γ2=1 (`SHRINK_1=SHRINK_2=1.0`) | 대각/비대각 분리 shrinkage, Goswami et al. NeurIPS 2023(arXiv:2309.14062) Eq.8 |

세 head 모두 **backprop 없음**(닫힌 형태 통계 갱신) — gradient 계산 0회.

### 2.4 실시간 파이프라인 (배포 구성)

| 항목 | 값 |
|---|---|
| 캡처 간격 | 200ms |
| 프레임 버퍼(ring buffer) | 16프레임 |
| 신규 인코딩 | 도착한 1프레임만(batch=1) |
| 목표 처리율 | 10fps(=100ms/frame 예산) |

### 2.5 기타

| 항목 | 값 |
|---|---|
| 실행 환경 | CPU-only (Apple Silicon 실측 기준, 하드웨어 의존) |
| 난수 시드 | 스크립트별 고정(`seed=0`, 예: HDC 랜덤 투영 행렬) — 시드 분산은 미실측 |

## 3. 평가 지표

### 3.1 정확도 계열

CIL 프로토콜은 48클래스를 8스테이지(스테이지당 6클래스)로 순차 노출한다.

| 지표 | 정의 | 용도 |
|---|---|---|
| stage accuracy | 스테이지 $s$까지 학습한 시점, 그때까지 본 모든 클래스에 대한 val accuracy | 학습 곡선의 한 점 |
| **avg_inc**(헤드라인) | $\frac{1}{S}\sum_{s=1}^{S}\text{acc}_s$ — 8개 스테이지 accuracy 평균 | 대부분의 리포트가 "정확도"로 보고하는 값 |
| last | 마지막 스테이지(48클래스 전부)의 accuracy | 가장 어려운/정직한 최종 성능, 클래스 수가 다른 벤치마크 간 비교 시 사용 |

### 3.2 평가 레짐 (class-IL vs task-IL)

- **FULL 48-way (class-IL)**: 48개 로짓 전체에서 argmax, 테스트 시 태스크(스테이지) ID
  미제공. chance = 1/48. **이 프로젝트의 기본 리포트 값.**
- **TASK-AWARE 6-way (task-IL)**: 샘플이 속한 스테이지의 6클래스로만 argmax 제한(태스크 ID
  제공 가정). chance = 1/6, 더 쉬움. `dev/compute_val_metrics.py`가 참고용으로 병행 계산.

### 3.3 mAP (보조 지표)

`average_precision_score`(macro, one-vs-rest). **주 지표로 쓰지 않음** — FeCAM의 원시
점수(음의 마할라노비스 거리)는 샘플마다 상수 오프셋이 섞여 있어 열(class) 방향 랭킹인
mAP만 왜곡시킨다(행 방향 argmax인 accuracy는 무관). 행별 z-score 정규화로 고치면
accuracy와 같은 순위로 돌아오지만(`ap_fps_sweep_result.md §6`), 다른 모든 리포트가
accuracy를 헤드라인으로 쓰므로 이 논문에서도 accuracy를 주 지표로, mAP는 참고용으로만
쓴다.

### 3.4 속도/자원 지표

| 지표 | 정의 |
|---|---|
| **FPS** | `1000 / (encode_1frame_ms + pool_ms + head_scores_ms)` — CPU, batch=1 스트리밍 기준(배치 처리량이 아님) |
| **fit time** | 헤드를 한 스테이지 분량 데이터로 갱신하는 wall-clock 시간(초) |

## 4. 정직한 한계

- SSv2 크라우드 작업자 수(1,300 vs 5,400)는 원 논문 PDF 미대조 상태 — §1.2 참고.
- 시드 분산(여러 랜덤 시드에 걸친 표준편차)은 이 프로젝트 어디에서도 측정한 적 없다 —
  모든 수치는 단일 시드 실행 기준.
- 하드웨어 의존 수치(FPS, fit time 절대값)는 Apple Silicon CPU 기준이며 다른 하드웨어에서는
  달라질 수 있다.
