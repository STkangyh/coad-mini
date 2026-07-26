# CPU-only 학습 방법 조사 — backprop-free 스트리밍/닫힌해 계열 실측

Generated: 2026-07 (auto, 2차 확장). 질문: "CPU로만 학습하는 다른 방법이 더 있나?" + "더 최신 방법은?"
→ 1차: 고전 계열(NCM/SLDA/RLS) 실측 — Deep SLDA가 A-GEM GRU와 task-aware 동률, class-IL +34% 우위.
→ **2차(§6): 최신(2023–26) 계열 실측 — FeCAM(shared)이 전 지표 신기록: task-aware 0.410,
full-48-way 0.157, 학습 42밀리초.** GRU+A-GEM(0.387/0.105/270초)을 모든 축에서 능가.

> ⚠️ **학습시간 수치 정정(2026-07).** 아래 표의 "학습(초)" 열은 러너 타이머
> (`dev/run_modern_cpu_methods.py:186-194`)가 stage 1의 `model.scores(Xva)` **평가 1회를
> 포함**해 측정한 값이라 **순수 학습 시간이 아니다**. FeCAM은 그 8.7초 중 **99.5%가 평가**이고
> 실제 fit은 **42.4 ms**다. 정확히 귀속된 fit-only 시간과 하드웨어 무관 FLOPs는
> [`reports/flops_result.md`](flops_result.md) 참조. 정확도 수치는 영향 없음.

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

## 6. 2차 확장 — 최신(2023~2026) 방법 조사 + 실측

### 6-1. 문헌 지형 (2024~2026)

| 방법 | venue | 핵심 아이디어 | 우리 스택 실측 |
|---|---|---|---|
| **FeCAM** | NeurIPS 2023 | Mahalanobis 분류 + 공분산 정규화 트릭(Tukey 변환·shrinkage·correlation 정규화) | ✅ 아래 |
| **RanDumb** | NeurIPS 2024 | "랜덤 표현(RBF 커널 근사 RFF)이 온라인 CL로 *학습한* 표현을 이긴다" — 고정 랜덤 임베딩 + 스트리밍 선형 분류기 | ✅ 아래 |
| **AnaCP** | NeurIPS 2025 | 분석적 대조 투영 — gradient 없이 feature 적응까지 수행, **joint-training 상한 도달** 주장 | 문헌만 (반복 투영 재현 무거움) |
| **StPR** | **ICLR 2026** | **exemplar-free 비디오 CIL** — frame-shared semantics 증류 + 시간분해 MoE 라우팅 (UCF101/HMDB51/K400 SOTA) | 문헌만 (backbone 학습 필요) |
| **CSTA** | 2025 | 인과적 시공간 적응 exemplar-free VCIL | 문헌만 |
| EFCIL for SSMs | 2025 | SSM 백본용 exemplar-free CL | 문헌만 (우리 SSM 실험과 연결) |

주목: 최신 흐름 자체가 우리 결론과 같은 방향 — **exemplar-free(버퍼 제거) + frozen/분석적 head**가
2024~26 프론티어이고, 비디오 CIL 최신(StPR)도 리허설을 버렸다.

### 6-2. 실측 (동일 8-stage 프로토콜) — `dev/run_modern_cpu_methods.py`

| 방법 | task-aware | S1 drop | full-48 acc | full F1 | full mAP | 학습(초) |
|---|---|---|---|---|---|---|
| **FeCAM shared-cov (NeurIPS'23)** | **0.410** | +0.000 | **0.157** | **0.144** | **0.120** | 8.7 |
| RanDumb-style RFF(2000)+SLDA (NeurIPS'24, 3 seeds) | 0.395±0.003 | −0.006 | 0.145 | 0.130 | 0.119 | 0.9 |
| Deep SLDA (1차 실측) | 0.390 | +0.005 | 0.141 | 0.123 | 0.102 | 2.1 |
| (참고) GRU + A-GEM | 0.387±0.018 | −0.061 | 0.105 | 0.075 | 0.105 | ~270 |
| FeCAM per-class cov | 0.320 | +0.000 | 0.094 | 0.092 | 0.084 | 8.8 |

### 6-3. 해석

1. **최신 기법의 정규화 트릭이 실제로 더 얹힌다.** FeCAM(shared)의 Tukey 변환 + 공분산
   정규화가 SLDA 대비 task-aware +2.0%p, full acc +1.6%p → **전 지표 신기록**. 이제
   backprop-free 계열이 GRU+A-GEM을 task-aware에서도 명확히 이김(0.410 vs 0.387).
2. **RanDumb 논문 주장이 우리 세팅에서도 재현됨** — 고정 RBF 랜덤 임베딩+선형 head(0.9초
   학습)가 backprop로 학습한 GRU를 이긴다. "온라인 CL에서 표현 *학습* 자체가 이득이 없다"는
   RanDumb의 도발적 결론과 우리 데이터가 일치. (단 1차의 ReLU 랜덤투영은 실패했는데 RBF-RFF는
   성공 — 커널 근사의 기하가 중요, 단순 차원 확장이 아님.)
3. **FeCAM per-class는 실패 — 이유가 유익함.** 클래스당 100개로 512-d 공분산을 클래스별
   추정하면 과적합(0.320). 우리 데이터 규모에선 **공유 공분산이 정답** — "작은 데이터에선
   단순한 쪽이 이긴다"는 프로젝트 전체 패턴의 또 하나의 사례.
4. 미실측 최신(AnaCP·StPR)은 방향 제시용: AnaCP는 "gradient 없이 joint-training 상한"까지
   주장하므로 후속 검증 가치가 있고, StPR은 Ego4D 확장 시 비디오 CIL 최신 비교군.

## 7. 우리 방법(GRU+A-GEM) 대비 차이 요약 (Δ = 대안 − 우리)

학습 시간은 **fit-only로 정정**한 값(평가 제외, `reports/flops_result.md`),
FLOPs는 하드웨어 무관 지표로 병기:

| 방법 | task-aware (Δ) | full-48 acc (Δ) | full F1 (Δ) | 학습(fit only) | train FLOPs | 버퍼 |
|---|---|---|---|---|---|---|
| **FeCAM shared** | 0.410 (**+0.023**) | 0.157 (**+0.052, +50%**) | 0.144 (**+0.069, +92%**) | **42 ms (6,400배↓)** | 2.85 G (1,628배↓) | 불필요 |
| RanDumb RFF | 0.395 (+0.008) | 0.145 (+0.040) | 0.130 (+0.055) | 532 ms (508배↓) | 64.3 G (72배↓) | 불필요 |
| Deep SLDA | 0.390 (+0.003) | 0.141 (+0.036) | 0.123 (+0.048) | 1.9초 (145배↓) | 6.61 G (702배↓) | 불필요 |
| Ridge RLS | 0.373 (−0.014) | 0.143 (+0.038) | 0.122 (+0.047) | 21 ms | 3.06 G (1,514배↓) | 불필요 |
| NCM | 0.378 (−0.009) | 0.124 (+0.019) | 0.105 (+0.030) | 2.4 ms | 44 M (104,791배↓) | 불필요 |
| FeCAM per-class | 0.320 (−0.067) | 0.094 (−0.011) | 0.092 (+0.017) | — | — | 불필요 |
| **우리 (GRU+A-GEM)** | **0.387 (기준)** | **0.105 (기준)** | **0.075 (기준)** | ~270초 | 4.64 T | 필요(50/stage) |

우리가 아직 이기는 축: **S1 backward-transfer**(−0.061, 리허설 고유 효과)와 시퀀스 입력 유지뿐.

## 8. 데모 앱 통합 (실행 완료)

조사에서 끝내지 않고 **FeCAM head를 데모에 실제 탑재**:

- `src/models/fecam_head.py` — FeCAMHead(observe/enroll_class/scores/save/load), 테스트 8개.
- `dev/build_fecam_head.py` — train feature로 head 생성(**fit 0.1초**), `checkpoints/fecam_head.npz`(2.1MB).
  val 재현: task-aware 0.410 / full-48-way 0.157 (실험치와 일치).
- `app.py` — 실시간 자막에 **FeCAM 세 번째 박스(파랑)** 추가(Baseline/A-GEM/FeCAM 삼자 비교),
  `/predict`·`/predict_rt` 응답에 `fecam` 키, `/health`에 fecam 상태.
- **밀리초 enrollment**: `/enroll`이 GRU 경로와 병행으로 FeCAM 프로토타입 등록 수행 —
  **실측 0.17ms**(클립 5개). 평균 벡터 1개 계산이라 기존 클래스 통계를 전혀 건드리지 않음
  = **망각이 구조적으로 불가능한 등록**. 응답에 `fecam_ms` 포함, UI에 표시.
- 재시작 persist(`fecam_head_enrolled.npz`) + `/reset_classes` 연동. Dockerfile이 checkpoints/를
  복사하므로 HF Spaces 재배포 시 자동 포함.

데모 서사: 같은 화면에서 **"순차학습(빨강, 망각) vs 리허설(초록) vs 통계 계열(파랑, 무망각)"**
삼자 비교 + 새 동작을 가르치면 파랑은 0.2ms 만에 배우는 대비를 라이브로 보여줄 수 있음.

## 참고 문헌

- FeCAM: Goswami et al., NeurIPS 2023 — https://arxiv.org/abs/2309.14062
- RanDumb: Prabhu et al., NeurIPS 2024 — https://arxiv.org/abs/2402.08823
- AnaCP: NeurIPS 2025 — https://arxiv.org/abs/2511.13880
- StPR (exemplar-free video CIL): ICLR 2026 — https://arxiv.org/abs/2505.13997
- CSTA (exemplar-free video CIL): 2025 — https://arxiv.org/abs/2501.07236
- Deep SLDA: Hayes & Kanan, CVPR-W 2020 — https://arxiv.org/abs/1909.01520
- SimpleCIL/APER: Zhou et al., IJCV 2024 — https://arxiv.org/abs/2303.07338
- RanPAC: McDonnell et al., NeurIPS 2023 — https://arxiv.org/abs/2307.02251
- ACIL (analytic CIL, recursive least squares): https://arxiv.org/abs/2205.14922
- GACL (exemplar-free generalized analytic CL): https://arxiv.org/abs/2403.15706
- XLDA (edge-scale LDA CL): https://arxiv.org/abs/2307.11317

## 재현

```bash
python3 dev/run_cpu_friendly_methods.py   # 1차: NCM/SLDA/RLS (~10초)
python3 dev/run_modern_cpu_methods.py     # 2차: FeCAM/RanDumb (~1분)
```
