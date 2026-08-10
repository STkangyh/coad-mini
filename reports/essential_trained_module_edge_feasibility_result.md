# ESSENTIAL 같은 학습형 모듈이 엣지에서 가능한가 — "언제 학습하느냐"의 문제

Generated: 2026-08-08. 근거: FeCAM 원문(§ 이전 리포트에서 확인), ESSENTIAL 원문
([arXiv:2508.10896](https://arxiv.org/abs/2508.10896), 이전 세션에서 PDF 직접 확인),
[`cpu_friendly_methods_result.md`](cpu_friendly_methods_result.md),
[`ssv2_accuracy_interpretation_result.md`](ssv2_accuracy_interpretation_result.md)
(새 실험 없음 — 기존 검증된 사실들을 조합한 분석 리포트)

**질문:** `ssv2_accuracy_interpretation_result.md`가 "우리는 선형 천장(24.61%)의 95.7%를
이미 달성했고, 더 올리려면 ESSENTIAL처럼 학습형 temporal encoder가 필요하다"고 결론 냈다.
그런데 **학습형 모듈은 이 프로젝트의 CPU-only 엣지 전제와 애초에 충돌하는 것 아닌가?**

## 1. ESSENTIAL이 실제로 하는 건 "한 번 학습 후 고정"이 아니다

ESSENTIAL 원문을 다시 확인하면:

> *"we employ a dedicated MR module for each task"*

**새 태스크(클래스 묶음)가 들어올 때마다 그 태스크 전용 MR 모듈을 새로 학습**한다(모듈
자체가 잊어버리는 걸 막으려고). 학습에는 별도의 static/temporal matching loss를 쓴다
(원문 §3.2.1). 즉 이건 "미리 한 번 학습해서 냉동 보관"이 아니라 **증분학습 매 세션마다
진짜 backprop이 다시 일어나는 구조**다.

## 2. 왜 이게 엣지에 문제가 되는가 — 이미 실측된 비용

이 프로젝트는 이미 backprop 기반 증분학습의 비용을 직접 쟀다
([`cpu_friendly_methods_result.md`](cpu_friendly_methods_result.md)):

| | 학습 방식 | 학습 시간 |
|---|---|---|
| FeCAM(닫힌 형태) | gradient 0회 | **42.4 ms** |
| GRU+A-GEM(backprop) | epoch 여러 번 | **270초** |

**6,400배 차이.** ESSENTIAL의 MR 모듈도 태스크마다 loss를 정의하고 epoch을 도는 구조라,
성격상 GRU+A-GEM류(수십~수백 초대)에 가까울 것으로 추정된다(ESSENTIAL 자체의 학습
시간은 원문에 없음 — §5의 한계 참고). 이는 이 프로젝트가 목표로 하는 세 축 전부와
정면으로 부딪힌다:

- **CPU-only**: backprop은 GPU 없이 매우 느리다(위 표가 같은 CPU 조건에서 보여줌).
- **실시간(10fps=100ms/frame)**: 새 클래스 등록마다 초 단위 학습이 끼어들면 실시간
  파이프라인 자체가 성립하지 않는다.
- **배터리**: gradient 계산은 순전파(forward)만 하는 것보다 에너지 소모가 훨씬 크다
  (역전파+ optimizer state 갱신 포함).

## 3. 중간 지점 — "한 번 학습 후 고정"은 가능하다

완전히 막힌 건 아니다. temporal encoder를 **한 번만**(오프라인, 베이스 클래스들로) 학습시켜
놓고 **그 이후엔 얼려서 추론 전용**으로 쓰는 건 구조적으로 가능하다 — 이건 사실 지금
CLIP을 쓰는 방식과 같은 패턴이다: CLIP도 어딘가에서 대규모로 한 번 학습됐을 뿐, 엣지에서
학습되는 게 아니다.

이 경우 주장이 어떻게 바뀌는지 명확히 구분해야 한다:

| 주장 | 우리 현재(FeCAM) | "한 번 학습 후 고정"(가상) | ESSENTIAL(실제) |
|---|---|---|---|
| 배포 후 학습 필요 없음 | ✅ | ✅ | ❌(태스크마다 재학습) |
| 표현이 한 번도 SSv2로 학습된 적 없음 | ✅ | ❌(temporal encoder는 학습됨) | ❌ |

**"배포 후 학습 필요 없다"는 핵심 주장은 유지되지만, "표현이 아예 학습된 적 없다"는 더
강한 주장은 포기해야 한다.** 이건 완전히 다른 트랙으로의 이동이다 — `sota_positioning_brief.md §1(f)`가
이미 정리했듯 TCD/STSP/CSTA는 "SSv2로 학습/적응된 표현" 트랙이고, 우리는 지금까지
"한 번도 SSv2를 본 적 없는 frozen CLIP" 트랙이었다. temporal encoder를 사전학습하는
순간 이 트랙 경계를 넘게 된다.

## 4. 이건 이미 열린 과제로 남아있던 아이디어다

`RESEARCH_LOG.md`의 미실행 항목 5번: *"학습형 temporal encoder를 선택적으로 —
backprop-free 주장은 포기하되 상한 확인용."* 이 리포트가 제안하는 "한 번 학습 후 고정"
방향이 정확히 이 항목이다 — 아직 실행되지 않았다.

## 5. 정직한 한계

- **ESSENTIAL 자신의 학습 시간이 원문에 없다.** MR 모듈 학습이 GRU+A-GEM급으로 느릴
  것이라는 §2의 추정은 **구조적 유사성에 근거한 추론이지 실측이 아니다** — 원문을
  다시 검색해도 wall-clock 학습 시간을 보고하지 않는다(`ap_fps_sweep_result.md`가 이미
  확인한 "CIL 문헌은 시간을 안 잰다" 패턴과 일치).
- **"얼린 temporal encoder"의 추론(inference) 비용도 이 프로젝트에서 측정한 적 없다.**
  작은 transformer라도 공짜는 아니다 — `ap_fps_sweep_result.md`에서 OpenCLIP L/14
  인코더가 196ms/frame까지 나왔던 걸 참고하면, temporal encoder를 CPU에서 추론만
  돌리는 것도 상당한 비용일 수 있다. 실제로 얼마인지는 구현하고 재봐야 안다.
- **"한 번 학습 후 고정"이 실제로 정확도를 얼마나 올려주는지도 미확인.** ESSENTIAL의
  48.9%(174클래스, 다른 스케일)를 그대로 기대할 근거는 없다 — 우리 스케일(48클래스)에서
  직접 구현해서 재야 한다.

## 재현

새 실험 없음 — 인용 근거:
```bash
# ESSENTIAL 원문의 "dedicated MR module for each task" 재확인
curl -sL -o essential.pdf https://arxiv.org/pdf/2508.10896
pdftotext -layout essential.pdf essential.txt
grep -in "dedicated MR module\|matching loss" essential.txt
```
