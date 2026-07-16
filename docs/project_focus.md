# 프로젝트 포커스 (공식 프레이밍)

Confirmed: 2026-07 (교수님 면담 후 확정).

## 정의

- **주제:**
  1. Video classification
  2. Video classification **with continual learning**
- **제약: CPU-only / Embedded GPU** — 스마트글래스, 로봇 등 **edge 단에서 continual
  learning을 수행**하는 것이 핵심. 클라우드/대형 GPU 학습은 스코프 밖.
- **목적:**
  1. **제품**: 어떤 상황이든/원하는 상황에서 잘 되는 것 (데모/시스템 완성도)
  2. **논문**: 우리가 제안하는 방법/세팅이 잘 된다는 것을 보여줌 (실증)

## 왜 이 프레이밍이 강한가 — 기존 증거가 전부 여기에 정렬됨

이 렌즈를 끼우면 지금까지의 발견들이 "흥미로운 부산물"에서 **논지의 핵심 증거**로 바뀐다:

| 기존 발견 | edge-CL 프레이밍에서의 의미 |
|---|---|
| 아키텍처 복잡도 무익 (Attention −0.103, SSM −0.040, 용량 양방향 무병목) | edge에선 애초에 큰 모델을 못 씀 — "작게 가도 손해가 없다"는 실증적 허가 |
| **backprop-free 계열이 GRU+A-GEM을 능가** (FeCAM 0.410/0.157, fit 0.1~9초) | **edge CL의 정답 후보**: backprop 불가/부담스러운 기기에서 통계 누적만으로 CL 완성 |
| CPU 실측 (604K params, 4.5분/seed→FeCAM 0.1초, RAM <1GB, 추론 6.4win/s) | 편의가 아니라 **제약 조건 하의 core evidence** |
| 밀리초 enrollment (0.17ms, 무망각) | 스마트글래스에서 "사용자가 새 동작을 현장에서 가르치는" 시나리오의 직접 구현 |
| frozen CLIP + 경량 head 레시피 | vCLIMB SOTA(PIVOT/ESSENTIAL)와 같은 재료 — 단, 그들은 무거운 모듈, 우리는 edge 예산 |
| 데이터가 지배적 레버 (+0.06~0.07) | edge에선 모델 키우기보다 **현장 데이터 수집·등록**이 옳은 투자라는 설계 지침 |
| 데모 (웹캠 실시간, few-shot 등록, OOD 배지, Docker/CPU) | 스마트글래스/로봇 시나리오의 **제품 프로토타입** (목적 1) |

## 논문 포지셔닝 (목적 2)

> **"Edge-feasible video continual learning: backprop-free analytic heads on frozen
> features match or beat gradient-based CL at a fraction of the compute."**

- 2024–26 문헌 흐름(RanDumb, analytic CL, exemplar-free)과 합류하되, 그 논문들에 없는
  **edge 제약 + video + 실측 시스템** 조합이 우리의 차별점.
- 정직한 비교축: 정확도 SOTA(StPR/ESSENTIAL, 무거운 backbone 학습)와 겨루지 않고,
  **"같은 edge 예산에서 무엇이 최선인가"**를 묻는다 — 이 질문엔 우리가 데이터를 갖고 있음.

## 이 프레이밍이 드러내는 갭 (다음 단계 후보)

1. **실제 embedded 하드웨어 실측 없음** — 지금 수치는 Mac CPU. Jetson(Orin Nano 등)/
   Raspberry Pi 급에서 학습·추론 시간, (가능하면) 전력 측정이 논문 설득력을 크게 올림.
2. **CLIP 인코더가 추론 병목** (156ms/window) — edge용 경량 비전 인코더(MobileCLIP,
   TinyCLIP 등)로 feature 교체 시 정확도-지연 트레이드오프 측정.
   **착수함 → `reports/mobileclip_result.md`.** CPU 지연 측정은 완료(중요한 반전:
   MobileCLIP-S0가 파라미터 8배 작은데도 일반 PyTorch CPU에선 CLIP B/32보다
   13~18배 느림 — CoreML/ANE 전용 커널을 겨냥한 설계라 범용 CPU에선 이점이 안 살아남).
   다운스트림 정확도 비교는 원본 SSv2 영상 접근이 막혀 **보류 중**(재개 커맨드 리포트에 있음).
3. **진짜 스트리밍/online 프로토콜** — 통계 계열은 이미 single-pass. epoch 반복 없는
   one-pass 세팅으로 전 방법 통일 비교하면 edge 서사가 완성됨.
4. **에너지/메모리 프로파일** — RAM peak은 있고(0.95GB), 전력은 미측정.
5. 데모의 embedded 배포 (예: Jetson에서 Docker 이미지 구동 확인).

## 관련 문서

- 실측 증거 총람: [`reports/measured_evidence.md`](../reports/measured_evidence.md)
- backprop-free 연구: [`reports/cpu_friendly_methods_result.md`](../reports/cpu_friendly_methods_result.md)
- SOTA 지형/포지셔닝: [`reports/sota_positioning_brief.md`](../reports/sota_positioning_brief.md)
- 연구 일지: [`research_log.md`](research_log.md)
