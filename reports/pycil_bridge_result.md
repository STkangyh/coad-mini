# PyCIL 표준 벤치마크 세팅 + 브리지 실험 결과 (CIFAR-100)

Generated: 2026-07 (auto). 목적: "우리 결론이 자체 SSv2 벤치에서만 성립하는 것 아니냐"는
표준 반론에 대비해, PyCIL(표준 CIL 툴박스)의 **동일 프로토콜/스플릿** 위에서 우리
edge-friendly analytic head들을 검증. 인프라는 `benchmarks/pycil/` 참고.

## 1. 구축된 것

| 구성요소 | 상태 |
|---|---|
| PyCIL 체크아웃 (pin f3509b8) + 비-CUDA 패치 | ✅ `setup_pycil.sh`로 전체 재현 (MPS device + cuda_shim 전역 폴백) |
| CIFAR-100 | ✅ 자동 다운로드(원 서버 느릴 때 md5-검증 미러 문서화) |
| PyCIL 스모크 (SimpleCIL, b0=50 inc=10, 2-epoch, MPS) | ✅ **6-task 완주** — top1 curve [10.96, 9.73, 8.59, 7.58, 7.04, 6.42] (에폭 축소 스모크라 낮은 게 정상; 파이프라인 검증 목적) |
| ImageNet-100/1000 | ✅ 스캐폴드 (`setup_imagenet.sh` + `make_imagenet100.py` + config) — 데이터는 라이선스상 수동 |
| 브리지 (우리 head ↔ PyCIL 스플릿) | ✅ 아래 결과 |

## 2. 브리지 결과 — CIFAR-100 class-IL, PyCIL과 동일 스플릿

세팅: **b0=50, increment=10 (6 tasks), seed 1993** — 클래스 순서를 PyCIL의
`DataManager`에서 직접 import (스모크 로그의 순서 `[68, 56, 78, 8, ...]`와 비트 단위
일치 검증 완료). Frozen **CLIP ViT-B/32** feature (512-d), 진짜 class-IL 평가
(task ID 없음, 매 태스크 후 지금까지 본 전체 클래스에서 top-1).

| head (backprop ✕, buffer ✕) | Avg Inc Acc | Last Acc | 학습(초) | per-task |
|---|---|---|---|---|
| NCM prototype | 0.698 | 0.660 | 0.2 | 0.739 → 0.660 |
| Deep SLDA | 0.725 | 0.686 | 0.4 | 0.766 → 0.686 |
| **FeCAM (shared cov)** | **0.769** | **0.733** | ~9* | 0.806 → 0.733 |

\* FeCAM의 1097초는 대부분 **평가**(100클래스 × 10k 테스트 × 6회 반복의 클래스별
Mahalanobis 루프) — 통계 축적(학습) 자체는 초 단위. 평가 루프는 배치화로 쉽게
최적화 가능(추후).

## 3. 핵심 시사점

1. **순서가 SSv2 결과와 완전히 일치: FeCAM > SLDA > NCM.**
   우리 비디오 벤치(0.157 > 0.141 > 0.124 full-48-way)에서 본 서열이 표준 이미지
   벤치에서 그대로 재현 — **우리 발견("frozen feature 위에선 통계 head가 강하고,
   공분산 정규화가 추가 이득")이 자체 벤치 특유 현상이 아님**을 보여주는 교차 검증.

2. **절대 성능도 준수.** FeCAM avg-inc 76.9%는 buffer 0, backprop 0, 학습 수 초로
   얻은 수치. (참고 맥락: from-scratch ResNet을 태스크마다 수 시간 학습 + 2000-exemplar
   buffer를 쓰는 PyCIL 고전 계열의 보고치와 같은 자릿수 — 단 **트랙이 다르므로 직접
   우열 비교는 금지**, 아래 §4.)

3. **잊는 정도가 완만.** FeCAM per-task 곡선 0.806→0.733 (−7.3%p over 5 incremental
   steps) — 클래스별 통계 독립 누적 덕분에 하락이 주로 "클래스 수 증가에 따른 난이도"
   에서 오고 gradient 간섭 망각이 없음.

4. **edge 서사 강화.** 스마트글래스급 예산(frozen encoder + 통계 head)으로 표준
   벤치에서도 이 수준이 나온다는 것 자체가 논문 §experiments의 한 축이 됨.

## 4. 공정성 주석 (논문에 반드시 명기)

- 우리 head는 **web-pretrained frozen CLIP** 사용 = **PTM 기반 CIL 트랙**
  (SimpleCIL/RanPAC/FeCAM 문헌). PyCIL 고전(iCaRL/DER/FOSTER)은 **from-scratch
  ResNet 트랙**. 두 트랙을 한 표에 섞어 우열을 주장하면 안 되고, 트랙 라벨을 달아
  병렬 보고할 것.
- CLIP의 사전학습 데이터에 CIFAR류 이미지가 노출되었을 가능성(데이터 오염)은 PTM
  트랙 공통의 알려진 caveat — 문헌 관례에 따라 명기.
- PyCIL 고전 baseline의 **본 실행**(160-200 epoch × ResNet)은 이 노트북에서 비실용적
  → 스모크로 파이프라인만 검증했고, 본 수치는 CUDA 머신에서 동일 config로 실행 예정.

## 5. 재현

```bash
./benchmarks/pycil/setup_pycil.sh                      # PyCIL + 패치
cd external/PyCIL && PYTORCH_ENABLE_MPS_FALLBACK=1 \
  python3 main.py --config ../../benchmarks/pycil/exps/simplecil_cifar100_smoke.json
python3 benchmarks/pycil/run_analytic_cifar100.py      # 브리지 (feature 캐시 자동)
python3 benchmarks/pycil/run_analytic_cifar100.py --init 10 --inc 10   # 표준 b0=10 변형
```

ImageNet: `benchmarks/pycil/README.md` 참고.
