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
Mahalanobis 루프) — 통계 축적(학습) 자체는 초 단위.
**→ 해결됨(2026-07):** `scores()`를 이차형식 전개 + 캐시로 바꿔 FLOPs 40배·실측 100배 이상
단축했다(정확도 불변, `reports/flops_result.md` 발견 4·5). 위 학습(초) 열은 최적화 **이전**
측정치이므로, 재실행하면 크게 줄어든다 — 정확도 수치는 영향 없음.

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

## 5. ⭐ 문헌 대조 (2026-08-08 추가) — PTM-frozen 트랙 SOTA와 나란히

**질문:** SSv2 AP/FPS 스윕([`ap_fps_sweep_result.md`](ap_fps_sweep_result.md))을 만들면서
"다른 모델들과 비교해달라"는 요청이 있었다. PyCIL의 실제 모델(SimpleCIL/ACIL/APER 등)을
SSv2에 직접 돌리는 방안도 검토했으나, **PyCIL은 비디오 데이터셋을 지원하지 않고**
(`DataManager`가 CIFAR/ImageNet 이미지 전용 — SSv2용 어댑터를 새로 짜야 해서 엔지니어링
비용이 큼), SSv2에서의 직접 실행은 보류하고 **문헌 수치 대조**로 방향을 잡았다. 그러면 이
비교는 SSv2가 아니라 **이 리포트가 이미 다루는 CIFAR-100**에서 해야 앞뒤가 맞는다 — PTM
계열 논문(SimpleCIL/APER, RanPAC)이 실제로 보고하는 벤치마크가 CIFAR-100이기 때문.

**출처(원문 PDF 직접 확인, 표절대 재인용 아님):**
- SimpleCIL/APER — Zhou et al., *Revisiting CIL with Pre-Trained Models*
  ([arXiv:2303.07338](https://arxiv.org/abs/2303.07338)), Table 1
- RanPAC — McDonnell et al. ([arXiv:2307.02251](https://arxiv.org/abs/2307.02251)), Table 1
- ACIL — Zhuang et al., NeurIPS 2022 ([arXiv:2205.14922](https://arxiv.org/abs/2205.14922)), §IV-C

### 5.1 결과 — PTM-frozen 트랙 (우리와 같은 트랙, CIFAR-100)

| method | backbone | Ā(avg inc) | 마지막 태스크 |
|---|---|---|---|
| SimpleCIL | ViT-B/16-IN21K, frozen | 87.57% | 81.26% |
| APER w/ Adapter (최고 변형) | 〃 + adapter 미세조정 | 90.65% | 85.15% |
| RanPAC | ViT-B/16-IN21K + PETL + random projection | — | **92.2%**\* |
| **FeCAM (우리, 위 §2)** | **frozen CLIP ViT-B/32** | **76.9%** | **73.3%** |

\* RanPAC 논문 Table 1은 **마지막 태스크 정확도만** 보고한다(원문 §5.2 "we report final
accuracy, A_T, in the main paper" — 확인함). SimpleCIL/APER처럼 avg/last를 병기하지 않으므로
직접 그 열에만 놓았다.

**해석:** 우리 FeCAM(76.9/73.3)은 이 표에서 가장 낮다. 그런데 조건이 동일하지 않다 —
① 우리 backbone은 **CLIP ViT-B/32**(512-d), 저들은 **ViT-B/16-IN21K**(768-d, ImageNet-21K
지도학습 사전학습) — 백본 자체가 다르다. ② 프로토콜도 다르다: 우리는 b0=50/inc=10(6task),
저들은 B0Inc5(20task, 첫 태스크부터 5클래스). 태스크가 잘게 쪼개질수록 초반 클래스 수가
적어 두 표의 숫자는 **직접 뺄셈 비교 대상이 아니다.** ③ RanPAC·APER는 **PETL로 소량
파라미터를 튜닝**한다(완전한 backprop-free가 아님) — 우리·SimpleCIL만 순수 backprop-free.

**그럼에도 남는 사실:** SimpleCIL이 우리 FeCAM보다 순수 "frozen + 통계 head" 조건에서
CIFAR-100을 10%p 이상 앞선다. 이건 **ViT-B/16-IN21K가 CLIP B/32보다 CIFAR류 자연 이미지에
더 잘 맞는 표현을 준다**는 뜻일 가능성이 높다(IN21K 지도학습이 CIFAR100의 상위 카테고리와
더 가까운 반면, CLIP은 웹 캡션 대조학습이라 다른 분포). **이건 head의 문제가 아니라
backbone 선택의 문제**라는 걸 문헌이 보여준다 — [`ap_fps_sweep_result.md §3`](ap_fps_sweep_result.md)
이 SSv2에서 "속도는 backbone이 결정한다"고 한 것과 같은 결의 결론이 정확도 축에서도 나온
셈: **backbone이 이 파이프라인에서 가장 레버리지가 큰 선택이다.**

### 5.2 참고용 — 다른 트랙 (ACIL, 직접 비교 금지)

| method | backbone | 학습 방식 | CIFAR-100 Ā |
|---|---|---|---|
| ACIL (5-phase) | ResNet-32, **from-scratch backprop 기반 학습 후** analytic 증분 | base task는 SGD 160epoch | 66.30% |
| ACIL (25-phase) | 〃 | 〃 | 65.95% |

ACIL은 **base task를 ResNet-32로 처음부터 backprop 학습**(160 epoch)한 뒤에만 analytic
증분이 붙는다 — "backprop-free"라는 이름값과 달리 base 단계는 무거운 학습이 필요하다.
frozen web-pretrained 표현만 쓰는 우리·SimpleCIL·RanPAC과는 **완전히 다른 트랙**이라
§5.1과 나란히 놓지 않았다(`pycil_bridge_result.md §4`가 이미 세운 원칙 그대로 적용).

### 5.3 ⭐ 속도 비교는 문헌으로 불가능하다

SimpleCIL/APER, RanPAC, ACIL **세 논문 모두 원문 전체를 검색했지만 FPS·inference
latency·ms/frame 수치가 단 하나도 없다** — "training time"이라는 정성적 언급(그림 반경
등)만 있을 뿐, 표로 정리된 속도 수치는 전무하다. 이건 이미 `pycil_survey_edge_gap.md`가
CIL 분야 전반에 대해 지적한 것과 정확히 같은 공백이다 — **CIL 문헌은 정확도만 보고하고
시간·에너지는 거의 안 잰다.** 그래서 이 세션의 [`ap_fps_sweep_result.md`](ap_fps_sweep_result.md)
같은 AP-vs-FPS 스윕을 문헌과 나란히 놓을 방법 자체가 없다 — 속도 축에서는 비교 대상이
없다는 것 자체가, 우리가 이 축을 실측한다는 것의 상대적 가치를 보여준다.

## 6. 재현

```bash
./benchmarks/pycil/setup_pycil.sh                      # PyCIL + 패치
cd external/PyCIL && PYTORCH_ENABLE_MPS_FALLBACK=1 \
  python3 main.py --config ../../benchmarks/pycil/exps/simplecil_cifar100_smoke.json
python3 benchmarks/pycil/run_analytic_cifar100.py      # 브리지 (feature 캐시 자동)
python3 benchmarks/pycil/run_analytic_cifar100.py --init 10 --inc 10   # 표준 b0=10 변형
```

ImageNet: `benchmarks/pycil/README.md` 참고.
