# 최근(2025-2026) CPU-only/backprop-free CL 문헌 — coad-mini에 써먹을 만한 것

Generated: 2026-08-08. 원문 직접 확인(PDF 다운로드 + grep): FoRo
([arXiv:2509.01533](https://arxiv.org/abs/2509.01533), ACM MM 2025), ImageHD
([arXiv:2604.21280](https://arxiv.org/pdf/2604.21280), 2026). 웹서치만 하고 원문
미확인(아래 §4): Mono-Forward([arXiv:2511.01061](https://arxiv.org/abs/2511.01061)),
AFCL/DeepAFL/APFL 계열. 새 실험 없음 — 문헌 조사 리포트.

**질문:** CPU-only/backprop-free CL 쪽 최근 논문 중 이 프로젝트(frozen CLIP + FeCAM,
video CIL, edge)에 실제로 가져다 쓸 만한 게 있는가?

## 1. ⭐ ImageHD — 우리가 가장 걱정하는 지점(O(D³) 공분산)을 아예 없앤 대안

[arXiv:2604.21280](https://arxiv.org/pdf/2604.21280), 2026. **Hyperdimensional Computing
(HDC)** 기반 — 데이터를 고차원 이진 벡터로 인코딩하고 bundling/binding 연산만으로
학습·추론한다. 원문이 명시적으로 겨냥하는 문제:

> *"introducing additional control state and cubic-time consolidation"* — 즉 기존
> 통계 기반 방법(FeCAM류)의 **O(D³) 역행렬 계산**을 문제로 지목한다.

**실측(원문 표 기준):** **CORe50**(우리와 성격이 가까운 video CIL 벤치마크 — 객체당
11개 비디오 세션, Kinect2 20fps, 우리 UCF101처럼 정적 편향에 가까움)에서 최적화된
CPU 대비 **40.4배 속도, 383배 에너지 효율**, GPU 대비도 4.84배/105.1배.

**왜 관련 있나:** 우리는 이번 세션에서 D=2560 chunks3+adjdiff의 "매 프레임 공분산
갱신"이 head 비용만 533ms(예산의 5배)까지 나온다는 걸 실측했다
(`ssv2_video_access_result.md §10`) — 그래서 "가끔만 갱신"으로 우회했다. HDC는 **애초에
공분산 자체가 없어서** 이 문제가 구조적으로 발생하지 않는다. 우리가 우회책을 찾은 문제를
HDC는 원천 제거한다.

**우리 head 계열(NCM→SLDA→FeCAM, "공분산 구조를 얼마나 쓰는가" 사다리,
`fecam_vs_slda_covariance_result.md`)에 이어 붙일 수 있는 네 번째 지점**이다 — 단, 이번엔
"공분산을 더 쓰는" 방향이 아니라 "통계 자체를 다른 표현(hypervector)으로 바꾸는" 완전히
다른 축이라 그 사다리에 그냥 끼워 넣을 순 없고 별도 비교가 필요하다.

**바로 못 쓰는 이유(정직하게):** HDC는 원문 자신이 인정하듯 *"HDC accuracy on complex
visual tasks still lags behind that [of other methods]"* — 속도·에너지는 압도적이지만
정확도는 아직 FeCAM류에 못 미치는 경향이 문헌에 알려져 있다. SSv2처럼 이미 천장(24.6%)이
낮은 벤치에서 정확도를 더 깎는 트레이드오프가 맞는 선택인지는 우리가 직접 재봐야 안다.

## 2. FoRo — 우리 head 계열과 직접 벤치마크가 겹친다

[arXiv:2509.01533](https://arxiv.org/abs/2509.01533), ACM MM 2025. Frozen **ViT-B/16**
위에 두 요소를 얹는다:
1. **Prompt tuning via CMA-ES**(gradient-free 진화 전략) — 입력 레이어에 학습 가능한
   prompt를 삽입해 backbone 가중치는 안 건드리고 태스크별 분포 이동을 흡수.
2. **Knowledge Encoding**: nonlinear random projection + **recursive least squares** —
   우리 프로젝트의 Ridge-RLS(`cpu_friendly_methods_result.md` 1세대 비교, 지금은
   은퇴시킨 head)와 정확히 같은 계열.

**실측(원문 Table 기준):** CIFAR-100·**ImageNet-R**·CUB-200 — 이 셋은
`pycil_bridge_result.md §5`가 SimpleCIL/APER/RanPAC과 비교했던 바로 그 벤치마크
집합이다. FoRo는 평균 정확도 **84.5%**로 비교군(Fine-tune, LwF, L2P, DSG, LayUP 등)
중 최고를 보고한다. ImageNet-R T=25(긴 태스크 시퀀스)에서 **76.7%** vs LwF **58.0%**.

**⚠️ 정직한 한계 — "gradient-free"가 "빠르다"를 보장하지 않는다.** CMA-ES는 매 태스크마다
**후보 prompt 집단(population)을 샘플링하고 반복(iteration)하며 fitness를 평가**하는
구조다(원문 Algorithm 부분 확인). 즉 backprop은 없지만 **여러 번의 forward pass를 반복하는
루프**라, FeCAM의 "한 번에 끝나는 닫힌 형태 fit(42ms)"과는 비용 구조가 다르다. 정확한
population size·iteration 수·wall-clock 시간이 원문에 없어(§4의 한계) 실제로 얼마나
느린지는 우리가 재보기 전엔 모른다.

## 3. 이 둘을 어떻게 위치시킬까

| | 우리(FeCAM) | FoRo | ImageHD |
|---|---|---|---|
| backbone | frozen CLIP | frozen ViT-B/16 | (원문에서 backbone 특정 안 확인) |
| head 학습 | 닫힌 형태(공분산 통계) | RLS + CMA-ES prompt | HDC bundling/binding |
| gradient 사용 | 0회 | 0회(단, CMA-ES는 반복 최적화) | 0회 |
| 강점 | 우리가 이미 실측 완료 | 표준 벤치마크에서 최고 정확도 | 압도적 속도·에너지 |
| 약점 | D³ 갱신 비용(§10에서 대응함) | 반복 최적화 비용 미상 | 정확도 열세(원문 자인) |

**셋 다 "gradient 없음"이라는 큰 우산 아래 있지만 서로 다른 트레이드오프를 진다.** 우리
FeCAM은 "한 번에 끝나는 통계 계산 vs D³ 갱신 비용"의 트레이드오프였고, FoRo는 "prompt로
표현력을 더 얻는 대신 반복 최적화 비용"을, ImageHD는 "속도·에너지를 극대화하는 대신
정확도를 희생"한다.

## 4. 원문 미확인 — 참고만 할 것

- **Mono-Forward**([arXiv:2511.01061](https://arxiv.org/abs/2511.01061)) — backprop
  대비 정확도 우위 + 에너지 41%↓ + 속도 34%↑를 주장하지만, **MLP 전용**으로 보이고(원문
  초록 기준) continual learning 적용 여부가 불명확 — CIL 적용 가능성은 별도 확인 필요.
- **AFCL/DeepAFL/APFL**(analytic federated CL 계열, arXiv:2505.12245 등) — RLS 기반
  닫힌 형태 갱신이라는 점에서 우리와 같은 계열이지만 **연합학습(federated)** 세팅이라
  "여러 클라이언트"라는 전제가 이 프로젝트(단일 엣지 디바이스)와 안 맞는다. 다만 그
  RLS 갱신 수식 자체는 참고할 여지가 있다.

## 5. 실행 가능한 다음 단계 (제안, 아직 미실행)

1. ~~**ImageHD의 HDC 인코딩을 4번째 head 후보로 직접 구현·비교**~~ — **완료.**
   [`hdc_comparison_result.md`](hdc_comparison_result.md): NCM보다는 낫지만(21.47%)
   SLDA(22.77%)·FeCAM(24.19%)을 못 넘었고, 속도도 FeCAM보다 9배 느렸다(numpy 구현이
   이진 hypervector의 비트 연산 이점을 못 살려서 — §3 참고). 채택 안 함.
2. **FoRo의 RLS 부분만 분리해서 비교** — CMA-ES prompt tuning 없이 knowledge encoding
   메커니즘만 떼어내면 사실상 Ridge-RLS의 변형이라, 이미 있던 비교(§1 표)에 다시
   끼워 넣기 쉽다. CMA-ES 전체를 도입하는 건 반복 최적화 비용부터 실측해야 판단 가능.
3. 두 경우 다 **먼저 사용자 확인 후 진행** — 이 세션에서 Ridge-RLS를 사용자 확인 없이
   되살렸다가 지적받은 전례가 있어, head 후보를 늘리는 결정은 특히 조심할 것.

## 정직한 한계

- **문헌 검색은 웹서치 기준**이라 최근 논문을 놓쳤을 수 있다 — arXiv 전수조사가 아니다.
- **FoRo·ImageHD 모두 이미지(정적) CIL 벤치마크에서만 검증됐다** — SSv2 같은 시간 편향
  벤치마크에서의 성능은 원문에 없다. 우리 세팅으로 옮기면 결과가 달라질 수 있다.
- **ImageHD의 backbone이 무엇인지, CLIP과 결합 가능한지 확인 못 했다** — PDF 일부만
  파싱됐고(1.1MB 바이너리, 표/그림이 많아 pdftotext -layout으로 다 못 건짐), §1의
  수치는 grep으로 찾은 부분만 반영했다.
- **FoRo의 CMA-ES 반복 비용(population×iteration×forward-pass)을 실제 숫자로
  확인 못 했다** — "gradient 없음"과 "빠름"을 동일시하지 않도록 §2에 명시했다.

## 재현(원문 확인 과정)

```bash
curl -sL -o foro.pdf https://www.arxiv.org/pdf/2509.01533
curl -sL -o imagehd.pdf https://arxiv.org/pdf/2604.21280
pdftotext -layout foro.pdf foro.txt && pdftotext -layout imagehd.pdf imagehd.txt
grep -in "CIFAR\|accuracy\|CMA-ES" foro.txt
grep -in "CORe50\|speedup\|energy" imagehd.txt
```
