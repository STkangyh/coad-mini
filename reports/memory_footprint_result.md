# 메모리는 어디로 가나 — 인코더 vs 헤드, 그리고 임시 할당

Generated: 2026-07-31. 스크립트: [`dev/bench_realtime_incremental.py --memory`](../dev/bench_realtime_incremental.py)
원시결과: `reports/realtime_memory_raw.json`

**질문(자체 감사 개선점 10):** 우리 RAM 근거는 **`0.95 GB` 하나뿐**이고, 그건 GRU 시절
**학습** 실행 수치다. 배포 시스템은 형태가 다르고, pooling을 바꾸면 헤드가 D²로 커진다.
"edge"를 주장하려면 바이트가 어디로 가는지 알아야 한다.

---

## 1. 배포 프로세스 RSS

| 단계 | RSS | 증가 |
|---|---|---|
| 임포트 후 (numpy·torch·transformers) | 211 MB | — |
| **+ CLIP vision tower** (87.5M params) | 470 MB | **+259 MB** |
| **+ FeCAM 헤드** (D=2048, 서빙 상태) | 703 MB | **+233 MB** |

**인코더와 헤드가 비슷한 규모**다. 시간 축에서는 인코더가 91%를 먹었지만
(`realtime_incremental_result.md` §3) **메모리에서는 그렇지 않다.**

## 2. 헤드가 실제로 들고 있는 것 — D×D 두 장이 전부

| 구성요소 | 크기 |
|---|---|
| `cov_sum` (D×D float64) | 33.6 MB |
| `precision` (D×D float64) | 33.6 MB |
| `means` (256×D) | 4.2 MB |
| mean 캐시 (48×D ×2) | 1.6 MB |
| 나머지 | ~0 |
| **합계** | **72.9 MB** |

**클래스 수가 아니라 D가 지배한다.** 256클래스 슬롯을 다 잡아도 means는 4.2 MB뿐이고,
D×D 두 장이 67 MB다. 실시간 측정에서 "클래스 500개까지 속도가 평평"했던 것과 같은 구조 —
**제품이 클래스를 늘려도 메모리도 안 깨진다.**

| pooling | D | 체크포인트 | 헤드 RAM |
|---|---|---|---|
| mean | 512 | 1.6 MB | 5.6 MB |
| **chunks4 (배포)** | **2048** | **12.6 MB** | **72.9 MB** |
| chunks3+adjdiff | 2560 | 18.4 MB | 112.1 MB |

D²이라 pooling을 키우면 **4배씩** 뛴다. chunks4를 고른 게 정확도뿐 아니라 메모리에서도
이득이었다(chunks3+adjdiff 대비 −39 MB).

## 3. ⭐ 측정하다 찾은 것 — 임시 할당이 실제 배열의 5배였다

헤드 배열 합계는 72.9 MB인데 **RSS 증가는 388 MB**였다. gc를 돌려도 안 줄었다.

원인은 누수가 아니라 **임시 D×D 배열의 양**이다. `_cov_terms()`가

```python
raw = cov_sum / n            off = raw - diag(diag(raw))
cov = raw + a*eye(d) + b*(1-eye(d))          # eye 2장 + 곱 2장 + 합 2장
prec = inv(cov / outer(sd,sd))               # outer, 나눗셈, inv
penalty = trace(prec @ (raw / outer(sd,sd))) # 나눗셈 + D×D 행렬곱
```

**10장 넘는 D×D 배열**을 거쳐가고, D=2048에서 장당 33.6 MB다. 할당자는 해제된 아레나를
OS에 돌려주지 않으므로 RSS가 그대로 남는다.

**수정 — 결과는 그대로, 할당만 줄임:**
- shrinkage를 in-place로: `eye(d)` 2장과 중간 합 제거
- 상관행렬 변환을 in-place로: `cov /= outer(sd,sd)`를 새 배열 없이
- **`tr(P @ S)`를 행렬곱 없이**: `tr(AB) = Σ A∘Bᵀ` → `einsum("ij,ji->", ...)`.
  D×D 할당이 사라지고 **O(D³) → O(D²)** 로도 줄어든다
- `load()`의 대칭화도 in-place (`cov + triu(cov,1).T`가 2장을 더 쓰고 있었다)

| 단계 | 수정 전 | 수정 후 | 절감 |
|---|---|---|---|
| 헤드 로드 | 173.9 MB | 115.6 MB | **−58 MB** |
| 첫 `scores()` | 388.2 MB | 297.0 MB | **−91 MB (−23%)** |

**정확도 불변 검증:** 15시드 합성 대조에서 precision 최대 상대차 **5.8e-16**(1 ULP),
sd **0.00**, penalty 9.98e-16, argmax 전부 일치, `_cov_sum` 무오염.
실데이터 4,702샘플에서 **argmax 100% 일치, 정확도 0.235644로 소수점 6자리까지 동일.**

남은 297 MB 중 224 MB는 LAPACK의 역행렬 워크스페이스와 잔여 임시분으로, 더 줄이려면
알고리즘을 바꿔야 한다(예: Cholesky 기반 solve로 명시적 역행렬 회피) — 미착수.

## 4. edge 주장과의 정합성 (물음표 9에 대한 답)

| 대상 | 총 RSS | 판정 |
|---|---|---|
| **Jetson AGX Orin** (32/64 GB) | 703 MB | 여유 |
| **Raspberry Pi 5** (4/8 GB) | 703 MB | 가능 |
| 폰 SoC (SparCL이 쓴 Galaxy S20급) | 703 MB | 빠듯하나 가능 |
| **MCU급 스마트글래스** (수십 MB) | 703 MB | **불가** |

**정직하게:** 우리 시스템은 **SBC/로봇 급**이지 **MCU 급이 아니다.** 703 MB 중 259 MB가
CLIP이고 233 MB가 헤드(임시 포함)라, 헤드만 줄여도 인코더가 남는다. 논문에서 "AI 글래스"를
말하려면 이 수치를 병기하거나 주장 대상을 로봇/SBC로 좁혀야 한다.

이건 Jetson 대여 계획과도 이어진다 — Orin은 이 예산에 여유롭지만, **그래서 글래스 급을
입증하지는 못한다.**

## 5. 정직한 한계

- **RSS는 할당자 정책에 의존한다.** macOS malloc 기준이고 Linux/musl에서는 다를 수 있다.
  Jetson 실측 때 같이 확인할 것.
- `max_classes=256` 슬롯을 미리 잡는 구조라 means가 실제 48클래스보다 5배 크다.
  줄이면 3.4 MB 절감 — 전체에서는 미미.
- 학습(전체 fit) 피크는 별도다. 여기 수치는 **서빙 경로**만이다.
- float32로 내리면 D×D가 절반이 되지만 역행렬 조건수에 영향이 있어 미검증.

## 재현

```bash
python3 dev/bench_realtime_incremental.py --memory
```
