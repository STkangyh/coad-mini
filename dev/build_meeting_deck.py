"""Assemble the meeting deck, inlining the figure SVGs (CSP blocks external files)."""
import json
from pathlib import Path

HERE = Path(__file__).parent
FIGS = json.loads((HERE / "figs.json").read_text())

HEAD = """<title>coad-mini 진행 보고 — 2026-07-20 이후</title>
<style>
:root{
  --ground:#fbfbfa; --surface:#ffffff; --ink:#14181c; --ink-2:#59636e;
  --ink-3:#8b959f; --accent:#2a78d6; --accent-soft:#eaf2fc;
  --rule:#e4e7e4; --warn:#9a5b00; --warn-soft:#fdf4e7; --figure-bg:#fcfcfb;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --ground:#15171a; --surface:#1c1f23; --ink:#eaeef2; --ink-2:#9aa5b1;
    --ink-3:#6c757f; --accent:#5b9df0; --accent-soft:#172536;
    --rule:#2a2f35; --warn:#d99a3c; --warn-soft:#2a2115;
  }
}
:root[data-theme="dark"]{
  --ground:#15171a; --surface:#1c1f23; --ink:#eaeef2; --ink-2:#9aa5b1;
  --ink-3:#6c757f; --accent:#5b9df0; --accent-soft:#172536;
  --rule:#2a2f35; --warn:#d99a3c; --warn-soft:#2a2115;
}
*{box-sizing:border-box}
body{
  margin:0; background:var(--ground); color:var(--ink);
  font-family:-apple-system,"Apple SD Gothic Neo","Pretendard",system-ui,"Segoe UI",sans-serif;
  font-size:16px; line-height:1.7; -webkit-font-smoothing:antialiased;
}
.wrap{max-width:920px; margin:0 auto; padding:56px 24px 96px}
.col{max-width:68ch}
h1,h2,h3{text-wrap:balance; line-height:1.25; margin:0}
h1{font-size:clamp(28px,4.4vw,40px); font-weight:760; letter-spacing:-.022em}
h2{font-size:22px; font-weight:700; letter-spacing:-.014em}
h3{font-size:16px; font-weight:680; letter-spacing:-.006em}
p{margin:0}
.eyebrow{
  font-size:12px; font-weight:640; letter-spacing:.1em; text-transform:uppercase;
  color:var(--ink-3);
}
.lede{color:var(--ink-2); font-size:17px}
.muted{color:var(--ink-2)}
.small{font-size:14px}
b,strong{font-weight:660}
.num{font-family:ui-monospace,"SF Mono",Menlo,monospace; font-variant-numeric:tabular-nums}

header{display:flex; flex-direction:column; gap:14px; padding-bottom:28px}
.meta{display:flex; flex-wrap:wrap; gap:8px 18px; color:var(--ink-3); font-size:13px}

/* headline stats — summary before detail */
.stats{
  display:grid; gap:1px; background:var(--rule); border:1px solid var(--rule);
  border-radius:10px; overflow:hidden; grid-template-columns:repeat(3,1fr);
  margin:28px 0 8px;
}
.stat{background:var(--surface); padding:20px 18px; display:flex; flex-direction:column; gap:5px}
.stat .v{font-size:30px; font-weight:720; letter-spacing:-.02em; color:var(--accent);
  font-family:ui-monospace,"SF Mono",Menlo,monospace; font-variant-numeric:tabular-nums}
.stat .k{font-size:13px; color:var(--ink-2); line-height:1.45}
@media (max-width:640px){ .stats{grid-template-columns:1fr} }

section{padding-top:44px; display:flex; flex-direction:column; gap:16px}
section > h2{display:flex; gap:12px; align-items:baseline}
.sn{color:var(--accent); font-weight:700; font-size:15px;
  font-family:ui-monospace,"SF Mono",Menlo,monospace}
hr{border:0; border-top:1px solid var(--rule); margin:0}

/* figures sit on their own light card in both themes — charts are documents */
figure{margin:0; background:var(--figure-bg); border:1px solid var(--rule);
  border-radius:10px; padding:14px 14px 8px; overflow-x:auto}
svg.fig{width:100%; height:auto; display:block}
figcaption{font-size:13px; color:#59636e; padding:8px 4px 4px; line-height:1.55}

.tablewrap{overflow-x:auto; border:1px solid var(--rule); border-radius:10px; background:var(--surface)}
table{border-collapse:collapse; width:100%; font-size:14.5px}
th,td{text-align:left; padding:10px 14px; border-bottom:1px solid var(--rule)}
tbody tr:last-child td{border-bottom:0}
th{font-size:12.5px; font-weight:640; color:var(--ink-2); letter-spacing:.02em;
  background:var(--ground)}
td.n{text-align:right; font-family:ui-monospace,"SF Mono",Menlo,monospace;
  font-variant-numeric:tabular-nums}
tr.us td{background:var(--accent-soft); font-weight:640}
tr.us td:first-child{box-shadow:inset 3px 0 0 var(--accent)}

.callout{border:1px solid var(--rule); border-left:3px solid var(--accent);
  background:var(--surface); border-radius:0 10px 10px 0; padding:16px 18px;
  display:flex; flex-direction:column; gap:8px}
.callout.warn{border-left-color:var(--warn); background:var(--warn-soft)}
.callout .t{font-weight:660; font-size:14.5px}
.callout p{font-size:14.5px; color:var(--ink-2)}
.callout.warn p{color:var(--ink)}

ul,ol{margin:0; padding-left:1.25em; display:flex; flex-direction:column; gap:7px}
li{padding-left:2px}
li::marker{color:var(--ink-3)}
code{font-family:ui-monospace,"SF Mono",Menlo,monospace; font-size:.88em;
  background:var(--accent-soft); padding:1px 5px; border-radius:4px}
a{color:var(--accent); text-underline-offset:2px}
a:focus-visible,summary:focus-visible{outline:2px solid var(--accent); outline-offset:3px; border-radius:3px}
footer{margin-top:56px; padding-top:20px; border-top:1px solid var(--rule);
  color:var(--ink-3); font-size:13px}
@media (prefers-reduced-motion:reduce){*{animation:none!important; transition:none!important}}
</style>
"""


def fig(name, caption):
    return f'<figure>{FIGS[name]}<figcaption>{caption}</figcaption></figure>'


BODY = f"""
<div class="wrap">
<header>
  <div class="eyebrow">coad-mini · 진행 보고</div>
  <h1>표준 벤치마크 진입 — UCF101에서 공개 방법 중 2위</h1>
  <p class="lede col">2026-07-20 이후 작업 정리. 지난 미팅의 결론(&ldquo;backprop 없는 통계 head가
  GRU+A-GEM보다 낫다&rdquo;)이 <b>우리 데이터에서만 성립하는지</b>를 검증하는 것이 목표였고,
  표준 벤치마크에서도 성립함을 확인했습니다.</p>
  <div class="meta">
    <span class="num">21 commits</span><span class="num">37 files</span>
    <span class="num">+4,266 / −31</span><span class="num">70 tests passed</span>
  </div>
</header>

<div class="stats">
  <div class="stat"><span class="v">88.84</span>
    <span class="k">UCF101 평균 증분 정확도<br>공개 방법 중 <b>2위</b></span></div>
  <div class="stat"><span class="v">+31.45</span>
    <span class="k">%p — 우리 baseline<br>GRU+A-GEM 대비</span></div>
  <div class="stat"><span class="v">0.3s</span>
    <span class="k">학습 시간<br>backprop·exemplar 0</span></div>
</div>

<section>
  <h2><span class="sn">01</span> 표준 벤치마크에 들어갔다</h2>
  <div class="col"><p>지금까지 우리 숫자는 전부 <b>직접 고른 48클래스 SSv2 subset</b>이라,
  남의 논문과 &ldquo;서열&rdquo;만 얘기할 수 있었고 숫자를 나란히 놓을 수 없었습니다.
  <b>UCF101</b>은 원문을 읽은 네 논문(TCD·STSP·CSTA·ESSENTIAL)이 전부 같은 프로토콜로
  보고하는 벤치마크입니다.</p></div>

  <div class="callout">
    <div class="t">프로토콜을 &ldquo;비슷하게&rdquo;가 아니라 <b>똑같이</b> 맞췄습니다</div>
    <p>TCD 저장소의 <code>class_list.pkl</code>이
    <code>RandomState(1000).permutation(101)</code>과 <b>101개 전부 일치</b> —
    우리가 쓴 클래스 순서가 그들이 실제로 쓴 순서입니다. 실행 스크립트에서
    base 51 · 증분 10 · exemplar 클래스당 5개까지 확인해 그대로 적용했습니다.</p>
  </div>

  <div class="tablewrap"><table>
    <thead><tr><th>방법</th><th>백본</th><th>exemplar</th><th style="text-align:right">UCF101 10×5</th></tr></thead>
    <tbody>
      <tr><td>TCD (ICCV&rsquo;21)</td><td>ResNet-34+TSM</td><td>있음</td><td class="n">74.89</td></tr>
      <tr><td>FrameMaker (NeurIPS&rsquo;22)</td><td>TSM</td><td>있음</td><td class="n">78.13</td></tr>
      <tr><td>STSP (ECCV&rsquo;24)</td><td>TSM</td><td>0</td><td class="n">81.15</td></tr>
      <tr><td>ST-prompt</td><td>CLIP</td><td>없음</td><td class="n">84.80</td></tr>
      <tr class="us"><td>우리 — frozen CLIP + FeCAM</td><td>frozen CLIP</td><td>0</td><td class="n">88.84</td></tr>
      <tr><td>ESSENTIAL (ICCV&rsquo;25 Highlight)</td><td>frozen CLIP</td><td>sparse+prompt</td><td class="n">95.10</td></tr>
    </tbody>
  </table></div>
  <p class="col small muted">TCD 대비 +14.0, STSP 대비 +7.7. <b>backprop 0회, exemplar 0개,
  백본 학습 0회, fit 0.3초</b>로 얻은 수치입니다.</p>
</section>

<section>
  <h2><span class="sn">02</span> 같은 벤치마크에서 우리 baseline과 직접 비교</h2>
  {fig("ucf101_tcd_curves", "노란 선(GRU+A-GEM)이 첫 증분에서 수직으로 떨어지는 반면 세 통계 head는 거의 평평합니다. TCD의 세 클래스 순서 평균, task ID 없는 진짜 class-IL.")}
  <div class="col"><p>중요한 건 최종 숫자가 아니라 <b>모양</b>입니다. GRU는 처음 51클래스를
  <b>84.6%</b>로 잘 배웁니다 — FeCAM(90.1)과 5.6점 차이뿐입니다. 그런데 클래스가 10개
  추가되는 순간 <b>60.1%로 떨어지고</b> 이후 50% 근처에 정체합니다.
  FeCAM은 같은 지점에서 <b>0.08점</b> 떨어집니다.</p></div>

  <div class="tablewrap"><table>
    <thead><tr><th>방법</th><th style="text-align:right">평균 증분 정확도</th><th style="text-align:right">최종</th><th style="text-align:right">학습 시간</th></tr></thead>
    <tbody>
      <tr><td>GRU + A-GEM</td><td class="n">57.39 ±0.16</td><td class="n">52.30</td><td class="n">480 s</td></tr>
      <tr class="us"><td>FeCAM</td><td class="n">88.84 ±0.52</td><td class="n">87.44</td><td class="n">0.3 s</td></tr>
      <tr><td class="muted">차이</td><td class="n">+31.45 %p</td><td class="n">+35.14 %p</td><td class="n">1,600×</td></tr>
    </tbody>
  </table></div>
  <p class="col"><b>문제는 표현력이 아니라 망각</b>이라는 것이 그림 한 장으로 드러납니다.</p>

  <div class="callout warn">
    <div class="t">정직하게 병기할 것</div>
    <p>우리 GRU+A-GEM 57.39는 위 표의 <b>모든 공개 방법보다 낮습니다</b>(최하위 iCaRL 70.6).
    당연합니다 — 저들은 비디오 CIL 전용 설계이고 우리 baseline은 단순 GRU+A-GEM입니다.
    주장은 &ldquo;우리 baseline이 강하다&rdquo;가 아니라
    <b>&ldquo;같은 frozen feature 위에서는 backprop을 없애는 쪽이 낫다&rdquo;</b>이며,
    이 비교가 정확히 그것을 보여줍니다.</p>
  </div>
</section>

<section>
  <h2><span class="sn">03</span> 우리만 가진 성질 — 증분 크기에 불변</h2>
  {fig("increment_sensitivity", "클래스를 10개씩 주다가 2개씩으로 잘게 쪼갤 때의 정확도 변화. 각 논문이 자기 표에 보고한 값.")}
  <div class="col"><p>클래스별 통계가 <b>독립적으로 누적</b>되므로 언제 몇 개씩 들어오는지가
  최종 상태에 영향을 주지 않습니다. gradient 간섭이 없어 세션 경계 자체가 무의미합니다.</p>
  <p style="margin-top:12px"><b>edge 시나리오에 직결됩니다</b> — 스마트글래스에서 사용자가
  새 동작을 <b>한 번에 하나씩</b> 가르쳐도 손해가 없다는 뜻입니다. 다른 방법은
  그럴수록 불리해집니다.</p></div>
</section>

<section>
  <h2><span class="sn">04</span> 적용 범위를 정량화했다</h2>
  <div class="col"><p>같은 head, 같은 인코더인데 결과가 극단적으로 다릅니다.
  이 대비가 오히려 우리 방법이 어디까지 통하는지를 규정합니다.</p></div>
  <div class="tablewrap"><table>
    <thead><tr><th>벤치마크</th><th>성격</th><th style="text-align:right">우리 FeCAM</th></tr></thead>
    <tbody>
      <tr class="us"><td>UCF101</td><td>static-biased — 외형으로 구별</td><td class="n">88.84</td></tr>
      <tr><td>SSv2 (48-class subset)</td><td>temporal-biased — 모션으로만 구별</td><td class="n">15.70</td></tr>
    </tbody>
  </table></div>
  <div class="col"><p class="small muted">&ldquo;static / temporal-biased&rdquo;는 우리가 만든 용어가 아니라
  <b>ESSENTIAL 논문이 자기 §4.1에서 쓰는 분류</b>입니다.</p>
  <p style="margin-top:12px">논문 문장: <i>&ldquo;외형으로 구별되는 행동에는 학습 없는 통계 head가
  무거운 학습 방법을 능가하고, 모션으로만 구별되는 행동에는 그렇지 않다.
  우리는 그 경계를 정량화한다.&rdquo;</i></p></div>
</section>

<section>
  <h2><span class="sn">05</span> 선행연구 검증 — 반례 확인, 그리고 주장 축소</h2>
  <div class="col">
    <h3>반례를 직접 재검증했습니다</h3>
    <p class="muted" style="margin-top:6px">TCD(ICCV&rsquo;21)가 SSv2에서 <b>우리와 반대 결과</b>를
    보고합니다 — prototype 방식이 CNN보다 못하다고요. 그 조건(base가 크고 증분이 작은 구조)을
    우리 데이터에 이식해 다시 돌렸으나 <b>반전은 없었고 격차가 오히려 벌어졌습니다</b>
    (+0.052 → +0.073). 이유도 원문에서 확정했습니다: TCD는 백본을 SSv2로 계속 학습시켜
    feature에 모션 정보가 들어있고 평균이 그것을 파괴합니다.
    <b>우리는 frozen이라 애초에 잃을 것이 없습니다.</b></p>
  </div>
  <div class="callout warn">
    <div class="t">주장을 좁혔습니다</div>
    <p>처음에는 &ldquo;아무도 edge CPU에서 CL 학습 비용을 재지 않았다&rdquo;고 판단했으나,
    확인해보니 <b>틀렸습니다</b>. SparCL(NeurIPS&rsquo;22)이 갤럭시 S20 CPU에서 실제로
    측정했습니다(3.1× 가속). 그래서 남는 것만 정리했습니다:</p>
    <ul class="small" style="margin-top:4px">
      <li><b>SparCL은 이미지</b> — 비디오 CIL 네 편은 CPU·지연·전력을 하나도 재지 않음</li>
      <li><b>SparCL은 backprop을 희소화</b>할 뿐 제거하지 않음 — 우리는 제거</li>
      <li><b>&ldquo;몇 초에 학습되는가&rdquo;</b>라는 절대값을 보고한 연구가 셋 다 없음</li>
    </ul>
    <p>→ 우리 기여는 <b>&ldquo;비디오 × backprop-free × 절대 CPU 시간&rdquo;의 교집합</b>입니다.</p>
  </div>
</section>

<section>
  <h2><span class="sn">06</span> 측정을 문헌 단위로 맞추고, 우리 오류를 고쳤다</h2>
  <div class="col"><p>효율 주장이 전부 wall-clock이라 다른 논문 옆에 놓을 수 없었습니다.
  SparCL·BudgetCL이 쓰는 <b>FLOPs</b>로 단위를 맞췄습니다 —
  <b>FeCAM이 GRU보다 연산량이 1,628배 적습니다.</b></p></div>
  <div class="callout warn">
    <div class="t">발견한 우리 수치 오류 (정확도는 영향 없음)</div>
    <p>학습 시간 타이머 안에 <b>평가 호출</b>이 들어가 있었습니다. FeCAM 학습을 8.7초로
    보고했는데 <b>그중 99.5%가 평가였고 실제는 42밀리초</b>입니다. &ldquo;30배 빠르다&rdquo;고
    한 것이 오히려 <b>과소평가</b>였고 실제로는 6,400배입니다.
    FLOPs를 따로 계산해보다 드러난 것이라, 앞으로 <b>두 지표를 함께 보고</b>하는 것이
    이런 실수를 잡는 장치가 됩니다.</p>
  </div>
  <div class="col"><p>이어서 FeCAM 채점 코드를 수식 전개로 최적화했습니다 —
  <b>윈도우당 14.4ms → 0.058ms (248배)</b>, 출력은 완전히 동일(최대 상대오차 6e-15,
  argmax 100% 일치). 회귀 테스트를 추가하고 데모에도 배포했습니다.</p></div>
</section>

<section>
  <h2><span class="sn">07</span> PyCIL 표준 벤치 — 다른 데이터셋에서도 같은 서열</h2>
  {fig("cifar100_bridge", "PyCIL의 동일 스플릿(클래스 순서를 DataManager에서 직접 import) 위에서 실행. FeCAM &gt; SLDA &gt; NCM 간격이 끝까지 유지됩니다.")}
  <div class="col"><p>SSv2에서 본 서열이 CIFAR-100에서 그대로 재현됩니다 —
  자체 벤치 특유의 현상이 아니라는 교차 검증입니다.
  곁다리로 PyCIL 자체의 상류 버그(정확도 행렬 크래시)도 발견해 수정했습니다.</p></div>
</section>

<section>
  <h2><span class="sn">08</span> 인코더 자체를 바꿔봤다</h2>
  <div class="col"><p>head는 다섯 개를 비교했으면서 <b>인코더는 한 번도 바꿔보지 않았습니다.</b>
  그런데 인코더가 <b>추론 비용의 99.98%, 학습 전체의 146배</b>를 차지합니다 —
  edge 성능을 실제로 결정하는 건 head가 아니라 인코더입니다.
  (원래 막혀 있던 실험인데, UCF101 원본 영상을 확보하면서 풀렸습니다.)</p></div>

  <div class="tablewrap"><table>
    <thead><tr><th>인코더</th><th style="text-align:right">파라미터</th><th style="text-align:right">UCF101 FeCAM</th><th style="text-align:right">CPU ms/window</th></tr></thead>
    <tbody>
      <tr class="us"><td>CLIP ViT-B/32 <span class="muted">(현재)</span></td><td class="n">87.5M</td><td class="n">88.84 ±0.52</td><td class="n">148</td></tr>
      <tr><td>MobileCLIP-S0</td><td class="n">10.9M</td><td class="n">88.63 ±0.67</td><td class="n">2,985</td></tr>
      <tr><td class="muted">차이</td><td class="n">8× 작음</td><td class="n">−0.21 %p</td><td class="n">20× 느림</td></tr>
    </tbody>
  </table></div>

  <div class="callout">
    <div class="t">파라미터 수가 edge 지연을 예측하지 못합니다</div>
    <p>파라미터 8배 작은 모델이 CPU에서 <b>20배 느립니다</b>. 오차가 아니라 <b>설계 의도의 결과</b>입니다 —
    MobileCLIP은 Apple의 CoreML/ANE 커널을 겨냥해 만들어져, 범용 PyTorch에서는 이점이
    전혀 살아나지 않습니다. 우리가 읽은 edge-CL 선행연구 어느 쪽도 이 지점을 다루지 않습니다
    (SparCL은 희소화 가속률을, BudgetCL은 하드웨어를 추상화한 iteration 수를 보고).</p>
  </div>

  <div class="col"><p><b>그리고 좋은 head는 인코더 요구사항을 낮춥니다.</b>
  약한 인코더로 바꿨을 때 손실이 NCM −1.75, SLDA −1.58인데 <b>FeCAM은 −0.21</b>입니다 —
  공분산 정규화가 feature 품질 저하를 상당 부분 흡수합니다.</p>
  <p style="margin-top:12px"><b>결론:</b> CLIP B/32를 계속 쓰는 것이 맞지만,
  <b>&ldquo;더 정확해서&rdquo;가 아니라</b>(0.21%p는 의미 없음)
  <b>우리가 실제로 가진 런타임에서 20배 빠르기 때문</b>입니다.
  CoreML/ANE 배포 경로를 잡으면 결론이 뒤집힐 수 있어 후속 과제로 남겼습니다.</p></div>
</section>

<section>
  <h2><span class="sn">08b</span> 자주 나올 질문 &mdash; 정확도가 너무 높은 거 아닌가?</h2>
  <div class="col"><p>UCF101 파일명은 <code>v_ApplyEyeMakeup_g08_c01.avi</code>처럼
  <b>group 번호</b>를 담고 있습니다. 같은 group의 클립들은 <b>원본 영상 하나를 잘라 만든 것</b>이라
  같은 사람·배경·조명이 거의 그대로 반복됩니다. 클립 단위로 무작위 분할하면 train/test에
  사실상 같은 영상이 섞여 정확도가 부풀려질 수 있습니다 &mdash; 공식 split은 이를 막기 위해
  <b>group을 통째로</b> 한쪽에만 배정합니다.</p></div>

  <div class="tablewrap"><table>
    <tbody>
      <tr><td>train group 수</td><td class="n">1,818</td></tr>
      <tr><td>test group 수</td><td class="n">707</td></tr>
      <tr class="us"><td>겹치는 group</td><td class="n">0</td></tr>
    </tbody>
  </table></div>
  <p class="col"><b>누수 없음.</b> 88.84는 데이터 누수로 부풀려진 수치가 아닙니다.</p>
</section>

<section>
  <h2><span class="sn">09</span> 정직한 caveat &amp; 다음 단계</h2>
  <div class="col">
    <h3>논문에 반드시 병기할 것</h3>
    <ul style="margin-top:8px">
      <li><b>트랙이 다릅니다</b> — 우리는 백본을 전혀 학습하지 않습니다. 위 표는
      위치 파악용이지 동일 조건 순위가 아닙니다.</li>
      <li><b>CLIP 사전학습 오염 가능성</b> — UCF101은 YouTube 기반, CLIP은 웹 스케일 학습.
      사전학습 모델을 쓰는 연구의 공통 caveat.</li>
      <li><b>ESSENTIAL도 같은 frozen CLIP으로 95.1</b> — 6.3%p 격차는 백본이 아니라
      그들의 학습형 temporal 모듈에서 옵니다. 개선 여지가 그만큼 있다는 뜻이기도 합니다.</li>
      <li><b>HMDB51 보류</b> — 영상은 구할 수 있으나 공식 split을 구할 수 없습니다.
      자체 분할은 비교 가능성이라는 목적 자체를 없애므로 중단했습니다.</li>
    </ul>
    <h3 style="margin-top:26px">다음 단계</h3>
    <ol style="margin-top:8px">
      <li><b>실제 임베디드 보드 실측</b>(Jetson / 라즈베리파이) — 지금은 맥북 CPU.
      &ldquo;edge&rdquo;를 주장하면서 측정이 노트북인 것이 가장 큰 취약점입니다.</li>
      <li><b>연산 예산 프로토콜</b>(BudgetCL 방식) 병기 — 대규모 결과와 직접 비교 가능해집니다.</li>
      <li><b>AUC 지표 채택</b> + 거기에 <b>연산 축 추가</b> ← 우리 차별점.</li>
      <li>HMDB51 — split 확보 시 즉시 재개(파이프라인 재사용).</li>
    </ol>
  </div>
</section>

<footer>
  상세 근거: <code>reports/ucf101_tcd_result.md</code> ·
  <code>reports/progress_since_2026-07-17.md</code> ·
  <code>reports/flops_result.md</code> ·
  <code>reports/pycil_survey_edge_gap.md</code><br>
  브랜치 <code>feature/pycil-benchmarks</code> · PR #2
</footer>
</div>
"""

(HERE / "index.html").write_text(HEAD + BODY)
print("wrote", (HERE / "index.html").stat().st_size // 1024, "KB")
