"""
Stage-based Continual Learning 실험  (16 classes × 4 stages)

비교:
  Baseline   : 아무 제약 없는 순수 SGD
  OrthGrad   : 이전 stage 평균 grad 방향 억제 (λ=0.5, 선행 sweep 결과)
  A-GEM      : 메모리 샘플로 실제 grad 충돌 시에만 투영 (Lopez-Paz 2017)

Stage1: Opening / Closing / Turning upside down / Putting onto
Stage2: Putting into / Taking out / Putting on surface / Taking from
Stage3: Moving up / down / toward cam / away cam
Stage4: Pushing L→R / R→L / Pulling L→R / R→L   ← Stage3과 유사
"""

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

from src.utils.orthogonal_grad import OrthogonalGradient
from src.utils.gem import AGEM
from src.trainer import (
    load_samples, make_model, set_seed,
    train_epoch,
    TRAIN_FEAT_DIR, VAL_FEAT_DIR,
)

SEED       = 0
NUM_EPOCHS = 15   # stage당 epoch 수
MEM_PER_STAGE = 30  # A-GEM 메모리 버퍼 크기 (stage 당)

STAGES = {
    1: [0,  1,  2,  3],
    2: [4,  5,  6,  7],
    3: [8,  9,  10, 11],
    4: [12, 13, 14, 15],
}


def filter_by_classes(samples: list, class_ids: list[int]) -> list:
    return [s for s in samples if s["class_id"] in class_ids]


@torch.no_grad()
def evaluate_stage(model, val_samples, feat_dir, device, class_ids):
    """stage 내 argmax 정확도 (stage-local evaluation)"""
    model.eval()
    correct = total = 0
    for s in val_samples:
        if s["class_id"] not in class_ids:
            continue
        feat_path = Path(feat_dir) / f"{s['id']}.npy"
        if not feat_path.exists():
            continue
        feat = np.load(feat_path)
        w = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(device)
        logits, _ = model(w, None)
        stage_logits = logits[0, class_ids]
        pred_class   = class_ids[stage_logits.argmax().item()]
        total   += 1
        correct += int(pred_class == s["class_id"])
    return correct / total if total > 0 else 0.0


# ── 실험 함수 ─────────────────────────────────────────────────────────────────
def run_experiment(method: str):
    """
    method: 'baseline' | 'orthgrad' | 'agem'
    """
    set_seed(SEED)
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    model     = make_model(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # 방법별 객체 초기화
    orth = None
    gem  = None
    if method == "orthgrad":
        orth = OrthogonalGradient(lam=0.5)   # λ sweep 결과 S2 drop 최소 지점
    elif method == "agem":
        gem = AGEM(mem_per_stage=MEM_PER_STAGE)

    train_all = load_samples("data/subset/train_mini.json")
    val_all   = load_samples("data/subset/val_mini.json")

    acc_table = {}   # stage_done → {stage_eval: acc}

    for stage_id, class_ids in STAGES.items():
        stage_train = filter_by_classes(train_all, class_ids)

        for ep in range(NUM_EPOCHS):
            # A-GEM: epoch 시작마다 참조 그래디언트를 1회 캐싱 (매 step 재계산 방지)
            if gem is not None:
                gem.precompute_ref(model, device)

            # A-GEM은 gem 객체를 orth 슬롯에 넘김 (apply 인터페이스 동일)
            train_epoch(
                model, stage_train, TRAIN_FEAT_DIR,
                criterion, optimizer, device,
                orth=gem if gem is not None else orth,
            )

        acc_table[stage_id] = {
            sid: evaluate_stage(model, val_all, VAL_FEAT_DIR, device, cids)
            for sid, cids in STAGES.items()
        }

        # stage 종료 후 메모리/방향 갱신
        if orth is not None:
            orth.commit_stage()
        if gem is not None:
            gem.add_stage(stage_train, TRAIN_FEAT_DIR)

    return acc_table


# ── sweep 실행 ────────────────────────────────────────────────────────────────
METHODS = ["baseline", "orthgrad", "agem"]
results = {}

for method in METHODS:
    label = {"baseline": "Baseline", "orthgrad": "OrthGrad(λ=0.5)", "agem": "A-GEM"}[method]
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    results[method] = run_experiment(method)

    t = results[method]
    header = "  After→  " + "  ".join(f"S{s}" for s in STAGES)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for after in STAGES:
        row = f"  S{after} done  "
        for sid in STAGES:
            mark = "←" if sid == after else "  "
            row += f"{t[after][sid]:.3f}{mark} "
        print(row)

# ── 최종 비교 테이블 ───────────────────────────────────────────────────────────
print(f"\n\n{'='*60}")
print("  Final Comparison — After S4")
print(f"{'='*60}")
print(f"  {'Method':<18}  {'S1 drop':>9}  {'S2 drop':>9}  {'Avg acc':>9}")
print(f"  {'-'*52}")

for method in METHODS:
    t      = results[method]
    drop1  = t[1][1] - t[4][1]
    drop2  = t[2][2] - t[4][2]
    avg    = np.mean([t[4][sid] for sid in STAGES])
    label  = {"baseline": "Baseline", "orthgrad": "OrthGrad(λ=0.5)", "agem": "A-GEM"}[method]
    print(f"  {label:<18}  {drop1:>+9.3f}  {drop2:>+9.3f}  {avg:>9.3f}")

