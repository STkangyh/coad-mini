"""
Few-shot enrollment 데모
=========================
checkpoints/agem_48cls.pt 를 로드해 48 → 49 클래스로 확장하고,
합성(synthetic) 새 클래스를 few-shot 으로 등록한 뒤
  - 기존 48 클래스 retention accuracy
  - 새 클래스(48) accuracy
를 출력한다.

외부 데이터 없이 동작한다 (모든 window 는 합성 (16, 512) 배열).

실행:
    python3 experiments/demo_few_shot.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from src.models.gru_detector import GRUDetector
from src.enroll.few_shot import expand_classifier, FewShotEnroller, retention_accuracy

CKPT = Path("checkpoints/agem_48cls.pt")
N_FRAMES = 16
SEED = 0


def load_agem_model():
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = GRUDetector(
        feature_dim=ckpt.get("feature_dim", 512),
        hidden_dim=ckpt.get("hidden_dim", 256),
        num_classes=ckpt.get("n_classes", 48),
        num_layers=ckpt.get("num_layers", 1),
        dropout=ckpt.get("dropout", 0.0),
        bidirectional=ckpt.get("bidirectional", False),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def synth_window(rng, feature_dim, center, noise=0.15):
    """center 벡터 주변의 (16, F) 합성 window (프레임 간 약한 상관)."""
    base = center + noise * rng.standard_normal((1, feature_dim)).astype(np.float32)
    w = base + 0.5 * noise * rng.standard_normal((N_FRAMES, feature_dim)).astype(np.float32)
    return w.astype(np.float32)


def main():
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    device = "cpu"

    if not CKPT.exists():
        print(f"[!] {CKPT} 없음 — experiments/save_checkpoints.py 를 먼저 실행하세요.")
        return

    model, ckpt = load_agem_model()
    feature_dim = model.feature_dim
    old_n = model.num_classes
    print(f"Loaded A-GEM checkpoint: {old_n} classes, feature_dim={feature_dim}")

    # ── 기존 클래스 평가셋 만들기 ──────────────────────────────────────────────
    # 모델이 실제로 잘 분류하는 합성 window 를 만들기 위해, 각 old 클래스마다
    # logit 을 키우는 방향(cls weight row)을 중심으로 window 를 합성한다.
    with torch.no_grad():
        cls_w = model.cls.weight.detach().cpu().numpy()  # (old_n, hidden)
    # hidden 공간이 아니라 입력 공간이 필요하므로, 실제 모델을 통과시켜 라벨을 정한다.
    eval_old = []
    per_class = 3
    for c in range(old_n):
        for _ in range(per_class):
            w = synth_window(rng, feature_dim, np.zeros(feature_dim, dtype=np.float32), noise=1.0)
            with torch.no_grad():
                logits, _ = model(torch.from_numpy(w).unsqueeze(0), None)
                lbl = int(logits.argmax(-1).item())
            eval_old.append((w, lbl))
    # 모델 자기예측 라벨로 평가 → 확장/학습 후 retention 측정의 기준점
    base_acc = retention_accuracy(model, eval_old, device)
    print(f"Old-class self-consistency (pre-enroll): {base_acc:.3f}  (n={len(eval_old)})")

    # exemplar buffer: 위 eval_old 를 그대로 replay 재료로 사용
    old_windows = [w for w, _ in eval_old]
    old_labels = [y for _, y in eval_old]

    # ── 49번째(새) 클래스 합성 ─────────────────────────────────────────────────
    new_class_id = old_n  # 48
    new_center = 4.0 * rng.standard_normal(feature_dim).astype(np.float32)
    new_train = [synth_window(rng, feature_dim, new_center, noise=0.1) for _ in range(8)]
    new_eval = [synth_window(rng, feature_dim, new_center, noise=0.1) for _ in range(8)]

    # ── 확장 ───────────────────────────────────────────────────────────────────
    model = expand_classifier(model, n_new=1)
    print(f"Expanded classifier: {old_n} -> {model.num_classes}")

    # 확장 직후 망각 0 확인
    acc_after_expand = retention_accuracy(model, eval_old, device)
    print(f"Old-class acc right after expand (should equal pre): {acc_after_expand:.3f}")

    # ── enroll ─────────────────────────────────────────────────────────────────
    enroller = FewShotEnroller(model, device=device, use_agem=True, seed=SEED)
    enroller.add_exemplars(old_windows, old_labels)
    model = enroller.enroll(
        new_train, [new_class_id] * len(new_train),
        epochs=25, lr=1e-3, replay_per_step=8,
    )

    # ── 결과 ───────────────────────────────────────────────────────────────────
    ret_acc = retention_accuracy(model, eval_old, device)
    new_acc = retention_accuracy(
        model, [(w, new_class_id) for w in new_eval], device
    )
    print("\n=== Few-shot enrollment result ===")
    print(f"Old-class retention accuracy : {ret_acc:.3f}  (was {base_acc:.3f})")
    print(f"New-class (id={new_class_id}) accuracy   : {new_acc:.3f}")
    drop = base_acc - ret_acc
    print(f"Retention drop               : {drop:+.3f}")
    print("New class learned" if new_acc >= 0.5 else "New class NOT learned",
          "| retention OK" if drop <= 0.15 else "| retention degraded")


if __name__ == "__main__":
    main()
