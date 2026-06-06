"""
Few-shot enrollment 테스트
==========================
모두 고정 seed 로 결정적이며, 외부 데이터 없이 합성 window 로 동작한다.
실제 checkpoints/agem_48cls.pt 를 사용하는 통합 테스트 1개 포함.

실행:
    python3 -m pytest tests/test_few_shot_enroll.py -q
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pytest
import torch

from src.models.gru_detector import GRUDetector
from src.enroll.few_shot import (
    expand_classifier,
    retention_accuracy,
    FewShotEnroller,
)

FEATURE_DIM = 512
HIDDEN_DIM = 256
N_FRAMES = 16
AGEM_CKPT = ROOT / "checkpoints" / "agem_48cls.pt"


def set_seed(seed=0):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def synth_window(rng, feature_dim, center, noise=0.1, n_frames=N_FRAMES):
    base = center + noise * rng.standard_normal((1, feature_dim)).astype(np.float32)
    w = base + 0.5 * noise * rng.standard_normal((n_frames, feature_dim)).astype(np.float32)
    return w.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 1. 구조 테스트
# ─────────────────────────────────────────────────────────────────────────────
def test_expand_classifier_grows_and_preserves_weights():
    set_seed(0)
    model = GRUDetector(feature_dim=FEATURE_DIM, hidden_dim=HIDDEN_DIM, num_classes=48)

    old_w = model.cls.weight.detach().clone()
    old_b = model.cls.bias.detach().clone()

    model = expand_classifier(model, n_new=1)

    # 출력 차원이 49 로 성장
    assert model.cls.out_features == 49
    assert model.num_classes == 49

    # 기존 weight/bias 행이 비트-정확하게 보존
    assert torch.equal(model.cls.weight[:48], old_w)
    assert torch.equal(model.cls.bias[:48], old_b)

    # forward 가 width 49 logits 를 낸다
    x = torch.randn(1, N_FRAMES, FEATURE_DIM)
    logits, _ = model(x, None)
    assert logits.shape == (1, 49)


def test_expand_classifier_invalid_n_new():
    model = GRUDetector(feature_dim=FEATURE_DIM, hidden_dim=HIDDEN_DIM, num_classes=4)
    with pytest.raises(ValueError):
        expand_classifier(model, n_new=0)


# ─────────────────────────────────────────────────────────────────────────────
# 2. no-forget 메커니즘 (자체 완결형 합성 continual 시나리오)
# ─────────────────────────────────────────────────────────────────────────────
def _build_small_scenario(seed=0):
    """
    클래스 0-3 각각 distinct mean vector. 작은 GRU 를 잠깐 학습해
    old 클래스를 분류할 수 있게 만든 뒤, 클래스 4 를 enroll 할 재료를 만든다.
    """
    rng = np.random.default_rng(seed)
    set_seed(seed)
    fdim = 32
    n_old = 4

    centers = [4.0 * rng.standard_normal(fdim).astype(np.float32) for _ in range(n_old + 1)]

    def make(cls, n, noise=0.15):
        return [(synth_window(rng, fdim, centers[cls], noise=noise), cls) for _ in range(n)]

    old_train = []
    for c in range(n_old):
        old_train += make(c, 20)
    old_eval = []
    for c in range(n_old):
        old_eval += make(c, 10)

    model = GRUDetector(feature_dim=fdim, hidden_dim=48, num_classes=n_old)

    # 작은 사전학습
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    crit = torch.nn.CrossEntropyLoss()
    model.train()
    for _ in range(40):
        rng.shuffle(old_train)
        for w, y in old_train:
            x = torch.from_numpy(w).unsqueeze(0)
            opt.zero_grad()
            logits, _ = model(x, None)
            loss = crit(logits, torch.tensor([y]))
            loss.backward()
            opt.step()
    model.eval()

    new_train = make(n_old, 10, noise=0.1)   # 클래스 4
    new_eval = make(n_old, 10, noise=0.1)
    return model, old_train, old_eval, new_train, new_eval, n_old


def test_enroll_retains_old_and_learns_new():
    model, old_train, old_eval, new_train, new_eval, n_old = _build_small_scenario(seed=0)

    base_acc = retention_accuracy(model, old_eval)
    assert base_acc >= 0.8, f"pretraining failed, base_acc={base_acc}"

    model = expand_classifier(model, n_new=1)
    new_id = n_old

    enroller = FewShotEnroller(model, device="cpu", use_agem=True, seed=0)
    enroller.add_exemplars([w for w, _ in old_train], [y for _, y in old_train])
    model = enroller.enroll(
        [w for w, _ in new_train], [new_id] * len(new_train),
        epochs=30, lr=3e-3, replay_per_step=8,
    )

    ret_acc = retention_accuracy(model, old_eval)
    new_acc = retention_accuracy(model, [(w, new_id) for w, _ in new_eval])

    # old 클래스 망각이 작아야 함
    assert base_acc - ret_acc <= 0.2, f"forgot too much: {base_acc} -> {ret_acc}"
    # 새 클래스가 학습되어야 함
    assert new_acc >= 0.7, f"new class not learned: {new_acc}"


def test_replay_forgets_less_than_naive_finetune():
    """replay(A-GEM) 가 naive fine-tuning 보다 망각이 적거나 같아야 한다."""
    # 동일 시나리오를 두 번 (replay vs naive) 학습해 비교
    model_a, old_train, old_eval, new_train, new_eval, n_old = _build_small_scenario(seed=1)
    import copy
    model_b = copy.deepcopy(model_a)

    base_acc = retention_accuracy(model_a, old_eval)
    new_id = n_old

    # (A) replay
    model_a = expand_classifier(model_a, n_new=1)
    en_a = FewShotEnroller(model_a, device="cpu", use_agem=True, seed=1)
    en_a.add_exemplars([w for w, _ in old_train], [y for _, y in old_train])
    model_a = en_a.enroll([w for w, _ in new_train], [new_id] * len(new_train),
                          epochs=30, lr=3e-3, replay_per_step=8)
    ret_replay = retention_accuracy(model_a, old_eval)

    # (B) naive fine-tune (replay buffer 비움 → 새 데이터만 학습)
    model_b = expand_classifier(model_b, n_new=1)
    en_b = FewShotEnroller(model_b, device="cpu", use_agem=False, seed=1)
    # exemplar 를 추가하지 않음 → replay 없음
    model_b = en_b.enroll([w for w, _ in new_train], [new_id] * len(new_train),
                          epochs=30, lr=3e-3, replay_per_step=8)
    ret_naive = retention_accuracy(model_b, old_eval)

    new_acc_replay = retention_accuracy(model_a, [(w, new_id) for w, _ in new_eval])
    assert new_acc_replay >= 0.7, f"replay failed to learn new class: {new_acc_replay}"
    # replay 가 naive 보다 망각이 적거나 같아야 한다
    assert ret_replay >= ret_naive - 1e-6, (
        f"replay should forget no more than naive: replay={ret_replay}, naive={ret_naive}, base={base_acc}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. 통합 테스트 (실제 agem_48cls.pt)
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.skipif(not AGEM_CKPT.exists(), reason="agem_48cls.pt not present")
def test_integration_real_checkpoint():
    set_seed(0)
    rng = np.random.default_rng(0)

    ckpt = torch.load(AGEM_CKPT, map_location="cpu", weights_only=False)
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

    old_n = model.num_classes
    assert old_n == 48
    fdim = model.feature_dim

    model = expand_classifier(model, n_new=1)
    assert model.num_classes == 49
    new_id = 48

    # 합성 class-48 window (뚜렷한 중심)
    center = 4.0 * rng.standard_normal(fdim).astype(np.float32)
    new_train = [synth_window(rng, fdim, center, noise=0.1) for _ in range(8)]
    new_eval = [synth_window(rng, fdim, center, noise=0.1) for _ in range(8)]

    # old replay exemplar: 모델이 잘 분류하는 합성 window 수집
    ex_w, ex_y = [], []
    for _ in range(200):
        if len(ex_w) >= 24:
            break
        w = rng.standard_normal((N_FRAMES, fdim)).astype(np.float32)
        with torch.no_grad():
            logits, _ = model(torch.from_numpy(w).unsqueeze(0), None)
        ex_w.append(w)
        ex_y.append(int(logits.argmax(-1).item()))

    enroller = FewShotEnroller(model, device="cpu", use_agem=True, seed=0)
    enroller.add_exemplars(ex_w, ex_y)
    model = enroller.enroll(new_train, [new_id] * len(new_train),
                            epochs=15, lr=1e-3, replay_per_step=8)

    # end-to-end 가 돌고, 새 클래스 window 가 48 로 분류되어야 함
    new_acc = retention_accuracy(model, [(w, new_id) for w in new_eval])
    assert new_acc >= 0.7, f"new class not classifiable as 48: {new_acc}"

    # forward 출력 폭이 49 인지 확인
    logits, _ = model(torch.from_numpy(new_eval[0]).unsqueeze(0), None)
    assert logits.shape[-1] == 49
