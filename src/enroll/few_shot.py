"""
Few-shot enrollment of NEW action classes
==========================================
기존 48-class GRUDetector를 망각 없이 48+K 클래스로 확장하는 모듈.

핵심 아이디어
  1. `expand_classifier` : 마지막 `cls` Linear 의 출력 차원을
     num_classes → num_classes + n_new 로 키우되,
     기존 weight/bias 행은 비트 단위로 그대로 복사하고
     새 클래스 행만 작게 초기화한다 (망각 0 에서 출발).
  2. `FewShotEnroller` : 기존 클래스의 (window, label) exemplar 를
     메모리에 들고 있다가, 새 클래스 few-shot 학습 시 함께 replay 한다.
     replay 방식은 (a) 단순 인터리브 + (b) A-GEM gradient projection 을
     모두 지원한다 (`use_agem` 플래그).

이 모듈은 파일이 아니라 **메모리 상의 feature window** 위에서 동작한다.
window 텐서는 모두 shape (T, feature_dim) 또는 (B, T, feature_dim).

참고: A-GEM gradient projection 로직은 src/utils/gem.py 의 수식을 재사용한다.
"""

from __future__ import annotations

import copy
import random
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# 1. classifier head 확장
# ─────────────────────────────────────────────────────────────────────────────
def expand_classifier(model: nn.Module, n_new: int) -> nn.Module:
    """
    model.cls (Linear) 의 출력 차원을 num_classes → num_classes + n_new 로 확장.

    - 기존 weight/bias 행은 **비트 단위로 그대로 복사**된다.
    - 새 클래스 행은 작은 정규분포로 초기화된다.
    - model.num_classes 를 갱신한다.
    - 새 Linear 는 기존과 같은 device / dtype 을 따른다.

    in-place 로 model 을 수정하고 같은 model 을 반환한다.
    """
    if n_new <= 0:
        raise ValueError(f"n_new must be positive, got {n_new}")

    old_cls: nn.Linear = model.cls
    in_features = old_cls.in_features
    old_out = old_cls.out_features
    new_out = old_out + n_new

    device = old_cls.weight.device
    dtype = old_cls.weight.dtype

    new_cls = nn.Linear(in_features, new_out, bias=old_cls.bias is not None)
    new_cls = new_cls.to(device=device, dtype=dtype)

    with torch.no_grad():
        # 새 행은 작게 초기화 (기존 출력 분포를 흔들지 않도록)
        nn.init.normal_(new_cls.weight, mean=0.0, std=0.01)
        if new_cls.bias is not None:
            nn.init.zeros_(new_cls.bias)
        # 기존 행 비트-정확 복사
        new_cls.weight[:old_out].copy_(old_cls.weight)
        if old_cls.bias is not None and new_cls.bias is not None:
            new_cls.bias[:old_out].copy_(old_cls.bias)

    model.cls = new_cls
    model.num_classes = new_out
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 헬퍼: window 정규화
# ─────────────────────────────────────────────────────────────────────────────
def _as_window_tensor(w, device) -> torch.Tensor:
    """(T, F) 또는 (1, T, F) → (1, T, F) float32 텐서"""
    if isinstance(w, np.ndarray):
        t = torch.from_numpy(w.astype(np.float32))
    elif torch.is_tensor(w):
        t = w.to(dtype=torch.float32)
    else:
        t = torch.tensor(np.asarray(w, dtype=np.float32))
    if t.dim() == 2:
        t = t.unsqueeze(0)
    elif t.dim() != 3:
        raise ValueError(f"window must be (T,F) or (B,T,F), got shape {tuple(t.shape)}")
    return t.to(device)


# ─────────────────────────────────────────────────────────────────────────────
# 2. retention / accuracy 헬퍼
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def retention_accuracy(model: nn.Module, eval_set: Sequence[tuple], device: str = "cpu") -> float:
    """
    eval_set : [(window, label), ...]  (window: (T,F) array/tensor, label: int)
    argmax 기준 top-1 accuracy 반환.
    """
    if len(eval_set) == 0:
        return 0.0
    was_training = model.training
    model.eval()
    correct = 0
    for window, label in eval_set:
        x = _as_window_tensor(window, device)
        logits, _ = model(x, None)
        pred = logits.argmax(dim=-1).item()
        correct += int(pred == int(label))
    if was_training:
        model.train()
    return correct / len(eval_set)


# ─────────────────────────────────────────────────────────────────────────────
# 3. FewShotEnroller
# ─────────────────────────────────────────────────────────────────────────────
class FewShotEnroller:
    """
    로드된 GRUDetector 를 감싸 few-shot 으로 새 클래스를 등록한다.

    사용 흐름:
        enroller = FewShotEnroller(model, device="cpu")
        enroller.add_exemplars(old_windows, old_labels)   # replay buffer 채우기
        model = enroller.enroll(new_windows, new_labels, epochs=20, lr=1e-3)

    replay buffer 는 기존(old) 클래스 exemplar 의 (window_tensor, label) 목록.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        buffer_size: int = 200,
        use_agem: bool = True,
        seed: int | None = None,
    ):
        self.model = model.to(device)
        self.device = device
        self.buffer_size = buffer_size
        self.use_agem = use_agem
        self._buffer: list[tuple[torch.Tensor, int]] = []
        self._rng = random.Random(seed)
        self.criterion = nn.CrossEntropyLoss()

    # ── replay buffer ────────────────────────────────────────────────────────
    def add_exemplars(self, windows: Sequence, labels: Sequence[int]) -> None:
        """기존 클래스 exemplar 를 reservoir 방식으로 buffer 에 추가."""
        for w, y in zip(windows, labels):
            t = _as_window_tensor(w, self.device).detach()
            item = (t, int(y))
            if len(self._buffer) < self.buffer_size:
                self._buffer.append(item)
            else:
                j = self._rng.randint(0, len(self._buffer))
                if j < self.buffer_size:
                    self._buffer[j] = item

    @property
    def buffer(self) -> list[tuple[torch.Tensor, int]]:
        return self._buffer

    # ── A-GEM gradient projection (src/utils/gem.py 수식 재사용) ──────────────
    def _flat_grad(self) -> torch.Tensor | None:
        grads = [p.grad.view(-1).detach()
                 for p in self.model.parameters() if p.grad is not None]
        return torch.cat(grads) if grads else None

    def _set_flat_grad(self, g_flat: torch.Tensor) -> None:
        idx = 0
        for p in self.model.parameters():
            if p.grad is None:
                continue
            n = p.grad.numel()
            with torch.no_grad():
                p.grad.copy_(g_flat[idx: idx + n].view_as(p.grad))
            idx += n

    def _compute_ref_grad(self, sample: list[tuple[torch.Tensor, int]]) -> torch.Tensor | None:
        """replay sample 로 참조 그래디언트 g_ref 계산 (현재 grad 보존)."""
        if not sample:
            return None
        saved = [p.grad.clone() if p.grad is not None else None
                 for p in self.model.parameters()]
        self.model.zero_grad()
        for window, label in sample:
            x = window.to(self.device)
            y = torch.tensor([int(label)], dtype=torch.long, device=self.device)
            logits, _ = self.model(x, None)
            loss = self.criterion(logits, y)
            loss.backward()
        g_ref = self._flat_grad()
        # 원래 grad 복원
        for p, g in zip(self.model.parameters(), saved):
            p.grad = g
        return g_ref

    def _project_agem(self, g_ref: torch.Tensor | None) -> None:
        """dot(g_cur, g_ref) < 0 이면 g_cur 를 g_ref 와 비충돌로 투영."""
        if g_ref is None:
            return
        g_cur = self._flat_grad()
        if g_cur is None:
            return
        dot = g_cur.dot(g_ref)
        if dot < 0:
            g_new = g_cur - (dot / (g_ref.dot(g_ref) + 1e-12)) * g_ref
            self._set_flat_grad(g_new)

    # ── 핵심: enroll ──────────────────────────────────────────────────────────
    def enroll(
        self,
        new_examples: Sequence,
        new_labels: Sequence[int],
        epochs: int = 20,
        lr: float = 1e-3,
        replay_per_step: int = 8,
    ) -> nn.Module:
        """
        새 클래스 few-shot 학습.

        new_examples : [(T,F) array/tensor, ...]  새 클래스 window 들
        new_labels   : 대응 정수 라벨 (보통 모두 동일한 새 class_id)
        epochs       : 새 데이터 위 epoch 수
        lr           : 학습률
        replay_per_step : 매 step 에 함께 학습/투영할 old exemplar 수

        반환: 학습된 model (in-place 로도 수정됨)

        동작:
          - use_agem=True  : 매 step 마다 replay sample 로 g_ref 를 구해
                             A-GEM projection 으로 망각을 억제 (gem.py 수식).
                             추가로 같은 sample 을 인터리브 학습해 안정화.
          - use_agem=False : 새 데이터와 old exemplar 를 인터리브 학습만 수행.
        """
        new_items = [(_as_window_tensor(w, self.device).detach(), int(y))
                     for w, y in zip(new_examples, new_labels)]
        if not new_items:
            raise ValueError("new_examples is empty")

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()

        for _ in range(epochs):
            order = new_items[:]
            self._rng.shuffle(order)
            for window, label in order:
                replay_sample = self._sample_replay(replay_per_step)

                # A-GEM 모드: 참조 그래디언트 미리 계산
                g_ref = None
                if self.use_agem and replay_sample:
                    g_ref = self._compute_ref_grad(replay_sample)

                optimizer.zero_grad()

                # 새 클래스 loss
                x = window.to(self.device)
                y = torch.tensor([label], dtype=torch.long, device=self.device)
                logits, _ = self.model(x, None)
                loss = self.criterion(logits, y)

                # 인터리브 replay (두 모드 모두 안정화에 도움)
                for r_window, r_label in replay_sample:
                    rx = r_window.to(self.device)
                    ry = torch.tensor([int(r_label)], dtype=torch.long, device=self.device)
                    r_logits, _ = self.model(rx, None)
                    loss = loss + self.criterion(r_logits, ry)

                loss.backward()

                if self.use_agem and g_ref is not None:
                    self._project_agem(g_ref)

                optimizer.step()

        self.model.eval()
        return self.model

    def _sample_replay(self, k: int) -> list[tuple[torch.Tensor, int]]:
        if not self._buffer or k <= 0:
            return []
        if k >= len(self._buffer):
            return list(self._buffer)
        return self._rng.sample(self._buffer, k)


__all__ = ["expand_classifier", "retention_accuracy", "FewShotEnroller"]
