"""
A-GEM (Averaged Gradient Episodic Memory)
------------------------------------------
각 stage 종료 후 add_stage()를 호출해 메모리에 샘플을 저장.
이후 매 step에서 apply() 호출 시:

  1. 현재 그래디언트 g_cur 수집
  2. 저장된 모든 메모리 샘플로 참조 그래디언트 g_ref 계산
  3. dot(g_cur, g_ref) < 0 (충돌) 이면

         g_new = g_cur - (g_cur·g_ref / g_ref·g_ref) * g_ref

     으로 투영 → 과거 task 손실 증가 방지

참고: Lopez-Paz & Ranzato, "Gradient Episodic Memory for Continual Learning", NeurIPS 2017
"""

import random
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path


class AGEM:
    def __init__(
        self,
        mem_per_stage: int = 30,
        selection: str = "random",   # "random" | "balanced" | "reservoir" | "hard" | "balanced_hard"
        replay_ratio: float = 1.0,   # fraction of memory used for g_ref (0 < r ≤ 1)
    ):
        """
        mem_per_stage : 각 stage에서 메모리에 저장할 샘플 수
        selection     : "random" (무작위) | "balanced" (클래스별 균등)
                        | "reservoir" (streaming uniform)
                        | "hard" (loss 상위 샘플)
                        | "balanced_hard" (클래스별 loss 상위 샘플)
        replay_ratio  : g_ref 계산 시 사용할 메모리 비율 (0 < r ≤ 1)
        """
        self.mem_per_stage = mem_per_stage
        self.selection     = selection
        self.replay_ratio  = float(replay_ratio)
        self._memory: list[tuple[list, Path]] = []  # (샘플 리스트, feat_dir)
        self._criterion = nn.CrossEntropyLoss()
        self._g_ref: torch.Tensor | None = None   # epoch당 1회 캐싱

    # ── stage 종료 시 호출 ───────────────────────────────────────────────────
    def add_stage(
        self,
        samples: list,
        feat_dir: Path,
        model: nn.Module | None = None,
        device: str | None = None,
    ):
        """현재 stage 샘플 중 mem_per_stage개를 메모리에 저장 (selection 전략 적용)"""
        if self.selection == "balanced":
            chosen = self._balanced_sample(samples, self.mem_per_stage)
        elif self.selection == "reservoir":
            chosen = self._reservoir_sample(samples, self.mem_per_stage)
        elif self.selection == "hard":
            chosen = self._hard_sample(samples, feat_dir, self.mem_per_stage, model, device)
        elif self.selection == "balanced_hard":
            chosen = self._balanced_hard_sample(samples, feat_dir, self.mem_per_stage, model, device)
        else:
            chosen = random.sample(samples, min(self.mem_per_stage, len(samples)))
        self._memory.append((chosen, Path(feat_dir)))
        self._g_ref = None   # 새 stage 추가 → 캐시 무효화

    def _balanced_sample(self, samples: list, total: int) -> list:
        """클래스별로 균등하게 total개 샘플링"""
        from collections import defaultdict
        buckets: dict = defaultdict(list)
        for s in samples:
            buckets[s["class_id"]].append(s)
        n_cls   = len(buckets)
        per_cls = max(1, total // n_cls)
        chosen  = []
        for cls_samples in buckets.values():
            chosen.extend(random.sample(cls_samples, min(per_cls, len(cls_samples))))
        # 부족한 경우 나머지 채우기
        if len(chosen) < total:
            pool = [s for s in samples if s not in chosen]
            chosen.extend(random.sample(pool, min(total - len(chosen), len(pool))))
        return chosen[:total]

    def _reservoir_sample(self, samples: list, total: int) -> list:
        """순차 스트림에서 uniform하게 total개 유지"""
        reservoir = []
        for i, sample in enumerate(samples):
            if len(reservoir) < total:
                reservoir.append(sample)
                continue
            j = random.randint(0, i)
            if j < total:
                reservoir[j] = sample
        return reservoir

    @torch.no_grad()
    def _score_losses(
        self,
        samples: list,
        feat_dir: Path,
        model: nn.Module | None,
        device: str | None,
    ) -> list[tuple[float, dict]]:
        """현재 모델 기준 loss를 계산해 hard-example selection에 사용"""
        if model is None or device is None:
            raise ValueError(
                f"AGEM(selection={self.selection!r}) requires model and device in add_stage()."
            )

        was_training = model.training
        model.eval()
        scored = []
        feat_dir = Path(feat_dir)

        for s in samples:
            feat_path = feat_dir / f"{s['id']}.npy"
            if not feat_path.exists():
                continue
            feat = np.load(feat_path)
            x = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(device)
            label = torch.tensor([s["class_id"]], dtype=torch.long).to(device)
            logits, _ = model(x, None)
            loss = self._criterion(logits, label)
            scored.append((float(loss.item()), s))

        if was_training:
            model.train()
        return scored

    def _hard_sample(
        self,
        samples: list,
        feat_dir: Path,
        total: int,
        model: nn.Module | None,
        device: str | None,
    ) -> list:
        """전체 stage에서 loss가 큰 샘플 우선 저장"""
        scored = self._score_losses(samples, feat_dir, model, device)
        if not scored:
            return random.sample(samples, min(total, len(samples)))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [sample for _, sample in scored[:total]]

    def _balanced_hard_sample(
        self,
        samples: list,
        feat_dir: Path,
        total: int,
        model: nn.Module | None,
        device: str | None,
    ) -> list:
        """클래스별로 loss가 큰 샘플을 우선 저장하고 부족분은 hard 순위로 채움"""
        from collections import defaultdict

        scored = self._score_losses(samples, feat_dir, model, device)
        if not scored:
            return self._balanced_sample(samples, total)

        buckets: dict[int, list[tuple[float, dict]]] = defaultdict(list)
        for loss, sample in scored:
            buckets[sample["class_id"]].append((loss, sample))

        n_cls = len(buckets)
        per_cls = max(1, total // n_cls)
        chosen = []
        chosen_ids = set()

        for cls_samples in buckets.values():
            cls_samples.sort(key=lambda item: item[0], reverse=True)
            for _, sample in cls_samples[:per_cls]:
                chosen.append(sample)
                chosen_ids.add(sample["id"])

        if len(chosen) < total:
            scored.sort(key=lambda item: item[0], reverse=True)
            for _, sample in scored:
                if sample["id"] in chosen_ids:
                    continue
                chosen.append(sample)
                chosen_ids.add(sample["id"])
                if len(chosen) >= total:
                    break

        return chosen[:total]

    # ── epoch 시작마다 1회 호출 ──────────────────────────────────────────────
    def precompute_ref(self, model: nn.Module, device: str):
        """메모리 샘플 (replay_ratio 적용)으로 참조 그래디언트를 미리 계산해 캐싱"""
        if not self._memory:
            self._g_ref = None
            return

        # replay_ratio에 따라 각 stage 메모리를 서브샘플링
        replay_memory = []
        for (mem_samples, feat_dir) in self._memory:
            k = max(1, round(len(mem_samples) * self.replay_ratio))
            subset = random.sample(mem_samples, min(k, len(mem_samples)))
            replay_memory.append((subset, feat_dir))

        saved = [p.grad.clone() if p.grad is not None else None
                 for p in model.parameters()]
        model.zero_grad()
        count = 0
        model.train()

        for (mem_samples, feat_dir) in replay_memory:
            for s in mem_samples:
                feat_path = feat_dir / f"{s['id']}.npy"
                if not feat_path.exists():
                    continue
                feat  = np.load(feat_path)
                x     = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(device)
                label = torch.tensor([s["class_id"]], dtype=torch.long).to(device)
                logits, _ = model(x, None)
                loss = self._criterion(logits, label)
                loss.backward()
                count += 1

        if count > 0:
            grads = [p.grad.view(-1).detach()
                     for p in model.parameters() if p.grad is not None]
            self._g_ref = torch.cat(grads) if grads else None
        else:
            self._g_ref = None

        # 원래 grad 복원
        for p, g in zip(model.parameters(), saved):
            p.grad = g

    # ── 매 step 호출 ─────────────────────────────────────────────────────────
    def apply(self, model: nn.Module, device: str):
        """충돌 시 현재 그래디언트를 메모리 방향과 비충돌로 투영 (캐시 사용)"""
        if self._g_ref is None:
            return

        g_cur = self._collect_grad(model)
        if g_cur is None:
            return

        dot = g_cur.dot(self._g_ref)
        if dot < 0:
            g_new = g_cur - (dot / (self._g_ref.dot(self._g_ref) + 1e-12)) * self._g_ref
            self._set_grad(model, g_new)

    # ── 내부 헬퍼 ────────────────────────────────────────────────────────────
    def _collect_grad(self, model: nn.Module) -> torch.Tensor | None:
        grads = [p.grad.view(-1).detach()
                 for p in model.parameters() if p.grad is not None]
        return torch.cat(grads) if grads else None

    def _set_grad(self, model: nn.Module, g_flat: torch.Tensor):
        idx = 0
        for p in model.parameters():
            if p.grad is None:
                continue
            n = p.grad.numel()
            with torch.no_grad():
                p.grad.copy_(g_flat[idx: idx + n].view_as(p.grad))
            idx += n
