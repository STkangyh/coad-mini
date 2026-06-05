import torch


class OrthogonalGradient:
    """
    각 stage 종료 후 commit_stage()를 호출하면
    해당 시점의 평균 그래디언트 방향을 기억.
    이후 apply() 시 기억된 방향마다

        g ← g - lam * proj * past_direction

    을 적용. lam=1.0 이면 완전 직교화, lam<1.0 이면 부분 억제.
    """

    def __init__(self, lam: float = 1.0):
        self.lam         = lam
        self.stage_grads = []   # 과거 stage 그래디언트 방향들 (단위벡터)
        self._accum      = None
        self._accum_count = 0

    # ── step마다 호출 ──────────────────────────────────────────────────────
    def apply(self, model, device=None):
        grads = []
        for p in model.parameters():
            if p.grad is not None:
                grads.append(p.grad.view(-1))
        if not grads:
            return

        g = torch.cat(grads)

        # 현재 step 그래디언트를 누적 (stage 평균 계산용)
        if self._accum is None:
            self._accum = g.detach().clone()
        else:
            self._accum = self._accum + g.detach()
        self._accum_count += 1

        # 과거 stage 방향들에 대해 lam 비율로 억제
        for past in self.stage_grads:
            proj = g.dot(past) / (past.dot(past) + 1e-8)
            g = g - self.lam * proj * past

        # 투영 후 norm이 극단적으로 작아지면 스케일 복원 (학습 안정화)
        g_norm = g.norm()
        if g_norm > 1e-8:
            pass  # 정상
        else:
            return  # 그래디언트가 거의 0 → 건너뜀

        # 수정된 그래디언트를 모델에 복사
        idx = 0
        for p in model.parameters():
            if p.grad is None:
                continue
            n = p.grad.numel()
            with torch.no_grad():
                p.grad.copy_(g[idx:idx + n].view_as(p.grad))
            idx += n

    # ── stage 종료 시 호출 ────────────────────────────────────────────────
    def commit_stage(self):
        """현재 stage의 평균 그래디언트 방향을 저장하고 누적기 초기화"""
        if self._accum is None or self._accum_count == 0:
            return
        avg = self._accum / self._accum_count
        avg_norm = avg.norm()
        if avg_norm > 1e-8:
            self.stage_grads.append((avg / avg_norm).detach())
        self._accum = None
        self._accum_count = 0
