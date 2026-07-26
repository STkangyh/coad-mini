"""
Analytic FLOPs accounting -- training AND inference -- for every head in our
8-stage SSv2 protocol.

Why: our efficiency claims so far are wall-clock only (e.g. "FeCAM trains in 8.7s
vs GRU+A-GEM's ~270s"). Wall-clock is hardware- and implementation-dependent, so
it cannot be placed next to the edge-CL literature. Both papers the CIL survey
cites as prior work in this direction report FLOPs instead:
  - SparCL (NeurIPS'22) reports "training FLOPs", up to 23x fewer than DER++
  - BudgetCL (CVPR'23) deliberately abstracts hardware away entirely
FLOPs are hardware-independent, so reporting them lets our numbers sit in the
same table as theirs. See reports/pycil_survey_edge_gap.md.

Why analytic rather than a profiler: the analytic heads (FeCAM/SLDA/NCM/Ridge/
RanDumb) are numpy statistical accumulations, not nn.Modules -- no FLOP counter
(fvcore/thop/ptflops) can trace them. Deriving all methods by hand keeps a single
consistent convention across backprop and backprop-free methods.

CONVENTION (stated explicitly because papers differ):
  * one multiply-accumulate (MAC) = 2 FLOPs (1 multiply + 1 add)
  * training FLOPs = forward + backward; backward = 2x forward (standard
    assumption: one pass for input grads, one for weight grads)
  * elementwise ops (activations, norms) counted at ~1 FLOP/element
  * papers that count a MAC as 1 FLOP would report exactly half of these numbers

Run: python3 dev/compute_flops.py
"""
import sys

sys.path.insert(0, ".")

import torch

from experiments.run_capacity_ablation import (
    MEM_PER_STAGE, REPLAY_RATIO, N_CLASSES, CLASSES_PER_STAGE,
    DEFAULT_FEATURE_DIM, ExperimentConfig, make_model, train_all,
)

# ── protocol constants (mirrors the main experiment) ─────────────────────────
D = DEFAULT_FEATURE_DIM          # 512, CLIP ViT-B/32 embedding width
T = 16                           # frames per window
H = 256                          # GRU hidden ("best_current_h256")
C = N_CLASSES                    # 48
EPOCHS = 15
N_STAGES = N_CLASSES // CLASSES_PER_STAGE          # 8
N_TRAIN = len(train_all)                           # 4800
PER_STAGE = N_TRAIN // N_STAGES                    # 600
RFF_DIM = 2000                   # RanDumb-style random Fourier features
RP_DIM = 2000                    # Ridge + random projection width

MAC = 2                          # FLOPs per multiply-accumulate
BWD_MULT = 3                     # fwd+bwd = 3x fwd


# ── forward-pass costs ────────────────────────────────────────────────────────
def gru_forward_flops(t=T, d=D, h=H, c=C):
    """One (T,D) window through nn.GRU(D->H) + Linear(H->C).

    GRU per timestep: 3 gates x (W_i x  +  W_h h) = 3(DH + H^2) MACs,
    plus ~10H elementwise (biases, 2 sigmoids, 1 tanh, r*(..), (1-z)n+zh).
    The classifier runs on the last timestep only.
    """
    per_step = 3 * (d * h + h * h) * MAC + 10 * h
    return t * per_step + (h * c * MAC + c)


def clip_vitb32_forward_flops(n_frames=T):
    """Frozen CLIP ViT-B/32 encoder, 224x224 -> 50 tokens, 12 layers, width 768.

    Included for context only: this cost is IDENTICAL for every head (the
    encoder is frozen and shared), and in our pipeline it is paid once per
    video and cached to .npy. It is not part of any head's training cost.
    """
    tok, w, layers = 50, 768, 12
    per_layer = (
        3 * tok * w * w * MAC          # Q,K,V projections
        + 2 * tok * tok * w * MAC      # QK^T and (attn)V
        + tok * w * w * MAC            # output projection
        + 2 * tok * w * (4 * w) * MAC  # MLP up + down
    )
    patch_embed = (tok - 1) * (3 * 32 * 32) * w * MAC
    proj_head = w * D * MAC
    return n_frames * (layers * per_layer + patch_embed + proj_head)


# ── method-level training costs ───────────────────────────────────────────────
def gru_agem_training_flops():
    """Full 8-stage GRU+A-GEM run: main updates + A-GEM reference grads + projection."""
    fwd = gru_forward_flops()
    step = BWD_MULT * fwd                                  # one sample, batch=1

    main = N_STAGES * PER_STAGE * EPOCHS * step

    # precompute_ref(): once per epoch, over subsampled memory of all PAST stages.
    # k per stored stage = max(1, round(MEM_PER_STAGE * REPLAY_RATIO))
    k = max(1, round(MEM_PER_STAGE * REPLAY_RATIO))
    past_stage_epochs = sum(range(N_STAGES))               # 0+1+...+7 = 28
    ref = past_stage_epochs * EPOCHS * k * step

    # apply(): two dot products over the flat gradient (+ one axpy when projecting)
    n_params = 3 * (D * H + H * H + 2 * H) + (H * C + C)
    proj = N_STAGES * PER_STAGE * EPOCHS * (6 * n_params)

    return {"main": main, "agem_ref": ref, "agem_proj": proj,
            "total": main + ref + proj, "n_params": n_params}


def mean_pool_flops(n=N_TRAIN, t=T, d=D):
    """(T,D) -> (D,) mean-pool, done once per sample by every analytic head."""
    return n * (t * d + d)


def tukey_prep_flops(n, d=D):
    """sign/abs/power Tukey transform + L2 normalize."""
    return n * d * 3 + n * (2 * d * MAC + d)


def fecam_training_flops(n=N_TRAIN, d=D):
    """Class means + ONE shared covariance, single streaming pass. No backprop."""
    prep = tukey_prep_flops(n, d)
    means = n * d                                  # running sums
    cov = n * d * d * MAC                          # D.T @ D  -- dominant term
    inv = (d ** 3) * MAC                           # one-off correlation inverse (cached)
    return {"prep": prep, "means": means, "cov": cov, "inv": inv,
            "total": prep + means + cov + inv}


def slda_training_flops(n=N_TRAIN, d=D):
    """NOTE: our SLDA (dev/run_cpu_friendly_methods.py) is a TRUE per-sample
    streaming implementation -- it loops over individual samples doing a rank-1
    outer product plus a full DxD read-modify-write each step, rather than one
    batched D.T@D. That costs more arithmetic than the batched form AND gets no
    BLAS batching, which is why its wall-clock is far worse than FeCAM's at
    comparable FLOPs. Counted here as implemented, not as idealized.
    """
    per_sample = (
        3 * d                 # running mean update
        + d                   # d = x - mean
        + d * d * MAC         # d @ d.T  (rank-1 outer product)
        + 3 * d * d           # (outer - cov), /n, cov +=
    )
    return n * per_sample + (d ** 3) * MAC     # + one inverse at scoring


def ncm_training_flops(n=N_TRAIN, d=D):
    """Prototype means only -- no covariance, no Tukey prep (FeCAM-specific)."""
    return n * d + N_CLASSES * d


def ridge_training_flops(n=N_TRAIN, d=D, rp=None):
    """Gram matrix G = H^T H and C = H^T Y, then one solve."""
    dim = rp if rp else d
    proj = (n * d * rp * MAC + n * rp) if rp else 0        # random projection + ReLU
    gram = n * dim * dim * MAC
    corr = n * dim * N_CLASSES * MAC
    solve = (dim ** 3) * MAC
    return proj + gram + corr + solve


def randumb_training_flops(n=N_TRAIN, d=D, rff=RFF_DIM):
    """RFF lift (cos(XW+b)) then an SLDA head in the lifted space."""
    lift = n * d * rff * MAC + n * rff * 3                 # matmul + bias + cos + scale
    head = n * rff + n * rff * rff * MAC + (rff ** 3) * MAC
    return lift + head


# ── INFERENCE: one (T,D) window -> one prediction ─────────────────────────────
# Deployment shape that matters for us is the demo's real-time path: a single
# window arrives, the head scores it, no batching. Reported in two parts:
#   steady-state  = per-window cost once any one-time precomputation is cached
#   per-call extra = precomputation our current code redoes on EVERY scores()
# FeCAM caches its precision matrix (self._cache); SLDA/Ridge/RanDumb do NOT --
# they recompute a DxD (or 2000x2000) inverse/solve inside every scores() call,
# which is invisible in batch evaluation but crippling for single-window
# real-time use. Flagged rather than silently idealized.
def gru_infer_flops():
    """Forward pass only -- no backward at inference."""
    return gru_forward_flops()


def fecam_infer_flops(d=D, k=C):
    """mean-pool + Tukey prep, then the expanded Mahalanobis quadratic form.

    scores() computes (x-m)'P(x-m) = x'Px - x'Pm - m'Px + m'Pm with the m-only
    term cached, so the per-window cost is one (D,)x(D,D) matvec plus two
    (D,)x(D,K) matvecs -- D^2 + 2KD MACs -- instead of the K*D^2 the old
    per-class loop cost. See src/models/fecam_head.py::scores.
    """
    pool = T * d + d
    prep = d * 3 + 2 * d * MAC + d
    quad = d * d * MAC + d              # x'Px
    cross = 2 * k * d * MAC             # x'Pm and m'Px
    return {"steady": pool + prep + quad + cross + 3 * k, "per_call": 0}


def fecam_infer_flops_loop(d=D, k=C):
    """Pre-optimization cost, kept for the before/after comparison in the report."""
    return (T * d + d) + (d * 3 + 2 * d * MAC + d) + k * (2 * d + (d * d + d) * MAC)


def ncm_infer_flops(d=D, k=C):
    """L2-normalize then one (D,) x (D,K) matvec."""
    return {"steady": T * d + d + 3 * d + d * k * MAC, "per_call": 0}


def slda_infer_flops(d=D, k=C):
    """Cached form is a single matvec; our code recomputes inv(cov) per call."""
    steady = T * d + d + d * k * MAC + k
    per_call = (d ** 3) * MAC + k * d * d * MAC        # inv(cov) + means @ prec
    return {"steady": steady, "per_call": per_call}


def ridge_infer_flops(d=D, k=C):
    """Cached W is a matvec; our code runs a DxD solve per call."""
    steady = T * d + d + d * k * MAC
    per_call = (d ** 3) * MAC                          # np.linalg.solve(G+lam I, C)
    return {"steady": steady, "per_call": per_call}


def randumb_infer_flops(d=D, rff=RFF_DIM, k=C):
    """RFF lift then matvec; our code inverts a 2000x2000 covariance per call."""
    lift = d * rff * MAC + rff * 3
    steady = T * d + d + lift + rff * k * MAC + k
    per_call = (rff ** 3) * MAC + k * rff * rff * MAC
    return {"steady": steady, "per_call": per_call}


# ── measured wall-clock, FIT ONLY ─────────────────────────────────────────────
# CORRECTION: the train_s figures in reports/cpu_friendly_methods_result.md and
# cpu_friendly/modern runners are NOT pure fit time -- their timer brackets a
# model.scores(Xva) evaluation call made at stage 1 (to record the s1_drop
# metric). For FeCAM that evaluation is 99.5% of the reported 8.7s; its actual
# fit is 42ms. Numbers below are fit-only, measured by timing observe() alone
# over the same 8 stages on the same real features (min of 3 runs).
WALL_S = {
    "GRU + A-GEM": 270.0,          # backprop loop; no eval inside the timer
    "FeCAM (shared cov)": 0.0424,
    "Deep SLDA": 1.860,
    "NCM prototype": 0.0024,
    "Ridge RLS (closed-form)": 0.0213,
    "RanDumb (RFF 2000)": 0.532,
}


def fmt(x):
    for unit, div in (("P", 1e15), ("T", 1e12), ("G", 1e9), ("M", 1e6), ("K", 1e3)):
        if x >= div:
            return f"{x / div:6.2f} {unit}"
    return f"{x:6.0f}  "


def main():
    # sanity check: our hand-derived parameter count must match the real model
    model = make_model(ExperimentConfig("best_current_h256", hidden_dim=H), D)
    real_params = sum(p.numel() for p in model.parameters())
    gru = gru_agem_training_flops()
    assert gru["n_params"] == real_params, (
        f"param formula {gru['n_params']} != actual {real_params}")

    print(f"protocol: {N_STAGES} stages x {PER_STAGE} samples x {EPOCHS} epochs "
          f"| D={D} T={T} H={H} C={C}")
    print(f"GRU param-count check: derived {gru['n_params']} == actual {real_params}  OK")
    print(f"convention: 1 MAC = {MAC} FLOPs, train = {BWD_MULT}x forward\n")

    rows = [
        ("GRU + A-GEM", gru["total"]),
        ("FeCAM (shared cov)", fecam_training_flops()["total"]),
        ("Deep SLDA", slda_training_flops()),
        ("NCM prototype", ncm_training_flops()),
        ("Ridge RLS (closed-form)", ridge_training_flops()),
        ("Ridge + RP 2000", ridge_training_flops(rp=RP_DIM)),
        ("RanDumb (RFF 2000)", randumb_training_flops()),
    ]
    # every analytic head also pays the one-time mean-pool
    pool = mean_pool_flops()
    rows = [(n, f if "GRU" in n else f + pool) for n, f in rows]

    base = gru["total"]
    print(f"{'method':26s} {'train FLOPs':>12s}  {'vs GRU':>9s}  "
          f"{'wall-clock':>10s}  {'FLOPs/s':>10s}")
    print("-" * 76)
    for name, f in rows:
        w = WALL_S.get(name)
        if w is None:
            ws, rate = "-", "-"
        else:
            ws = f"{w:.1f} s" if w >= 1 else f"{w * 1000:.1f} ms"
            rate = fmt(f / w).strip() + "/s"
        print(f"{name:26s} {fmt(f):>12s}  {base / f:8.0f}x  {ws:>10s}  {rate:>11s}")

    print("\nGRU + A-GEM breakdown:")
    for k in ("main", "agem_ref", "agem_proj"):
        print(f"  {k:12s} {fmt(gru[k]):>12s}  ({100 * gru[k] / base:4.1f}%)")

    fe = fecam_training_flops()
    print("\nFeCAM breakdown:")
    for k in ("prep", "means", "cov", "inv"):
        print(f"  {k:12s} {fmt(fe[k]):>12s}  ({100 * fe[k] / fe['total']:4.1f}%)")

    enc = clip_vitb32_forward_flops()
    enc_all = enc * N_TRAIN
    print(f"\ncontext -- frozen CLIP ViT-B/32 encoder (shared by ALL heads, "
          f"paid once per video and cached):")
    print(f"  per {T}-frame window      {fmt(enc)}")
    print(f"  over the {N_TRAIN} train windows {fmt(enc_all)}")
    print(f"  = {enc_all / base:.0f}x the entire GRU+A-GEM training, "
          f"{enc_all / fecam_training_flops()['total']:.0f}x FeCAM's")

    # ── inference ────────────────────────────────────────────────────────────
    print(f"\n\nINFERENCE -- one {T}-frame window -> one prediction "
          f"({C} classes enrolled)")
    print(f"{'head':26s} {'steady-state':>12s}  {'per-call extra':>14s}  "
          f"{'% of encoder':>12s}")
    print("-" * 72)
    inf = [
        ("GRU head", {"steady": gru_infer_flops(), "per_call": 0}),
        ("FeCAM (shared cov)", fecam_infer_flops()),
        ("NCM prototype", ncm_infer_flops()),
        ("Deep SLDA", slda_infer_flops()),
        ("Ridge RLS (closed-form)", ridge_infer_flops()),
        ("RanDumb (RFF 2000)", randumb_infer_flops()),
    ]
    for name, d in inf:
        extra = fmt(d["per_call"]).strip() if d["per_call"] else "-- (cached)"
        print(f"{name:26s} {fmt(d['steady']):>12s}  {extra:>14s}  "
              f"{100 * d['steady'] / enc:11.4f}%")
    print(f"{'frozen CLIP encoder':26s} {fmt(enc):>12s}  {'--':>14s}  "
          f"{100.0:11.4f}%")
    print("\n  'per-call extra' = precomputation our code currently redoes inside")
    print("  every scores() call (a DxD / 2000x2000 inverse or solve). Harmless")
    print("  when scoring a big batch, dominant for single-window real-time use.")
    print("  FeCAM caches it; SLDA/Ridge/RanDumb do not -- see reports/flops_result.md.")

    loop = fecam_infer_flops_loop()
    now = fecam_infer_flops()["steady"]
    print(f"\n  FeCAM scoring was optimized (per-class loop -> expanded quadratic form):")
    print(f"    before {fmt(loop).strip()}  ->  after {fmt(now).strip()}   "
          f"({loop / now:.0f}x fewer FLOPs, accuracy bit-identical)")


if __name__ == "__main__":
    main()
