"""
Result figures for the README / Pages / reports.

Generates (dark theme, matching the demo):
  docs/assets/fig_data_scale.png  — data-fraction scaling curve, B/32 vs L/14
  docs/assets/fig_levers.png      — effect of each lever on Avg Acc (data dominates)

Numbers come from reports/{data_scale,openclip_l14,gru_attention}_result.md.

Run: python3 dev/make_result_figs.py
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/assets"
OUT.mkdir(parents=True, exist_ok=True)

BG = "#0d1117"; FG = "#e6edf3"; MUTED = "#8b949e"; GRID = "#21262d"
BLUE = "#58a6ff"; GREEN = "#3fb950"; RED = "#f85149"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": FG, "axes.labelcolor": FG, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": GRID, "font.size": 12,
})

# ── data from reports ─────────────────────────────────────────────────────────
fracs = [25, 50, 100]
b32      = [0.314, 0.370, 0.388]
b32_std  = [0.009, 0.008, 0.020]
l14      = [0.339, 0.394, 0.401]
l14_std  = [0.008, 0.017, 0.033]
BASELINE = 0.388  # B/32 + GRU @ full data

# ── Figure A: scaling curve ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.4, 4.2))
ax.errorbar(fracs, l14, yerr=l14_std, marker="o", color=GREEN, capsize=4,
            lw=2.2, label="OpenCLIP ViT-L/14 (768d)")
ax.errorbar(fracs, b32, yerr=b32_std, marker="o", color=BLUE, capsize=4,
            lw=2.2, label="CLIP ViT-B/32 (512d)")
ax.set_xlabel("Training data used (%)")
ax.set_ylabel("Avg accuracy (8-stage continual)")
ax.set_title("Data quantity is the dominant lever", color=FG, fontweight="bold")
ax.set_xticks(fracs); ax.set_ylim(0.28, 0.44)
ax.grid(True, color=GRID, lw=0.8)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG, loc="lower right")
ax.annotate("still rising at 100%\n(not saturated)", xy=(100, 0.401), xytext=(63, 0.345),
            color=MUTED, fontsize=10,
            arrowprops=dict(arrowstyle="->", color=MUTED))
fig.tight_layout()
fig.savefig(OUT / "fig_data_scale.png", dpi=150)
print("saved", OUT / "fig_data_scale.png")

# ── Figure B: lever effects (Δ Avg Acc vs B/32+GRU baseline) ──────────────────
levers = [
    ("More data\n(25%→100%)", 0.074, GREEN),
    ("Backbone\n(B/32→L/14)", 0.013, BLUE),
    ("GRU capacity\n(256→512)", 0.000, MUTED),
    ("Temporal\n(GRU→Attention)", -0.103, RED),
]
labels = [x[0] for x in levers]
vals   = [x[1] for x in levers]
cols   = [x[2] for x in levers]

fig, ax = plt.subplots(figsize=(6.8, 4.2))
bars = ax.barh(range(len(levers)), vals, color=cols, height=0.62)
ax.set_yticks(range(len(levers))); ax.set_yticklabels(labels)
ax.invert_yaxis()
ax.axvline(0, color=MUTED, lw=1)
ax.set_xlabel("Δ Avg accuracy vs simple GRU + A-GEM baseline (0.388)")
ax.set_title("What actually moves the needle", color=FG, fontweight="bold")
ax.set_xlim(-0.13, 0.10)
ax.grid(True, axis="x", color=GRID, lw=0.8)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
for i, v in enumerate(vals):
    ax.text(v + (0.004 if v >= 0 else -0.004), i, f"{v:+.3f}",
            va="center", ha="left" if v >= 0 else "right", color=FG, fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "fig_levers.png", dpi=150)
print("saved", OUT / "fig_levers.png")
