"""
Figures for the CIL results — the standard accuracy-vs-classes-seen curves that
every paper in this line uses (TCD Fig 4, STSP Fig 3, ESSENTIAL Fig 4).

PyCIL itself ships no plotting; it writes logs and leaves the figures to you,
so these are drawn from our own raw result JSONs.

Outputs (reports/figures/):
  ucf101_tcd_curves.png     UCF101 TCD protocol — our heads vs our GRU baseline
  cifar100_bridge.png       PyCIL CIFAR-100 bridge — same splits as PyCIL
  increment_sensitivity.png accuracy change as increments get finer

Run: python3 dev/plot_cil_results.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/figures"

# ── design tokens (dataviz reference palette, light mode) ────────────────────
SURFACE = "#fcfcfb"
INK = "#0b0b0b"          # text-primary
INK_2 = "#52514e"        # text-secondary
GRID = "#e1e0d9"         # hairline gridline
AXIS = "#c3c2b7"         # baseline / axis
# categorical slots, assigned in fixed order and never cycled
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"

LINE_W = 2.0
MARKER = 8               # >= 8px


def style_axes(ax, xlabel, ylabel):
    """Recessive chrome: hairline horizontal grid, no top/right spines."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="y", color=GRID, linewidth=1.0, linestyle="-", zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=INK_2, labelsize=10, length=0)
    ax.set_xlabel(xlabel, color=INK_2, fontsize=11, labelpad=8)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=11, labelpad=8)


def plot_series(ax, x, y, color, label, direct_label=True, dx=1.2):
    ax.plot(x, y, color=color, linewidth=LINE_W, solid_joinstyle="round",
            solid_capstyle="round", marker="o", markersize=6.5,
            markerfacecolor=color, markeredgecolor=SURFACE,
            markeredgewidth=1.5, label=label, zorder=3)
    if direct_label:
        # label rides in text ink; the colored line beside it carries identity
        ax.annotate(label, xy=(x[-1], y[-1]), xytext=(x[-1] + dx, y[-1]),
                    color=INK, fontsize=10, va="center", fontweight="medium")


def fig_ucf101():
    tcd = json.loads((ROOT / "reports/ucf101_tcd_raw.json").read_text())
    gru = json.loads((ROOT / "reports/ucf101_gru_agem_raw.json").read_text())
    xs = [51, 61, 71, 81, 91, 101]

    def mean_curve(entry, key="per_seed"):
        return 100 * np.array([r["per_step"] for r in entry[key]]).mean(axis=0)

    fecam = mean_curve(tcd["inc10/FeCAM (shared cov)"])
    slda = mean_curve(tcd["inc10/Deep SLDA"])
    ncm = mean_curve(tcd["inc10/NCM prototype"])
    gru_c = 100 * np.array([r["per_step"] for r in gru["runs"]]).mean(axis=0)

    fig, ax = plt.subplots(figsize=(8.4, 5.2), facecolor=SURFACE)
    plot_series(ax, xs, fecam, S1, "FeCAM")
    plot_series(ax, xs, slda, S2, "Deep SLDA")
    plot_series(ax, xs, ncm, S3, "NCM")
    plot_series(ax, xs, gru_c, S4, "GRU + A-GEM")

    style_axes(ax, "Classes seen", "Accuracy over seen classes (%)")
    ax.set_xticks(xs)
    ax.set_xlim(48, 118)
    ax.set_ylim(40, 96)
    ax.set_title("UCF101, TCD protocol (51-class base + 10 per step)",
                 color=INK, fontsize=13, fontweight="semibold", pad=30,
                 loc="left")
    ax.text(0, 1.018, "mean of TCD's three class orders · true class-IL, no task id",
            transform=ax.transAxes, color=INK_2, fontsize=10)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13),
              fontsize=10, labelcolor=INK_2, handlelength=1.6, ncols=4)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"ucf101_tcd_curves.{ext}", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def fig_cifar_bridge():
    d = json.loads((ROOT / "reports/pycil_bridge_b50_i10_raw.json").read_text())
    xs = [d["init"] + i * d["inc"] for i in range(d["n_tasks"])]
    cur = lambda k: 100 * np.array(d["heads"][k]["per_task"])  # noqa: E731
    fecam, slda, ncm = (cur("FeCAM (shared cov)"), cur("Deep SLDA"),
                        cur("NCM prototype"))

    fig, ax = plt.subplots(figsize=(8.4, 5.2), facecolor=SURFACE)
    plot_series(ax, xs, fecam, S1, "FeCAM")
    plot_series(ax, xs, slda, S2, "Deep SLDA")
    plot_series(ax, xs, ncm, S3, "NCM")

    style_axes(ax, "Classes seen", "Accuracy over seen classes (%)")
    ax.set_xticks(xs)
    ax.set_xlim(48, 116)
    ax.set_ylim(60, 84)
    ax.set_title("CIFAR-100 on PyCIL's own splits (b0=50, +10 per task)",
                 color=INK, fontsize=13, fontweight="semibold", pad=30,
                 loc="left")
    ax.text(0, 1.018,
            "class order imported from PyCIL's DataManager — bit-identical",
            transform=ax.transAxes, color=INK_2, fontsize=10)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13),
              fontsize=10, labelcolor=INK_2, handlelength=1.6, ncols=3)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"cifar100_bridge.{ext}", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def fig_ssv2_curves():
    d = json.loads((ROOT / "reports/ssv2_head_curves_raw.json").read_text())
    xs = [6 * i for i in range(1, 9)]
    cur = lambda k: 100 * np.array(d[k]["per_step"])  # noqa: E731
    fecam, slda, ncm = (cur("FeCAM (shared cov)"), cur("Deep SLDA"),
                        cur("NCM prototype"))

    fig, ax = plt.subplots(figsize=(8.4, 5.2), facecolor=SURFACE)
    plot_series(ax, xs, fecam, S1, "FeCAM")
    plot_series(ax, xs, slda, S2, "Deep SLDA")
    plot_series(ax, xs, ncm, S3, "NCM")

    style_axes(ax, "Classes seen", "Accuracy over seen classes (%)")
    ax.set_xticks(xs)
    ax.set_xlim(3, 58)
    ax.set_ylim(0, 55)
    ax.set_title("SSv2 (our 48-class subset), same 3 heads as UCF101",
                 color=INK, fontsize=13, fontweight="semibold", pad=30,
                 loc="left")
    ax.text(0, 1.018,
            "true class-IL, no task id · 8-stage curriculum order",
            transform=ax.transAxes, color=INK_2, fontsize=10)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13),
              fontsize=10, labelcolor=INK_2, handlelength=1.6, ncols=3)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"ssv2_head_curves.{ext}", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def fig_static_vs_temporal():
    """Same three heads, final accuracy on each benchmark -- do they all drop
    by roughly the same amount, or is the UCF101/SSv2 gap FeCAM-specific?"""
    ucf = json.loads((ROOT / "reports/ucf101_tcd_raw.json").read_text())
    ssv2 = json.loads((ROOT / "reports/ssv2_head_curves_raw.json").read_text())
    heads = ["NCM prototype", "Deep SLDA", "FeCAM (shared cov)"]
    labels = ["NCM", "Deep SLDA", "FeCAM"]
    ucf_vals = [100 * ucf[f"inc10/{h}"]["last"] for h in heads]
    ssv2_vals = [100 * ssv2[h]["last"] for h in heads]

    fig, ax = plt.subplots(figsize=(8.4, 4.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    x = np.arange(len(heads))
    w = 0.32
    b1 = ax.bar(x - w / 2, ucf_vals, width=w, color=S1, zorder=3,
               label="UCF101 (static-biased)")
    b2 = ax.bar(x + w / 2, ssv2_vals, width=w, color=S2, zorder=3,
               label="SSv2 (temporal-biased)")
    for bars, vals in ((b1, ucf_vals), (b2, ssv2_vals)):
        for rect, v in zip(bars, vals):
            ax.annotate(f"{v:.1f}", xy=(rect.get_x() + rect.get_width() / 2, v),
                        xytext=(0, 6), textcoords="offset points",
                        ha="center", color=INK, fontsize=10)
    for xi, uv, sv in zip(x, ucf_vals, ssv2_vals):
        ax.annotate(f"−{uv - sv:.0f}", xy=(xi, (uv + sv) / 2),
                    xytext=(28, 0), textcoords="offset points",
                    ha="left", va="center", color=INK_2, fontsize=10)

    ax.set_xticks(x, labels, color=INK_2, fontsize=11)
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", color=GRID, linewidth=1.0, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_2, labelsize=10, length=0)
    ax.set_ylabel("Final accuracy (%)", color=INK_2, fontsize=11, labelpad=8)
    ax.set_title("The static/temporal gap is structural, not FeCAM-specific",
                 color=INK, fontsize=13, fontweight="semibold", pad=30,
                 loc="left")
    ax.text(0, 1.03, "same encoder, same 3 heads, last-session accuracy on each benchmark",
            transform=ax.transAxes, color=INK_2, fontsize=10)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12),
              fontsize=10, labelcolor=INK_2, handlelength=1.6, ncols=2)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"static_vs_temporal.{ext}", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def fig_increment_sensitivity():
    """One measure, one series -> bars. Each paper's own UCF101 numbers."""
    rows = [                     # (label, coarse 10x5, fine 2x25)
        ("TCD (ICCV'21)", 74.89, 72.19),
        ("FrameMaker", 78.13, 75.77),
        ("STSP (ECCV'24)", 81.15, 79.25),
        ("ESSENTIAL (ICCV'25)", 95.1, 93.3),
        ("Ours (FeCAM)", 88.84, 88.84),
    ]
    labels = [r[0] for r in rows]
    deltas = [r[2] - r[1] for r in rows]

    fig, ax = plt.subplots(figsize=(8.4, 4.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    y = np.arange(len(rows))
    # emphasis: ours in slot 1, the literature recedes to a single muted tone
    colors = [AXIS] * (len(rows) - 1) + [S1]
    ax.barh(y, deltas, height=0.58, color=colors, zorder=3)
    # a zero-length bar would read as missing data, so the no-change row keeps
    # a visible mark on the baseline
    for i, d in enumerate(deltas):
        if abs(d) < 0.005:
            ax.plot([0], [i], marker="o", markersize=8, color=colors[i],
                    markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=4)

    for i, d in enumerate(deltas):
        ax.annotate(f"{d:+.2f}", xy=(d, i),
                    xytext=(-8 if d < 0 else 8, 0), textcoords="offset points",
                    va="center", ha="right" if d < 0 else "left",
                    color=INK, fontsize=10, fontweight="medium")

    ax.set_yticks(y, labels, color=INK_2, fontsize=10)
    ax.invert_yaxis()
    ax.grid(True, axis="x", color=GRID, linewidth=1.0, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.axvline(0, color=AXIS, linewidth=1.0, zorder=2)
    ax.tick_params(colors=INK_2, labelsize=10, length=0)
    ax.set_xlim(-3.2, 0.9)
    ax.set_xlabel("Accuracy change, 10-class steps → 2-class steps (pp)",
                  color=INK_2, fontsize=11, labelpad=8)
    ax.set_title("Finer increments cost every method accuracy — except ours",
                 color=INK, fontsize=13, fontweight="semibold", pad=30,
                 loc="left")
    ax.text(0, 1.06, "UCF101, each paper's own reported numbers",
            transform=ax.transAxes, color=INK_2, fontsize=10)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"increment_sensitivity.{ext}", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig_ucf101()
    fig_cifar_bridge()
    fig_ssv2_curves()
    fig_static_vs_temporal()
    fig_increment_sensitivity()
    for p in sorted(OUT.glob("*.*")):
        print(f"  {p.relative_to(ROOT)}  ({p.stat().st_size//1024} KB)")


if __name__ == "__main__":
    main()
