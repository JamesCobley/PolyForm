# ============================================================================
# PolyForm — Figure: the modal PTM lattice as a graded binomial object
# (revised layout: leader-line labels + relaxed placement so no text overlaps)
# ============================================================================

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from math import lgamma, log10, comb
from itertools import product

# ----------------------------------------------------------------------------
# User parameters
# ----------------------------------------------------------------------------

R_MAIN = 300         # main modal lattice shown as graded binomial object
R_EXAMPLE = 393      # p53-style example for low-grade cone
KMAX_CONE = 15       # show cumulative low-grade region up to this k
R_TOY = 5            # explicit Hamming graph toy example
DPI = 300

OUT_PNG = "polyform_modal_lattice_figure.png"
OUT_PDF = "polyform_modal_lattice_figure.pdf"

# ----------------------------------------------------------------------------
# Plot style
# ----------------------------------------------------------------------------

mpl.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "axes.labelsize": 9.5,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.frameon": False,
})

INK    = "#1b2a41"
TEAL   = "#0f9b8e"
AMBER  = "#e08a2e"
VIOLET = "#7d5ba6"
GREY   = "#b8bec8"
ROSE   = "#c44e52"
BLUE   = "#4c78a8"

BOX = dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=GREY, alpha=0.96)

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def log10_comb(n, k):
    """Stable log10(binomial(n, k)) without explicit huge integers."""
    if k < 0 or k > n:
        return np.nan
    if k == 0 or k == n:
        return 0.0
    return (lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)) / np.log(10)

def total_states_log10(R):
    return R * log10(2)

def cumulative_low_grade_counts(R, kmax):
    ks = np.arange(0, kmax + 1)
    exact = []
    running = 0
    for k in ks:
        running += comb(R, int(k))
        exact.append(running)
    return ks, np.array(exact, dtype=object)

def state_to_str(bits):
    return "".join(str(b) for b in bits)

def hamming_distance(a, b):
    return sum(x != y for x, y in zip(a, b))

def relax_labels(targets, ymin, ymax, min_gap):
    """Spread label y-positions so consecutive labels are >= min_gap apart,
    while staying as close as possible to their target positions."""
    y = np.array(sorted(targets), dtype=float)
    # forward pass: push up when too close
    for _ in range(500):
        moved = False
        for i in range(1, len(y)):
            if y[i] - y[i - 1] < min_gap:
                shift = (min_gap - (y[i] - y[i - 1])) / 2.0
                y[i - 1] -= shift
                y[i] += shift
                moved = True
        y[0] = max(y[0], ymin)
        y[-1] = min(y[-1], ymax)
        # re-clamp interior after boundary clamp
        for i in range(1, len(y)):
            if y[i] - y[i - 1] < min_gap:
                y[i] = y[i - 1] + min_gap
        if not moved:
            break
    return y

# ----------------------------------------------------------------------------
# Panel A — exact graded modal lattice for large R
# ----------------------------------------------------------------------------

ks_main = np.arange(0, R_MAIN + 1)
log10_g = np.array([log10_comb(R_MAIN, int(k)) for k in ks_main])
max_log = log10_g.max()
widths = 0.02 + 0.98 * (log10_g / max_log)

# ----------------------------------------------------------------------------
# Panel B — cumulative low-grade cone
# ----------------------------------------------------------------------------

ks_cone, cum_counts = cumulative_low_grade_counts(R_EXAMPLE, KMAX_CONE)
log10_cum = np.array([log10(int(x)) for x in cum_counts], dtype=float)
log10_total_example = total_states_log10(R_EXAMPLE)
log10_strata = np.array([log10_comb(R_EXAMPLE, int(k)) for k in ks_cone])

# ----------------------------------------------------------------------------
# Panel C — explicit toy Hamming graph
# ----------------------------------------------------------------------------

toy_states = [tuple(bits) for bits in product([0, 1], repeat=R_TOY)]
toy_states = sorted(toy_states, key=lambda x: (sum(x), x))

toy_by_grade = {}
for s in toy_states:
    toy_by_grade.setdefault(sum(s), []).append(s)

positions = {}
for k, states_k in toy_by_grade.items():
    n = len(states_k)
    xs = np.linspace(-n / 2, n / 2, n) if n > 1 else np.array([0.0])
    for x, s in zip(xs, states_k):
        positions[s] = (x, k)

toy_edges = []
for i in range(len(toy_states)):
    for j in range(i + 1, len(toy_states)):
        if hamming_distance(toy_states[i], toy_states[j]) == 1:
            toy_edges.append((toy_states[i], toy_states[j]))

# ----------------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------------

fig = plt.figure(figsize=(13.5, 8.4))
gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.0], height_ratios=[1.0, 1.0],
                      hspace=0.36, wspace=0.30)

# ============================================================================
# (a) Graded modal lattice as diamond-shaped binomial object
# ============================================================================
ax = fig.add_subplot(gs[:, 0])

for k, w in zip(ks_main, widths):
    ax.hlines(y=k, xmin=-w, xmax=w, color=TEAL, lw=1.1)

ax.vlines(0, 0, R_MAIN, color=GREY, lw=0.8, alpha=0.8)

# --- leader-line labels on the RIGHT, relaxed so they never overlap ---------
annot_ks = [0, 1, 2, 5, 10, 25, 50, 100, 150, 200, 250, 275, 299, 300]
label_x = 1.14                       # where the text column starts
y_txt = relax_labels(annot_ks, ymin=0, ymax=R_MAIN, min_gap=21.0)

for k, yt in zip(annot_ks, y_txt):
    lg = log10_g[k]
    if k in (0, R_MAIN):
        count_label = "1"
    else:
        count_label = rf"$10^{{{lg:.1f}}}$"
    # thin leader line from the stratum edge to the text
    ax.plot([widths[k] + 0.01, label_x - 0.02], [k, yt],
            color=GREY, lw=0.6, alpha=0.9, zorder=1)
    ax.text(label_x, yt, rf"$k={k}$   $|\mathcal{{M}}_k|\approx${count_label}",
            va="center", ha="left", fontsize=7.4, color=INK)

ax.set_ylim(-6, R_MAIN + 6)
ax.set_xlim(-1.28, 2.05)
ax.set_xlabel("normalized width  ~  log$_{10}$ binomial stratum size")
ax.set_ylabel("grade $k$")
ax.set_title("a   The modal PTM lattice is a graded binomial object", loc="left")

text_a = (
    rf"$\mathcal{{M}}=\{{0,1\}}^R$" "\n"
    rf"$|\mathcal{{M}}_k|=\binom{{R}}{{k}}$,  $|\mathcal{{M}}|=2^R$" "\n"
    rf"For $R={R_MAIN}$:  $|\mathcal{{M}}|=2^{{{R_MAIN}}}\approx 10^{{{total_states_log10(R_MAIN):.1f}}}$" "\n"
    "Rows are exact grade strata;\n"
    "the full node set is not enumerated."
)
ax.text(0.015, 0.015, text_a, transform=ax.transAxes, ha="left", va="bottom",
        fontsize=7.6, color=INK, bbox=BOX)

# ============================================================================
# (b) Local navigability / low-grade cone
# ============================================================================
ax = fig.add_subplot(gs[0, 1])

ax.plot(ks_cone, log10_cum, marker="o", ms=4.5, lw=1.6, color=AMBER,
        label=r"cumulative cone  $\sum_{i=0}^{k}\binom{R}{i}$")
ax.plot(ks_cone, log10_strata, marker="s", ms=4, lw=1.2, color=VIOLET,
        label=r"single stratum  $\binom{R}{k}$")
ax.axhline(log10_total_example, color=GREY, lw=1.0, ls="--")

# label the full-space line directly (keeps it out of the legend)
ax.text(KMAX_CONE, log10_total_example - 4,
        rf"full space $2^{{R}}\approx10^{{{log10_total_example:.1f}}}$  ($R={R_EXAMPLE}$)",
        ha="right", va="top", fontsize=7.4, color="#5f6672")

ax.set_xlabel("maximum grade $k$ included")
ax.set_ylabel(r"$\log_{10}$ number of states")
ax.set_ylim(-6, log10_total_example + 12)
ax.set_xlim(-0.7, KMAX_CONE + 0.7)
ax.set_title("b   Low-grade regions are exactly enumerable", loc="left")
ax.legend(fontsize=7.6, loc="upper left", handlelength=1.6,
          borderaxespad=0.6, labelspacing=0.5)

cone_text = (
    rf"Example: human p53-like length $R={R_EXAMPLE}$" "\n"
    rf"Full space:  $2^{{{R_EXAMPLE}}}\approx 10^{{{log10_total_example:.1f}}}$" "\n"
    rf"One-step neighbourhood of any state:  $R={R_EXAMPLE}$" "\n"
    rf"Low-grade cone to $k\leq{KMAX_CONE}$:" "\n"
    rf"$\quad\sum_{{i=0}}^{{{KMAX_CONE}}}\binom{{{R_EXAMPLE}}}{{i}}\approx 10^{{{log10_cum[-1]:.1f}}}$"
)
ax.text(0.97, 0.44, cone_text, transform=ax.transAxes, ha="right", va="top",
        fontsize=7.5, color=INK, bbox=BOX)

# ============================================================================
# (c) Toy Hamming graph
# ============================================================================
ax = fig.add_subplot(gs[1, 1])

for u, v in toy_edges:
    x1, y1 = positions[u]
    x2, y2 = positions[v]
    ax.plot([x1, x2], [y1, y2], color=GREY, lw=0.7, alpha=0.6, zorder=1)

grades = [sum(s) for s in toy_states]
cmap = mpl.cm.viridis
norm = mpl.colors.Normalize(vmin=0, vmax=R_TOY)

for s in toy_states:
    x, y = positions[s]
    ax.scatter(x, y, s=46, color=cmap(norm(sum(s))),
               edgecolor="white", linewidth=0.6, zorder=2)

# label only the extreme grades; nudge k=1 labels below the row so they
# don't collide with the k=0/k=2 nodes or each other
for s in toy_states:
    x, y = positions[s]
    kk = sum(s)
    if kk in (0, R_TOY):
        ax.text(x, y + 0.22, state_to_str(s), ha="center", va="bottom",
                fontsize=6.6, color=INK)
    elif kk == 1:
        ax.text(x, y - 0.26, state_to_str(s), ha="center", va="top",
                fontsize=6.4, color=INK)

ax.set_title("c   Hamming-1 adjacency on a toy modal lattice", loc="left")
ax.set_xlabel("states arranged within each grade")
ax.set_ylabel("grade $k$")
ax.set_yticks(range(0, R_TOY + 1))
ax.set_xlim(-8.5, 8.5)
ax.set_ylim(-0.9, R_TOY + 1.7)   # extra headroom for the info box strip

toy_text = (
    rf"Toy example: $R={R_TOY}$,  $|\mathcal{{M}}|=2^{{{R_TOY}}}={2**R_TOY}$ states" "\n"
    "Edges connect states differing at exactly one site;\n"
    "each state has at most $R$ Hamming-1 neighbours."
)
ax.text(0.5, 0.985, toy_text, transform=ax.transAxes, ha="center", va="top",
        fontsize=7.3, color=INK, bbox=BOX)

sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, fraction=0.045, pad=0.02)
cbar.set_label("grade $k$")

# ----------------------------------------------------------------------------
# Suptitle and save
# ----------------------------------------------------------------------------

fig.suptitle(
    "PolyForm: the proteoform modal lattice is globally combinatorial but locally navigable",
    y=0.995, fontsize=12,
)

fig.savefig(OUT_PNG, dpi=DPI)
fig.savefig(OUT_PDF)
print(f"Saved: {OUT_PNG}")
print(f"Saved: {OUT_PDF}")
