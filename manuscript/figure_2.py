# ============================================================================
# PolyForm Figure 2
# Recursive Hamming structure of the modal proteoform lattice
#
# Core message:
#   The modal lattice is globally exponential but locally navigable.
#   Its Hamming-1 adjacency is recursively generated and exactly compressible
#   by grade, even when the full 2^R graph cannot be enumerated.
# ============================================================================

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from math import lgamma, log10
from matplotlib.colors import ListedColormap

# ----------------------------------------------------------------------------
# User parameters
# ----------------------------------------------------------------------------

REPO_DIR = "/content/PolyForm"
OUTDIR = os.path.join(REPO_DIR, "manuscript", "figures")

if not os.path.exists(REPO_DIR):
    OUTDIR = "/content"

os.makedirs(OUTDIR, exist_ok=True)

DPI = 300

R_PROOF = 8          # recursive proof panel
R_MOTIFS = [4, 6, 8, 10]
R_LARGE = 350        # large R for exact compressed grade adjacency

OUT_PNG = os.path.join(OUTDIR, "figure02_recursive_hamming_structure.png")
OUT_PDF = os.path.join(OUTDIR, "figure02_recursive_hamming_structure.pdf")


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


# ============================================================================
# Helper functions
# ============================================================================

def log10_comb(n, k):
    """
    Stable log10 binomial coefficient.
    """
    if k < 0 or k > n:
        return np.nan
    if k == 0 or k == n:
        return 0.0
    return (lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)) / np.log(10)


def adjacency_direct_binary_order(R):
    """
    Direct Hamming-1 adjacency in binary integer order.

    State i is connected to state j if j = i XOR 2^bit
    for exactly one bit.
    """
    n = 2 ** R
    A = np.zeros((n, n), dtype=np.int8)

    for i in range(n):
        for bit in range(R):
            j = i ^ (1 << bit)
            A[i, j] = 1

    return A


def adjacency_recursive(R):
    """
    Recursive construction of the R-dimensional hypercube adjacency matrix.

    A_R = [[A_{R-1}, I],
           [I,       A_{R-1}]]

    This assumes binary integer ordering.
    """
    if R < 1:
        raise ValueError("R must be >= 1")

    A = np.array([[0, 1],
                  [1, 0]], dtype=np.int8)

    if R == 1:
        return A

    for r in range(2, R + 1):
        n_prev = A.shape[0]
        I = np.eye(n_prev, dtype=np.int8)
        A = np.block([
            [A, I],
            [I, A],
        ])

    return A


def adjacency_binary_by_grade_order(R):
    """
    Build adjacency matrix after sorting states by grade k and then binary value.
    This is for visualization of k-layered motifs.
    """
    n = 2 ** R
    states = list(range(n))

    def grade(i):
        return int(i).bit_count()

    states = sorted(states, key=lambda i: (grade(i), i))
    index = {state: idx for idx, state in enumerate(states)}

    A = np.zeros((n, n), dtype=np.int8)

    for state in states:
        i = index[state]
        for bit in range(R):
            neigh = state ^ (1 << bit)
            j = index[neigh]
            A[i, j] = 1

    boundaries = []
    running = 0
    for k in range(R + 1):
        count_k = sum(1 for s in states if grade(s) == k)
        running += count_k
        if k < R:
            boundaries.append(running)

    return A, boundaries


def log10_edges_between_grades(R):
    """
    Hamming-1 edges between grade k and k+1:

        E_{k,k+1} = C(R,k) * (R-k)
                  = C(R,k+1) * (k+1)

    Returns k=0..R-1 and log10 edge counts.
    """
    ks = np.arange(0, R)
    vals = np.array([
        log10_comb(R, int(k)) + log10(R - int(k))
        for k in ks
    ])
    return ks, vals


def compressed_grade_adjacency_matrix(R):
    """
    Matrix M where M[k,k+1] and M[k+1,k] contain
    log10 number of Hamming-1 edges between neighbouring grades.
    """
    K = R + 1
    M = np.full((K, K), np.nan)

    ks, vals = log10_edges_between_grades(R)

    for k, v in zip(ks, vals):
        M[k, k + 1] = v
        M[k + 1, k] = v

    return M


def add_panel_label(ax, label):
    ax.text(
        -0.08, 1.08, label,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=13,
        fontweight="bold",
        color=INK,
    )


# ============================================================================
# Build proof matrices
# ============================================================================

A_direct = adjacency_direct_binary_order(R_PROOF)
A_rec = adjacency_recursive(R_PROOF)
A_diff = A_direct - A_rec
max_abs_diff = np.max(np.abs(A_diff))

print(f"Recursive proof R={R_PROOF}: max absolute difference = {max_abs_diff}")


# ============================================================================
# Build compressed large-R grade adjacency
# ============================================================================

M_grade = compressed_grade_adjacency_matrix(R_LARGE)
ks_edges, log_edges = log10_edges_between_grades(R_LARGE)

log10_vertices_large = R_LARGE * log10(2)
n_grades_large = R_LARGE + 1
n_grade_transitions = R_LARGE


# ============================================================================
# Figure
# ============================================================================

fig = plt.figure(figsize=(14.2, 9.2))
outer = fig.add_gridspec(
    2, 2,
    width_ratios=[1.25, 1.0],
    height_ratios=[1.0, 1.0],
    hspace=0.38,
    wspace=0.30,
)

# ----------------------------------------------------------------------------
# Panel a — recursive proof
# ----------------------------------------------------------------------------

gs_a = outer[0, 0].subgridspec(1, 3, wspace=0.10)
axes_a = [fig.add_subplot(gs_a[0, i]) for i in range(3)]

proof_cmap = ListedColormap(["#06136d", "#f2f80f"])

axes_a[0].imshow(A_direct, cmap=proof_cmap, interpolation="nearest", origin="upper")
axes_a[0].set_title(f"Direct Hamming\nR={R_PROOF}", fontsize=9)

axes_a[1].imshow(A_rec, cmap=proof_cmap, interpolation="nearest", origin="upper")
axes_a[1].set_title("Recursive\nconstruction", fontsize=9)

axes_a[2].imshow(A_diff, cmap="coolwarm", interpolation="nearest", origin="upper", vmin=-1, vmax=1)
axes_a[2].set_title(f"Difference\nmax |Δ|={max_abs_diff}", fontsize=9)

for ax in axes_a:
    ax.set_xticks([])
    ax.set_yticks([])

add_panel_label(axes_a[0], "a")

# Matplotlib mathtext does not support LaTeX pmatrix.
# Use Unicode/plain text to show the recursive block structure.
axes_a[0].text(
    0.0, -0.18,
    "Aᴿ = [ Aᴿ⁻¹   I ]\n     [  I    Aᴿ⁻¹ ]",
    transform=axes_a[0].transAxes,
    ha="left",
    va="top",
    fontsize=9.5,
    color=INK,
    family="DejaVu Sans Mono",
)

# ----------------------------------------------------------------------------
# Panel b — recurring motifs as R increases
# ----------------------------------------------------------------------------

gs_b = outer[1, 0].subgridspec(1, len(R_MOTIFS), wspace=0.08)
axes_b = [fig.add_subplot(gs_b[0, i]) for i in range(len(R_MOTIFS))]

for ax, R in zip(axes_b, R_MOTIFS):
    A, boundaries = adjacency_binary_by_grade_order(R)
    ax.imshow(A, cmap=proof_cmap, interpolation="nearest", origin="upper")

    for b in boundaries:
        ax.axhline(b - 0.5, color="white", lw=0.35, alpha=0.7)
        ax.axvline(b - 0.5, color="white", lw=0.35, alpha=0.7)

    ax.set_title(f"R={R}\nn={2**R}", fontsize=8.5)
    ax.set_xticks([])
    ax.set_yticks([])

add_panel_label(axes_b[0], "b")

axes_b[0].text(
    0.0, -0.18,
    "The same Hamming-1 motif recurs as the lattice grows.",
    transform=axes_b[0].transAxes,
    ha="left", va="top",
    fontsize=8.7,
    color=INK,
)

# ----------------------------------------------------------------------------
# Panel c — compressed k-grade adjacency matrix for large R
# ----------------------------------------------------------------------------

ax_c = fig.add_subplot(outer[0, 1])

cmap_c = mpl.cm.plasma.copy()
cmap_c.set_bad(color="white")

im_c = ax_c.imshow(
    M_grade,
    cmap=cmap_c,
    interpolation="nearest",
    origin="upper",
)

ax_c.set_title(f"Exact compressed grade adjacency, R={R_LARGE}", loc="left")
ax_c.set_xlabel("grade k")
ax_c.set_ylabel("grade k")

# Use sparse ticks
tick_vals = [0, 50, 100, 150, 200, 250, 300, 350]
tick_vals = [t for t in tick_vals if t <= R_LARGE]
ax_c.set_xticks(tick_vals)
ax_c.set_yticks(tick_vals)

cbar_c = fig.colorbar(im_c, ax=ax_c, fraction=0.046, pad=0.03)
cbar_c.set_label(r"$\log_{10}$ Hamming-1 edges")

add_panel_label(ax_c, "c")

text_c = (
    rf"Full graph vertices: $2^{{{R_LARGE}}}\approx 10^{{{log10_vertices_large:.1f}}}$" "\n"
    rf"Compressed grades: {n_grades_large}" "\n"
    rf"Adjacent grade transitions: {n_grade_transitions}" "\n"
    rf"Edges only connect $k$ to $k+1$"
)

ax_c.text(
    0.03, 0.97,
    text_c,
    transform=ax_c.transAxes,
    ha="left", va="top",
    fontsize=8.2,
    color=INK,
    bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor=GREY, alpha=0.95),
)

# ----------------------------------------------------------------------------
# Panel d — edge counts between adjacent grades
# ----------------------------------------------------------------------------

ax_d = fig.add_subplot(outer[1, 1])

ax_d.plot(
    ks_edges,
    log_edges,
    color=TEAL,
    lw=1.8,
)

ax_d.fill_between(
    ks_edges,
    log_edges,
    np.nanmin(log_edges),
    color=TEAL,
    alpha=0.18,
)

ax_d.set_title("Neighbouring-grade edge counts are exactly computable", loc="left")
ax_d.set_xlabel("lower grade k in transition k → k+1")
ax_d.set_ylabel(r"$\log_{10} E_{k,k+1}$")

ax_d.axvline(R_LARGE / 2, color=GREY, lw=1.0, ls="--")
ax_d.text(
    R_LARGE / 2 + 4,
    np.nanmax(log_edges) * 0.92,
    "maximal near\nmiddle grades",
    fontsize=8,
    color=INK,
    ha="left",
    va="top",
)

eq_text = (
    r"$E_{k,k+1}=\binom{R}{k}(R-k)$" "\n"
    r"$=\binom{R}{k+1}(k+1)$" "\n\n"
    rf"Local neighbours per state: $R={R_LARGE}$"
)

ax_d.text(
    0.03, 0.97,
    eq_text,
    transform=ax_d.transAxes,
    ha="left", va="top",
    fontsize=8.5,
    color=INK,
    bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor=GREY, alpha=0.95),
)

add_panel_label(ax_d, "d")

# ----------------------------------------------------------------------------
# Global title and save
# ----------------------------------------------------------------------------

fig.suptitle(
    "Recursive Hamming structure makes the proteoform modal lattice locally navigable",
    y=0.992,
    fontsize=13,
)

fig.savefig(OUT_PNG, dpi=DPI)
fig.savefig(OUT_PDF)

plt.show()

print(f"Saved PNG: {OUT_PNG}")
print(f"Saved PDF: {OUT_PDF}")


# ============================================================================
# Manuscript reporting values
# ============================================================================

print("\n" + "=" * 80)
print("REPORTING VALUES")
print("=" * 80)

print(f"R_PROOF = {R_PROOF}")
print(f"Recursive proof max absolute difference: {max_abs_diff}")
print(f"R_LARGE = {R_LARGE}")
print(f"log10 full vertices 2^R: {log10_vertices_large:.2f}")
print(f"compressed grades: {n_grades_large}")
print(f"compressed adjacent grade transitions: {n_grade_transitions}")
print(f"local Hamming-1 neighbours per state: {R_LARGE}")
print(f"max log10 edges between adjacent grades: {np.nanmax(log_edges):.2f}")
print(f"k at max edge count: {int(ks_edges[np.nanargmax(log_edges)])}")
