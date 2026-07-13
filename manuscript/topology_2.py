# ============================================================================
# PolyForm 03 PATCH — improved tissue sharing + lattice heatmaps
#
# Run this AFTER polyform_03_tissue_support_topology.py
# ============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from collections import defaultdict

PATCH_OUTDIR = os.path.join(OUTDIR, "patch_improved_figures")
os.makedirs(PATCH_OUTDIR, exist_ok=True)

# ----------------------------------------------------------------------------
# 1. Accession-level sharing
# ----------------------------------------------------------------------------
# The original script used lattice_id = Accession + sequence_hash.
# That is correct for topology, but too strict for tissue sharing.
# Here we also compute sharing at Accession + state_key level.

STATE_ACC = (
    STATE_SUPPORT
    .drop_duplicates(["Tissue", "Accession", "Uniprot_Id", "Description", "state_key_string"])
    .copy()
)

acc_sharing = (
    STATE_ACC
    .groupby(["Accession", "state_key_string"])
    .agg(
        n_tissues_acc=("Tissue", "nunique"),
        tissues_acc=("Tissue", lambda x: ";".join(sorted(set(map(str, x))))),
    )
    .reset_index()
)

STATE_ACC = STATE_ACC.merge(
    acc_sharing,
    on=["Accession", "state_key_string"],
    how="left",
)

STATE_ACC["sharing_class_acc"] = np.where(
    STATE_ACC["n_tissues_acc"] == 1,
    "tissue-specific",
    "shared",
)

# Tissue-level corrected sharing summary.
acc_summary_rows = []

for tissue, g in STATE_ACC.groupby("Tissue"):
    n_states = len(g)
    n_unique = int((g["sharing_class_acc"] == "tissue-specific").sum())
    n_shared = int((g["sharing_class_acc"] == "shared").sum())

    acc_summary_rows.append({
        "Tissue": tissue,
        "n_accession_level_states": n_states,
        "n_tissue_specific_acc_states": n_unique,
        "n_shared_acc_states": n_shared,
        "unique_state_fraction_acc": n_unique / n_states if n_states else np.nan,
        "shared_state_fraction_acc": n_shared / n_states if n_states else np.nan,
    })

TISSUE_SHARING_ACC = pd.DataFrame(acc_summary_rows)

TISSUE_SUMMARY_PATCH = TISSUE_SUMMARY.merge(
    TISSUE_SHARING_ACC,
    on="Tissue",
    how="left",
)

display(TISSUE_SUMMARY_PATCH)


# ----------------------------------------------------------------------------
# 2. Extra topology summaries
# ----------------------------------------------------------------------------

anchored_patch = LATTICE_TISSUE_METRICS[
    LATTICE_TISSUE_METRICS["mode"] == "anchored"
].copy()

empirical_patch = LATTICE_TISSUE_METRICS[
    LATTICE_TISSUE_METRICS["mode"] == "empirical"
].copy()

extra_rows = []

for tissue, g in anchored_patch.groupby("Tissue"):
    n = len(g)

    frac_fully_connected = (
        (g["fraction_observed_connected_to_k0"] == 1.0).sum() / n
        if n else np.nan
    )

    frac_any_k2plus = (
        (g["n_k2plus_observed"] > 0).sum() / n
        if n else np.nan
    )

    mean_k_weighted = np.average(
        g["mean_distance_to_k0"],
        weights=g["n_observed_states"]
    ) if g["n_observed_states"].sum() > 0 else np.nan

    extra_rows.append({
        "Tissue": tissue,
        "fraction_lattices_fully_k0_connected": frac_fully_connected,
        "fraction_lattices_with_k2plus": frac_any_k2plus,
        "weighted_mean_k": mean_k_weighted,
    })

EXTRA_TOPOLOGY = pd.DataFrame(extra_rows)

TISSUE_SUMMARY_PATCH = TISSUE_SUMMARY_PATCH.merge(
    EXTRA_TOPOLOGY,
    on="Tissue",
    how="left",
)

display(TISSUE_SUMMARY_PATCH)


# ----------------------------------------------------------------------------
# 3. Improved tissue summary figure
# ----------------------------------------------------------------------------

ts = TISSUE_SUMMARY_PATCH.copy()
ts = ts.sort_values("n_observed_states", ascending=False).reset_index(drop=True)

tissues = ts["Tissue"].tolist()
x = np.arange(len(tissues))

fig = plt.figure(figsize=(13.2, 8.6))
gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.34)

# a — support size
ax = fig.add_subplot(gs[0, 0])
ax.bar(x - 0.18, ts["n_records"], width=0.36, color=GREY, label="records")
ax.bar(x + 0.18, ts["n_observed_states"], width=0.36, color=TEAL, label="accessed states")
ax.set_xticks(x)
ax.set_xticklabels(tissues, rotation=30, ha="right")
ax.set_ylabel("count")
ax.set_title("a  tissue atlas support size", loc="left")
ax.legend(fontsize=7.5)

# b — grade composition
ax = fig.add_subplot(gs[0, 1])
bottom = np.zeros(len(ts))

for col, label, color in [
    ("frac_k0_observed", "k=0", INK),
    ("frac_k1_observed", "k=1", AMBER),
    ("frac_k2plus_observed", "k≥2", VIOLET),
]:
    vals = ts[col].fillna(0).values
    ax.bar(x, vals, bottom=bottom, color=color, label=label)
    bottom += vals

ax.set_xticks(x)
ax.set_xticklabels(tissues, rotation=30, ha="right")
ax.set_ylabel("fraction of observed states")
ax.set_ylim(0, 1.02)
ax.set_title("b  low-grade support dominates", loc="left")
ax.legend(fontsize=7.5)

# c — boundary connectivity
ax = fig.add_subplot(gs[0, 2])
ax.bar(
    x,
    ts["weighted_fraction_observed_connected_to_k0_anchor"],
    color=ROSE,
)
ax.set_xticks(x)
ax.set_xticklabels(tissues, rotation=30, ha="right")
ax.set_ylim(0, 1.02)
ax.set_ylabel("fraction connected to k=0 anchor")
ax.set_title("c  boundary-attached support", loc="left")

# d — fraction of protein lattices fully connected to k0
ax = fig.add_subplot(gs[1, 0])
ax.bar(
    x,
    ts["fraction_lattices_fully_k0_connected"],
    color=TEAL,
)
ax.set_xticks(x)
ax.set_xticklabels(tissues, rotation=30, ha="right")
ax.set_ylim(0, 1.02)
ax.set_ylabel("fraction of protein lattices")
ax.set_title("d  fully k=0-connected lattices", loc="left")

# e — corrected accession-level tissue sharing
ax = fig.add_subplot(gs[1, 1])
ax.bar(
    x,
    ts["shared_state_fraction_acc"],
    color=BLUE,
    label="shared",
)
ax.bar(
    x,
    ts["unique_state_fraction_acc"],
    bottom=ts["shared_state_fraction_acc"],
    color=GREY,
    label="tissue-specific",
)
ax.set_xticks(x)
ax.set_xticklabels(tissues, rotation=30, ha="right")
ax.set_ylim(0, 1.02)
ax.set_ylabel("fraction of accession-level states")
ax.set_title("e  corrected tissue sharing", loc="left")
ax.legend(fontsize=7.5)

# f — weighted mean distance to k0
ax = fig.add_subplot(gs[1, 2])
ax.bar(
    x,
    ts["weighted_mean_k"],
    color=AMBER,
)
ax.set_xticks(x)
ax.set_xticklabels(tissues, rotation=30, ha="right")
ax.set_ylabel("weighted mean distance to k=0")
ax.set_title("f  boundary distance", loc="left")

fig.suptitle(
    "Tissue-resolved support topology with accession-level sharing",
    y=0.99,
    fontsize=12,
)

fig.savefig(os.path.join(PATCH_OUTDIR, "polyform_03_patch_improved_tissue_summary.png"), dpi=DPI)
fig.savefig(os.path.join(PATCH_OUTDIR, "polyform_03_patch_improved_tissue_summary.pdf"))
plt.show()

print("Saved improved summary figure:")
print(os.path.join(PATCH_OUTDIR, "polyform_03_patch_improved_tissue_summary.png"))
print(os.path.join(PATCH_OUTDIR, "polyform_03_patch_improved_tissue_summary.pdf"))


# ----------------------------------------------------------------------------
# 4. Select variable and similar accession-level examples
# ----------------------------------------------------------------------------

def jaccard(a, b):
    a = set(a)
    b = set(b)
    if len(a | b) == 0:
        return np.nan
    return len(a & b) / len(a | b)


accession_rows = []

for acc, g in STATE_ACC.groupby("Accession"):
    tissue_sets = {
        tissue: set(sub["state_key_string"])
        for tissue, sub in g.groupby("Tissue")
    }

    if len(tissue_sets) < 2:
        continue

    pairwise = []
    tissues_here = sorted(tissue_sets)

    for i in range(len(tissues_here)):
        for j in range(i + 1, len(tissues_here)):
            pairwise.append(jaccard(tissue_sets[tissues_here[i]], tissue_sets[tissues_here[j]]))

    pairwise = [v for v in pairwise if np.isfinite(v)]

    if not pairwise:
        continue

    all_states = set(g["state_key_string"])
    grades = sorted(set(g["k"]))

    desc = str(g["Description"].iloc[0])
    uid = str(g["Uniprot_Id"].iloc[0])

    accession_rows.append({
        "Accession": acc,
        "Uniprot_Id": uid,
        "Description": desc,
        "n_tissues": len(tissue_sets),
        "n_states_total": len(all_states),
        "grades": tuple(grades),
        "delta_k": max(grades) - min(grades) if grades else 0,
        "mean_pairwise_jaccard": float(np.mean(pairwise)),
        "min_pairwise_jaccard": float(np.min(pairwise)),
        "max_pairwise_jaccard": float(np.max(pairwise)),
        "state_counts_by_tissue": ";".join(
            f"{t}:{len(tissue_sets[t])}" for t in tissues_here
        ),
    })

ACCESSION_VARIATION = pd.DataFrame(accession_rows)

# Avoid totally trivial single-state examples.
candidate = ACCESSION_VARIATION[
    (ACCESSION_VARIATION["n_tissues"] >= 2) &
    (ACCESSION_VARIATION["n_states_total"] >= 3)
].copy()

if len(candidate) == 0:
    candidate = ACCESSION_VARIATION.copy()

VARIABLE_EXAMPLES = (
    candidate
    .sort_values(
        ["mean_pairwise_jaccard", "n_states_total", "delta_k"],
        ascending=[True, False, False],
    )
    .head(3)
)

SIMILAR_EXAMPLES = (
    candidate
    .sort_values(
        ["mean_pairwise_jaccard", "n_states_total", "delta_k"],
        ascending=[False, False, False],
    )
    .head(3)
)

SELECTED_HEATMAP_EXAMPLES = pd.concat([
    VARIABLE_EXAMPLES.assign(example_class="variable"),
    SIMILAR_EXAMPLES.assign(example_class="similar"),
], ignore_index=True).drop_duplicates("Accession")

print("\nSelected heatmap examples:")
display(SELECTED_HEATMAP_EXAMPLES)

ACCESSION_VARIATION.to_csv(
    os.path.join(PATCH_OUTDIR, "polyform_03_patch_accession_variation.csv"),
    index=False,
)

SELECTED_HEATMAP_EXAMPLES.to_csv(
    os.path.join(PATCH_OUTDIR, "polyform_03_patch_selected_heatmap_examples.csv"),
    index=False,
)


# ----------------------------------------------------------------------------
# 5. Lattice heatmap helper
# ----------------------------------------------------------------------------

def parse_state_string(s):
    if pd.isna(s):
        return tuple()
    s = str(s)
    if s in {"0", "0^L", ""}:
        return tuple()
    out = []
    for z in s.split(","):
        z = z.strip()
        if z == "":
            continue
        try:
            out.append(int(z))
        except Exception:
            out.append(z)
    return tuple(out)


def anchored_components_for_accession(acc):
    """
    Build anchored components over the union of states for an accession.
    """
    sub = STATE_ACC[STATE_ACC["Accession"] == acc].copy()

    states = set(parse_state_string(s) for s in sub["state_key_string"])
    states.add(tuple())  # k=0 anchor

    states = sorted(states, key=lambda s: (len(s), str(s)))
    comps, comp_map = connected_components_for_states(states)

    return states, comps, comp_map


def make_accession_heatmap_matrix(acc):
    sub = STATE_ACC[STATE_ACC["Accession"] == acc].copy()

    tissues_here = sorted(sub["Tissue"].unique())

    states, comps, comp_map = anchored_components_for_accession(acc)

    # State order: component, grade, state string.
    states = sorted(
        states,
        key=lambda s: (
            comp_map[tuple(s)],
            len(s),
            str(s)
        )
    )

    # Matrix value:
    # 0 = absent
    # 1 = observed
    # 0.35 = k=0 anchor only
    mat = np.zeros((len(states), len(tissues_here)))

    observed_pairs = set(
        (row["Tissue"], tuple(parse_state_string(row["state_key_string"])))
        for _, row in sub.iterrows()
    )

    for i, state in enumerate(states):
        for j, tissue in enumerate(tissues_here):
            if (tissue, tuple(state)) in observed_pairs:
                mat[i, j] = 1.0
            elif len(state) == 0:
                mat[i, j] = 0.35

    row_labels = []
    for state in states:
        cid = comp_map[tuple(state)]
        k = len(state)
        if k == 0:
            label = f"C{cid} | k=0 | anchor"
        else:
            label = f"C{cid} | k={k} | {state_to_string(state)}"
        row_labels.append(label)

    return mat, tissues_here, states, row_labels, comp_map


# ----------------------------------------------------------------------------
# 6. Lattice heatmaps
# ----------------------------------------------------------------------------

n_examples = len(SELECTED_HEATMAP_EXAMPLES)
if n_examples == 0:
    print("No heatmap examples selected.")
else:
    fig_h = max(2.5 * n_examples, 7.0)
    fig, axes = plt.subplots(
        n_examples,
        1,
        figsize=(10.8, fig_h),
        squeeze=False,
    )

    cmap = ListedColormap(["white", "#d8c7ef", "#1b2a41"])
    norm = BoundaryNorm([-0.1, 0.1, 0.7, 1.1], cmap.N)

    for idx, (_, ex) in enumerate(SELECTED_HEATMAP_EXAMPLES.iterrows()):
        acc = ex["Accession"]
        cls = ex["example_class"]

        mat, tissues_here, states, row_labels, comp_map = make_accession_heatmap_matrix(acc)

        ax = axes[idx, 0]
        im = ax.imshow(mat, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)

        ax.set_xticks(np.arange(len(tissues_here)))
        ax.set_xticklabels(tissues_here, rotation=30, ha="right")

        max_rows_label = 35
        if len(row_labels) <= max_rows_label:
            yticks = np.arange(len(row_labels))
            ylabels = row_labels
        else:
            yticks = np.linspace(0, len(row_labels) - 1, max_rows_label).astype(int)
            ylabels = [row_labels[i] for i in yticks]

        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels, fontsize=6.5)

        title = (
            f"{idx+1}. {cls}: {acc} / {ex['Uniprot_Id']} "
            f"| states={ex['n_states_total']} | tissues={ex['n_tissues']} "
            f"| mean Jaccard={ex['mean_pairwise_jaccard']:.2f}"
        )
        ax.set_title(title, loc="left", fontsize=9)

        ax.set_ylabel("native component | grade | state")

        # grid lines
        ax.set_xticks(np.arange(-0.5, len(tissues_here), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
        ax.grid(which="minor", color="#eeeeee", linestyle="-", linewidth=0.5)
        ax.tick_params(which="minor", bottom=False, left=False)

    axes[-1, 0].set_xlabel("tissue")

    # Custom legend
    from matplotlib.patches import Patch
    legend_items = [
        Patch(facecolor="#1b2a41", edgecolor="none", label="observed state"),
        Patch(facecolor="#d8c7ef", edgecolor="none", label="k=0 anchor only"),
        Patch(facecolor="white", edgecolor="#999999", label="not observed"),
    ]
    axes[0, 0].legend(
        handles=legend_items,
        loc="upper right",
        fontsize=7.5,
        frameon=False,
    )

    fig.suptitle(
        "Tissue-resolved proteoform support on native Hamming-1 lattice components",
        y=0.995,
        fontsize=12,
    )

    fig.savefig(os.path.join(PATCH_OUTDIR, "polyform_03_patch_lattice_heatmaps.png"), dpi=DPI)
    fig.savefig(os.path.join(PATCH_OUTDIR, "polyform_03_patch_lattice_heatmaps.pdf"))
    plt.show()

    print("Saved heatmap figures:")
    print(os.path.join(PATCH_OUTDIR, "polyform_03_patch_lattice_heatmaps.png"))
    print(os.path.join(PATCH_OUTDIR, "polyform_03_patch_lattice_heatmaps.pdf"))
