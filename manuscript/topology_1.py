# ============================================================================
# PolyForm — tissue-resolved support topology in a top-down human atlas
#
# Purpose:
#   Instantiate Hamming-1 support topology on a multi-tissue top-down atlas.
#
#   The script computes:
#     1. Empirical support topology using only observed proteoform states.
#     2. Anchored support topology after adding the unmodified k=0 state as a
#        reference boundary vertex for each observed protein-sequence/tissue pair.
#
#   The k=0 anchor is NOT treated as observed unless it is present in the data.
#   It is included to measure boundary connectivity to the unmodified state.
#
# Outputs:
#   TISSUE_SUMMARY
#   LATTICE_TISSUE_METRICS
#   COMPONENTS
#   STATE_SUPPORT
#   STATE_SHARING
#
# Figures:
#   polyform_03_tissue_support_topology.png/pdf
# ============================================================================
import os
import re
import glob
import hashlib
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from collections import defaultdict
from IPython.display import display

# ----------------------------------------------------------------------------
# User settings
# ----------------------------------------------------------------------------

XLSX = "pr2c00034_si_002 (1).xlsx"

REPO_DIR = "/content/PolyForm"
OUTDIR = os.path.join(REPO_DIR, "manuscript", "polyform_03_tissue_support_topology_outputs")

if not os.path.exists(REPO_DIR):
    OUTDIR = "polyform_03_tissue_support_topology_outputs"

os.makedirs(OUTDIR, exist_ok=True)

SHEET_NAME = "All_Tissues"

GROUP_BY_SEQUENCE = True
ADD_K0_ANCHOR = True

DPI = 300
RANDOM_SEED = 7
np.random.seed(RANDOM_SEED)

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

INK = "#1b2a41"
TEAL = "#0f9b8e"
AMBER = "#e08a2e"
VIOLET = "#7d5ba6"
GREY = "#b8bec8"
ROSE = "#c44e52"
BLUE = "#4c78a8"
GREEN = "#59a14f"


# ============================================================================
# 1. Input handling
# ============================================================================

def find_xlsx(path):
    if os.path.exists(path):
        return path

    candidates = []
    candidates += glob.glob("*.xlsx")
    candidates += glob.glob("/content/*.xlsx")

    atlas_candidates = [
        c for c in candidates
        if "pr2c00034" in os.path.basename(c).lower()
    ]

    if atlas_candidates:
        return sorted(atlas_candidates)[0]

    if candidates:
        return sorted(candidates)[0]

    raise FileNotFoundError(
        "No XLSX file found. Upload the tissue atlas workbook or set XLSX."
    )


XLSX = find_xlsx(XLSX)

print(f"Reading workbook: {XLSX}")

xls = pd.ExcelFile(XLSX)
print("Sheets:", xls.sheet_names)

if SHEET_NAME in xls.sheet_names:
    raw = pd.read_excel(XLSX, sheet_name=SHEET_NAME)
else:
    # Fallback: concatenate tissue sheets if All_Tissues is unavailable.
    tissue_sheets = [s for s in xls.sheet_names if s.lower() not in {"shared"}]
    frames = []
    for s in tissue_sheets:
        tmp = pd.read_excel(XLSX, sheet_name=s)
        if "Tissue" not in tmp.columns:
            tmp["Tissue"] = s
        frames.append(tmp)
    raw = pd.concat(frames, ignore_index=True)

print(f"Rows read: {len(raw):,}")
display(raw.head())


# ============================================================================
# 2. Column checks
# ============================================================================

required_cols = [
    "Tissue",
    "Accession",
    "Uniprot_Id",
    "Description",
    "Sequence",
    "Modifications",
    "Modification_Codes",
    "N_Terminal_Modification_Code",
    "C_Terminal_Modification_Code",
]

missing = [c for c in required_cols if c not in raw.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df = raw.copy()

# Clean core columns.
for c in ["Tissue", "Accession", "Uniprot_Id", "Description", "Sequence"]:
    df[c] = df[c].astype(str).str.strip()

df = df[df["Sequence"].notna()]
df = df[df["Sequence"].astype(str).str.len() > 0].copy()
df["L"] = df["Sequence"].astype(str).str.len()


# ============================================================================
# 3. Modification parsing
# ============================================================================

def is_blank(x):
    if pd.isna(x):
        return True
    s = str(x).strip()
    return s == "" or s.upper() in {"NA", "N/A", "NAN", "NONE", "NULL"}


def coord_sort_key(x):
    """
    Stable sorting for mixed coordinates:
      NTERM first, residue positions, CTERM last.
    """
    if x == "NTERM":
        return (-1, -1)
    if x == "CTERM":
        return (10**12, 10**12)
    return (int(x), int(x))


def parse_modifications(row):
    """
    Parses top-down atlas modification text.

    Examples:
      alpha-amino acetylated residue@N
      phosphorylation@17
      some modification@C

    Coordinates:
      @N -> NTERM
      @C -> CTERM
      @number -> integer residue coordinate as reported

    Returns:
      valid_mods = tuple((coord, label), ...)
      state_key = tuple(unique coordinates)
      fiber_key = tuple((coord, label), ...)
    """
    mods = []

    mod_text = row.get("Modifications", None)

    if not is_blank(mod_text):
        text = str(mod_text)

        # Match chunks ending in @N, @C, or @integer.
        # Works for simple single modifications and many pipe/semicolon separated strings.
        matches = re.finditer(r"([^|;]+?)@([NC]|\d+)", text)

        for m in matches:
            label = m.group(1).strip()
            pos_raw = m.group(2).strip()

            if pos_raw == "N":
                coord = "NTERM"
            elif pos_raw == "C":
                coord = "CTERM"
            else:
                coord = int(pos_raw)

            mods.append((coord, label))

    # Terminal modification codes are retained if the text field did not already
    # supply that terminal coordinate.
    existing_coords = {coord for coord, label in mods}

    n_code = row.get("N_Terminal_Modification_Code", None)
    if not is_blank(n_code) and "NTERM" not in existing_coords:
        mods.append(("NTERM", str(n_code).strip()))

    c_code = row.get("C_Terminal_Modification_Code", None)
    if not is_blank(c_code) and "CTERM" not in existing_coords:
        mods.append(("CTERM", str(c_code).strip()))

    # Remove exact duplicate fibre labels.
    mods = sorted(set(mods), key=lambda z: (coord_sort_key(z[0]), str(z[1])))

    state_key = tuple(sorted({coord for coord, label in mods}, key=coord_sort_key))
    fiber_key = tuple(mods)

    return pd.Series({
        "state_key": state_key,
        "fiber_key": fiber_key,
        "k_state": len(state_key),
        "k_fiber": len(fiber_key),
        "is_unmodified_observed": len(state_key) == 0,
    })


df = pd.concat([df, df.apply(parse_modifications, axis=1)], axis=1)


def short_hash(s, n=10):
    return hashlib.md5(str(s).encode("utf-8")).hexdigest()[:n]


df["sequence_hash"] = df["Sequence"].apply(short_hash)

if GROUP_BY_SEQUENCE:
    df["lattice_id"] = (
        df["Accession"].astype(str)
        + "|"
        + df["sequence_hash"].astype(str)
    )
else:
    df["lattice_id"] = df["Accession"].astype(str)

print("\nParsed records:")
print(f"  rows after sequence filter: {len(df):,}")
print(f"  tissues: {df['Tissue'].nunique():,}")
print(f"  accessions: {df['Accession'].nunique():,}")
print(f"  lattice units: {df['lattice_id'].nunique():,}")
print(f"  observed unmodified rows: {int(df['is_unmodified_observed'].sum()):,}")
print(f"  modified rows: {int((~df['is_unmodified_observed']).sum()):,}")

display(df[[
    "Tissue", "Accession", "Uniprot_Id", "Description",
    "L", "Modifications", "state_key", "k_state", "fiber_key"
]].head(12))


# ============================================================================
# 4. Hamming topology helpers
# ============================================================================

def state_to_string(state):
    if len(state) == 0:
        return "0"
    return ",".join(map(str, state))


def tuple_to_string(x):
    if isinstance(x, tuple):
        return ";".join(map(str, x))
    return str(x)


def hamming_distance_state(a, b):
    """
    Hamming distance between two binary states encoded as tuples of coordinates.
    """
    return len(set(a).symmetric_difference(set(b)))


def connected_components_for_states(states):
    """
    Compute connected components under Hamming-1 adjacency.

    States are tuples of modified coordinates.
    Edge exists when symmetric difference size is 1.
    """
    states = list(states)

    if len(states) == 0:
        return [], {}

    n = len(states)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    state_sets = [set(s) for s in states]

    for i in range(n):
        for j in range(i + 1, n):
            if len(state_sets[i].symmetric_difference(state_sets[j])) == 1:
                union(i, j)

    groups = defaultdict(list)
    for i, s in enumerate(states):
        groups[find(i)].append(s)

    comps = list(groups.values())
    comps = sorted(comps, key=lambda c: (-len(c), min(len(x) for x in c), str(c[0])))

    comp_map = {}
    for cid, comp in enumerate(comps, start=1):
        for s in comp:
            comp_map[tuple(s)] = cid

    return comps, comp_map


def component_distance(comp_a, comp_b):
    best = np.inf
    for a in comp_a:
        for b in comp_b:
            d = hamming_distance_state(a, b)
            if d < best:
                best = d
    return best


def min_inter_component_distance(comps):
    if len(comps) <= 1:
        return np.nan

    dmin = np.inf
    for i in range(len(comps)):
        for j in range(i + 1, len(comps)):
            dmin = min(dmin, component_distance(comps[i], comps[j]))

    return dmin if np.isfinite(dmin) else np.nan


def grade_gap_count(K):
    if len(K) <= 1:
        return 0
    K = sorted(K)
    full = set(range(min(K), max(K) + 1))
    return len(full - set(K))


# ============================================================================
# 5. Build observed state support
# ============================================================================

support_rows = []

group_cols = [
    "Tissue",
    "lattice_id",
    "Accession",
    "Uniprot_Id",
    "Description",
    "Sequence",
    "sequence_hash",
    "L",
]

for keys, grp in df.groupby(group_cols, dropna=False):
    info = dict(zip(group_cols, keys))

    state_counts = grp.groupby("state_key").size().reset_index(name="n_records_for_state")

    for _, sr in state_counts.iterrows():
        state = tuple(sr["state_key"])
        support_rows.append({
            **info,
            "state_key": state,
            "state_key_string": state_to_string(state),
            "k": len(state),
            "observed": True,
            "anchor": False,
            "n_records_for_state": int(sr["n_records_for_state"]),
        })

STATE_SUPPORT = pd.DataFrame(support_rows)

print("\nObserved support states:")
print(f"  rows: {len(STATE_SUPPORT):,}")
print(f"  tissue-lattice pairs: {STATE_SUPPORT[['Tissue','lattice_id']].drop_duplicates().shape[0]:,}")
display(STATE_SUPPORT.head())


# ============================================================================
# 6. Per tissue-lattice topology metrics
# ============================================================================

metric_rows = []
component_rows = []

for keys, grp in STATE_SUPPORT.groupby([
    "Tissue",
    "lattice_id",
    "Accession",
    "Uniprot_Id",
    "Description",
    "sequence_hash",
    "L",
], dropna=False):

    Tissue, lattice_id, Accession, Uniprot_Id, Description, sequence_hash, L = keys
    L = int(L)

    observed_states = sorted(
        {tuple(s) for s in grp["state_key"]},
        key=lambda s: (len(s), str(s))
    )

    for mode in ["empirical", "anchored"]:
        if mode == "empirical":
            graph_states = list(observed_states)
            anchor_added = False
        else:
            graph_states = list(observed_states)
            anchor_added = tuple() not in graph_states
            if anchor_added:
                graph_states.append(tuple())

        graph_states = sorted(set(graph_states), key=lambda s: (len(s), str(s)))
        comps, comp_map = connected_components_for_states(graph_states)

        n_observed_states = len(observed_states)
        n_graph_states = len(graph_states)

        observed_K = sorted({len(s) for s in observed_states})
        k_min = min(observed_K) if observed_K else np.nan
        k_max = max(observed_K) if observed_K else np.nan
        delta_k = k_max - k_min if observed_K else np.nan

        n_k0_obs = sum(1 for s in observed_states if len(s) == 0)
        n_k1_obs = sum(1 for s in observed_states if len(s) == 1)
        n_k2plus_obs = sum(1 for s in observed_states if len(s) >= 2)

        frac_k0_obs = n_k0_obs / n_observed_states if n_observed_states else np.nan
        frac_k1_obs = n_k1_obs / n_observed_states if n_observed_states else np.nan
        frac_k2plus_obs = n_k2plus_obs / n_observed_states if n_observed_states else np.nan

        # Component containing the unmodified state if present in graph.
        if tuple() in comp_map:
            k0_component_id = comp_map[tuple()]
            observed_in_k0_component = sum(
                1 for s in observed_states
                if comp_map.get(tuple(s), None) == k0_component_id
            )
            graph_in_k0_component = sum(
                1 for s in graph_states
                if comp_map.get(tuple(s), None) == k0_component_id
            )
        else:
            k0_component_id = np.nan
            observed_in_k0_component = 0
            graph_in_k0_component = 0

        frac_observed_connected_to_k0 = (
            observed_in_k0_component / n_observed_states
            if n_observed_states else np.nan
        )

        largest_component_size_graph = max([len(c) for c in comps]) if comps else 0
        largest_component_fraction_graph = (
            largest_component_size_graph / n_graph_states
            if n_graph_states else np.nan
        )

        observed_counts_by_component = defaultdict(int)
        for s in observed_states:
            observed_counts_by_component[comp_map[tuple(s)]] += 1

        largest_component_observed_count = (
            max(observed_counts_by_component.values())
            if observed_counts_by_component else 0
        )
        largest_component_fraction_observed = (
            largest_component_observed_count / n_observed_states
            if n_observed_states else np.nan
        )

        dmin = min_inter_component_distance(comps)

        mean_distance_to_k0 = (
            float(np.mean([len(s) for s in observed_states]))
            if observed_states else np.nan
        )
        median_distance_to_k0 = (
            float(np.median([len(s) for s in observed_states]))
            if observed_states else np.nan
        )

        metric_rows.append({
            "Tissue": Tissue,
            "lattice_id": lattice_id,
            "Accession": Accession,
            "Uniprot_Id": Uniprot_Id,
            "Description": Description,
            "sequence_hash": sequence_hash,
            "L": L,
            "mode": mode,
            "n_observed_states": n_observed_states,
            "n_graph_states": n_graph_states,
            "anchor_added": bool(anchor_added),
            "n_components": len(comps),
            "largest_component_size_graph": largest_component_size_graph,
            "largest_component_fraction_graph": largest_component_fraction_graph,
            "largest_component_observed_count": largest_component_observed_count,
            "largest_component_fraction_observed": largest_component_fraction_observed,
            "k0_component_id": k0_component_id,
            "observed_states_connected_to_k0": observed_in_k0_component,
            "fraction_observed_connected_to_k0": frac_observed_connected_to_k0,
            "graph_states_connected_to_k0": graph_in_k0_component,
            "K_observed": tuple(observed_K),
            "k_min_observed": k_min,
            "k_max_observed": k_max,
            "delta_k_observed": delta_k,
            "grade_gap_count": grade_gap_count(observed_K),
            "n_k0_observed": n_k0_obs,
            "n_k1_observed": n_k1_obs,
            "n_k2plus_observed": n_k2plus_obs,
            "frac_k0_observed": frac_k0_obs,
            "frac_k1_observed": frac_k1_obs,
            "frac_k2plus_observed": frac_k2plus_obs,
            "mean_distance_to_k0": mean_distance_to_k0,
            "median_distance_to_k0": median_distance_to_k0,
            "min_inter_component_distance": dmin,
        })

        for cid, comp in enumerate(comps, start=1):
            comp_states = [tuple(s) for s in comp]
            comp_observed = [s for s in comp_states if s in observed_states]
            comp_anchor_present = tuple() in comp_states and tuple() not in observed_states

            grades_all = [len(s) for s in comp_states]
            grades_obs = [len(s) for s in comp_observed]

            component_rows.append({
                "Tissue": Tissue,
                "lattice_id": lattice_id,
                "Accession": Accession,
                "Uniprot_Id": Uniprot_Id,
                "Description": Description,
                "sequence_hash": sequence_hash,
                "L": L,
                "mode": mode,
                "component_id": cid,
                "component_size_graph": len(comp_states),
                "component_observed_state_count": len(comp_observed),
                "component_anchor_present": bool(comp_anchor_present),
                "component_contains_observed_k0": any(len(s) == 0 for s in comp_observed),
                "component_boundary_attached_k0": any(len(s) == 0 for s in comp_states),
                "k_min_graph": min(grades_all) if grades_all else np.nan,
                "k_max_graph": max(grades_all) if grades_all else np.nan,
                "k_min_observed_component": min(grades_obs) if grades_obs else np.nan,
                "k_max_observed_component": max(grades_obs) if grades_obs else np.nan,
            })

LATTICE_TISSUE_METRICS = pd.DataFrame(metric_rows)
COMPONENTS = pd.DataFrame(component_rows)

print("\nPer tissue-lattice metrics:")
display(LATTICE_TISSUE_METRICS.head(20))

print("\nComponents:")
display(COMPONENTS.head(20))


# ============================================================================
# 7. Tissue-specific versus shared states
# ============================================================================

# State identity is defined within a fixed lattice unit.
STATE_SUPPORT["lattice_state_id"] = (
    STATE_SUPPORT["lattice_id"].astype(str)
    + "|"
    + STATE_SUPPORT["state_key_string"].astype(str)
)

sharing = (
    STATE_SUPPORT
    .groupby(["lattice_id", "state_key_string"])
    .agg(
        n_tissues=("Tissue", "nunique"),
        tissues=("Tissue", lambda x: ";".join(sorted(set(map(str, x))))),
        n_records_total=("n_records_for_state", "sum"),
    )
    .reset_index()
)

STATE_SHARING = STATE_SUPPORT.merge(
    sharing,
    on=["lattice_id", "state_key_string"],
    how="left",
)

STATE_SHARING["sharing_class"] = np.where(
    STATE_SHARING["n_tissues"] == 1,
    "tissue-specific",
    "shared",
)

print("\nState sharing:")
display(STATE_SHARING.head(20))


# ============================================================================
# 8. Tissue summary
# ============================================================================

anchored = LATTICE_TISSUE_METRICS[LATTICE_TISSUE_METRICS["mode"] == "anchored"].copy()
empirical = LATTICE_TISSUE_METRICS[LATTICE_TISSUE_METRICS["mode"] == "empirical"].copy()

def weighted_mean(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not mask.any():
        return np.nan

    return float(np.sum(values[mask] * weights[mask]) / np.sum(weights[mask]))


summary_rows = []

for tissue, g in anchored.groupby("Tissue"):
    g_emp = empirical[empirical["Tissue"] == tissue]
    ss = STATE_SHARING[STATE_SHARING["Tissue"] == tissue]

    n_records = int(df[df["Tissue"] == tissue].shape[0])
    n_accessions = int(df[df["Tissue"] == tissue]["Accession"].nunique())
    n_lattice_units = int(g["lattice_id"].nunique())
    n_observed_states = int(g["n_observed_states"].sum())

    frac_connected_to_k0_weighted = weighted_mean(
        g["fraction_observed_connected_to_k0"],
        g["n_observed_states"],
    )

    frac_k0 = g["n_k0_observed"].sum() / n_observed_states if n_observed_states else np.nan
    frac_k1 = g["n_k1_observed"].sum() / n_observed_states if n_observed_states else np.nan
    frac_k2plus = g["n_k2plus_observed"].sum() / n_observed_states if n_observed_states else np.nan

    n_unique_states = int((ss["sharing_class"] == "tissue-specific").sum())
    n_shared_states = int((ss["sharing_class"] == "shared").sum())
    unique_state_fraction = n_unique_states / len(ss) if len(ss) else np.nan

    summary_rows.append({
        "Tissue": tissue,
        "n_records": n_records,
        "n_accessions": n_accessions,
        "n_lattice_units": n_lattice_units,
        "n_observed_states": n_observed_states,
        "n_unique_states": n_unique_states,
        "n_shared_states": n_shared_states,
        "unique_state_fraction": unique_state_fraction,
        "frac_k0_observed": frac_k0,
        "frac_k1_observed": frac_k1,
        "frac_k2plus_observed": frac_k2plus,
        "weighted_fraction_observed_connected_to_k0_anchor": frac_connected_to_k0_weighted,
        "median_components_empirical": float(g_emp["n_components"].median()) if len(g_emp) else np.nan,
        "median_components_anchored": float(g["n_components"].median()),
        "mean_components_anchored": float(g["n_components"].mean()),
        "median_delta_k_observed": float(g["delta_k_observed"].median()),
        "median_distance_to_k0": float(g["median_distance_to_k0"].median()),
        "mean_distance_to_k0_weighted": weighted_mean(g["mean_distance_to_k0"], g["n_observed_states"]),
        "n_anchor_added": int(g["anchor_added"].sum()),
    })

TISSUE_SUMMARY = pd.DataFrame(summary_rows)
TISSUE_SUMMARY = TISSUE_SUMMARY.sort_values("n_observed_states", ascending=False).reset_index(drop=True)

print("\nTissue summary:")
display(TISSUE_SUMMARY)


# ============================================================================
# 9. Export tables
# ============================================================================

def export_tuple_cols(tab):
    out = tab.copy()
    for c in out.columns:
        if out[c].apply(lambda x: isinstance(x, tuple)).any():
            out[c] = out[c].apply(tuple_to_string)
    return out


export_tuple_cols(STATE_SUPPORT).to_csv(
    os.path.join(OUTDIR, "polyform_03_state_support.csv"),
    index=False,
)

export_tuple_cols(STATE_SHARING).to_csv(
    os.path.join(OUTDIR, "polyform_03_state_sharing.csv"),
    index=False,
)

export_tuple_cols(LATTICE_TISSUE_METRICS).to_csv(
    os.path.join(OUTDIR, "polyform_03_lattice_tissue_metrics.csv"),
    index=False,
)

export_tuple_cols(COMPONENTS).to_csv(
    os.path.join(OUTDIR, "polyform_03_components.csv"),
    index=False,
)

TISSUE_SUMMARY.to_csv(
    os.path.join(OUTDIR, "polyform_03_tissue_summary.csv"),
    index=False,
)

print("\nSaved tables to:")
print(f"  {OUTDIR}")


# ============================================================================
# 10. Figure: tissue-resolved support topology
# ============================================================================

ts = TISSUE_SUMMARY.copy()
tissues = ts["Tissue"].tolist()
x = np.arange(len(tissues))

fig = plt.figure(figsize=(13.2, 8.6))
gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.34)

# a — records and accessed states
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
ax.set_title("b  observed grade composition", loc="left")
ax.legend(fontsize=7.5)

# c — k0 anchor connectivity
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
ax.set_title("c  boundary connectivity", loc="left")

# d — component counts
ax = fig.add_subplot(gs[1, 0])
ax.plot(
    x,
    ts["median_components_empirical"],
    marker="o",
    lw=1.4,
    color=GREY,
    label="empirical",
)
ax.plot(
    x,
    ts["median_components_anchored"],
    marker="o",
    lw=1.4,
    color=TEAL,
    label="anchored",
)
ax.set_xticks(x)
ax.set_xticklabels(tissues, rotation=30, ha="right")
ax.set_ylabel("median component count")
ax.set_title("d  native support fragmentation", loc="left")
ax.legend(fontsize=7.5)

# e — tissue-specific states
ax = fig.add_subplot(gs[1, 1])
ax.bar(x, ts["unique_state_fraction"], color=BLUE)
ax.set_xticks(x)
ax.set_xticklabels(tissues, rotation=30, ha="right")
ax.set_ylim(0, 1.02)
ax.set_ylabel("fraction tissue-specific")
ax.set_title("e  tissue-specific accessed states", loc="left")

# f — per lattice support and boundary connectivity
ax = fig.add_subplot(gs[1, 2])
anch = anchored.copy()
for tissue in tissues:
    sub = anch[anch["Tissue"] == tissue]
    ax.scatter(
        sub["n_observed_states"],
        sub["fraction_observed_connected_to_k0"],
        s=18,
        alpha=0.55,
        label=tissue,
        edgecolor="none",
    )

ax.set_xscale("log")
ax.set_ylim(-0.03, 1.03)
ax.set_xlabel("observed states per protein-sequence")
ax.set_ylabel("fraction connected to k=0")
ax.set_title("f  lattice-level boundary connectivity", loc="left")
ax.legend(fontsize=6.5, ncol=2)

fig.suptitle(
    "Tissue-resolved support topology of a top-down human proteoform atlas",
    y=0.99,
    fontsize=12,
)

fig.savefig(os.path.join(OUTDIR, "polyform_03_tissue_support_topology.png"), dpi=DPI)
fig.savefig(os.path.join(OUTDIR, "polyform_03_tissue_support_topology.pdf"))
plt.show()

print("\nSaved figures:")
print(f"  {OUTDIR}/polyform_03_tissue_support_topology.png")
print(f"  {OUTDIR}/polyform_03_tissue_support_topology.pdf")


# ============================================================================
# 11. Top examples: tissue-specific disconnected support
# ============================================================================

examples = (
    anchored
    .sort_values(
        [
            "n_observed_states",
            "n_components",
            "fraction_observed_connected_to_k0",
            "delta_k_observed",
        ],
        ascending=[False, False, True, False],
    )
    .head(25)
)

print("\nTop tissue-lattice examples for inspection:")
display(examples[[
    "Tissue",
    "Accession",
    "Uniprot_Id",
    "Description",
    "L",
    "n_observed_states",
    "K_observed",
    "delta_k_observed",
    "n_components",
    "fraction_observed_connected_to_k0",
    "frac_k1_observed",
    "frac_k2plus_observed",
    "mean_distance_to_k0",
]])

export_tuple_cols(examples).to_csv(
    os.path.join(OUTDIR, "polyform_03_top_topology_examples.csv"),
    index=False,
)


# ============================================================================
# 12. Manuscript-safe reporting sentences
# ============================================================================

print("\n" + "=" * 86)
print("REPORTING SENTENCES")
print("=" * 86)

print(
    "We analysed the tissue atlas in empirical support mode using only observed "
    "proteoform states, and in anchored support mode after adding the unmodified "
    "k=0 state as a reference boundary vertex."
)

print(
    "The k=0 state was not treated as observed unless it was present in the atlas; "
    "when absent, it was included only as an anchor for measuring boundary connectivity."
)

print(
    "This allowed tissue-resolved proteoform support to be represented as native "
    "Hamming-1 support graphs, with connected components, component counts, "
    "largest component fractions and k=0 boundary connectivity computed directly "
    "on the modal lattice."
)

print(
    "Tissue-specificity was assessed by comparing whether the same binary state "
    "within the same protein-sequence lattice was observed in one tissue or shared "
    "across multiple tissues."
)
