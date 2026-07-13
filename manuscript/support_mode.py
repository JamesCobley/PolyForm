# ============================================================================
# PolyForm — global support-mode structural metrics on the choice-free modal lattice
#
# Mathematical object:
#   For a protein of length L, the choice-free modal lattice is
#       M = {0,1}^L
#   Each residue coordinate is unmodified (0) or modified (1).
#
# Support mode:
#   The human proteoform database provides observed proteoform identities,
#   not injective quantitative occupancy weights. Therefore this script computes
#   support/access metrics only.
#
# Valid global metrics:
#   - global catalogue size
#   - global possible state-space size: sum_i 2^L_i
#   - global accessed support: sum_i |S_obs,i|
#   - global state-access fraction
#   - protein-level accessed states
#   - protein-level grade span
#   - global observed grade distribution
#   - global grade coverage relative to theoretical degeneracy
#   - fibre refinement over binary states
#
# Dataframes produced:
#   df              = parsed record-level dataframe
#   METS            = protein-level structural metrics
#   GRADE           = protein-by-grade structural metrics
#   GRADE_GLOBAL    = global grade-level summary
#   GLOBAL          = one-row global summary
#   RESIDUES        = modified residue summary
#   CODE_RESIDUE    = RESID code -> residue empirical mapping
#   TOP_ACCESS      = proteins with most accessed binary states
#   TOP_SPAN        = proteins with widest grade span
#   TOP_FIBRE       = proteins with strongest fibre refinement
#
# Files saved:
#   polyform_global_structural_metrics.png
#   polyform_global_structural_metrics.pdf
#   polyform_global_summary.csv
#   polyform_protein_metrics.csv
#   polyform_grade_metrics.csv
#   polyform_global_grade_metrics.csv
#   polyform_residue_counts.csv
#   polyform_resid_code_residue_map.csv
#   polyform_top_accessed_states.csv
#   polyform_top_grade_span.csv
#   polyform_top_fibre_refinement.csv
# ============================================================================

import os
import glob
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from math import log10, lgamma
from collections import Counter, defaultdict
from IPython.display import display

# ----------------------------------------------------------------------------
# User settings
# ----------------------------------------------------------------------------

CSV = "proteoforms.csv"          # if missing, script will try to find a similar file
FASTA_PATH = None                # optional, e.g. "/content/Homo_sapiens_Sp_canonical.fasta"
POS_BASE = 0                     # set to 1 if RESID positions are 1-based
DPI = 300

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


# ============================================================================
# 1. Input handling
# ============================================================================

def find_csv(path):
    if os.path.exists(path):
        return path

    candidates = []
    candidates += glob.glob("*.csv")
    candidates += glob.glob("/content/*.csv")

    proteoform_candidates = [
        c for c in candidates
        if "proteoform" in os.path.basename(c).lower()
    ]

    if proteoform_candidates:
        return sorted(proteoform_candidates)[0]

    if candidates:
        return sorted(candidates)[0]

    raise FileNotFoundError(
        "No CSV file found. Upload proteoforms.csv or set CSV to the correct path."
    )


CSV = find_csv(CSV)
print(f"Reading: {CSV}")

df = pd.read_csv(CSV)

required = ["Entry Accession", "PTMs"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

NAME_COL = "Entry Description" if "Entry Description" in df.columns else None
ISOFORM_COL = "Isoform Sequence" if "Isoform Sequence" in df.columns else None
PROTEOFORM_SEQ_COL = "Proteoform Sequence" if "Proteoform Sequence" in df.columns else ISOFORM_COL

if ISOFORM_COL is None and FASTA_PATH is None:
    raise ValueError(
        "Need either an Isoform Sequence column or FASTA_PATH to resolve protein lengths."
    )


# ============================================================================
# 2. Parsing and sequence resolution
# ============================================================================

def parse_ptms(s, pos_base=0):
    """
    Parse PTM strings of the form:
        'RESID:55@3|RESID:76@8'

    Returns:
        [(position, mark_code), ...]
    where position is adjusted by POS_BASE.
    """
    if pd.isna(s) or not str(s).strip():
        return []

    out = []

    for tok in str(s).split("|"):
        tok = tok.strip()
        if not tok:
            continue

        if tok.startswith("RESID:"):
            body = tok[len("RESID:"):]
            if "@" not in body:
                continue

            mark, pos = body.split("@", 1)

            try:
                pos = int(pos) - pos_base
                out.append((pos, str(mark)))
            except Exception:
                continue

    return out


def read_fasta(path):
    """
    Read a UniProt-style FASTA.
    Returns dictionary accession -> sequence.
    """
    seqs = {}
    acc = None
    buf = []

    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")

            if line.startswith(">"):
                if acc is not None:
                    seqs[acc] = "".join(buf)

                header = line[1:]
                parts = header.split("|")

                if len(parts) >= 2:
                    acc = parts[1]
                else:
                    acc = header.split()[0]

                buf = []
            else:
                buf.append(line.strip())

        if acc is not None:
            seqs[acc] = "".join(buf)

    return seqs


df["mods_raw"] = df["PTMs"].apply(lambda s: parse_ptms(s, pos_base=POS_BASE))

fasta = read_fasta(FASTA_PATH) if FASTA_PATH else {}

canon = {}
for acc, grp in df.groupby("Entry Accession"):
    if acc in fasta:
        canon[acc] = fasta[acc]
    elif ISOFORM_COL is not None:
        seqs = grp[ISOFORM_COL].dropna()
        if len(seqs):
            canon[acc] = str(seqs.iloc[0])

print(
    f"Canonical sequences resolved for {len(canon):,} / "
    f"{df['Entry Accession'].nunique():,} proteins "
    f"(source: {'FASTA' if fasta else ISOFORM_COL})"
)


def build_keys_for_record(row):
    """
    Build binary state and fibre keys using the resolved canonical length.

    Binary state:
        unique modified residue positions.

    Fibre:
        residue position plus modification identity.

    Positions outside the resolved canonical sequence are excluded from the
    binary state and counted as invalid.
    """
    acc = row["Entry Accession"]

    if acc not in canon:
        return pd.Series({
            "L": np.nan,
            "valid_mods": tuple(),
            "invalid_mods": tuple(row["mods_raw"]),
            "state_key": tuple(),
            "fiber_key": tuple(),
            "k_state": np.nan,
            "k_fiber": np.nan,
            "n_invalid_positions": len(row["mods_raw"]),
        })

    L = len(canon[acc])
    valid = []
    invalid = []

    for pos, code in row["mods_raw"]:
        if 0 <= pos < L:
            valid.append((pos, code))
        else:
            invalid.append((pos, code))

    state_key = tuple(sorted(set(pos for pos, code in valid)))
    fiber_key = tuple(sorted(valid))

    return pd.Series({
        "L": L,
        "valid_mods": tuple(valid),
        "invalid_mods": tuple(invalid),
        "state_key": state_key,
        "fiber_key": fiber_key,
        "k_state": len(state_key),
        "k_fiber": len(fiber_key),
        "n_invalid_positions": len(invalid),
    })


df = pd.concat([df, df.apply(build_keys_for_record, axis=1)], axis=1)


# ============================================================================
# 3. Mathematical helpers
# ============================================================================

LOG10_2 = log10(2)


def log10_comb(n, k):
    """
    log10 binomial coefficient using lgamma for numerical stability.
    """
    if k < 0 or k > n:
        return -np.inf

    return (lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)) / np.log(10)


def log10sumexp_base10(values):
    """
    log10(sum_i 10^values_i), stable for large logs.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan

    m = np.max(values)
    return m + log10(np.sum(10 ** (values - m)))


def safe_log10_state_fraction(n_acc, L):
    """
    log10(n_acc / 2^L), avoiding overflow/underflow.
    """
    if n_acc <= 0 or L <= 0:
        return np.nan

    return log10(n_acc) - L * LOG10_2


def state_access_fraction_float(n_acc, L):
    """
    Floating version of n_acc / 2^L.

    For large proteins this will underflow; use log10_A_state as primary.
    """
    logv = safe_log10_state_fraction(n_acc, L)

    if np.isnan(logv):
        return np.nan

    if logv < -300:
        return 0.0

    return 10 ** logv


def tuple_to_string(t):
    if isinstance(t, tuple):
        return ";".join(map(str, t))
    return str(t)


# ============================================================================
# 4. Protein-level support metrics
# ============================================================================

def protein_support_metrics(acc, grp):
    """
    Compute exact support-mode structural metrics for one protein.

    No biological occupancy weights are inferred.
    """
    if acc not in canon:
        return None

    L = len(canon[acc])
    name = str(grp[NAME_COL].iloc[0]) if NAME_COL else acc

    # Observed binary states.
    S_obs = set(grp["state_key"])

    # Observed fibres over binary states.
    fibres = set(grp["fiber_key"])

    n_acc = len(S_obs)
    if n_acc == 0:
        return None

    K = sorted({len(x) for x in S_obs})
    k_min = min(K)
    k_max = max(K)
    delta_k = k_max - k_min

    log10_M = L * LOG10_2
    log10_A_state = safe_log10_state_fraction(n_acc, L)
    A_state = state_access_fraction_float(n_acc, L)

    A_grade = delta_k / L if L > 0 else np.nan
    F_grade = len(K) / (L + 1) if L >= 0 else np.nan

    n_by_grade = Counter(len(x) for x in S_obs)

    # Fibre multiplicity over binary states.
    state_to_fibres = defaultdict(set)
    for _, r in grp.iterrows():
        state_to_fibres[r["state_key"]].add(r["fiber_key"])

    fibres_per_state = [len(v) for v in state_to_fibres.values()]

    return {
        "acc": acc,
        "name": name[:100],
        "L": L,
        "log10_M": log10_M,
        "n_records": len(grp),
        "n_accessed_states": n_acc,
        "n_fibres": len(fibres),
        "mean_fibres_per_state": float(np.mean(fibres_per_state)),
        "median_fibres_per_state": float(np.median(fibres_per_state)),
        "max_fibres_per_state": int(np.max(fibres_per_state)),
        "log10_A_state": log10_A_state,
        "A_state_float": A_state,
        "K": tuple(K),
        "n_occupied_grades": len(K),
        "k_min": k_min,
        "k_max": k_max,
        "delta_k": delta_k,
        "A_grade": A_grade,
        "F_grade": F_grade,
        "n_invalid_positions": int(grp["n_invalid_positions"].sum()),
        "has_unmodified_state": tuple() in S_obs,
    }


metric_rows = []
for acc, grp in df.groupby("Entry Accession"):
    out = protein_support_metrics(acc, grp)
    if out is not None:
        metric_rows.append(out)

METS = pd.DataFrame(metric_rows)

if len(METS) == 0:
    raise ValueError("No protein metrics computed. Check sequence resolution and input columns.")

METS = METS.sort_values(["n_accessed_states", "L"], ascending=[False, True]).reset_index(drop=True)

METS["log10_n_accessed_states"] = np.log10(METS["n_accessed_states"])
METS["log10_n_fibres"] = np.log10(METS["n_fibres"])


# ============================================================================
# 5. Protein-by-grade dataframe
# ============================================================================

grade_rows = []

for acc, grp in df.groupby("Entry Accession"):
    if acc not in canon:
        continue

    L = len(canon[acc])
    name = str(grp[NAME_COL].iloc[0]) if NAME_COL else acc

    S_obs = set(grp["state_key"])
    n_by_grade = Counter(len(x) for x in S_obs)

    for k, n_acc_k in sorted(n_by_grade.items()):
        lg_gk = log10_comb(L, k)
        lg_cov = log10(n_acc_k) - lg_gk if n_acc_k > 0 else np.nan
        coverage_float = 0.0 if lg_cov < -300 else 10 ** lg_cov

        grade_rows.append({
            "acc": acc,
            "name": name[:100],
            "L": L,
            "k": k,
            "n_accessed_states_k": n_acc_k,
            "log10_g_k": lg_gk,
            "log10_grade_coverage": lg_cov,
            "grade_coverage_float": coverage_float,
        })

GRADE = pd.DataFrame(grade_rows).sort_values(["acc", "k"]).reset_index(drop=True)


# ============================================================================
# 6. Global grade-level summary
# ============================================================================

global_grade_rows = []
observed_ks = sorted(GRADE["k"].unique())

for k in observed_ks:
    obs = int(GRADE.loc[GRADE["k"] == k, "n_accessed_states_k"].sum())

    possible_logs = [
        log10_comb(int(L), int(k))
        for L in METS["L"].values
        if int(L) >= int(k)
    ]

    log10_possible = log10sumexp_base10(possible_logs)
    log10_observed = log10(obs) if obs > 0 else np.nan
    log10_coverage = log10_observed - log10_possible if obs > 0 else np.nan
    coverage_float = 0.0 if log10_coverage < -300 else 10 ** log10_coverage

    global_grade_rows.append({
        "k": int(k),
        "global_observed_states_k": obs,
        "log10_global_observed_states_k": log10_observed,
        "log10_global_possible_states_k": log10_possible,
        "log10_global_grade_coverage": log10_coverage,
        "global_grade_coverage_float": coverage_float,
        "n_proteins_with_possible_grade": int((METS["L"] >= k).sum()),
        "n_proteins_with_observed_grade": int((GRADE.loc[GRADE["k"] == k, "acc"]).nunique()),
    })

GRADE_GLOBAL = pd.DataFrame(global_grade_rows)


# ============================================================================
# 7. Global summary dataframe
# ============================================================================

log10_total_possible = log10sumexp_base10(METS["log10_M"].values)
total_accessed_states = int(METS["n_accessed_states"].sum())
log10_total_accessed = log10(total_accessed_states)
log10_global_access_fraction = log10_total_accessed - log10_total_possible
global_access_fraction_float = 0.0 if log10_global_access_fraction < -300 else 10 ** log10_global_access_fraction

GLOBAL = pd.DataFrame([{
    "records": len(df),
    "proteins_in_input": df["Entry Accession"].nunique(),
    "proteins_with_resolved_sequence": len(canon),
    "proteins_analysed": len(METS),
    "ptm_bearing_records_valid_binary": int((df["k_state"] > 0).sum()),
    "unmodified_records_binary_k0": int((df["k_state"] == 0).sum()),
    "invalid_ptm_positions_excluded": int(df["n_invalid_positions"].sum()),
    "total_accessed_binary_states": total_accessed_states,
    "log10_total_accessed_binary_states": log10_total_accessed,
    "log10_total_possible_state_space_sum_2L": log10_total_possible,
    "log10_global_state_access_fraction": log10_global_access_fraction,
    "global_state_access_fraction_float": global_access_fraction_float,
    "total_observed_fibres": int(METS["n_fibres"].sum()),
    "median_accessed_states_per_protein": float(METS["n_accessed_states"].median()),
    "max_accessed_states_per_protein": int(METS["n_accessed_states"].max()),
    "median_log10_state_access_fraction": float(METS["log10_A_state"].median()),
    "median_occupied_grades": float(METS["n_occupied_grades"].median()),
    "median_delta_k": float(METS["delta_k"].median()),
    "max_delta_k": int(METS["delta_k"].max()),
    "median_A_grade": float(METS["A_grade"].median()),
    "median_F_grade": float(METS["F_grade"].median()),
    "median_fibres_per_state": float(METS["mean_fibres_per_state"].median()),
    "max_fibres_per_state": int(METS["max_fibres_per_state"].max()),
}])


# ============================================================================
# 8. Residue and RESID-code summaries
# ============================================================================

aa_all = Counter()
aa_by_code = defaultdict(Counter)
unmapped_residue_calls = 0

for _, r in df.iterrows():
    seq = None

    if PROTEOFORM_SEQ_COL is not None and pd.notna(r.get(PROTEOFORM_SEQ_COL, np.nan)):
        seq = str(r[PROTEOFORM_SEQ_COL])
    elif r["Entry Accession"] in canon:
        seq = canon[r["Entry Accession"]]

    if seq is None:
        continue

    for pos, code in r["valid_mods"]:
        if 0 <= pos < len(seq):
            aa = seq[pos]
            aa_all[aa] += 1
            aa_by_code[code][aa] += 1
        else:
            unmapped_residue_calls += 1

total_aa_calls = sum(aa_all.values())

RESIDUES = pd.DataFrame([
    {
        "residue": aa,
        "n_modification_calls": n,
        "fraction": n / total_aa_calls if total_aa_calls else np.nan,
    }
    for aa, n in aa_all.most_common()
])

code_rows = []
for code, cnt in aa_by_code.items():
    n = sum(cnt.values())
    aa, m = cnt.most_common(1)[0]
    code_rows.append({
        "RESID_code": code,
        "n_calls": n,
        "dominant_residue": aa,
        "dominant_residue_calls": m,
        "purity": m / n if n else np.nan,
        "residue_counts": dict(cnt),
    })

CODE_RESIDUE = (
    pd.DataFrame(code_rows)
    .sort_values("n_calls", ascending=False)
    .reset_index(drop=True)
)


# ============================================================================
# 9. Top global tables
# ============================================================================

TOP_ACCESS = (
    METS.sort_values("n_accessed_states", ascending=False)
    .head(25)
    .reset_index(drop=True)
)

TOP_SPAN = (
    METS.sort_values(["delta_k", "n_accessed_states"], ascending=False)
    .head(25)
    .reset_index(drop=True)
)

TOP_FIBRE = (
    METS.sort_values(["max_fibres_per_state", "n_fibres"], ascending=False)
    .head(25)
    .reset_index(drop=True)
)


# ============================================================================
# 10. Console report
# ============================================================================

g = GLOBAL.iloc[0]

print("\n" + "=" * 86)
print("POLYFORM GLOBAL SUPPORT-MODE STRUCTURAL SUMMARY")
print("=" * 86)
print(f"Records                                      {g.records:,.0f}")
print(f"Proteins analysed                            {g.proteins_analysed:,.0f}")
print(f"PTM-bearing records, valid binary             {g.ptm_bearing_records_valid_binary:,.0f}")
print(f"Binary k=0 records                            {g.unmodified_records_binary_k0:,.0f}")
print(f"Total accessed binary states                  {g.total_accessed_binary_states:,.0f}")
print(f"Total observed fibres                         {g.total_observed_fibres:,.0f}")
print(f"log10 total possible state space sum_i 2^L    {g.log10_total_possible_state_space_sum_2L:,.2f}")
print(f"log10 global accessed support                 {g.log10_total_accessed_binary_states:,.2f}")
print(f"log10 global state-access fraction            {g.log10_global_state_access_fraction:,.2f}")
print(f"Median accessed states per protein            {g.median_accessed_states_per_protein:,.0f}")
print(f"Max accessed states per protein               {g.max_accessed_states_per_protein:,.0f}")
print(f"Median log10 state-access fraction            {g.median_log10_state_access_fraction:,.2f}")
print(f"Median occupied grades                        {g.median_occupied_grades:,.0f}")
print(f"Median grade span delta_k                     {g.median_delta_k:,.0f}")
print(f"Max grade span delta_k                        {g.max_delta_k:,.0f}")
print(f"Median normalized grade span A_grade          {g.median_A_grade:,.4f}")
print(f"Median grade-access fraction F_grade          {g.median_F_grade:,.4f}")
print(f"Max fibres per binary state                   {g.max_fibres_per_state:,.0f}")
print(f"Invalid PTM positions excluded                {g.invalid_ptm_positions_excluded:,.0f}")

print("\nTop proteins by accessed binary states:")
display(TOP_ACCESS[[
    "acc", "name", "L", "n_accessed_states", "n_fibres",
    "delta_k", "A_grade", "log10_A_state"
]].head(15))

print("\nTop proteins by grade span:")
display(TOP_SPAN[[
    "acc", "name", "L", "n_accessed_states", "n_fibres",
    "delta_k", "A_grade", "K"
]].head(15))

print("\nTop proteins by fibre refinement:")
display(TOP_FIBRE[[
    "acc", "name", "L", "n_accessed_states", "n_fibres",
    "mean_fibres_per_state", "max_fibres_per_state"
]].head(15))


# ============================================================================
# 11. Export dataframes
# ============================================================================

METS_EXPORT = METS.copy()
METS_EXPORT["K"] = METS_EXPORT["K"].apply(tuple_to_string)

CODE_EXPORT = CODE_RESIDUE.copy()
if len(CODE_EXPORT):
    CODE_EXPORT["residue_counts"] = CODE_EXPORT["residue_counts"].apply(str)

GLOBAL.to_csv("polyform_global_summary.csv", index=False)
METS_EXPORT.to_csv("polyform_protein_metrics.csv", index=False)
GRADE.to_csv("polyform_grade_metrics.csv", index=False)
GRADE_GLOBAL.to_csv("polyform_global_grade_metrics.csv", index=False)
RESIDUES.to_csv("polyform_residue_counts.csv", index=False)
CODE_EXPORT.to_csv("polyform_resid_code_residue_map.csv", index=False)
TOP_ACCESS.to_csv("polyform_top_accessed_states.csv", index=False)
TOP_SPAN.to_csv("polyform_top_grade_span.csv", index=False)
TOP_FIBRE.to_csv("polyform_top_fibre_refinement.csv", index=False)

print("\nSaved tables:")
print("  polyform_global_summary.csv")
print("  polyform_protein_metrics.csv")
print("  polyform_grade_metrics.csv")
print("  polyform_global_grade_metrics.csv")
print("  polyform_residue_counts.csv")
print("  polyform_resid_code_residue_map.csv")
print("  polyform_top_accessed_states.csv")
print("  polyform_top_grade_span.csv")
print("  polyform_top_fibre_refinement.csv")


# ============================================================================
# 12. Global figure
# ============================================================================

fig = plt.figure(figsize=(12.4, 10.2))
gs = fig.add_gridspec(3, 3, hspace=0.48, wspace=0.38)

# a — global headline text
ax = fig.add_subplot(gs[0, 0])
ax.axis("off")
headline = (
    "Global support-mode summary\n\n"
    f"records: {int(g.records):,}\n"
    f"proteins analysed: {int(g.proteins_analysed):,}\n"
    f"accessed binary states: {int(g.total_accessed_binary_states):,}\n"
    f"observed fibres: {int(g.total_observed_fibres):,}\n\n"
    f"log10 sum_i 2^L: {g.log10_total_possible_state_space_sum_2L:.1f}\n"
    f"log10 accessed support: {g.log10_total_accessed_binary_states:.1f}\n"
    f"log10 global access fraction: {g.log10_global_state_access_fraction:.1f}\n\n"
    f"median n_accessed: {g.median_accessed_states_per_protein:.0f}\n"
    f"median delta_k: {g.median_delta_k:.0f}\n"
    f"median A_grade: {g.median_A_grade:.3f}"
)
ax.text(
    0.02, 0.98, headline,
    transform=ax.transAxes,
    va="top", ha="left",
    fontsize=9,
    family="DejaVu Sans Mono",
    color=INK,
)
ax.set_title("a  catalogue-scale metrics", loc="left")

# b — protein length distribution
ax = fig.add_subplot(gs[0, 1])
ax.hist(METS["L"], bins=60, color=TEAL, alpha=0.88)
ax.set_xlabel("protein length L")
ax.set_ylabel("proteins")
ax.set_title("b  protein lengths", loc="left")

# c — possible state-space distribution
ax = fig.add_subplot(gs[0, 2])
ax.hist(METS["log10_M"], bins=60, color=AMBER, alpha=0.88)
ax.set_xlabel(r"$\log_{10}|\mathcal{M}| = L\log_{10}2$")
ax.set_ylabel("proteins")
ax.set_title("c  possible state-space scale", loc="left")

# d — accessed states per protein
ax = fig.add_subplot(gs[1, 0])
max_log_access = max(1, int(np.ceil(METS["log10_n_accessed_states"].max())))
bins = np.linspace(0, max_log_access, 40)
ax.hist(METS["log10_n_accessed_states"], bins=bins, color=TEAL, alpha=0.88)
ax.set_xlabel(r"$\log_{10}|S_{\mathrm{obs}}|$")
ax.set_ylabel("proteins")
ax.set_title("d  accessed binary states", loc="left")

# e — log10 state access fraction
ax = fig.add_subplot(gs[1, 1])
finite_access = METS["log10_A_state"].replace([np.inf, -np.inf], np.nan).dropna()
ax.hist(finite_access, bins=70, color=VIOLET, alpha=0.88)
ax.set_xlabel(r"$\log_{10}(|S_{\mathrm{obs}}|/|\mathcal{M}|)$")
ax.set_ylabel("proteins")
ax.set_title("e  state-access fraction", loc="left")

# f — grade span distribution
ax = fig.add_subplot(gs[1, 2])
ax.hist(METS["delta_k"], bins=np.arange(METS["delta_k"].min(), METS["delta_k"].max() + 2) - 0.5,
        color=AMBER, alpha=0.88)
ax.set_xlabel(r"grade span $\Delta k$")
ax.set_ylabel("proteins")
ax.set_title("f  realized grade span", loc="left")

# g — global observed grade distribution
ax = fig.add_subplot(gs[2, 0])
ax.bar(
    GRADE_GLOBAL["k"],
    GRADE_GLOBAL["global_observed_states_k"],
    color=TEAL,
    width=0.72,
)
ax.set_xlabel("grade k")
ax.set_ylabel("accessed binary states")
ax.set_title("g  global observed grades", loc="left")

# h — observed versus possible states by grade
ax = fig.add_subplot(gs[2, 1])
ax.plot(
    GRADE_GLOBAL["k"],
    GRADE_GLOBAL["log10_global_possible_states_k"],
    color=GREY,
    lw=1.4,
    label="possible"
)
ax.scatter(
    GRADE_GLOBAL["k"],
    GRADE_GLOBAL["log10_global_observed_states_k"],
    color=TEAL,
    s=18,
    label="observed"
)
ax.set_xlabel("grade k")
ax.set_ylabel(r"$\log_{10}$ states")
ax.set_title("h  observed support over degeneracy", loc="left")
ax.legend(fontsize=7.5)

# i — fibre refinement over binary states
ax = fig.add_subplot(gs[2, 2])
ax.scatter(
    METS["log10_n_accessed_states"],
    METS["log10_n_fibres"],
    s=8,
    color=TEAL,
    alpha=0.50,
    edgecolor="none",
)
lims = [
    0,
    max(METS["log10_n_accessed_states"].max(), METS["log10_n_fibres"].max()) * 1.05
]
ax.plot(lims, lims, ls="--", color=GREY, lw=0.9, label="fibres = binary states")
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xlabel(r"$\log_{10}$ accessed binary states")
ax.set_ylabel(r"$\log_{10}$ fibres")
ax.set_title("i  modification identity refines binary support", loc="left")
ax.legend(fontsize=7.2, loc="upper left")

fig.suptitle(
    "PolyForm global support-mode analysis of human proteoform observations",
    x=0.5,
    y=0.995,
    fontsize=12,
)

fig.savefig("polyform_global_structural_metrics.png", dpi=DPI)
fig.savefig("polyform_global_structural_metrics.pdf")
plt.show()

print("\nSaved figures:")
print("  polyform_global_structural_metrics.png")
print("  polyform_global_structural_metrics.pdf")


# ============================================================================
# 13. Display core dataframes
# ============================================================================

print("\nGLOBAL dataframe")
display(GLOBAL)

print("\nMETS dataframe: protein-level structural metrics")
display(METS.head(20))

print("\nGRADE_GLOBAL dataframe: global grade metrics")
display(GRADE_GLOBAL.head(30))

print("\nGRADE dataframe: protein-by-grade metrics")
display(GRADE.head(20))

print("\nRESIDUES dataframe")
display(RESIDUES.head(20))

print("\nCODE_RESIDUE dataframe")
display(CODE_RESIDUE.head(20))
