import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from math import log10, lgamma
from collections import Counter, defaultdict


def display(*args, **kwargs):  # notebook no-op
    return None


def run_weight_mode(csv="proteoforms.csv", fasta_path=None, pos_base=0,
                    outdir="polyform_02_demo_occupancy_outputs",
                    select_n=8, min_states=5, min_delta_k=3,
                    make_figures=True, dpi=300, random_seed=7):
    """Demonstration occupancy metrics on real observed proteoform supports.

    Selects proteins with broad observed k-range access, assigns explicit
    demonstration weight vectors over the observed binary support, and
    computes occupancy-dependent metrics. Returns a dict of DataFrames
    and writes CSVs + figures into `outdir`."""
    FASTA_PATH = fasta_path
    POS_BASE = pos_base
    SELECT_N = select_n
    MIN_STATES = min_states
    MIN_DELTA_K = min_delta_k
    DPI = dpi
    RANDOM_SEED = random_seed
    OUTDIR = outdir
    os.makedirs(outdir, exist_ok=True)
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


    CSV = find_csv(csv)
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
        Binary state:
            unique modified residue positions.

        Fibre:
            residue position plus modification identity.
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


    def safe_log10_state_fraction(n_acc, L):
        if n_acc <= 0 or L <= 0:
            return np.nan
        return log10(n_acc) - L * LOG10_2


    def tuple_to_string(t):
        if isinstance(t, tuple):
            return ";".join(map(str, t))
        return str(t)


    def state_to_string(state):
        if len(state) == 0:
            return "0^L"
        return ",".join(map(str, state))


    def hamming_distance_state(a, b):
        """
        Hamming distance between binary states encoded as tuples of modified positions.
        """
        return len(set(a).symmetric_difference(set(b)))


    def normalized_unevenness(rho):
        """
        U_state = 0 when weights are even across accessed states.
        U_state = 1 when all weight is concentrated into one state.

        rho must sum to 1 over accessed states.
        """
        rho = np.asarray(rho, dtype=float)
        rho = rho[rho > 0]

        n = len(rho)

        if n == 0:
            return np.nan
        if n == 1:
            return 1.0

        q = rho / rho.sum()
        concentration = np.sum(q ** 2)

        return (concentration - 1 / n) / (1 - 1 / n)


    def shannon_entropy(rho):
        """
        Statistical entropy of a discrete occupancy vector.
        Not physical entropy.
        """
        rho = np.asarray(rho, dtype=float)
        rho = rho[rho > 0]

        if len(rho) == 0:
            return np.nan

        return -np.sum(rho * np.log(rho))


    def normalized_entropy(rho):
        """
        Entropy normalized by log(number of accessed states).
        Returns 1 for an even distribution over n>1 states.
        """
        rho = np.asarray(rho, dtype=float)
        rho = rho[rho > 0]
        n = len(rho)

        if n == 0:
            return np.nan
        if n == 1:
            return 0.0

        return shannon_entropy(rho) / np.log(n)


    # ============================================================================
    # 4. Protein-level support dataframe
    # ============================================================================

    def protein_support_metrics(acc, grp):
        if acc not in canon:
            return None

        L = len(canon[acc])
        name = str(grp[NAME_COL].iloc[0]) if NAME_COL else acc

        S_obs = set(grp["state_key"])
        fibres = set(grp["fiber_key"])

        n_acc = len(S_obs)
        if n_acc == 0:
            return None

        K = sorted({len(x) for x in S_obs})
        k_min = min(K)
        k_max = max(K)
        delta_k = k_max - k_min

        state_to_fibres = defaultdict(set)
        for _, r in grp.iterrows():
            state_to_fibres[r["state_key"]].add(r["fiber_key"])

        fibres_per_state = [len(v) for v in state_to_fibres.values()]

        return {
            "acc": acc,
            "name": name[:100],
            "L": L,
            "log10_M": L * LOG10_2,
            "n_records": len(grp),
            "n_accessed_states": n_acc,
            "n_fibres": len(fibres),
            "mean_fibres_per_state": float(np.mean(fibres_per_state)),
            "max_fibres_per_state": int(np.max(fibres_per_state)),
            "K": tuple(K),
            "n_occupied_grades": len(K),
            "k_min": k_min,
            "k_max": k_max,
            "delta_k": delta_k,
            "A_grade": delta_k / L if L > 0 else np.nan,
            "F_grade": len(K) / (L + 1) if L >= 0 else np.nan,
            "log10_A_state": safe_log10_state_fraction(n_acc, L),
        }


    metric_rows = []
    for acc, grp in df.groupby("Entry Accession"):
        out = protein_support_metrics(acc, grp)
        if out is not None:
            metric_rows.append(out)

    METS = pd.DataFrame(metric_rows)

    if len(METS) == 0:
        raise ValueError("No protein metrics computed.")

    METS = METS.sort_values(["delta_k", "n_accessed_states"], ascending=False).reset_index(drop=True)


    # ============================================================================
    # 5. Select proteins for demonstration
    # ============================================================================

    candidates = METS[
        (METS["n_accessed_states"] >= MIN_STATES) &
        (METS["delta_k"] >= MIN_DELTA_K)
    ].copy()

    if len(candidates) < SELECT_N:
        candidates = METS[
            METS["n_accessed_states"] >= max(2, MIN_STATES // 2)
        ].copy()

    if len(candidates) < SELECT_N:
        candidates = METS.copy()

    SELECTED_PROTEINS = (
        candidates
        .sort_values(["delta_k", "n_accessed_states"], ascending=False)
        .head(SELECT_N)
        .reset_index(drop=True)
    )

    print("\n" + "=" * 86)
    print("SELECTED PROTEINS FOR DEMONSTRATION OCCUPANCY")
    print("=" * 86)
    display(SELECTED_PROTEINS[[
        "acc", "name", "L", "n_accessed_states", "n_fibres",
        "K", "delta_k", "A_grade", "log10_A_state"
    ]])


    # ============================================================================
    # 6. Support graph and native components
    # ============================================================================

    def connected_components_for_states(states):
        """
        Compute connected components of observed support under Hamming adjacency.

        States are tuples of modified positions.
        Edge exists when Hamming distance = 1.

        For selected proteins, pairwise computation is transparent and sufficient.
        """
        states = list(states)
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

        # Sort components by size descending, then minimum grade.
        comps = sorted(comps, key=lambda c: (-len(c), min(len(x) for x in c)))

        comp_map = {}
        for cid, comp in enumerate(comps, start=1):
            for s in comp:
                comp_map[s] = cid

        return comps, comp_map


    def component_distance(comp_a, comp_b):
        """
        Minimum Hamming distance between two components.
        """
        best = np.inf

        for a in comp_a:
            for b in comp_b:
                d = hamming_distance_state(a, b)
                if d < best:
                    best = d

        return best


    # ============================================================================
    # 7. Demonstration weight profiles
    # ============================================================================

    PROFILES = [
        "even",
        "low_grade",
        "high_grade",
        "dominant_state",
    ]


    def assign_demo_weights(states, profile):
        """
        Assign explicit demonstration weights over observed support.

        These weights are not inferred from the catalogue. They are supplied only
        to demonstrate occupancy-dependent PolyForm metrics.
        """
        states = list(states)
        k = np.array([len(s) for s in states], dtype=float)

        if len(states) == 0:
            return np.array([])

        if profile == "even":
            raw = np.ones(len(states), dtype=float)

        elif profile == "low_grade":
            # Emphasize lower-k observed states.
            beta = 1.25
            raw = np.exp(-beta * (k - k.min()))

        elif profile == "high_grade":
            # Emphasize higher-k observed states.
            beta = 1.25
            raw = np.exp(beta * (k - k.min()))

        elif profile == "dominant_state":
            # Concentrate most weight into a single observed state.
            # Choose the lowest-grade state, then lexicographic order.
            raw = np.ones(len(states), dtype=float) * 0.02
            order = sorted(range(len(states)), key=lambda i: (len(states[i]), states[i]))
            raw[order[0]] = 1.0

        else:
            raise ValueError(f"Unknown profile: {profile}")

        rho = raw / raw.sum()
        return rho


    # ============================================================================
    # 8. Build demonstration state-weight dataframe
    # ============================================================================

    state_rows = []
    component_rows = []
    metric_rows = []
    grade_rows = []

    for _, prow in SELECTED_PROTEINS.iterrows():
        acc = prow["acc"]
        grp = df[df["Entry Accession"] == acc].copy()

        L = int(prow["L"])
        name = prow["name"]

        states = sorted(set(grp["state_key"]), key=lambda s: (len(s), s))
        comps, comp_map = connected_components_for_states(states)

        # Component-level topology independent of weights.
        comp_distance_min = np.nan
        if len(comps) > 1:
            distances = []
            for i in range(len(comps)):
                for j in range(i + 1, len(comps)):
                    distances.append(component_distance(comps[i], comps[j]))
            comp_distance_min = min(distances) if distances else np.nan

        for profile in PROFILES:
            rho = assign_demo_weights(states, profile)

            tmp = pd.DataFrame({
                "state_key": states,
                "rho": rho,
            })

            tmp["k"] = tmp["state_key"].apply(len)

            # Tuple keys, especially the empty tuple (), can confuse pandas .map().
            # Explicit lookup keeps each tuple as one state object.
            tmp["component_id"] = tmp["state_key"].apply(lambda s: comp_map[tuple(s)])

            n_acc = len(tmp)
            K = sorted(tmp["k"].unique())
            k_min = min(K)
            k_max = max(K)
            delta_k = k_max - k_min

            U_state = normalized_unevenness(tmp["rho"].values)
            H_state = shannon_entropy(tmp["rho"].values)
            H_norm = normalized_entropy(tmp["rho"].values)
            N_eff = 1.0 / np.sum(tmp["rho"].values ** 2)

            mean_k = float(np.sum(tmp["rho"].values * tmp["k"].values))
            mean_k_norm = mean_k / L if L > 0 else np.nan

            # Grade occupancy.
            gdist = tmp.groupby("k")["rho"].sum().reset_index()
            for _, gr in gdist.iterrows():
                grade_rows.append({
                    "acc": acc,
                    "name": name,
                    "profile": profile,
                    "L": L,
                    "k": int(gr["k"]),
                    "rho_grade": float(gr["rho"]),
                })

            # Component occupancy.
            cdist = tmp.groupby("component_id")["rho"].sum().reset_index()
            cdist = cdist.sort_values("rho", ascending=False)

            largest_component_weight = float(cdist["rho"].max())
            n_components = len(cdist)

            boundary_component_weight = 0.0
            interior_component_weight = 0.0

            for cid, cgrp in tmp.groupby("component_id"):
                comp_states = list(cgrp["state_key"])
                comp_weight = float(cgrp["rho"].sum())
                comp_grades = [len(s) for s in comp_states]
                touches_k0 = any(k == 0 for k in comp_grades)
                touches_kL = any(k == L for k in comp_grades)

                if touches_k0 or touches_kL:
                    boundary_component_weight += comp_weight
                else:
                    interior_component_weight += comp_weight

                component_rows.append({
                    "acc": acc,
                    "name": name,
                    "profile": profile,
                    "L": L,
                    "component_id": int(cid),
                    "component_size_states": int(len(comp_states)),
                    "component_weight": comp_weight,
                    "k_min_component": int(min(comp_grades)),
                    "k_max_component": int(max(comp_grades)),
                    "touches_k0": touches_k0,
                    "touches_kL": touches_kL,
                })

            metric_rows.append({
                "acc": acc,
                "name": name,
                "profile": profile,
                "L": L,
                "n_accessed_states": n_acc,
                "K": tuple(K),
                "k_min": k_min,
                "k_max": k_max,
                "delta_k": delta_k,
                "A_grade": delta_k / L if L > 0 else np.nan,
                "mean_k": mean_k,
                "mean_k_norm": mean_k_norm,
                "U_state": U_state,
                "H_state": H_state,
                "H_norm": H_norm,
                "N_eff_states": N_eff,
                "n_components": n_components,
                "largest_component_weight": largest_component_weight,
                "boundary_component_weight": boundary_component_weight,
                "interior_component_weight": interior_component_weight,
                "min_inter_component_distance": comp_distance_min,
            })

            # State-level rows.
            tmp = tmp.sort_values(["component_id", "k", "state_key"]).reset_index(drop=True)
            for rank, sr in tmp.iterrows():
                state_rows.append({
                    "acc": acc,
                    "name": name,
                    "profile": profile,
                    "L": L,
                    "state_rank": rank + 1,
                    "state_key": sr["state_key"],
                    "state_key_string": state_to_string(sr["state_key"]),
                    "k": int(sr["k"]),
                    "component_id": int(sr["component_id"]),
                    "rho": float(sr["rho"]),
                })

    DEMO_STATE_WEIGHTS = pd.DataFrame(state_rows)
    DEMO_METRICS = pd.DataFrame(metric_rows)
    DEMO_GRADE = pd.DataFrame(grade_rows)
    DEMO_COMPONENTS = pd.DataFrame(component_rows)

    print("\n" + "=" * 86)
    print("DEMONSTRATION OCCUPANCY METRICS")
    print("=" * 86)
    display(DEMO_METRICS.head(20))


    # ============================================================================
    # 9. Export dataframes
    # ============================================================================

    SELECTED_EXPORT = SELECTED_PROTEINS.copy()
    SELECTED_EXPORT["K"] = SELECTED_EXPORT["K"].apply(tuple_to_string)

    METRICS_EXPORT = DEMO_METRICS.copy()
    METRICS_EXPORT["K"] = METRICS_EXPORT["K"].apply(tuple_to_string)

    STATE_EXPORT = DEMO_STATE_WEIGHTS.copy()
    STATE_EXPORT["state_key"] = STATE_EXPORT["state_key"].apply(tuple_to_string)

    SELECTED_EXPORT.to_csv(os.path.join(OUTDIR, "polyform_02_selected_proteins.csv"), index=False)
    STATE_EXPORT.to_csv(os.path.join(OUTDIR, "polyform_02_demo_state_weights.csv"), index=False)
    METRICS_EXPORT.to_csv(os.path.join(OUTDIR, "polyform_02_demo_metrics.csv"), index=False)
    DEMO_GRADE.to_csv(os.path.join(OUTDIR, "polyform_02_demo_grade_distributions.csv"), index=False)
    DEMO_COMPONENTS.to_csv(os.path.join(OUTDIR, "polyform_02_demo_components.csv"), index=False)

    print("\nSaved tables:")
    print(f"  {OUTDIR}/polyform_02_selected_proteins.csv")
    print(f"  {OUTDIR}/polyform_02_demo_state_weights.csv")
    print(f"  {OUTDIR}/polyform_02_demo_metrics.csv")
    print(f"  {OUTDIR}/polyform_02_demo_grade_distributions.csv")
    print(f"  {OUTDIR}/polyform_02_demo_components.csv")


    # ============================================================================
    # 10. Main demonstration figure
    # ============================================================================

    profile_order = PROFILES
    profile_labels = {
        "even": "even",
        "low_grade": "low-grade",
        "high_grade": "high-grade",
        "dominant_state": "dominant",
    }

    fig = plt.figure(figsize=(12.4, 8.1))
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.36)

    # a — selected proteins in support space
    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(
        METS["delta_k"],
        np.log10(METS["n_accessed_states"]),
        s=9,
        color=GREY,
        alpha=0.45,
        edgecolor="none",
        label="all proteins",
    )
    ax.scatter(
        SELECTED_PROTEINS["delta_k"],
        np.log10(SELECTED_PROTEINS["n_accessed_states"]),
        s=34,
        color=AMBER,
        edgecolor="none",
        label="selected",
    )
    ax.set_xlabel(r"support grade span $\Delta k$")
    ax.set_ylabel(r"$\log_{10}|S_{\mathrm{obs}}|$")
    ax.set_title("a  selected broad-support proteins", loc="left")
    ax.legend(fontsize=7.2)

    # b — unevenness by profile
    ax = fig.add_subplot(gs[0, 1])
    for i, prof in enumerate(profile_order):
        vals = DEMO_METRICS.loc[DEMO_METRICS.profile == prof, "U_state"].values
        x = np.full_like(vals, i, dtype=float) + np.random.normal(0, 0.035, size=len(vals))
        ax.scatter(x, vals, s=24, color=TEAL, alpha=0.78, edgecolor="none")
        ax.plot([i - 0.18, i + 0.18], [np.median(vals), np.median(vals)], color=INK, lw=1.2)
    ax.set_xticks(range(len(profile_order)))
    ax.set_xticklabels([profile_labels[p] for p in profile_order], rotation=20, ha="right")
    ax.set_ylabel(r"occupancy unevenness $U_{\mathrm{state}}$")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("b  supplied weights define unevenness", loc="left")

    # c — weighted mean grade
    ax = fig.add_subplot(gs[0, 2])
    for i, prof in enumerate(profile_order):
        vals = DEMO_METRICS.loc[DEMO_METRICS.profile == prof, "mean_k"].values
        x = np.full_like(vals, i, dtype=float) + np.random.normal(0, 0.035, size=len(vals))
        ax.scatter(x, vals, s=24, color=VIOLET, alpha=0.78, edgecolor="none")
        ax.plot([i - 0.18, i + 0.18], [np.median(vals), np.median(vals)], color=INK, lw=1.2)
    ax.set_xticks(range(len(profile_order)))
    ax.set_xticklabels([profile_labels[p] for p in profile_order], rotation=20, ha="right")
    ax.set_ylabel(r"weighted mean grade $\sum_x \rho(x)k(x)$")
    ax.set_title("c  weight shifts grade occupancy", loc="left")

    # d — largest component weight
    ax = fig.add_subplot(gs[1, 0])
    for i, prof in enumerate(profile_order):
        vals = DEMO_METRICS.loc[DEMO_METRICS.profile == prof, "largest_component_weight"].values
        x = np.full_like(vals, i, dtype=float) + np.random.normal(0, 0.035, size=len(vals))
        ax.scatter(x, vals, s=24, color=AMBER, alpha=0.78, edgecolor="none")
        ax.plot([i - 0.18, i + 0.18], [np.median(vals), np.median(vals)], color=INK, lw=1.2)
    ax.set_xticks(range(len(profile_order)))
    ax.set_xticklabels([profile_labels[p] for p in profile_order], rotation=20, ha="right")
    ax.set_ylabel("largest component weight")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("d  principal component weight", loc="left")

    # e — boundary versus interior component weight
    ax = fig.add_subplot(gs[1, 1])
    for i, prof in enumerate(profile_order):
        vals = DEMO_METRICS.loc[DEMO_METRICS.profile == prof, "boundary_component_weight"].values
        x = np.full_like(vals, i, dtype=float) + np.random.normal(0, 0.035, size=len(vals))
        ax.scatter(x, vals, s=24, color=ROSE, alpha=0.78, edgecolor="none")
        ax.plot([i - 0.18, i + 0.18], [np.median(vals), np.median(vals)], color=INK, lw=1.2)
    ax.set_xticks(range(len(profile_order)))
    ax.set_xticklabels([profile_labels[p] for p in profile_order], rotation=20, ha="right")
    ax.set_ylabel("boundary-component weight")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("e  boundary-attached occupancy", loc="left")

    # f — grade distributions for the most grade-broad selected protein
    ax = fig.add_subplot(gs[1, 2])
    focus_acc = SELECTED_PROTEINS.iloc[0]["acc"]
    focus_grade = DEMO_GRADE[DEMO_GRADE.acc == focus_acc].copy()

    for prof in profile_order:
        sub = focus_grade[focus_grade.profile == prof].sort_values("k")
        ax.plot(
            sub["k"],
            sub["rho_grade"],
            marker="o",
            lw=1.2,
            ms=3.5,
            label=profile_labels[prof],
        )

    ax.set_xlabel("grade k")
    ax.set_ylabel(r"grade occupancy $p_k$")
    ax.set_title(f"f  grade occupancy example: {focus_acc}", loc="left")
    ax.legend(fontsize=7.2)

    fig.suptitle(
        "PolyForm occupancy-mode demonstration on real observed proteoform supports",
        x=0.5,
        y=0.985,
        fontsize=12,
    )

    if make_figures: fig.savefig(os.path.join(OUTDIR, "polyform_02_demo_occupancy_metrics.png"), dpi=DPI)
    if make_figures: fig.savefig(os.path.join(OUTDIR, "polyform_02_demo_occupancy_metrics.pdf"))
    pass

    print("\nSaved figures:")
    print(f"  {OUTDIR}/polyform_02_demo_occupancy_metrics.png")
    print(f"  {OUTDIR}/polyform_02_demo_occupancy_metrics.pdf")


    # ============================================================================
    # 11. Focus state-weight heatmap
    # ============================================================================

    focus_states = DEMO_STATE_WEIGHTS[DEMO_STATE_WEIGHTS.acc == focus_acc].copy()

    # Order states by component, grade, then state rank.
    state_order = (
        focus_states[focus_states.profile == "even"]
        .sort_values(["component_id", "k", "state_key_string"])
        ["state_key_string"]
        .tolist()
    )

    heat = []
    for prof in profile_order:
        sub = focus_states[focus_states.profile == prof].set_index("state_key_string")
        heat.append([sub.loc[s, "rho"] if s in sub.index else 0.0 for s in state_order])

    heat = np.array(heat)

    fig, ax = plt.subplots(figsize=(12.4, 3.1))

    im = ax.imshow(heat, aspect="auto", interpolation="nearest")

    ax.set_yticks(range(len(profile_order)))
    ax.set_yticklabels([profile_labels[p] for p in profile_order])

    # Avoid too many x labels.
    max_labels = 25
    if len(state_order) <= max_labels:
        xticks = range(len(state_order))
        xlabels = state_order
    else:
        xticks = np.linspace(0, len(state_order) - 1, max_labels).astype(int)
        xlabels = [state_order[i] for i in xticks]

    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=90, fontsize=6)

    ax.set_xlabel("observed binary states ordered by native support component and grade")
    ax.set_title(
        f"State-level supplied occupancy over observed support: {focus_acc}",
        loc="left",
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label(r"supplied occupancy $\rho(x)$")

    if make_figures: fig.savefig(os.path.join(OUTDIR, "polyform_02_focus_state_weight_heatmap.png"), dpi=DPI)
    if make_figures: fig.savefig(os.path.join(OUTDIR, "polyform_02_focus_state_weight_heatmap.pdf"))
    pass

    print(f"  {OUTDIR}/polyform_02_focus_state_weight_heatmap.png")
    print(f"  {OUTDIR}/polyform_02_focus_state_weight_heatmap.pdf")


    # ============================================================================
    # 12. Manuscript-safe reporting sentences
    # ============================================================================

    print("\n" + "=" * 86)
    print("REPORTING SENTENCES")
    print("=" * 86)

    print(
        "Because the human proteoform catalogue does not provide valid quantitative "
        "occupancy weights, the global database analysis was performed in support mode."
    )

    print(
        f"To demonstrate occupancy-dependent metrics, we selected {len(SELECTED_PROTEINS)} "
        "proteins with broad observed support and supplied explicit demonstration "
        "weight profiles over their observed binary states."
    )

    print(
        "These profiles are not inferred biological abundances; they instantiate the "
        "mathematical behaviour of PolyForm when a valid occupancy vector is supplied."
    )

    print(
        "The supplied profiles show that the same observed support can have different "
        "state unevenness, weighted mean grade, component weights, and boundary-attached "
        "occupancy while preserving the same native modal lattice."
    )
    return {
        "selected": SELECTED_PROTEINS,
        "state_weights": DEMO_STATE_WEIGHTS,
        "metrics": DEMO_METRICS,
        "grade": DEMO_GRADE,
        "components": DEMO_COMPONENTS,
    }
