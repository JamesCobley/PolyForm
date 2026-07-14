"""
polyform.core
=============

Pure, stateless primitives shared across PolyForm's analyses, exposed as a
convenience API. These functions are copied verbatim from the manuscript
analysis scripts so that library use and manuscript reproduction stay identical.

Nothing here reads global state or performs I/O beyond the explicit file
readers (`find_csv`, `read_fasta`).
"""
import os
import glob
import numpy as np
import pandas as pd
from math import log10, lgamma
from collections import defaultdict

__all__ = [
    "find_csv", "read_fasta", "parse_ptms", "log10_comb",
    "log10sumexp_base10", "safe_log10_state_fraction", "tuple_to_string",
    "state_to_string", "hamming_distance_state", "normalized_unevenness",
    "shannon_entropy", "normalized_entropy",
    "connected_components_for_states", "component_distance",
]

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
