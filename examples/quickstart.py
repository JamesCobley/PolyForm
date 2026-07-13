"""
PolyForm quickstart.

Run from a directory containing `proteoforms.csv` (support/weight modes) and/or
the multi-tissue atlas spreadsheet (topology mode), or pass explicit paths.

    python examples/quickstart.py
"""
import polyform

# 1) Global support-mode structural metrics --------------------------------
support = polyform.run_support_mode(
    csv="proteoforms.csv",
    outdir="out_support",
    make_figures=True,
)
g = support["global"].iloc[0]
print(f"proteins analysed:        {int(g.proteins_analysed):,}")
print(f"accessed binary states:   {int(g.total_accessed_binary_states):,}")
print(f"observed fibres:          {int(g.total_observed_fibres):,}")

# 2) Demonstration occupancy metrics ---------------------------------------
weight = polyform.run_weight_mode(
    csv="proteoforms.csv",
    outdir="out_weight",
    select_n=8,
)
print(f"selected demo proteins:   {len(weight['selected'])}")

# 3) Tissue-resolved support topology --------------------------------------
# topology = polyform.run_topology(
#     xlsx="pr2c00034_si_002 (1).xlsx",
#     outdir="out_topology",
# )
# print(f"tissues: {len(topology['tissue_summary'])}")

# Core primitives are importable directly ----------------------------------
from polyform.core import parse_ptms, hamming_distance_state

print("parse_ptms demo:", parse_ptms("RESID:55@3|RESID:76@8"))
print("hamming demo:", hamming_distance_state((1, 3, 5), (1, 4, 5)))
