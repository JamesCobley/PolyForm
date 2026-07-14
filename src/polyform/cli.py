"""
Command-line interface for PolyForm.

    polyform support   --csv proteoforms.csv       --outdir out_support
    polyform weight    --csv proteoforms.csv       --outdir out_weight
    polyform topology  --xlsx atlas.xlsx           --outdir out_topology

Each subcommand runs the corresponding analysis, writes CSVs (and, unless
``--no-figures`` is passed, figures) to ``--outdir``, and prints a short
summary of the result tables.
"""
import argparse
import sys

from . import __version__


def _summarise(results):
    for key, df in results.items():
        try:
            print(f"  {key:28s} {len(df):>8d} rows x {len(df.columns):>3d} cols")
        except Exception:
            print(f"  {key:28s} {type(df).__name__}")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)

    parser = argparse.ArgumentParser(
        prog="polyform",
        description="Structured analysis of proteoform distributions on modal lattices.",
    )
    parser.add_argument("--version", action="version",
                        version=f"polyform {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    # support --------------------------------------------------------------
    p_sup = sub.add_parser("support", help="global support-mode structural metrics")
    p_sup.add_argument("--csv", default="proteoforms.csv",
                       help="proteoform catalogue CSV (default: proteoforms.csv)")
    p_sup.add_argument("--fasta", default=None,
                       help="optional UniProt FASTA to resolve lengths")
    p_sup.add_argument("--pos-base", type=int, default=0,
                       help="set to 1 if RESID positions are 1-based")
    p_sup.add_argument("--outdir", default="polyform_support_outputs")
    p_sup.add_argument("--dpi", type=int, default=300)
    p_sup.add_argument("--no-figures", action="store_true")

    # weight ---------------------------------------------------------------
    p_w = sub.add_parser("weight", help="demonstration occupancy metrics")
    p_w.add_argument("--csv", default="proteoforms.csv")
    p_w.add_argument("--fasta", default=None)
    p_w.add_argument("--pos-base", type=int, default=0)
    p_w.add_argument("--outdir", default="polyform_02_demo_occupancy_outputs")
    p_w.add_argument("--select-n", type=int, default=8)
    p_w.add_argument("--min-states", type=int, default=5)
    p_w.add_argument("--min-delta-k", type=int, default=3)
    p_w.add_argument("--dpi", type=int, default=300)
    p_w.add_argument("--seed", type=int, default=7)
    p_w.add_argument("--no-figures", action="store_true")

    # topology -------------------------------------------------------------
    p_t = sub.add_parser("topology", help="tissue-resolved support topology")
    p_t.add_argument("--xlsx", default="pr2c00034_si_002 (1).xlsx",
                     help="multi-tissue atlas spreadsheet")
    p_t.add_argument("--outdir", default="polyform_03_tissue_support_topology_outputs")
    p_t.add_argument("--sheet", default="All_Tissues")
    p_t.add_argument("--no-group-by-sequence", action="store_true")
    p_t.add_argument("--no-k0-anchor", action="store_true")
    p_t.add_argument("--dpi", type=int, default=300)
    p_t.add_argument("--seed", type=int, default=7)
    p_t.add_argument("--no-figures", action="store_true")

    args = parser.parse_args(argv)

    if args.command == "support":
        from .support_mode import run_support_mode
        results = run_support_mode(
            csv=args.csv, fasta_path=args.fasta, pos_base=args.pos_base,
            outdir=args.outdir, make_figures=not args.no_figures, dpi=args.dpi,
        )
    elif args.command == "weight":
        from .weight_mode import run_weight_mode
        results = run_weight_mode(
            csv=args.csv, fasta_path=args.fasta, pos_base=args.pos_base,
            outdir=args.outdir, select_n=args.select_n, min_states=args.min_states,
            min_delta_k=args.min_delta_k, make_figures=not args.no_figures,
            dpi=args.dpi, random_seed=args.seed,
        )
    elif args.command == "topology":
        from .topology import run_topology
        results = run_topology(
            xlsx=args.xlsx, outdir=args.outdir, sheet_name=args.sheet,
            group_by_sequence=not args.no_group_by_sequence,
            add_k0_anchor=not args.no_k0_anchor,
            make_figures=not args.no_figures, dpi=args.dpi, random_seed=args.seed,
        )
    else:  # pragma: no cover
        parser.error(f"unknown command {args.command!r}")

    print(f"\nResult tables (written to {args.outdir}):")
    _summarise(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
