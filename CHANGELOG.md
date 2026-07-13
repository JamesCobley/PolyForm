# Changelog

All notable changes to this project are documented here. This project adheres
to [Semantic Versioning](https://semver.org/).

## [0.1.0] — 2026-07-13

### Added
- First packaged release of PolyForm.
- `polyform.run_support_mode` — global support-mode structural metrics on the
  choice-free modal lattice (state access, grade distributions, fibre refinement).
- `polyform.run_weight_mode` — demonstration occupancy metrics on real observed
  proteoform supports.
- `polyform.run_topology` — tissue-resolved Hamming-1 support topology on a
  top-down human atlas (empirical + k=0-anchored), including accession-level
  sharing and lattice heatmaps.
- `polyform.core` — pure, importable primitives (PTM parsing, FASTA reading,
  Hamming topology, information measures) copied verbatim from the manuscript
  analysis scripts.
- `polyform` command-line interface with `support`, `weight`, and `topology`
  subcommands.

### Notes
- The computational bodies of the original manuscript scripts (in `manuscript/`)
  are preserved unchanged as the provenance record. The packaged functions were
  validated to reproduce their reference outputs exactly.
