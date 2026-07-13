# PolyForm

PolyForm is a Python package for structured analysis of proteoform
distributions. It maps measured proteoform catalogues onto finite, bounded,
*k*-graded modal lattices and computes structure-derived metrics including
state identity, occupancy, grade spread, support topology, native clustering,
principal occupied components, and statistical surprise.

## Install

```bash
pip install polyform-lattice
```

> The **import** name is `polyform`; the **PyPI distribution** name is
> `polyform-lattice` (the plain name `polyform` was already taken by an
> unrelated project). So you `pip install polyform-lattice` but `import polyform`.

From source (for development):

```bash
git clone https://github.com/JamesCobley/PolyForm.git
cd PolyForm
pip install -e .
```

## Usage

### Python API

Each analysis is a single call that returns a dict of pandas DataFrames and,
by default, writes CSVs and figures to an output directory.

```python
import polyform

# Global support-mode structural metrics on the choice-free modal lattice
support = polyform.run_support_mode(csv="proteoforms.csv", outdir="out_support")
support["global"]           # one-row global summary
support["protein_metrics"]  # protein-level structural metrics
support["grade"]            # protein-by-grade metrics

# Demonstration occupancy metrics on real observed supports
weight = polyform.run_weight_mode(csv="proteoforms.csv", outdir="out_weight")
weight["selected"]          # proteins selected for demonstration
weight["metrics"]           # occupancy-dependent metrics

# Tissue-resolved Hamming-1 support topology on a top-down atlas
topology = polyform.run_topology(xlsx="atlas.xlsx", outdir="out_topology")
topology["tissue_summary"]
topology["components"]
```

Set `make_figures=False` to skip figure rendering (faster; CSVs still written).

### Command line

```bash
polyform support   --csv proteoforms.csv --outdir out_support
polyform weight    --csv proteoforms.csv --outdir out_weight
polyform topology  --xlsx atlas.xlsx     --outdir out_topology

polyform support --help      # per-command options
polyform --version
```

### Core primitives

The lattice/topology/information primitives are importable directly:

```python
from polyform.core import (
    parse_ptms, read_fasta, log10_comb, hamming_distance_state,
    connected_components_for_states, shannon_entropy, normalized_entropy,
)
```

## Inputs

- **Support / weight modes** expect a proteoform catalogue CSV with at least
  `Entry Accession` and `PTMs` columns; protein lengths are resolved from an
  `Isoform Sequence` column or an optional `--fasta`. PTM strings use the
  `RESID:<code>@<position>` form, `|`-separated.
- **Topology mode** expects the multi-tissue top-down atlas spreadsheet
  (`--sheet`, default `All_Tissues`).

## Outputs

Each mode writes a set of `.csv` tables plus publication `.png`/`.pdf` figures
into its `--outdir`. See each function's docstring for the exact file list.

## Reproducibility

The original manuscript analysis scripts are preserved unchanged under
[`manuscript/`](manuscript/) as the provenance record. The packaged `run_*`
functions wrap those exact computational bodies (only parameterising
configuration and routing outputs) and were validated to reproduce the
manuscript reference outputs.

## Releasing (maintainers)

Releases publish to PyPI via **Trusted Publishing** (OIDC — no API tokens).
One-time setup: on PyPI, add a trusted publisher for the project pointing at
this repo's `publish.yml` workflow and a `pypi` environment.

To cut a release:

```bash
# bump version in pyproject.toml and src/polyform/__init__.py, update CHANGELOG.md
git commit -am "Release v0.1.0"
git tag v0.1.0
git push origin main --tags
```

Pushing the tag triggers `.github/workflows/publish.yml`, which builds the
sdist + wheel, runs `twine check`, and publishes to PyPI.

## License

MIT © 2026 James N. Cobley
