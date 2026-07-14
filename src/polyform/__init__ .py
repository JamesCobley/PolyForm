"""
PolyForm
========

Structured analysis of proteoform distributions on finite, bounded, k-graded
modal lattices.

PolyForm maps measured proteoform catalogues onto native modal lattices and
computes structure-derived metrics: state identity, occupancy, grade spread,
support topology, native clustering, principal occupied components, and
statistical surprise.

Public entry points
-------------------
    run_support_mode(...)   support/access metrics on the choice-free lattice
    run_weight_mode(...)    demonstration occupancy metrics on observed supports
    run_topology(...)       tissue-resolved Hamming-1 support topology

Each returns a dict of pandas DataFrames and, by default, writes CSVs and
figures to an output directory.

Core primitives (lattice parsing, Hamming topology, information measures) are
available under ``polyform.core``.
"""
from .support_mode import run_support_mode
from .weight_mode import run_weight_mode
from .topology import run_topology
from . import core

__version__ = "0.1.1"

__all__ = [
    "run_support_mode",
    "run_weight_mode",
    "run_topology",
    "core",
    "__version__",
]
