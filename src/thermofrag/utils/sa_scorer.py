"""Synthetic accessibility (SA) score wrapper.

Re-exports ``calculateScore`` from RDKit's ``Contrib/SA_Score/sascorer.py``
(Ertl & Schuffenhauer, J. Cheminform. 2009). The Contrib directory is not on
``sys.path`` by default in an anaconda RDKit install, so we add it here.

The canonical SA score is in [1, 10] with lower = easier to synthesize. We
return the raw score; call sites decide whether to normalize.
"""
from __future__ import annotations

import os
import sys

import rdkit

_CONTRIB = os.path.join(os.path.dirname(rdkit.__file__), "Contrib", "SA_Score")
if _CONTRIB not in sys.path:
    sys.path.insert(0, _CONTRIB)

import sascorer  # noqa: E402  (path-dependent)

calculateScore = sascorer.calculateScore


def sa_score(mol) -> float:
    """SA score of an RDKit Mol, in [1, 10]. Lower is easier."""
    return float(calculateScore(mol))
