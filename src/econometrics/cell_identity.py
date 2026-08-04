"""Recovers which of the 13 matrix-runner cells a persisted
`LLMDecisionRecord` row came from.

`LLMDecisionRecord.scenario`/`.domestic_or_cross_border` cannot distinguish
all 13 cells (see docs/superpowers/specs/
2026-08-02-phase3-plan5-econometrics-design.md Sec 1: `domestic_or_cross_
border` is unconditionally "unknown" in production, and a sandbox's
domestic/cross-border cells share the same `scenario` value) -- only
`.simulation_id` (== the matrix runner's `run_id`,
"{matrix_run_id}-{cell_key}-seed{seed}") does.
"""

import re

from src.simulation.matrix_runner import _build_cell_specs

# Sorted longest-first so a shorter key can never accidentally match before
# a longer one that shares a suffix/prefix.
_VALID_CELL_KEYS = sorted((spec.key for spec in _build_cell_specs()), key=len, reverse=True)

_SEED_SUFFIX = re.compile(r"-seed\d+$")


def cell_key_from_run_id(run_id: str) -> str:
    """Extracts the matrix-runner cell key (e.g. "master",
    "liquidity_vs_governance_domestic") from a `run_id` of the form
    "{matrix_run_id}-{cell_key}-seed{seed}". `matrix_run_id` itself may
    contain hyphens (it's caller-supplied or `generate_id`-produced), so
    this matches against the KNOWN set of 13 valid cell keys rather than
    naively splitting on "-". Raises `ValueError` if no known cell key
    matches (e.g. a `run_id` from outside `run_matrix`).
    """
    seed_match = _SEED_SUFFIX.search(run_id)
    prefix = run_id[: seed_match.start()] if seed_match else run_id
    for key in _VALID_CELL_KEYS:
        if prefix.endswith(f"-{key}"):
            return key
    raise ValueError(f"run_id {run_id!r} does not match any known matrix-runner cell key")
