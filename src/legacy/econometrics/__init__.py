"""Archived (2026-08-14): the clustered-logistic-regression hypothesis-
testing pipeline built in Phase 3 Plans 5-6, per docs/superpowers/specs/
2026-08-02-phase3-plan5-econometrics-design.md and 2026-08-04-phase3-plan6-
concurrency-and-sandbox-hypotheses-design.md.

Superseded by the research-methodology pivot from `New info.pdf` (see
docs/superpowers/specs/2026-08-14-hypothesis-sandboxes-pivot-design.md
§0): equilibrium-holdings tables (src/economy/equilibrium_holdings.py) and
the equivalence/indifference-search framework (src/economy/
equivalence_framework.py) now cover this research ground directly against
a live post-run Environment, rather than via regression over persisted
LLMDecisionRecord rows.

Not called by matrix_runner or any other active runner path -- kept here,
still independently testable (tests/legacy/), for reference. Its H1-H5 and
H7-H11 are the OLD hypothesis numbering (H1-H5 from the original master-
simulation design; H7-H11 a since-superseded "sandbox-preference" set from
Plan 6) -- NOT the same hypotheses as the pivot's new H1-H11.
"""
