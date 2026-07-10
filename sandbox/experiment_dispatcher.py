"""Dispatches multiple experiment configs across multiple sandboxes.

Not implemented: this conceptually belongs to orchestrating experiments/*.py
files, and experiments/ is explicitly out of scope for this backend
implementation phase.
"""


def dispatch_experiments(*args, **kwargs):
    raise NotImplementedError("Experiment dispatch is out of scope -- see experiments/ (excluded from this phase)")
