"""Append-only record of every shock that has fired during a run.

Plain in-memory accumulator -- src/simulation/timestep.py has no database
dependency (see docs/superpowers/specs/2026-07-29-phase3-plan2-shock-engine-design.md
Sec 3.5), so persisting this to intervention_logs is a later plan's job,
reading TimestepResult.fired_shocks (this task's other change) rather than
this class directly.
"""

from src.economy.shocks import ShockEvent


# NOTE: EventLog has no production consumer as of Phase 3 Plan 2 -- it is
# exercised only by tests. Plan 3/4 (agent population + matrix runner) is
# expected to instantiate EventLog for real runs (e.g. to back a stdev()
# helper for perceived trust). This is an intentional, recorded hand-off,
# not dead code left by oversight.
class EventLog:
    def __init__(self) -> None:
        self._events: list[ShockEvent] = []

    def record(self, shock: ShockEvent) -> None:
        self._events.append(shock)

    def all_events(self) -> list[ShockEvent]:
        return list(self._events)
