"""Schedules scenario shocks by the day they're due to fire."""

from src.economy.shocks import ShockEvent


class EventQueue:
    def __init__(self, shocks: list[ShockEvent]):
        self._by_day: dict[int, list[ShockEvent]] = {}
        for shock in shocks:
            self._by_day.setdefault(shock.day, []).append(shock)

    def pop_due(self, day: int) -> list[ShockEvent]:
        return self._by_day.pop(day, [])
