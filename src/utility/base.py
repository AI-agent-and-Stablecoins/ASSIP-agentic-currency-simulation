"""Shared abstract interface for every utility function.

Not part of the original module list, but every utility implementation
(CRRA, CARA, multi-attribute) needs a common interface for agents and the
routing engine to depend on -- kept here rather than duplicated per file.
"""

from abc import ABC, abstractmethod

from src.blockchain.routing_engine import CurrencyChainOption


class UtilityFunction(ABC):
    @abstractmethod
    def evaluate(self, option: CurrencyChainOption, **kwargs: float) -> float:
        """Higher is better. kwargs carries agent-specific context (e.g. wealth)."""


def choose_best(options: list[CurrencyChainOption], utility_fn: UtilityFunction, **kwargs: float) -> CurrencyChainOption:
    if not options:
        raise ValueError("No candidate currency/chain options to choose from")
    return max(options, key=lambda option: utility_fn.evaluate(option, **kwargs))
