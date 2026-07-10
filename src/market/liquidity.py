"""Order-book depth tracking per (good, currency), used to flag thin markets."""


class MarketLiquidity:
    def __init__(self):
        self._depth: dict[tuple[str, str], float] = {}

    def set_depth(self, good_name: str, currency_symbol: str, depth: float) -> None:
        self._depth[(good_name, currency_symbol)] = depth

    def get_depth(self, good_name: str, currency_symbol: str, default: float = 1.0) -> float:
        return self._depth.get((good_name, currency_symbol), default)

    def is_thin(self, good_name: str, currency_symbol: str, threshold: float = 0.1) -> bool:
        return self.get_depth(good_name, currency_symbol) < threshold
