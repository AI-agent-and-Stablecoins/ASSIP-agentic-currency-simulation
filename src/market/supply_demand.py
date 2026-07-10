"""Simple supply/demand price elasticity."""


def adjust_price(base_price: float, supply: float, demand: float) -> float:
    if supply <= 0:
        raise ValueError("supply must be positive")
    return base_price * (demand / supply)
