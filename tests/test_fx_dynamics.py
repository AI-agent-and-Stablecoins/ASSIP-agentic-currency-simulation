from src.economy.fx_dynamics import advance_eur_usd_rate
from src.economy.macro_state import MacroState


def test_advance_eur_usd_rate_perturbs_the_rate_by_a_small_amount():
    state = MacroState()
    updated = advance_eur_usd_rate(state, day=5)

    original = state.peg_reference_rates["EUR"]
    new_rate = updated.peg_reference_rates["EUR"]
    assert new_rate != original
    # A single day's move should be small (well under 5%), not a jump.
    assert abs(new_rate - original) / original < 0.05


def test_advance_eur_usd_rate_is_deterministic_per_day():
    state = MacroState()
    first = advance_eur_usd_rate(state, day=42)
    second = advance_eur_usd_rate(state, day=42)
    assert first.peg_reference_rates["EUR"] == second.peg_reference_rates["EUR"]


def test_advance_eur_usd_rate_differs_across_days():
    state = MacroState()
    day_1 = advance_eur_usd_rate(state, day=1)
    day_2 = advance_eur_usd_rate(state, day=2)
    assert day_1.peg_reference_rates["EUR"] != day_2.peg_reference_rates["EUR"]


def test_advance_eur_usd_rate_does_not_mutate_the_input_state():
    state = MacroState()
    original_rate = state.peg_reference_rates["EUR"]
    advance_eur_usd_rate(state, day=10)
    assert state.peg_reference_rates["EUR"] == original_rate


def test_advance_eur_usd_rate_only_touches_eur_not_other_pegs():
    state = MacroState()
    updated = advance_eur_usd_rate(state, day=7)
    assert updated.peg_reference_rates["USD"] == state.peg_reference_rates["USD"]
    assert updated.peg_reference_rates["XAU"] == state.peg_reference_rates["XAU"]


def test_advance_eur_usd_rate_produces_real_variance_over_many_days():
    """The whole point: a rolling window of daily rates must have genuine,
    non-zero variance -- confirming this isn't just noise that happens to
    cancel out to a constant."""
    import statistics

    state = MacroState()
    rates = [advance_eur_usd_rate(state, day=d).peg_reference_rates["EUR"] for d in range(30)]
    assert statistics.stdev(rates) > 0.0
