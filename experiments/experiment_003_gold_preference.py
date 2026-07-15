"""Experiment 003: Gold preference under a gold price rally.

Research question: when the gold price itself rallies (rather than fiat
inflation), do agents shift toward gold-backed currencies (PAXG, XAUT) as a
store of value, and is that shift concentrated among agents who already
hold gold (investors) or does it spread to the wider population?

Protocol:
    1. Run configs/scenarios/gold_boom.yaml (XAU price +25% on day 15 --
       see src/economy/shocks.py's GOLD_RALLY handling, which updates
       macro_state.gold_price and peg_reference_rates["XAU"]).
    2. Track the gold-backed adoption curve across the whole run.
    3. Split the ledger by agent risk_profile (investor agents hold PAXG/XAUT
       from configs/agent_profiles/investor.yaml; consumers/merchants don't)
       and compare gold-backed share within each group.

Uses configs/scenarios/gold_boom.yaml and existing metrics only.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments._common import (
    agent_ids_by_predicate,
    ledger_subset,
    load_simulation_config,
    make_run_dir,
    print_header,
    print_kv,
    run_simulation,
    save_run_artifacts,
    standard_metrics,
)
from metrics.currency_usage import market_share

EXPERIMENT_NAME = "experiment_003_gold_preference"
GOLD_BACKED_SYMBOLS = {"PAXG", "XAUT"}


def _gold_share(shares: dict[str, float]) -> float:
    return sum(v for symbol, v in shares.items() if symbol in GOLD_BACKED_SYMBOLS)


def main() -> None:
    parser = argparse.ArgumentParser(description=EXPERIMENT_NAME)
    parser.add_argument("--config", default="medium_test", choices=["small_test", "medium_test", "large_scale"])
    parser.add_argument("--persist", action="store_true")
    args = parser.parse_args()

    config = load_simulation_config(args.config).model_copy(update={"scenario": "gold_boom"})

    print_header("Experiment 003: Gold preference during a gold price rally")
    print_kv("agent_mix", config.agent_mix)

    env, sim_result = run_simulation(config, persist=args.persist)
    metrics = standard_metrics(env, sim_result.timesteps)

    gold_series = {day: _gold_share(shares) for day, shares in metrics["adoption_curve_series"].items()}
    print_kv("gold_share_series", gold_series)

    shock_day = 15
    pre = [gold_series[d] for d in gold_series if d < shock_day]
    post = [gold_series[d] for d in gold_series if d >= shock_day]
    pre_avg = sum(pre) / len(pre) if pre else 0.0
    post_avg = sum(post) / len(post) if post else 0.0
    print_kv("avg_gold_share_pre_rally", pre_avg)
    print_kv("avg_gold_share_post_rally", post_avg)

    if post_avg <= pre_avg:
        print(
            "  note: CRRA utility (src/utility/crra.py) weighs governance/liquidity/peg_error, not "
            "price appreciation -- a gold_rally shock moves peg_reference_rates but not those score "
            "fields, so Phase 1 rule-based agents may show no shift even though gold's USD value rose. "
            "This is a real, reportable finding about the current utility formulation, not a bug in "
            "this experiment; see the note in metrics/metrics.json."
        )

    investor_ids = agent_ids_by_predicate(env, lambda agent: agent.profile_name in {"investor", "institution"})
    non_investor_ids = set(env.agents.keys()) - investor_ids
    investor_gold_share = _gold_share(market_share(ledger_subset(env.ledger, investor_ids)))
    non_investor_gold_share = _gold_share(market_share(ledger_subset(env.ledger, non_investor_ids)))
    print_kv("gold_share_among_investors", investor_gold_share)
    print_kv("gold_share_among_non_investors", non_investor_gold_share)

    paths = make_run_dir(EXPERIMENT_NAME)
    save_run_artifacts(
        paths,
        config,
        env,
        sim_result.timesteps,
        metrics,
        extra_metadata={
            "research_question": "Does a gold price rally shift adoption toward gold-backed currencies, and for whom?",
            "shock_day": shock_day,
            "avg_gold_share_pre_rally": pre_avg,
            "avg_gold_share_post_rally": post_avg,
            "gold_share_among_investors": investor_gold_share,
            "gold_share_among_non_investors": non_investor_gold_share,
            "note": (
                "CRRA utility (src/utility/crra.py) evaluates governance_score * liquidity_score * "
                "(1 - peg_error), which a gold_rally shock does not change -- it only moves "
                "macro_state.gold_price / peg_reference_rates['XAU']. A null/small shift here is an "
                "expected consequence of the current utility formulation, not an experiment defect."
            ),
        },
    )
    print(f"\nSaved run artifacts to {paths['base']}")


if __name__ == "__main__":
    main()
