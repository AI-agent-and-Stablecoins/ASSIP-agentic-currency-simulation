"""Experiment 004: Bank run / banking crisis.

Research question: during a banking-confidence shock, do agents flee
bank-issued tokenized deposits (TDUSD, FDIC-insured but bank-linked) toward
non-bank stablecoins or gold-backed currencies?

Protocol:
    1. Run configs/scenarios/banking_crisis.yaml (a BANK_FAILURE shock on day
       5, magnitude 0.30 -- see src/economy/shocks.py, which reduces
       macro_state.confidence_index).
    2. Track TDUSD's (Tokenized_Deposits.yaml) adoption share before/after
       day 5, alongside overall market share.

Important, honestly-reported limitation: as of this backend version,
confidence_index is set by BANK_FAILURE shocks (src/economy/shocks.py) but
is not read by any utility function, currency config, or routing logic
(grep confirms the only references are macro_state.py's field definition
and shocks.py's assignment). So Phase 1 rule-based agents cannot yet react
to a confidence shock -- this experiment establishes that baseline/null
result and documents the extension point (wiring confidence_index into
governance_score or into a utility term) for whoever implements Phase 2/3.
This script does not attempt that wiring itself, since src/ is out of scope
for this layer.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments._common import (
    load_simulation_config,
    make_run_dir,
    print_header,
    print_kv,
    run_simulation,
    save_run_artifacts,
    standard_metrics,
)

EXPERIMENT_NAME = "experiment_004_bank_run"
BANK_LINKED_SYMBOL = "TDUSD"


def main() -> None:
    parser = argparse.ArgumentParser(description=EXPERIMENT_NAME)
    parser.add_argument("--config", default="medium_test", choices=["small_test", "medium_test", "large_scale"])
    parser.add_argument("--persist", action="store_true")
    args = parser.parse_args()

    config = load_simulation_config(args.config).model_copy(update={"scenario": "banking_crisis"})

    print_header("Experiment 004: Bank run / banking crisis")
    print_kv("agent_mix", config.agent_mix)

    env, sim_result = run_simulation(config, persist=args.persist)
    metrics = standard_metrics(env, sim_result.timesteps)

    tdusd_series = {
        day: shares.get(BANK_LINKED_SYMBOL, 0.0) for day, shares in metrics["adoption_curve_series"].items()
    }
    print_kv("tdusd_share_series", tdusd_series)

    shock_day = 5
    pre = [tdusd_series[d] for d in tdusd_series if d < shock_day]
    post = [tdusd_series[d] for d in tdusd_series if d >= shock_day]
    pre_avg = sum(pre) / len(pre) if pre else 0.0
    post_avg = sum(post) / len(post) if post else 0.0
    print_kv("avg_tdusd_share_pre_shock", pre_avg)
    print_kv("avg_tdusd_share_post_shock", post_avg)
    print_kv(
        "confidence_index_final",
        env.macro_state.confidence_index,
    )
    print(
        "  note: confidence_index is currently unread by any utility function or currency config "
        "(see this file's module docstring) -- a flat tdusd_share before/after is the expected, "
        "documented Phase 1 result, not a bug in this experiment."
    )

    paths = make_run_dir(EXPERIMENT_NAME)
    save_run_artifacts(
        paths,
        config,
        env,
        sim_result.timesteps,
        metrics,
        extra_metadata={
            "research_question": "Do agents flee bank-linked tokenized deposits during a confidence shock?",
            "shock_day": shock_day,
            "confidence_index_final": env.macro_state.confidence_index,
            "avg_tdusd_share_pre_shock": pre_avg,
            "avg_tdusd_share_post_shock": post_avg,
            "known_limitation": (
                "confidence_index is set by BANK_FAILURE shocks but not consumed by any utility "
                "function, currency config, or routing logic in this backend version -- Phase 1 "
                "rule-based agents cannot yet react to it. Wiring confidence_index into "
                "governance_score or a utility term is a Phase 2/3 extension point, not implemented "
                "by this experiments/ layer."
            ),
        },
    )
    print(f"\nSaved run artifacts to {paths['base']}")


if __name__ == "__main__":
    main()
