"""Optional Weights & Biases logging for simulation runs.

Not part of project_instructions.md's spec -- added because the team has a
W&B account and wants cross-run/cross-scenario comparison. wandb is
lazy-imported inside WandbRunLogger.__init__ so nothing else in the codebase
requires it installed (pip install -e ".[observability]" to use this).

Plugs into the same on_timestep hook SimulationRunner.run() already exposes
for database persistence -- no changes needed to simulation_runner.py.
"""

from typing import Optional

from metrics.chain_selection import chain_usage_share
from src.llm.hallucination_detector import HallucinationDirection, HallucinationResult
from metrics.compliance_effects import compliance_adoption_share
from metrics.currency_usage import market_share
from metrics.gas_fee_sensitivity import average_gas_fee_paid
from metrics.governance_preference import governance_preference
from metrics.liquidity_sensitivity import liquidity_sensitivity
from metrics.transaction_stats import negotiation_length, transaction_success_rate
from metrics.wealth_distribution import wealth_distribution_from_agents
from src.simulation.environment import Environment
from src.simulation.timestep import TimestepResult


class WandbRunLogger:
    def __init__(self, project: str = "assip-future-of-finance", run_name: Optional[str] = None, config: Optional[dict] = None):
        import wandb

        self._wandb = wandb
        self._run = wandb.init(project=project, name=run_name, config=config or {})

    def on_timestep(self, env: Environment, result: TimestepResult) -> None:
        self._wandb.log(
            {
                "market_share": market_share(env.ledger),
                "chain_usage_share": chain_usage_share(env.ledger),
                "transaction_success_rate": transaction_success_rate(env.ledger),
                "avg_gas_fee_paid": average_gas_fee_paid(env.ledger),
                "governance_preference": governance_preference(env.ledger, env.currencies),
                "compliance_adoption_share": compliance_adoption_share(env.ledger, env.currencies),
                "liquidity_sensitivity": liquidity_sensitivity(env.ledger, env.currencies),
                "negotiation_length": negotiation_length(result.negotiations),
                "wealth_distribution": wealth_distribution_from_agents(env.agents, env.exchange_rates),
            },
            step=result.day,
        )

    def log_llm_metrics(
        self,
        hallucination_results: list[HallucinationResult],
        model_attempts_by_call: list[list[str]],
        step: int,
    ) -> None:
        """Additive: logs LLM-path-specific metrics alongside whatever the
        caller already logs via on_timestep(). Only the LLM path calls this;
        Phase 1 rule-based runs never do."""
        total = len(hallucination_results)
        hallucination_rate = (
            sum(1 for result in hallucination_results if result.direction != HallucinationDirection.ACCURATE) / total
            if total
            else 0.0
        )
        fallback_rate = (
            sum(1 for attempts in model_attempts_by_call if len(attempts) > 1) / len(model_attempts_by_call)
            if model_attempts_by_call
            else 0.0
        )
        self._wandb.log(
            {"llm_hallucination_rate": hallucination_rate, "llm_fallback_rate": fallback_rate},
            step=step,
        )

    def finish(self) -> None:
        self._run.finish()
