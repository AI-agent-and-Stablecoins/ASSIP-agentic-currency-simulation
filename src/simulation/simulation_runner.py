"""The main entry point: orchestrates the day loop.

Persistence (lifecycle step 14) is not hardwired here -- callers (e2b's
sandbox_launcher, or a notebook/script) pass an on_timestep callback that
writes to the database, so this module has no dependency on database/.
"""

import random
from typing import Callable, Optional

from pydantic import BaseModel, Field

from src.simulation.environment import Environment
from src.simulation.timestep import TimestepResult, run_timestep
from src.utils.constants import DEFAULT_RANDOM_SEED


class SimulationConfig(BaseModel):
    agent_mix: dict[str, int]
    num_days: int
    scenario: str
    random_seed: int = DEFAULT_RANDOM_SEED
    max_negotiation_rounds: int = 10
    agreement_tolerance: float = 0.01
    concession_rate: float = 0.3


class SimulationResult(BaseModel):
    config: SimulationConfig
    timesteps: list[TimestepResult] = Field(default_factory=list)


class SimulationRunner:
    def run(
        self,
        config: SimulationConfig,
        on_timestep: Optional[Callable[[Environment, TimestepResult], None]] = None,
    ) -> SimulationResult:
        env = Environment.build(config.scenario, config.agent_mix)
        rng = random.Random(config.random_seed)

        timesteps: list[TimestepResult] = []
        for day in range(config.num_days):
            result = run_timestep(
                env,
                day,
                rng,
                max_negotiation_rounds=config.max_negotiation_rounds,
                agreement_tolerance=config.agreement_tolerance,
                concession_rate=config.concession_rate,
            )
            timesteps.append(result)
            if on_timestep is not None:
                on_timestep(env, result)

        return SimulationResult(config=config, timesteps=timesteps)
