"""Starts a simulation inside a sandbox.

No simulation business logic lives here -- this only marshals a
SimulationConfig in and a SimulationResult out, per the requirement that
business logic stays outside e2b wrappers.
"""

from typing import Optional

from e2b.sandbox_manager import SandboxHandle, SandboxManager
from src.simulation.simulation_runner import SimulationConfig, SimulationRunner


def launch(config: SimulationConfig, manager: Optional[SandboxManager] = None) -> SandboxHandle:
    manager = manager or SandboxManager()
    handle = manager.create_sandbox()
    handle.result = SimulationRunner().run(config)
    return handle
