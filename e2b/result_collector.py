"""Pulls simulation results back from a sandbox handle."""

from e2b.sandbox_manager import SandboxHandle
from src.simulation.simulation_runner import SimulationResult


def collect(handle: SandboxHandle) -> SimulationResult:
    if handle.result is None:
        raise RuntimeError(f"Sandbox {handle.handle_id} has no result yet")
    return handle.result
