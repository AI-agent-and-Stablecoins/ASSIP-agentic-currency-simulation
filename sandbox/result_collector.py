"""Pulls simulation results back from a sandbox."""

from e2b import Sandbox

from sandbox.sandbox_manager import SANDBOX_ROOT
from src.simulation.simulation_runner import SimulationResult


def collect(sandbox: Sandbox) -> SimulationResult:
    raw = sandbox.files.read(f"{SANDBOX_ROOT}/result.json", format="text")
    return SimulationResult.model_validate_json(raw)
