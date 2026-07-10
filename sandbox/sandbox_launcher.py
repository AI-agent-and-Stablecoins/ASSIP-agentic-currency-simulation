"""Starts a simulation inside a real E2B sandbox.

Packages the repo's business-logic directories (src/, database/, metrics/,
configs/) into the sandbox and runs a tiny driver script that imports
simulation_runner and executes it there. No simulation business logic lives
here -- this only marshals a SimulationConfig in and a SimulationResult out,
per the requirement that business logic stays outside sandbox wrappers.
"""

import io
import tarfile
from typing import Optional

from e2b import Sandbox

from sandbox.sandbox_manager import SANDBOX_ROOT, SandboxManager
from src.simulation.simulation_runner import SimulationConfig
from src.utils.constants import REPO_ROOT

_CODE_DIRS = ("src", "database", "metrics", "configs")

_DRIVER_SCRIPT = """
import sys
sys.path.insert(0, "{root}")
from src.simulation.simulation_runner import SimulationConfig, SimulationRunner

with open("{config_path}") as f:
    config = SimulationConfig.model_validate_json(f.read())
result = SimulationRunner().run(config)
with open("{result_path}", "w") as f:
    f.write(result.model_dump_json())
""".strip()


def _exclude_bytecode(tarinfo: tarfile.TarInfo) -> Optional[tarfile.TarInfo]:
    if "__pycache__" in tarinfo.name or tarinfo.name.endswith((".pyc", ".pyo")):
        return None
    return tarinfo


def _package_repo_code() -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for dirname in _CODE_DIRS:
            tar.add(REPO_ROOT / dirname, arcname=dirname, filter=_exclude_bytecode)
    return buffer.getvalue()


def _provision(sandbox: Sandbox) -> None:
    sandbox.commands.run(f"mkdir -p {SANDBOX_ROOT}", timeout=30)
    sandbox.files.write(f"{SANDBOX_ROOT}/code.tar.gz", _package_repo_code())
    sandbox.commands.run(f"tar -xzf {SANDBOX_ROOT}/code.tar.gz -C {SANDBOX_ROOT}", timeout=60)
    sandbox.commands.run("pip install pydantic sqlalchemy pyyaml python-dotenv pandas", timeout=180)


def launch(config: SimulationConfig, manager: Optional[SandboxManager] = None) -> Sandbox:
    manager = manager or SandboxManager()
    sandbox = manager.create_sandbox()
    _provision(sandbox)

    config_path = f"{SANDBOX_ROOT}/config.json"
    result_path = f"{SANDBOX_ROOT}/result.json"
    driver_path = f"{SANDBOX_ROOT}/driver.py"

    sandbox.files.write(config_path, config.model_dump_json())
    sandbox.files.write(
        driver_path,
        _DRIVER_SCRIPT.format(root=SANDBOX_ROOT, config_path=config_path, result_path=result_path),
    )
    sandbox.commands.run(f"cd {SANDBOX_ROOT} && python3 driver.py", timeout=280)

    return sandbox
