import hashlib
import subprocess
from pathlib import Path

from src.agents.base_agent import BaseAgent


def compute_git_commit_hash() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def compute_config_hash(paths: list[Path]) -> str:
    hasher = hashlib.sha256()
    for path in sorted(paths):
        hasher.update(path.read_bytes())
    return hasher.hexdigest()


def model_roster_summary_for(agents: list[BaseAgent]) -> str:
    distinct_models = {a.assigned_model for a in agents if a.assigned_model is not None}
    return f"{len(agents)} agents across {len(distinct_models)} OpenRouter models"
