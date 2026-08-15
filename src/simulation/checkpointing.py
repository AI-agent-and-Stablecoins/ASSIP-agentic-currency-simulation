import pickle
import random
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from src.simulation.environment import Environment
from src.simulation.timestep import TimestepResult


class CellSeedCheckpoint(BaseModel):
    """Pickled per-cell/seed resumability snapshot -- see `run_matrix`'s
    `checkpoint_dir` docstring paragraph. Not a database model: this is a
    side-channel file, not a persisted table, since it holds live Python
    objects (`Environment`, `random.Random`) with no natural relational
    shape."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    env: Environment
    rng: random.Random
    next_day: int
    daily_results: list[TimestepResult]
    num_days_completed: int
    total_transactions: int
    total_llm_decisions: int


def checkpoint_path(checkpoint_dir: Path, run_id: str) -> Path:
    return checkpoint_dir / f"{run_id}.pkl"


def save_checkpoint(checkpoint_dir: Path, run_id: str, checkpoint: CellSeedCheckpoint) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_path(checkpoint_dir, run_id)
    tmp_path = path.with_suffix(".pkl.tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump(checkpoint, f)
    tmp_path.replace(path)  # atomic on both POSIX and Windows -- never leaves a half-written checkpoint


def load_checkpoint(checkpoint_dir: Path, run_id: str) -> CellSeedCheckpoint | None:
    path = checkpoint_path(checkpoint_dir, run_id)
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def delete_checkpoint(checkpoint_dir: Path, run_id: str) -> None:
    path = checkpoint_path(checkpoint_dir, run_id)
    path.unlink(missing_ok=True)
