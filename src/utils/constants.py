"""Non-economic constants shared across the codebase.

No economic values (fees, weights, scores) belong here -- those are
config-driven per the project's "no hardcoded economic constants" rule.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "configs"

DEFAULT_RANDOM_SEED = 42
DEFAULT_DATABASE_URL = "sqlite:///./assip.db"
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
