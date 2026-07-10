"""Creates and tracks real E2B cloud sandboxes.

Wraps the e2b SDK behind a small interface so callers (sandbox_launcher,
result_collector, sandbox_cleanup) don't touch the SDK directly.

This folder was renamed from e2b/ to sandbox/: the e2b PyPI package itself
is imported as `from e2b import Sandbox`, which collided with this folder
once it was also named e2b/ and the repo root sat on sys.path (needed for
`from src...`/`from database...` imports everywhere else) -- two top-level
packages can't share the name `e2b` in the same environment.
"""

import os
from typing import Optional

from e2b import Sandbox

DEFAULT_TEMPLATE = "base"
DEFAULT_TIMEOUT_SECONDS = 300
SANDBOX_ROOT = "/home/user/assip"


class SandboxManager:
    def __init__(self, api_key: Optional[str] = None, template: str = DEFAULT_TEMPLATE):
        self.api_key = api_key or os.getenv("E2B_API_KEY")
        self.template = template
        self._sandboxes: dict[str, Sandbox] = {}

    def create_sandbox(self, timeout: int = DEFAULT_TIMEOUT_SECONDS) -> Sandbox:
        sandbox = Sandbox.create(template=self.template, timeout=timeout, api_key=self.api_key)
        self._sandboxes[sandbox.sandbox_id] = sandbox
        return sandbox

    def get(self, sandbox_id: str) -> Sandbox:
        return self._sandboxes[sandbox_id]

    def destroy(self, sandbox_id: str) -> None:
        sandbox = self._sandboxes.pop(sandbox_id, None)
        if sandbox is not None:
            sandbox.kill(api_key=self.api_key)
