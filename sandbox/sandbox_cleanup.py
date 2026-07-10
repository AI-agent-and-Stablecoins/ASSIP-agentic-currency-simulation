"""Deletes finished sandboxes."""

from sandbox.sandbox_manager import SandboxManager


def cleanup(manager: SandboxManager, sandbox_id: str) -> None:
    manager.destroy(sandbox_id)
