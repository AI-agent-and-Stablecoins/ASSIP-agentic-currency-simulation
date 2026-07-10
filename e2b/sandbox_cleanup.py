"""Deletes finished sandboxes."""

from e2b.sandbox_manager import SandboxManager


def cleanup(manager: SandboxManager, handle_id: str) -> None:
    manager.destroy(handle_id)
