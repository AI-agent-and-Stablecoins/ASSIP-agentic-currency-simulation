"""Creates and tracks sandbox handles.

Phase 1 stub: runs simulations as a local in-process call rather than a real
E2B sandbox, behind the same interface a real E2B-SDK-backed implementation
would expose, so swapping in real sandboxes later doesn't change callers.
"""

from dataclasses import dataclass
from typing import Any, Optional

from src.utils.helpers import generate_id


@dataclass
class SandboxHandle:
    handle_id: str
    result: Optional[Any] = None


class SandboxManager:
    def __init__(self):
        self._handles: dict[str, SandboxHandle] = {}

    def create_sandbox(self) -> SandboxHandle:
        handle = SandboxHandle(handle_id=generate_id("sandbox"))
        self._handles[handle.handle_id] = handle
        return handle

    def get(self, handle_id: str) -> SandboxHandle:
        return self._handles[handle_id]

    def destroy(self, handle_id: str) -> None:
        self._handles.pop(handle_id, None)
