"""Shared logging configuration.

Every module should call get_logger(__name__) instead of configuring its
own handlers, so log formatting stays consistent and handlers are never
attached more than once (which would duplicate every log line).
"""

import logging
import os

from src.utils.constants import LOG_FORMAT

_CONFIGURED = False


def _configure_root() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=getattr(logging, level_name, logging.INFO), format=LOG_FORMAT)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger with the project's shared configuration."""
    _configure_root()
    return logging.getLogger(name)
