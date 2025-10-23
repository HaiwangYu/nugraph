# pynuml/__init__.py
"""Standardised ML input processing for particle physics"""
from __future__ import annotations
import os as _os, importlib as _importlib

__version__ = "25.4.dev0"

# If disabled, expose nothing and avoid importing heavy deps at import-time.
if _os.getenv("PYNUML_DISABLE", "").lower() in ("1", "true", "yes", "y"):
    __all__: list[str] = []
else:
    # Lazy: only import subpackages when actually accessed
    __all__ = ["io", "labels", "data", "process", "plot"]

    def __getattr__(name: str):
        if name in __all__:
            return _importlib.import_module(f".{name}", __name__)
        raise AttributeError(name)
