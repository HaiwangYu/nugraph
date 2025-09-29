#!/usr/bin/env python3
"""Thin wrapper around ``pywcml.cli`` for backward-compatible script usage."""
from __future__ import annotations

from pywcml.cli import main


if __name__ == "__main__":  # pragma: no cover
    main()
