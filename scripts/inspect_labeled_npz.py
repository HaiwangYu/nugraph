#!/usr/bin/env python3
"""Inspect arrays stored in a labeled or inference-only WCML NPZ file."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path, help="Path to a WCML NPZ file")
    args = parser.parse_args()

    if not args.file.is_file():
        raise SystemExit(f"File not found: {args.file}")

    with np.load(args.file, allow_pickle=False) as data:
        print(f"=== {args.file} ===")
        print("Keys:")
        for key in data.files:
            array = data[key]
            print(f"  - {key}: shape={array.shape}, dtype={array.dtype}")

        vertex_keys = [
            key
            for key in data.files
            if any(token in key.lower() for token in ("vtx", "vertex", "nu_v", "nuvtx"))
        ]
        print("\nVertex-like keys:")
        if not vertex_keys:
            print("  (none)")
        for key in vertex_keys:
            array = data[key]
            print(f"  - {key}: shape={array.shape}, dtype={array.dtype}")
            if array.size <= 20:
                print(f"    values={array}")


if __name__ == "__main__":
    main()
