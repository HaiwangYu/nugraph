#!/usr/bin/env python3
"""Command-line entry point for converting WCML NPZ files into NuGraph HDF5 datasets."""
from __future__ import annotations

import argparse
from pathlib import Path

from pywcml import ConversionConfig, WCMLConverter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path,
                        help="Path to a WCML NPZ file or a directory containing NPZ files")
    parser.add_argument("output", type=Path,
                        help="Destination HDF5 file to create")
    parser.add_argument("--tolerance", type=float, default=5.0,
                        help="Drift-distance tolerance (mm) for associating CTPC points with a blob [default: %(default)s]")
    parser.add_argument("--projection-tolerance", type=float, default=10.0,
                        help="Acceptable mm difference between projected blob corners and CTPC wires")
    parser.add_argument("--train-frac", type=float, default=0.7,
                        help="Fraction of samples assigned to the training split when batching multiple files")
    parser.add_argument("--val-frac", type=float, default=0.15,
                        help="Fraction of samples assigned to the validation split")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ConversionConfig(
        x_tolerance=args.tolerance,
        projection_tolerance=args.projection_tolerance,
        train_fraction=args.train_frac,
        val_fraction=args.val_frac,
    )
    converter = WCMLConverter(config)
    if args.source.is_dir():
        graphs = converter.convert_many(sorted(args.source.glob("*.npz")))
        converter.write_hdf5(graphs, args.output)
    else:
        name, graph = converter.convert(args.source)
        converter.write_hdf5({name: graph}, args.output)


if __name__ == "__main__":
    main()
