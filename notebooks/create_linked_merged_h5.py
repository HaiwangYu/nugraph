#!/usr/bin/env python
import argparse
from pathlib import Path

import h5py
import numpy as np


def read_split(file_handle, split):
    if split == "validation":
        key = "validation" if "validation" in file_handle["samples"] else "val"
    else:
        key = split
    return [str(value) for value in file_handle[f"samples/{key}"].asstr()[()]]


def copy_metadata(input_file, output_file):
    for key in ("planes", "semantic_classes", "gen", "event_classes"):
        if key in input_file and key not in output_file:
            input_file.copy(input_file[key], output_file, key)


def main():
    parser = argparse.ArgumentParser(
        description="Create a merged NuGraph H5 using external links to source H5 files."
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("inputs", nargs="+")
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")

    merged_splits = {"train": [], "validation": [], "test": []}

    with h5py.File(output, "w", libver="latest") as fout:
        dataset = fout.create_group("dataset")

        for file_index, input_name in enumerate(args.inputs, start=1):
            input_path = Path(input_name).resolve()
            prefix = input_path.stem
            print(f"[links] Adding {input_path}", flush=True)

            with h5py.File(input_path, "r") as fin:
                copy_metadata(fin, fout)
                source_splits = {
                    "train": read_split(fin, "train"),
                    "validation": read_split(fin, "validation"),
                    "test": read_split(fin, "test"),
                }

                for split, names in source_splits.items():
                    for name in names:
                        linked_name = f"{prefix}__{name}"
                        dataset[linked_name] = h5py.ExternalLink(str(input_path), f"/dataset/{name}")
                        merged_splits[split].append(linked_name)

                print(
                    "[links]   "
                    + " ".join(f"{split}={len(names)}" for split, names in source_splits.items()),
                    flush=True,
                )

        string_dtype = h5py.string_dtype(encoding="utf-8")
        samples = fout.create_group("samples")
        for split, names in merged_splits.items():
            samples.create_dataset(split, data=np.asarray(names, dtype=object), dtype=string_dtype)

        datasize = fout.create_group("datasize")
        datasize.create_dataset("train", data=np.ones(len(merged_splits["train"]), dtype=np.int64))
        datasize.create_dataset("validation", data=np.asarray(len(merged_splits["validation"]), dtype=np.int64))
        datasize.create_dataset("test", data=np.asarray(len(merged_splits["test"]), dtype=np.int64))

        print("[links] Final split sizes:", flush=True)
        for split, names in merged_splits.items():
            print(f"[links]   {split}: {len(names)}", flush=True)
        print(f"[links] Wrote {output}", flush=True)


if __name__ == "__main__":
    main()
