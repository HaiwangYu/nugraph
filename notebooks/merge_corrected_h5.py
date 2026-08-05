#!/usr/bin/env python
import argparse
from pathlib import Path

import h5py
import numpy as np


def read_split_names(file_handle, split):
    split_key = "validation" if split == "val" and "validation" in file_handle["samples"] else split
    values = file_handle[f"samples/{split_key}"].asstr()[()]
    return [str(value) for value in values]


def copy_metadata(input_file, output_file):
    for key in ("planes", "semantic_classes", "gen", "event_classes"):
        if key in input_file and key not in output_file:
            input_file.copy(input_file[key], output_file, key)


def merge_files(inputs, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")

    split_names = {"train": [], "validation": [], "test": []}
    seen = set()

    with h5py.File(output, "w", libver="latest") as fout:
        dataset_out = fout.create_group("dataset")

        for file_index, input_path in enumerate(inputs, start=1):
            input_path = Path(input_path)
            print(f"[merge] Opening {input_path}", flush=True)
            with h5py.File(input_path, "r") as fin:
                copy_metadata(fin, fout)

                source_splits = {
                    "train": read_split_names(fin, "train"),
                    "validation": read_split_names(fin, "val"),
                    "test": read_split_names(fin, "test"),
                }
                total = sum(len(names) for names in source_splits.values())
                copied = 0

                for split, names in source_splits.items():
                    for name in names:
                        output_name = name
                        if output_name in seen:
                            output_name = f"file{file_index}__{name}"
                        seen.add(output_name)
                        fin.copy(fin[f"dataset/{name}"], dataset_out, output_name)
                        split_names[split].append(output_name)
                        copied += 1
                        if copied % 10000 == 0:
                            print(
                                f"[merge] {input_path.name}: copied {copied}/{total}",
                                flush=True,
                            )

                print(f"[merge] {input_path.name}: copied {copied}/{total}", flush=True)

        string_dtype = h5py.string_dtype(encoding="utf-8")
        samples = fout.create_group("samples")
        samples.create_dataset("train", data=np.asarray(split_names["train"], dtype=object), dtype=string_dtype)
        samples.create_dataset("validation", data=np.asarray(split_names["validation"], dtype=object), dtype=string_dtype)
        samples.create_dataset("test", data=np.asarray(split_names["test"], dtype=object), dtype=string_dtype)

        datasize = fout.create_group("datasize")
        datasize.create_dataset("train", data=np.ones(len(split_names["train"]), dtype=np.int64))
        datasize.create_dataset("validation", data=np.asarray(len(split_names["validation"]), dtype=np.int64))
        datasize.create_dataset("test", data=np.asarray(len(split_names["test"]), dtype=np.int64))

        print("[merge] Final split sizes:", flush=True)
        for split, names in split_names.items():
            print(f"[merge]   {split}: {len(names)}", flush=True)
        print(f"[merge] Wrote {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Merge corrected NuGraph HDF5 files.")
    parser.add_argument("--output", required=True, help="Merged HDF5 output path.")
    parser.add_argument("inputs", nargs="+", help="Input HDF5 files to merge.")
    args = parser.parse_args()
    merge_files(args.inputs, args.output)


if __name__ == "__main__":
    main()
