#!/usr/bin/env python
# filename: inspect_nugraph_batch_sem.py
"""
Sanity check for NuGraph4 + sidecar H5.

- Confirms sidecar /sem_features/sp is present and aligned.
- Prints hit.x shape (so we know the correct --in-features).
- Splits hit.x into [base | sidecar] and prints simple stats.

Usage:
  python inspect_nugraph_batch_sem.py \
      --data-path /lus/eagle/projects/neutrinoGPU/abhat/sbnd/clustering/23334072_nug4_vertex_test_sem.h5
"""

import argparse
from pathlib import Path

import h5py
import torch

from nugraph.data import H5DataModule
from nugraph.models import NuGraph4


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data-path",
        required=True,
        help="NuGraph HDF5 file with sidecar (/sem_features/sp).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for the sanity dataloader.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    path = Path(args.data_path)
    if not path.is_file():
        raise FileNotFoundError(path)

    # --- Open H5 and inspect semantic_classes + sidecar layout ---
    with h5py.File(path, "r") as f:
        if "semantic_classes" in f:
            semantic_classes = f["semantic_classes"].asstr()[()].tolist()
            print("[Info] semantic_classes:", semantic_classes)
        else:
            print("[Info] semantic_classes: MISSING")

        # pick a train sample name
        if "samples/train" in f:
            train_keys = f["samples/train"][()]
            train_keys = [k.decode() if isinstance(k, bytes) else k for k in train_keys]
        else:
            train_keys = list(f["dataset"].keys())

        if not train_keys:
            raise RuntimeError("No samples found in file.")

        first_key = train_keys[0]
        print(f"[Info] First train sample: {first_key}")

        # check sidecar
        has_sidecar = False
        sidecar_dim = None
        if "sem_features" in f and "sp" in f["sem_features"]:
            sp_group = f["sem_features"]["sp"]
            if first_key in sp_group:
                arr = sp_group[first_key][()]
                if arr.ndim == 2:
                    has_sidecar = True
                    sidecar_dim = arr.shape[1]
                    print(f"[Info] sidecar /sem_features/sp/{first_key} shape: {arr.shape}")
                else:
                    print(f"[WARN] sidecar /sem_features/sp/{first_key} has unexpected shape {arr.shape}")
            else:
                print(f"[WARN] sidecar group has no entry for {first_key}")
        else:
            print("[Info] No /sem_features/sp group found.")

    # --- Build DataModule + get one train batch (after transforms + dataset.py) ---
    dm = H5DataModule(
        data_path=str(path),
        model=NuGraph4,
        batch_size=args.batch_size,
        num_workers=0,
        shuffle="random",
        balance_frac=0.0,
    )
    dm.setup("fit")
    print(
        "train / val / test sizes:",
        len(dm.train_dataset),
        len(dm.val_dataset),
        len(dm.test_dataset),
    )

    batch = next(iter(dm.train_dataloader()))
    print("\nNode types in batch:", list(batch.node_types))

    hit = batch["hit"]

    # --- Hit features shape ---
    print("\n==== [hit] store ====")
    print("[hit.x] shape:", tuple(hit.x.shape))
    print("[hit.x] dtype:", hit.x.dtype, "device:", hit.x.device)

    # global stats
    x = hit.x
    print(
        "[hit.x] min / max: {:.4g} / {:.4g}".format(
            float(x.min().item()), float(x.max().item())
        )
    )

    # --- If we know sidecar_dim, try to separate base vs sidecar ---
    if has_sidecar and sidecar_dim is not None and sidecar_dim > 0:
        D = x.shape[1]
        if D > sidecar_dim:
            base_dim = D - sidecar_dim
            x_base = x[:, :base_dim]
            x_side = x[:, base_dim:]

            print(f"\n[Decomposition] total D = {D}, base_dim ≈ {base_dim}, sidecar_dim = {sidecar_dim}")

            # Base stats
            print("[hit.x (base part)] shape:", tuple(x_base.shape))
            print(
                "[hit.x (base)] min / max: {:.4g} / {:.4g}".format(
                    float(x_base.min().item()), float(x_base.max().item())
                )
            )

            # Sidecar stats
            print("[hit.x (sidecar part)] shape:", tuple(x_side.shape))
            print(
                "[hit.x (sidecar)] min / max: {:.4g} / {:.4g}".format(
                    float(x_side.min().item()), float(x_side.max().item())
                )
            )
            # show first row of sidecar
            print("[hit.x (sidecar) first row]:", x_side[0].tolist())
        else:
            print(
                "[WARN] D (hit.x.shape[1]) <= sidecar_dim. "
                "This would mean sidecar was NOT appended as expected."
            )
    else:
        print("\n[Info] No usable sidecar_dim found; hit.x is whatever NuGraphData.load produced.")

    # --- y_semantic and y_instance sanity ---
    y_sem = hit.y_semantic if hasattr(hit, "y_semantic") else None
    if y_sem is not None:
        uniques, counts = torch.unique(y_sem, return_counts=True)
        print("\n[hit] y_semantic uniques and counts:")
        for u, c in zip(uniques.tolist(), counts.tolist()):
            print(f"  class {u}: {c} hits")

    if hasattr(hit, "y_instance"):
        y_inst = hit.y_instance
        uniques, counts = torch.unique(y_inst, return_counts=True)
        print("\n[hit] y_instance uniques (up to 20):", uniques[:20].tolist())
        print("[hit] # labeled hits (y_instance >= 0):", int((y_inst >= 0).sum().item()))
    else:
        print("\n[hit] has no y_instance attribute")

    print("\n[Info] Use --in-features =", hit.x.shape[1], "in train.py for this file.")


if __name__ == "__main__":
    main()
