#!/usr/bin/env python

"""
Minimal single-node debug driver for NuGraph4 + edge loss.

Modes:
- devices=1  -> single GPU, no DDP (strategy="auto").
- devices>1  -> multi-GPU on ONE node using Lightning's ddp_spawn
                (no mpiexec / MPI involved).

Goal:
- Prove that NuGraph4 + edge head/loss is DDP-safe at the PyTorch level.
- Use tiny batch/epoch limits to avoid OOM and long runs.
"""

import os
import argparse

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger

from nugraph.models.nugraph4.nugraph4 import NuGraph4
from nugraph.data import H5DataModule as DataModule


def parse_args():
    p = argparse.ArgumentParser(
        description="Single-node debug run for NuGraph4 with edge loss"
    )

    # --- Data paths ---
    p.add_argument(
        "--data-path",
        required=True,
        help="Path to HDF5 file (same as --data-path you use in train.py)",
    )

    # --- Batch / loader ---
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)

    # --- Devices (1 = single GPU, >1 = ddp_spawn) ---
    p.add_argument(
        "--devices",
        type=int,
        default=1,
        help="Number of GPUs on this node to use. "
             "1 => no DDP, >1 => ddp_spawn.",
    )

    # --- Model hyperparams (match train.py defaults where relevant) ---
    p.add_argument("--in-features", type=int, default=8)
    p.add_argument("--hit-features", type=int, default=256)
    p.add_argument("--nexus-features", type=int, default=64)
    p.add_argument("--instance-features", type=int, default=32)
    p.add_argument("--interaction-features", type=int, default=32)
    p.add_argument("--num-iters", type=int, default=7)

    # semantic class weights (nu, cosmic)
    p.add_argument(
        "--semantic-class-weight",
        type=float,
        nargs=2,
        default=[30.0, 1.0],
        help="Weights for semantic classes [w_nu, w_cosmic].",
    )

    # edge loss knobs
    p.add_argument("--edge-hidden-dim", type=int, default=32)
    p.add_argument("--embed-dim", type=int, default=64)
    p.add_argument("--edge-pos-weight", type=float, default=10.0)
    p.add_argument("--lambda-edge", type=float, default=0.5)

    # --- Optimizer ---
    p.add_argument("--lr", type=float, default=2e-5)

    # --- Debug trainer controls ---
    p.add_argument("--max-epochs", type=int, default=1)
    p.add_argument("--limit-train-batches", type=int, default=10)
    p.add_argument("--limit-val-batches", type=int, default=4)
    p.add_argument("--limit-test-batches", type=int, default=2)

    p.add_argument(
        "--output-dir",
        default="log/nug4_debug",
        help="Where to write CSV logs & checkpoints",
    )

    return p.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(1337, workers=True)

    # --- Environment safety knobs ---
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        "expandable_segments:True,max_split_size_mb:256",
    )
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")

    print("===== NuGraph4 single-node debug run =====")
    print(f"  data-path     : {args.data_path}")
    print(f"  batch-size    : {args.batch_size}")
    print(f"  devices       : {args.devices}")
    print(f"  max-epochs    : {args.max_epochs}")
    print(f"  train batches : {args.limit_train_batches}")
    print(f"  val batches   : {args.limit_val_batches}")
    print(f"  test batches  : {args.limit_test_batches}")
    print("==========================================")

    # --- DataModule (mirror train.py as much as possible) ---
    nudata = DataModule(
        model=NuGraph4,
        data_path=args.data_path,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        min_nu_hits=0,
        shuffle="weighted",
        balance_frac=0.10,
    )

    print("[debug] Setting up DataModule...")
    nudata.setup("fit")
    print("[debug] DataModule setup complete.")
    print(
        f"[debug] Split sizes: "
        f"train={len(nudata.train_dataset)}, "
        f"val={len(nudata.val_dataset)}, "
        f"test={len(nudata.test_dataset)}"
    )
    print(f"[debug] semantic_classes: {getattr(nudata, 'semantic_classes', None)}")
    print(f"[debug] event_classes   : {getattr(nudata, 'event_classes', None)}")

    # --- Model (NuGraph4) ---
    semantic_classes = getattr(nudata, "semantic_classes", None)

    nugraph = NuGraph4(
        in_features=args.in_features,
        hit_features=args.hit_features,
        nexus_features=args.nexus_features,
        instance_features=args.instance_features,
        interaction_features=args.interaction_features,
        semantic_classes=semantic_classes,
        semantic_class_weight=args.semantic_class_weight,
        event_classes=getattr(nudata, "event_classes", None),
        num_iters=args.num_iters,
        event_head=False,
        semantic_head=True,
        filter_head=True,          # you used FilterDecoder in the run output
        vertex_head=False,
        instance_head=False,
        use_checkpointing=True,
        lr=args.lr,
        edge_hidden_dim=args.edge_hidden_dim,
        embed_dim=args.embed_dim,
        lambda_edge=args.lambda_edge,
        edge_pos_weight=args.edge_pos_weight,
    )

    print(nugraph)
    print(f"Total params: {sum(p.numel() for p in nugraph.parameters())}")

    # --- Logger: simple CSV, no W&B for this debug ---
    csv_logger = CSVLogger(
        save_dir=args.output_dir,
        name="nug4_single_node_debug",
    )

    # --- Trainer configuration: single- vs multi-GPU ---
    if args.devices <= 1:
        accelerator = "gpu" if torch.cuda.is_available() else "cpu"
        strategy = "auto"
        devices = 1
        print("[debug] Using single-GPU (no DDP), strategy='auto'")
    else:
        accelerator = "gpu"
        strategy = "ddp_spawn"
        devices = args.devices
        print("[debug] Using ddp_spawn on a single node "
              f"with devices={devices}")

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        num_nodes=1,
        strategy=strategy,
        max_epochs=args.max_epochs,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        limit_test_batches=args.limit_test_batches,
        precision="32-true",          # keep it simple for DDP debug
        logger=csv_logger,
        log_every_n_steps=5,
        enable_checkpointing=False,
        enable_progress_bar=True,
        detect_anomaly=False,
        gradient_clip_val=0.0,
        num_sanity_val_steps=0,       # avoid extra passes
    )

    trainer.fit(nugraph, datamodule=nudata)
    print("[debug] Training finished, starting test...")
    try:
        trainer.test(nugraph, datamodule=nudata)
    except Exception as e:
        print("[debug] Test failed:", e)


if __name__ == "__main__":
    main()
