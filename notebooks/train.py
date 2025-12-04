#!/usr/bin/env python
import os
os.environ.setdefault("TORCH_DIST_SKIP_PARAM_VALIDATION", "1")
# HDF5 parallel read friendliness
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
# Keep NCCL chatter reasonable (optional)
# Use TORCH_NCCL_ASYNC_ERROR_HANDLING=1 instead
os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")

import multiprocessing as mp
try:
    # Set start method before importing torch/lightning if possible
    mp.set_start_method("spawn", force=True)
    print(f"[PID {os.getpid()}] Set multiprocessing start method to spawn.")
except RuntimeError as e:
    # Might already be set or not allowed to change
    print(f"[PID {os.getpid()}] Warning: Could not set multiprocessing start method: {e}")
    pass

import argparse
from pathlib import Path
from datetime import timedelta  # Needed for DDPStrategy timeout
import torch
torch.set_float32_matmul_precision("high")

# Patch PyTorch DDP cross-rank param verification (fragile on some builds)
try:
    import torch.distributed.utils as dist_utils
    import torch.nn.parallel.distributed as ddp_module
    def _nugraph_skip_verify(*args, **kwargs): return
    dist_utils._verify_param_shape_across_processes = _nugraph_skip_verify
    ddp_module._verify_param_shape_across_processes = _nugraph_skip_verify
    print("[patch] Disabled DDP param verification in utils AND nn.parallel.distributed.")
except Exception as _e:
    print(f"[patch] Could not disable DDP param verification: {_e}")

import torch.nn as nn
import torch.distributed as dist  # Import dist for barrier (currently unused)
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import WandbLogger, CSVLogger
import nugraph as ng
from pytorch_lightning.plugins.environments import ClusterEnvironment



class ExternalMPIEnvironment(ClusterEnvironment):
    def __init__(self) -> None:
        self._world_size = int(os.environ.get("PMI_SIZE", os.environ.get("WORLD_SIZE", "1")))
        self._global_rank = int(os.environ.get("PMI_RANK", os.environ.get("RANK", "0")))
        self._main_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
        self._main_port = int(os.environ.get("MASTER_PORT", "29500"))

    @property
    def creates_processes_externally(self) -> bool:
        return True

    @property
    def main_address(self) -> str:
        return self._main_addr

    @property
    def main_port(self) -> int:
        return self._main_port

    @staticmethod
    def detect() -> bool:
        return True

    def world_size(self) -> int:
        return self._world_size

    def set_world_size(self, size: int) -> None:
        return  # Lightning shouldn’t override MPI’s size

    def global_rank(self) -> int:
        return self._global_rank

    def set_global_rank(self, rank: int) -> None:
        return

    def local_rank(self) -> int:
        return int(os.environ.get("PMI_LOCAL_RANK", os.environ.get("LOCAL_RANK", "0")))
    
    def node_rank(self) -> int:
        return int(os.environ.get("PMI_NODE_RANK", os.environ.get("NODE_RANK", "0")))






def get_last_linear(module: nn.Module) -> nn.Linear:
    """
    Walk all submodules and return the last nn.Linear encountered.
    Raises if none found.
    """
    last_linear = None
    for m in module.modules():
        if isinstance(m, nn.Linear):
            last_linear = m
    if last_linear is None:
        raise RuntimeError("No nn.Linear layer found.")
    return last_linear


def _get_semantic_last_linear(model: nn.Module) -> nn.Linear | None:
    """
    Helper that finds the final Linear layer of the semantic head for both
    NuGraph3 and NuGraph4.

    - NuGraph3: tries model.semantic_decoder (Sequential or similar)
    - NuGraph4: uses model.semantic_head (plain nn.Linear), if present
    """
    # NuGraph3-style: semantic_decoder present
    if hasattr(model, "semantic_decoder"):
        try:
            last = get_last_linear(getattr(model, "semantic_decoder"))
            if isinstance(last, nn.Linear):
                return last
        except Exception:
            pass

    # NuGraph4-style: semantic_head is a simple Linear
    if hasattr(model, "semantic_head"):
        head = getattr(model, "semantic_head")
        if isinstance(head, nn.Linear):
            return head

    return None


def main(args):
    # Ranks from mpiexec / Cray MPICH
    global_rank = int(os.environ.get("PMI_RANK", os.environ.get("RANK", "0")))
    world_size  = int(os.environ.get("PMI_SIZE", os.environ.get("WORLD_SIZE", "1")))
    local_rank = int(os.environ.get("PMI_LOCAL_RANK", os.environ.get("LOCAL_RANK", "0")))
    devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if devices:
        logical = [d.strip() for d in devices.split(",")]
        os.environ["CUDA_VISIBLE_DEVICES"] = logical[local_rank % len(logical)]
        torch.cuda.set_device(0)
    else:
        torch.cuda.set_device(local_rank % torch.cuda.device_count())


    print(f"[Rank {global_rank}] MPI/Env: WORLD_SIZE={world_size}, RANK={global_rank}, LOCAL_RANK={local_rank}")

    # Make sure Lightning also sees torchrun-style env
    os.environ["RANK"]       = str(global_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(local_rank)

    if global_rank == 0:
        import sys
        print("Python:", sys.executable)
        print("First sys.path entries:", sys.path[:5])
        print("Torch:", torch.__version__, "CUDA:", torch.version.cuda, "file:", torch.__file__)
        print("Lightning:", pl.__version__, "file:", pl.__file__)

    # --- Logger (Instantiate on ALL ranks, Lightning handles gating) ---
    log_base_dir = Path(args.log_dir); log_base_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.run_name
    logdir = log_base_dir / run_name

    # W&B paths before WandbLogger is constructed
    os.environ.setdefault("WANDB_DIR",           str(logdir / "wandb"))
    os.environ.setdefault("WANDB_CACHE_DIR",     str(logdir / "wandb_cache"))
    os.environ.setdefault("WANDB_ARTIFACTS_DIR", str(logdir / "wandb_artifacts"))
    os.environ.setdefault("WANDB_DISABLE_CODE",  "true")   # optional space saver

    Path(os.environ["WANDB_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["WANDB_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["WANDB_ARTIFACTS_DIR"]).mkdir(parents=True, exist_ok=True)

    # Create dir on all ranks for consistency before logger tries to
    if not logdir.exists():
        try:
            logdir.mkdir(parents=True, exist_ok=True)
        except OSError:  # Handle potential race condition
            if not logdir.is_dir():
                raise

    logger = None
    try:
        # Try creating WandbLogger on all ranks. It self-gates internally.
        wandb_mode = os.environ.get("WANDB_MODE", "online")
        logger = WandbLogger(
            save_dir=logdir,
            project="nugraph3",   # project name unchanged for now
            name=run_name,
            log_model=False,
            offline=(wandb_mode == "offline")
        )
        if global_rank == 0:
            print(f"[Rank 0] WandbLogger initialized (mode: {wandb_mode}).")
    except Exception as e:
        # Fallback to CSVLogger on ALL ranks if W&B fails
        if global_rank == 0:
            print(f"[Rank 0] WandbLogger init failed ({e}). Using CSVLogger.")
        logger = CSVLogger(save_dir=logdir, name=run_name)

    # --- Data ---
    from nugraph.data import H5DataModule as DataModule

    # --- Model choice (NuGraph3 vs NuGraph4) ---
    model_name = getattr(args, "model", "nugraph3").lower()
    if model_name == "nugraph4":
        Model = ng.models.NuGraph4
    else:
        Model = ng.models.NuGraph3

    # Add print statement to verify num_workers
    print(f"[Rank {global_rank}] Initializing DataModule with num_workers = {args.num_workers}")
    nudata = DataModule(
        model=Model,
        data_path=args.data_path,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        min_nu_hits=args.min_nu_hits,
        shuffle=args.shuffle,
        balance_frac=args.balance_frac,
        train_fraction=getattr(args, "train_fraction", 1.0),
    )

    # Make dataloading conservative for multi-node HDF5
    for attr, val in [
        ("drop_last", True),
        ("train_drop_last", True),
        ("persistent_workers", False),        # Ensure this is False, esp if num_workers > 0
        ("pin_memory", False),                # Often False is safer with HDF5/multinode
        ("prefetch_factor", None if args.num_workers == 0 else 2),
        ("timeout", 0),                       # Default DataLoader timeout
    ]:
        if hasattr(nudata, attr) and (val is not None):
            print(f"[Rank {global_rank}] Setting nudata.{attr} = {val}")
            setattr(nudata, attr, val)

    # Setup datamodule AFTER potentially changing loader settings
    print(f"[Rank {global_rank}] Setting up DataModule...")
    nudata.setup("fit")
    print(f"[Rank {global_rank}] DataModule setup complete.")
    print(
        f"[Rank {global_rank}] Split sizes after min_nu_hits={args.min_nu_hits}: "
        f"train={len(nudata.train_dataset)}, "
        f"val={len(nudata.val_dataset)}, "
        f"test={len(nudata.test_dataset)}"
    )

    # === SANITY A: confirm semantic class order from file ===
    if global_rank == 0:
        print("[Sanity] semantic_classes from H5:", getattr(nudata, "semantic_classes", None))
        print("[Sanity] event_classes from H5:", getattr(nudata, "event_classes", None))
        print(
            "[Sanity] You passed --semantic-class-weight =",
            args.semantic_class_weight,
            "(interpreted in the EXACT order printed above)."
        )

    # === SANITY B: inspect one train batch (transformed) ===
    if global_rank == 0:
        try:
            td = nudata.train_dataloader()
            batch = next(iter(td))  # one batch
            y = batch["hit"].y_semantic
            uniques, counts = torch.unique(y, return_counts=True)
            print(
                "[Sanity] y_semantic unique values (with counts):",
                {int(u.item()): int(c.item()) for u, c in zip(uniques, counts)}
            )

            sc = getattr(nudata, "semantic_classes", None)
            if sc is not None:
                mask = y >= 0
                if mask.any():
                    yv = y[mask]
                    for idx in range(len(sc)):
                        n = int((yv == idx).sum().item())
                        print(f"[Sanity] hits with class index {idx} ('{sc[idx]}'): {n}")
                else:
                    print("[Sanity] No labeled hits (all -1) in this sampled batch.")
        except Exception as e:
            print("[Sanity] Could not inspect a batch:", e)

    model_semantic_classes = getattr(nudata, "semantic_classes", None)
    if args.semantic_head and model_semantic_classes is None:
        print(f"[Rank {global_rank}] ERROR: semantic head requested but datamodule has no `semantic_classes`.")
        raise SystemExit(1)

    if global_rank == 0:
        print("[Rank 0] Initializing model...")

    # Build model; only pass NuGraph4-specific kwargs when using NuGraph4
    common_kwargs = dict(
        in_features=args.in_features,
        hit_features=args.hit_features,
        nexus_features=args.nexus_features,
        instance_features=args.instance_features,
        interaction_features=args.interaction_features,
        semantic_classes=model_semantic_classes,
        semantic_class_weight=args.semantic_class_weight,
        event_classes=getattr(nudata, 'event_classes', None),
        num_iters=args.num_iters,
        event_head=args.event_head,
        semantic_head=args.semantic_head,
        filter_head=args.filter_head,
        vertex_head=args.vertex_head,
        instance_head=args.instance_head,
        use_checkpointing=args.use_checkpointing,
        lr=args.learning_rate,
    )

    if model_name == "nugraph4":
        nugraph = Model(
            **common_kwargs,
            edge_hidden_dim=getattr(args, "edge_hidden_dim", 32),
            embed_dim=getattr(args, "embed_dim", 64),
            lambda_edge=getattr(args, "lambda_edge", 0.5),
            edge_pos_weight=getattr(args, "edge_pos_weight", 10.0),
            lambda_embed=getattr(args, "lambda_embed", 0.2),
            lambda_coh=getattr(args, "lambda_coh", 0.0),
            coh_edge_thr=getattr(args, "coh_edge_thr", 0.7),
            coh_min_cluster=getattr(args, "coh_min_cluster", 2),
        )
    else:
        nugraph = Model(**common_kwargs)

    # === SANITY C: head output dimension matches class count (NuGraph3 & NuGraph4) ===
    if global_rank == 0:
        try:
            last_fc = _get_semantic_last_linear(nugraph)
            sc = getattr(nudata, "semantic_classes", None)
            out_dim = last_fc.out_features if last_fc is not None else None
            print(
                f"[Sanity] semantic head out_features = {out_dim} "
                f"vs len(semantic_classes) = {len(sc) if sc else None}"
            )
        except Exception as e:
            print("[Sanity] Could not read semantic head output dim:", e)

    # Ensure FP32 params
    for p in nugraph.parameters():
        if p.dtype != torch.float32:
            p.data = p.data.float()
            if p.grad is not None:
                p.grad = p.grad.float()

    # Initialize semantic priors on CPU (works for both models)
    if model_semantic_classes:
        last_fc = _get_semantic_last_linear(nugraph)
        if last_fc is not None and isinstance(last_fc, nn.Linear):
            priors = torch.ones(len(model_semantic_classes), dtype=torch.float) / len(model_semantic_classes)
            with torch.no_grad():
                last_fc.bias.copy_(priors.log())
            if global_rank == 0:
                print("[Rank 0] Initialized semantic bias with uniform priors (on CPU)")

    # --- Callbacks (Instantiate on ALL ranks) ---
    checkpoint_dir = logdir / "checkpoints"
    ckpt_dir = str(checkpoint_dir)
    if not checkpoint_dir.exists():
        try:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
        except OSError:  # Handle potential race condition
            if not checkpoint_dir.is_dir():
                raise

    callbacks = [
        pl.callbacks.EarlyStopping(
            monitor="semantic/f1-macro-val",
            mode="max",
            patience=20,
            min_delta=1e-4,
            verbose=(global_rank == 0),
        ),
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
        pl.callbacks.ModelCheckpoint(
            monitor="semantic/f1-macro-val",
            mode="max",
            save_top_k=1,
            filename="best-f1",
            dirpath=ckpt_dir,
            save_last=True,
        ),
        pl.callbacks.ModelCheckpoint(
            monitor="semantic/recall-macro-val",
            mode="max",
            save_top_k=1,
            filename="best-recall",
            dirpath=ckpt_dir,
        ),
    ]


    # --- Trainer configuration ---
    use_gpu = torch.cuda.is_available()

    if world_size > 1:
        # Launched by mpiexec: one MPI rank == one GPU.
        accelerator = "gpu" if use_gpu else "cpu"
        ranks_per_node = int(os.environ.get("PMI_LOCAL_SIZE", os.environ.get("LOCAL_WORLD_SIZE", "1")))
        ranks_per_node = max(1, ranks_per_node)

        devices = ranks_per_node if use_gpu else 0        # IMPORTANT: 1 GPU per MPI process
        num_nodes = max(1, world_size // ranks_per_node)       # Lightning sees this as a "single node"; MPI spans nodes

        print(
            f"[Rank {global_rank}] Setting up DDP Trainer with external MPI "
            f"(world_size={world_size}, devices={devices}, num_nodes={num_nodes})"
        )

        strategy = DDPStrategy(
            process_group_backend="nccl",
            find_unused_parameters=True,
            timeout=timedelta(minutes=10),
            cluster_environment=ExternalMPIEnvironment(),  # <-- key line
        )
    else:
        # Single-process (no mpiexec): you can still run multi-GPU here if you want
        accelerator = "gpu" if use_gpu else "cpu"
        if use_gpu:
            devices = min(args.gpus_per_node, torch.cuda.device_count())
        else:
            devices = 0
        num_nodes = 1
        strategy = "auto"

        print(
            f"[Rank {global_rank}] Setting up single-process Trainer "
            f"(devices={devices}, num_nodes={num_nodes})"
        )

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        num_nodes=num_nodes,
        strategy=strategy,
        num_sanity_val_steps=0,
        # plugins=[ExternalMPIEnvironment()],

        # let your CLI flags control these if you have them;
        # otherwise they default to 1.0 (full epoch)
        limit_train_batches=getattr(args, "limit_train_batches", 1.0),
        limit_val_batches=getattr(args, "limit_val_batches", 1.0),
        limit_test_batches=getattr(args, "limit_test_batches", 1.0),

        logger=logger,
        callbacks=callbacks,
        precision="32-true",
        max_epochs=args.max_epochs,
        enable_progress_bar=(global_rank == 0),
        enable_checkpointing=True,
        sync_batchnorm=False,
        use_distributed_sampler=True,
    )



    print(f"[Rank {global_rank}] Starting training...")
    ckpt_path = args.resume_from
    if ckpt_path:
        print(f"[Rank {global_rank}] Resuming training from checkpoint: {ckpt_path}")
        trainer.fit(nugraph, datamodule=nudata, ckpt_path=ckpt_path)
    else:
        print(f"[Rank {global_rank}] Starting a new training run (no checkpoint specified).")
        trainer.fit(nugraph, datamodule=nudata)

    # --- Testing (run on ALL ranks) ---
    print("[Rank %d] Starting testing..." % global_rank)
    try:
        trainer.test(model=nugraph, datamodule=nudata, ckpt_path="best")
    except Exception as e:
        print(f"[Rank {global_rank}] Test failed: {e}")

    # --- Finish / cleanup (rank 0 only) ---
    if global_rank == 0:
        try:
            if isinstance(logger, WandbLogger):
                import wandb
                wandb.finish()
        except Exception:
            pass
        print(f"[Rank 0] Run {run_name} complete. Logs/checkpoints in {logdir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="NuGraph Training Script (Polaris, mpiexec, Lightning DDP)"
    )
    # --- Paths ---
    p.add_argument("--data-path", type=str, required=True,
                   help="Path to HDF5 data file.")
    p.add_argument("--log-dir", type=str, required=True,
                   help="Base directory for logs and checkpoints.")
    p.add_argument("--run-name", type=str, required=True,
                   help="Specific name for this run.")

    # --- Hardware & Dataloading ---
    p.add_argument("--gpus-per-node", type=int, default=4,
                   help="Number of GPUs per node (Polaris=4).")
    p.add_argument("--num-workers", type=int, default=2,
                   help="DataLoader workers per rank (0 often safest for HDF5).")
    p.add_argument("--batch-size", type=int, default=32,
                   help="Batch size per GPU (per rank).")

    # --- Training Hyperparameters ---
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--max-epochs", type=int, default=50)

    # --- Model selection ---
    p.add_argument("--model", type=str, default="nugraph3",
                   choices=["nugraph3", "nugraph4"],
                   help="Choose model architecture.")

    # --- Model Configuration ---
    p.add_argument("--in-features", type=int, default=8,
                   help="Number of input node features.")
    p.add_argument("--hit-features", type=int, default=256,
                   help="Dimension of hit node embeddings.")
    p.add_argument("--nexus-features", type=int, default=64,
                   help="Dimension of nexus node embeddings.")
    p.add_argument("--instance-features", type=int, default=32,
                   help="Dimension for instance clustering head.")
    p.add_argument("--interaction-features", type=int, default=32,
                   help="Dimension for interaction classification head.")
    p.add_argument("--num-iters", type=int, default=7,
                   help="Number of message passing iterations.")

    # --- Heads & Outputs ---
    p.add_argument("--semantic-class-weight", type=float, nargs=2,
                   default=[30.0, 1.0],
                   help="Weights for semantic classes.")
    p.add_argument("--event-head", action="store_true", default=False,
                   help="Enable event classification head.")
    p.add_argument("--semantic-head", dest="semantic_head", action="store_true",
                   default=True,
                   help="Enable semantic segmentation head (default).")
    p.add_argument("--no-semantic-head", dest="semantic_head", action="store_false",
                   help="Disable semantic segmentation head.")
    p.add_argument("--filter-head", action="store_true", default=False,
                   help="Enable filter head.")
    p.add_argument("--vertex-head", action="store_true", default=False,
                   help="Enable vertex regression head.")
    p.add_argument("--instance-head", action="store_true", default=False,
                   help="Enable instance clustering head.")

    # --- NuGraph4-specific ---
    p.add_argument("--edge-hidden-dim", type=int, default=32,
                help="Edge MLP hidden size (NuGraph4).")
    p.add_argument("--embed-dim", type=int, default=64,
                help="Embedding decoder dim (NuGraph4).")
    p.add_argument("--lambda-edge", type=float, default=0.0,
                help="Weight of the edge loss term in the total loss (NuGraph4).")
    p.add_argument("--edge-pos-weight", type=float, default=1.0,
                help="Positive-class weight for edge BCE loss (nu–nu edges) in NuGraph4.")
    p.add_argument("--lambda-embed", type=float, default=0.2,
                help="Weight of instance embedding loss (NuGraph4).")
    p.add_argument("--lambda-coh", type=float, default=0.0,
                 help="Weight of semantic coherence loss over predicted clusters (NuGraph4).")
    p.add_argument("--coh-edge-thr", type=float, default=0.7,
                 help="p_same threshold for building clusters used in coherence loss.")
    p.add_argument("--coh-min-cluster", type=int, default=2,
                 help="Minimum cluster size for coherence loss.")


    # --- Other ---
    p.add_argument("--use-checkpointing", dest="use_checkpointing",
                   action="store_true", default=True,
                   help="Enable gradient checkpointing (default).")
    p.add_argument("--no-checkpointing", dest="use_checkpointing",
                   action="store_false",
                   help="Disable gradient checkpointing.")
    p.add_argument("--resume-from", type=str, default=None,
                   help="Path to checkpoint file to resume training from (e.g., .../last.ckpt)")
    p.add_argument(
        "--min-nu-hits",
        type=int,
        default=0,
        help="Require at least this many 'nu' hits per event (U+V+Y). 0 disables the cut."
    )
    p.add_argument(
        "--shuffle", type=str, default="weighted",
        choices=["random", "balance", "weighted"],
        help="Training sampler: random | balance (datasize-based) | weighted (nu-hit aware)."
    )
    p.add_argument("--balance-frac", type=float, default=0.10,
                   help="Fraction for BalanceSampler.")
    
    p.add_argument(
        "--limit-train-batches",
        type=float,
        default=1.0,
        help="Fraction or number of train batches per epoch (Lightning semantics).",
    )
    p.add_argument(
        "--limit-val-batches",
        type=float,
        default=1.0,
        help="Fraction or number of val batches per epoch.",
    )
    p.add_argument(
        "--limit-test-batches",
        type=float,
        default=1.0,
        help="Fraction or number of test batches.",
    )
    p.add_argument(
        "--train-fraction",
        type=float,
        default=1.0,
        help="Fraction of the train split to use (0 < f <= 1.0).",
    )

    args = p.parse_args()
    pl.seed_everything(1337, workers=True)
    main(args)
