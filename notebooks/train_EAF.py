#!/usr/bin/env python
"""
train.py — NuGraph3/NuGraph4 training script

Works in:
  - 1 GPU single-process (EAF MIG, no mpiexec)
  - multi-rank external MPI (Polaris-style mpiexec, one rank per GPU)

Key fixes vs your current version:
  - Only apply DDP verification patch + DDP/MPI imports when world_size>1
  - Do NOT rewrite CUDA_VISIBLE_DEVICES in single-GPU mode
  - use_distributed_sampler = (world_size>1)
  - DDP find_unused_parameters=False (your standardized baseline)
  - Fix args.num_iters mismatch (parser dest now matches usage)
  
NEW: Added --use-sp-features and --use-vtx-features flags for fair model comparison
"""

import os
from pathlib import Path
from datetime import timedelta
import argparse
import multiprocessing as mp

# --- Environment defaults ---
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")

# If you ever hit annoying user-site contamination, uncomment:
# os.environ.setdefault("PYTHONNOUSERSITE", "1")

# Try to set mp start method (safe on EAF/Polaris)
try:
    mp.set_start_method("spawn", force=True)
    print(f"[PID {os.getpid()}] Set multiprocessing start method to spawn.")
except RuntimeError as e:
    print(f"[PID {os.getpid()}] Warning: Could not set multiprocessing start method: {e}")

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, CSVLogger

import nugraph as ng


# -------------------------
# Optional: External MPI env
# -------------------------
try:
    from pytorch_lightning.plugins.environments import ClusterEnvironment
except Exception:
    ClusterEnvironment = object  # fallback for type checks


class ExternalMPIEnvironment(ClusterEnvironment):
    """Lightning ClusterEnvironment adapter for externally-launched MPI ranks."""

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
        return

    def global_rank(self) -> int:
        return self._global_rank

    def set_global_rank(self, rank: int) -> None:
        return

    def local_rank(self) -> int:
        return int(os.environ.get("PMI_LOCAL_RANK", os.environ.get("LOCAL_RANK", "0")))

    def node_rank(self) -> int:
        return int(os.environ.get("PMI_NODE_RANK", os.environ.get("NODE_RANK", "0")))


# -------------------------
# Helpers
# -------------------------
def get_last_linear(module: nn.Module) -> nn.Linear:
    last_linear = None
    for m in module.modules():
        if isinstance(m, nn.Linear):
            last_linear = m
    if last_linear is None:
        raise RuntimeError("No nn.Linear layer found.")
    return last_linear


def _get_semantic_last_linear(model: nn.Module) -> nn.Linear | None:
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


def _maybe_patch_ddp_param_verify(world_size: int) -> None:
    """
    Patch PyTorch DDP cross-rank param verification (only when using DDP).
    """
    if world_size <= 1:
        return

    os.environ.setdefault("TORCH_DIST_SKIP_PARAM_VALIDATION", "1")

    try:
        import torch.distributed.utils as dist_utils
        import torch.nn.parallel.distributed as ddp_module

        def _nugraph_skip_verify(*args, **kwargs):
            return

        dist_utils._verify_param_shape_across_processes = _nugraph_skip_verify
        ddp_module._verify_param_shape_across_processes = _nugraph_skip_verify
        print("[patch] Disabled DDP param verification in utils AND nn.parallel.distributed.")
    except Exception as e:
        print(f"[patch] Could not disable DDP param verification: {e}")


def _setup_device_for_rank(world_size: int, local_rank: int) -> None:
    """
    - If world_size>1 (MPI ranks): assume one rank == one visible GPU, set device=0.
      (MPI launcher should set CUDA_VISIBLE_DEVICES per rank; do not rewrite it here.)
    - If world_size==1: set device 0 if CUDA available.
    """
    if not torch.cuda.is_available():
        return

    if world_size > 1:
        # External MPI rank should already have a single device visible.
        torch.cuda.set_device(0)
    else:
        # Single process: use device 0.
        torch.cuda.set_device(0)


def main(args):
    # Detect ranks from MPI or torchrun-style env
    global_rank = int(os.environ.get("PMI_RANK", os.environ.get("RANK", "0")))
    world_size = int(os.environ.get("PMI_SIZE", os.environ.get("WORLD_SIZE", "1")))
    local_rank = int(os.environ.get("PMI_LOCAL_RANK", os.environ.get("LOCAL_RANK", "0")))

    # Only patch DDP verify if we actually do DDP
    _maybe_patch_ddp_param_verify(world_size)

    # Matmul precision setting
    torch.set_float32_matmul_precision("high")

    # GPU device selection
    _setup_device_for_rank(world_size, local_rank)

    print(f"[Rank {global_rank}] MPI/Env: WORLD_SIZE={world_size}, RANK={global_rank}, LOCAL_RANK={local_rank}")
    if torch.cuda.is_available():
        dev = torch.cuda.current_device()
        print(f"[Rank {global_rank}] CUDA device={dev} name={torch.cuda.get_device_name(dev)}")

    # Make sure Lightning also sees torchrun-style env variables
    os.environ["RANK"] = str(global_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(local_rank)

    if global_rank == 0:
        import sys
        print("Python:", sys.executable)
        print("First sys.path entries:", sys.path[:5])
        print("Torch:", torch.__version__, "CUDA:", torch.version.cuda, "file:", torch.__file__)
        print("Lightning:", pl.__version__, "file:", pl.__file__)
        
        # NEW: Log feature configuration prominently
        print("\n" + "="*70)
        print("FEATURE CONFIGURATION:")
        print("="*70)
        print(f"  --use-sp-features: {args.use_sp_features}")
        print(f"  --use-vtx-features: {args.use_vtx_features}")
        if args.use_vtx_features:
            print("  ⚠️  WARNING: Vertex features ENABLED - using MC truth vertex info!")
            print("  ⚠️  This is for ABLATION STUDY only - not fair for production!")
        else:
            print("  ✅ Fair mode: only using charge and hit count (no MC truth)")
        print("="*70 + "\n")

    # --- Logger ---
    log_base_dir = Path(args.log_dir)
    log_base_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.run_name
    logdir = log_base_dir / run_name
    logdir.mkdir(parents=True, exist_ok=True)

    # W&B paths. Keep all W&B writes off $HOME.
    wandb_base = Path(os.environ.get("WANDB_BASE", "/lus/eagle/projects/neutrinoGPU/abhat/wandb"))
    os.environ.setdefault("WANDB_DIR", str(wandb_base / "run"))
    os.environ.setdefault("WANDB_CACHE_DIR", str(wandb_base / "cache"))
    os.environ.setdefault("WANDB_ARTIFACTS_DIR", str(wandb_base / "artifacts"))
    os.environ.setdefault("WANDB_ARTIFACT_DIR", str(wandb_base / "artifacts"))
    os.environ.setdefault("WANDB_CONFIG_DIR", str(wandb_base / "config"))
    os.environ.setdefault("WANDB_DATA_DIR", str(wandb_base / "data"))
    os.environ.setdefault("WANDB_DISABLE_CODE", "true")

    for key in ("WANDB_DIR", "WANDB_CACHE_DIR", "WANDB_ARTIFACTS_DIR", "WANDB_ARTIFACT_DIR", "WANDB_CONFIG_DIR", "WANDB_DATA_DIR"):
        Path(os.environ[key]).mkdir(parents=True, exist_ok=True)

    logger = None
    try:
        wandb_mode = os.environ.get("WANDB_MODE", "online")
        logger = WandbLogger(
            save_dir=str(logdir),
            project="nugraph3",
            name=run_name,
            log_model=False,
            offline=(wandb_mode == "offline"),
        )
        if global_rank == 0:
            print(f"[Rank 0] WandbLogger initialized (mode: {wandb_mode}).")
    except Exception as e:
        if global_rank == 0:
            print(f"[Rank 0] WandbLogger init failed ({e}). Using CSVLogger.")
        logger = CSVLogger(save_dir=str(logdir), name=run_name)

    # --- Data ---
    from nugraph.data.data_module import NuGraphDataModule as DataModule

    model_name = getattr(args, "model", "nugraph3").lower()
    Model = ng.models.NuGraph4 if model_name == "nugraph4" else ng.models.NuGraph3

    print(f"[Rank {global_rank}] Initializing DataModule with num_workers={args.num_workers}")
    nudata = DataModule(
        model=Model,
        data_path=args.data_path,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        min_nu_hits=args.min_nu_hits,
        shuffle=args.shuffle,
        balance_frac=args.balance_frac,
        train_fraction=args.train_fraction,
        in_features=args.in_features,   # <-- NEW
    )

    # Conservative loader knobs for HDF5
    for attr, val in [
        ("drop_last", True),
        ("train_drop_last", True),
        ("persistent_workers", True),
        ("pin_memory", False),
        ("prefetch_factor", None if args.num_workers == 0 else 2),
        ("timeout", 0),
    ]:
        if hasattr(nudata, attr) and (val is not None):
            print(f"[Rank {global_rank}] Setting nudata.{attr}={val}")
            setattr(nudata, attr, val)

    print(f"[Rank {global_rank}] Setting up DataModule...")
    nudata.setup("fit")
    print(f"[Rank {global_rank}] DataModule setup complete.")
    print(
        f"[Rank {global_rank}] Split sizes after min_nu_hits={args.min_nu_hits}: "
        f"train={len(nudata.train_dataset)}, val={len(nudata.val_dataset)}, test={len(nudata.test_dataset)}"
    )

    if global_rank == 0:
        print("[Sanity] semantic_classes from H5:", getattr(nudata, "semantic_classes", None))
        print("[Sanity] event_classes from H5:", getattr(nudata, "event_classes", None))
        print("[Sanity] --semantic-class-weight =", args.semantic_class_weight)

    # --- SANITY: inspect one batch and infer in_features ---
    batch = None
    try:
        td = nudata.train_dataloader()
        print("\n================ TRAINING LOOP DEBUG ================")
        print("len(train_dataset)   =", len(nudata.train_dataset))
        print("len(train_dataloader)=", len(td))
        print("=====================================================\n")
        batch = next(iter(td))
    except Exception as e:
        if global_rank == 0:
            print("[Sanity] Could not inspect a batch:", e)

    if batch is not None:
        # ---------------------------
        # Batch-0 smoke logs + asserts
        # ---------------------------
        try:
            sp = batch["sp"]
            y = sp.y_semantic.long()
            uniq, cnt = torch.unique(y, return_counts=True)
            stats = {int(u.item()): int(c.item()) for u, c in zip(uniq, cnt)}

            # Core semantic counts
            n_ghost = int((y == -1).sum().item())
            n_nu    = int((y == 0).sum().item())   # your convention
            n_bkg   = int((y == 1).sum().item())   # your convention
            n_tot   = int(y.numel())

            if global_rank == 0:
                print("[B0] y_semantic counts:", stats)
                print(f"[B0] totals: N={n_tot}  ghost(-1)={n_ghost}  nu(0)={n_nu}  bkg(1)={n_bkg}")
                if (n_nu + n_bkg) > 0:
                    print(f"[B0] nu fraction (non-ghost): {n_nu/(n_nu+n_bkg):.4f}")
                print("[B0] semantic_class_weight (nu,bkg):", args.semantic_class_weight)
                
                # NEW: Log SP features info
                if hasattr(sp, 'features'):
                    print(f"[B0] sp.features shape: {tuple(sp.features.shape)}")
                    print(f"[B0] SP feature columns: col0=charge, col1=hit_count, col2-5=vtx_dist/dx/dy/dz")
                    if args.use_vtx_features:
                        print(f"[B0] ⚠️  Using ALL SP features including vertex (cols 0-5)")
                    else:
                        print(f"[B0] ✅ Using only fair SP features (cols 0-1: charge, hit_count)")


            # Optional: instance labels MUST be on 'sp' if we index with supervision edges
            if hasattr(sp, "y_instance"):
                yi = sp.y_instance.long()
            else:
                yi = None
            

            # Supervision edge store should exist because dataset.py moves it
            sup_key = ("sp", "supervision", "sp")
            if sup_key in batch.edge_types:
                es = batch[sup_key]
                sei = es.edge_index.long()
                el  = es.edge_labelable.long()
                ey  = es.edge_y.long()

                m = (el == 1)
                E_lab = int(m.sum().item())
                E_tot = int(el.numel())

                if global_rank == 0:
                    print(f"[B0] supervision edges: E_total={E_tot}  E_labelable={E_lab}")

                if E_lab > 0:
                    a = sei[0, m]
                    b = sei[1, m]

                    # Assert: labelable edges never touch ghosts
                    bad_touch = int(((y[a] == -1) | (y[b] == -1)).sum().item())
                    assert bad_touch == 0, f"Labelable edges touch ghosts: {bad_touch}"

                    # Assert: labelable edges must have instance labels on both endpoints
                    if yi is not None:
                        bad_inst = int(((yi[a] < 0) | (yi[b] < 0)).sum().item())
                        assert bad_inst == 0, f"Labelable edges have yi<0 endpoints: {bad_inst}"

                    # Log pos-rate on labelable supervision edges
                    pos_rate = float(ey[m].float().mean().item())
                    n_pos = int((ey[m] == 1).sum().item())
                    if global_rank == 0:
                        print(f"[B0] labelable edge pos-rate = {pos_rate:.6f}  (pos={n_pos} / {E_lab})")

        except Exception as e:
            if global_rank == 0:
                print("[B0] Smoke batch checks failed:", e)

        # Sanity: check transform produced the requested feature dim (do NOT override)
        try:
            x = batch["hit"].x
            feat_dim = int(x.shape[-1])
            if global_rank == 0:
                print(f"[Sanity] hit.x feature dim={feat_dim} (requested --in-features={args.in_features})")
            if feat_dim != int(args.in_features):
                raise RuntimeError(f"Transform produced hit.x dim={feat_dim} but requested in_features={args.in_features}")
        except Exception as e:
            if global_rank == 0:
                print("[Sanity] Feature-dim check failed:", e)
            raise
    


    model_semantic_classes = getattr(nudata, "semantic_classes", None)
    if args.semantic_head and model_semantic_classes is None:
        print(f"[Rank {global_rank}] ERROR: semantic head requested but datamodule has no semantic_classes.")
        raise SystemExit(1)

    if global_rank == 0:
        print("[Rank 0] Initializing model...")

    common_kwargs = dict(
        in_features=args.in_features,
        hit_features=args.hit_features,
        nexus_features=args.nexus_features,
        instance_features=args.instance_features,
        interaction_features=args.interaction_features,
        semantic_classes=model_semantic_classes,
        semantic_class_weight=args.semantic_class_weight,
        event_classes=getattr(nudata, "event_classes", None),
        num_iters=args.num_iters,
        event_head=args.event_head,
        semantic_head=args.semantic_head,
        filter_head=args.filter_head,
        vertex_head=args.vertex_head,
        instance_head=args.instance_head,
        use_checkpointing=args.use_checkpointing,
        use_sp_features=args.use_sp_features,      # NEW
        use_vtx_features=args.use_vtx_features,    # NEW
        lr=args.learning_rate,
    )

    if model_name == "nugraph4":
        nugraph = Model(
            **common_kwargs,
            edge_hidden_dim=args.edge_hidden_dim,
            embed_dim=args.embed_dim,
            lambda_edge=args.lambda_edge,
            edge_pos_weight=args.edge_pos_weight,
            lambda_embed=args.lambda_embed,
            lambda_coh=args.lambda_coh,
            coh_edge_thr=args.coh_edge_thr,
            coh_min_cluster=args.coh_min_cluster,
        )
    else:
        nugraph = Model(**common_kwargs)

    # --- SANITY: semantic head out dim ---
    if global_rank == 0:
        try:
            last_fc = _get_semantic_last_linear(nugraph)
            sc = getattr(nudata, "semantic_classes", None)
            out_dim = last_fc.out_features if last_fc is not None else None
            print(f"[Sanity] semantic head out_features={out_dim} vs len(semantic_classes)={len(sc) if sc else None}")
        except Exception as e:
            print("[Sanity] Could not read semantic head output dim:", e)

    # Ensure FP32 params
    for p in nugraph.parameters():
        if p.dtype != torch.float32:
            p.data = p.data.float()
            if p.grad is not None:
                p.grad = p.grad.float()

    # Initialize semantic bias with uniform priors
    if model_semantic_classes:
        last_fc = _get_semantic_last_linear(nugraph)
        if last_fc is not None and isinstance(last_fc, nn.Linear):
            priors = torch.ones(len(model_semantic_classes), dtype=torch.float32) / len(model_semantic_classes)
            with torch.no_grad():
                last_fc.bias.copy_(priors.log())
            if global_rank == 0:
                print("[Rank 0] Initialized semantic bias with uniform priors (CPU).")

        # --- Callbacks ---
    checkpoint_dir = logdir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        pl.callbacks.EarlyStopping(
            monitor="semantic/f1-macro-val",
            mode="max",
            patience=20,
            min_delta=1e-4,
            verbose=(global_rank == 0),
        ),
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
        
        # Semantic F1 checkpoint (primary for semantic-only training)
        pl.callbacks.ModelCheckpoint(
            monitor="semantic/f1-macro-val",
            mode="max",
            save_top_k=3,
            filename="best-semantic-{epoch:02d}-{semantic/f1-macro-val:.4f}",
            dirpath=str(checkpoint_dir),
            save_last=True,
            verbose=(global_rank == 0),
        ),
        
        # Semantic recall checkpoint
        pl.callbacks.ModelCheckpoint(
            monitor="semantic/recall-macro-val",
            mode="max",
            save_top_k=1,
            filename="best-recall-{epoch:02d}-{semantic/recall-macro-val:.4f}",
            dirpath=str(checkpoint_dir),
            verbose=(global_rank == 0),
        ),
        
        # Instance/Edge F1 checkpoint (primary for instance segmentation)
        # Only active when lambda_edge > 0 (will show warnings in Phase 1, but harmless)
        pl.callbacks.ModelCheckpoint(
            monitor="edge/f1",
            mode="max",
            save_top_k=3,
            filename="best-instance-{epoch:02d}-{edge/f1:.4f}",
            dirpath=str(checkpoint_dir),
            verbose=(global_rank == 0),
        ),
        
        # Loss checkpoint (fallback)
        pl.callbacks.ModelCheckpoint(
            monitor="loss/val",
            mode="min",
            save_top_k=1,
            filename="best-loss-{epoch:02d}-{loss/val:.4f}",
            dirpath=str(checkpoint_dir),
            verbose=(global_rank == 0),
        ),
    ]

    # --- Trainer config ---
    use_gpu = torch.cuda.is_available()
    accelerator = "gpu" if use_gpu else "cpu"

    # IMPORTANT: distributed sampler only when actually distributed
    use_distributed_sampler = (world_size > 1)

    if world_size > 1:
        # External MPI: 1 rank == 1 GPU (visible device 0 inside rank)
        ranks_per_node = int(os.environ.get("PMI_LOCAL_SIZE", os.environ.get("LOCAL_WORLD_SIZE", "1")))
        ranks_per_node = max(1, ranks_per_node)
        devices = 1 if use_gpu else 0
        num_nodes = max(1, world_size // ranks_per_node)

        print(
            f"[Rank {global_rank}] DDP via external MPI: world_size={world_size}, "
            f"ranks_per_node={ranks_per_node}, devices={devices}, num_nodes={num_nodes}"
        )

        from pytorch_lightning.strategies import DDPStrategy

        strategy = DDPStrategy(
            process_group_backend="nccl",
            find_unused_parameters=False,
            timeout=timedelta(minutes=10),
            cluster_environment=ExternalMPIEnvironment(),
        )
    else:
        # Single process (EAF)
        devices = 1 if use_gpu else 0
        num_nodes = 1
        strategy = "auto"
        print(f"[Rank {global_rank}] Single-process Trainer: devices={devices}, num_nodes={num_nodes}")

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        num_nodes=num_nodes,
        strategy=strategy,
        num_sanity_val_steps=0,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        limit_test_batches=args.limit_test_batches,
        logger=logger,
        callbacks=callbacks,
        precision="32-true",
        max_epochs=args.max_epochs,
        enable_progress_bar=(global_rank == 0),
        enable_checkpointing=True,
        sync_batchnorm=False,
        use_distributed_sampler=use_distributed_sampler,
    )

    print(f"[Rank {global_rank}] Starting training...")
    ckpt_path = args.resume_from
    if ckpt_path:
        print(f"[Rank {global_rank}] Resuming from checkpoint: {ckpt_path}")
        trainer.fit(nugraph, datamodule=nudata, ckpt_path=ckpt_path)
    else:
        trainer.fit(nugraph, datamodule=nudata)

    # Testing
    print(f"[Rank {global_rank}] Starting testing...")
    try:
        trainer.test(model=nugraph, datamodule=nudata, ckpt_path="best")
    except Exception as e:
        print(f"[Rank {global_rank}] Test failed: {e}")

    if global_rank == 0:
        try:
            if isinstance(logger, WandbLogger):
                import wandb
                wandb.finish()
        except Exception:
            pass
        print(f"[Rank 0] Run {run_name} complete. Logs/checkpoints in {logdir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="NuGraph Training Script (EAF 1GPU + Polaris mpiexec)")

    # --- Paths ---
    p.add_argument("--data-path", type=str, required=True, help="Path to HDF5 data file.")
    p.add_argument("--log-dir", type=str, required=True, help="Base directory for logs and checkpoints.")
    p.add_argument("--run-name", type=str, required=True, help="Specific name for this run.")

    # --- Hardware & Dataloading ---
    p.add_argument("--gpus-per-node", type=int, default=4, help="GPUs per node (Polaris=4). Unused on EAF.")
    p.add_argument("--num-workers", type=int, default=2, help="DataLoader workers per rank (0 often safest for HDF5).")
    p.add_argument("--batch-size", type=int, default=32, help="Batch size per GPU (per rank).")

    # --- Training ---
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--max-epochs", type=int, default=50)

    # --- Model selection ---
    p.add_argument("--model", type=str, default="nugraph3", choices=["nugraph3", "nugraph4"])

    # --- Model config ---
    p.add_argument("--in-features", type=int, default=10, help="Will be overridden from batch['hit'].x if mismatch.")
    p.add_argument("--hit-features", type=int, default=256)
    p.add_argument("--nexus-features", type=int, default=64)
    p.add_argument("--instance-features", type=int, default=32)
    p.add_argument("--interaction-features", type=int, default=32)
    p.add_argument("--num-iters", dest="num_iters", type=int, default=7, help="Message passing iterations.")  # FIXED

    # --- Heads ---
    p.add_argument("--semantic-class-weight", type=float, nargs=2, default=[21.0, 1.0])
    p.add_argument("--event-head", action="store_true", default=False)
    p.add_argument("--semantic-head", dest="semantic_head", action="store_true", default=True)
    p.add_argument("--no-semantic-head", dest="semantic_head", action="store_false")
    p.add_argument("--filter-head", action="store_true", default=False)
    p.add_argument("--vertex-head", action="store_true", default=False)
    p.add_argument("--instance-head", action="store_true", default=False)

    # --- NuGraph4-specific ---
    p.add_argument("--edge-hidden-dim", type=int, default=32)
    p.add_argument("--embed-dim", type=int, default=64)
    p.add_argument("--lambda-edge", type=float, default=0.0)
    p.add_argument("--edge-pos-weight", type=float, default=1.0)
    p.add_argument("--lambda-embed", type=float, default=0.2)
    p.add_argument("--lambda-coh", type=float, default=0.0)
    p.add_argument("--coh-edge-thr", type=float, default=0.7)
    p.add_argument("--coh-min-cluster", type=int, default=2)

    # --- NEW: SP Feature Control ---
    p.add_argument("--use-sp-features", action="store_true", default=True,
                   help="Use SP-level features (charge, hit count) in encoder (default: True)")
    p.add_argument("--no-sp-features", action="store_false", dest="use_sp_features",
                   help="Disable SP-level features entirely")
    p.add_argument("--use-vtx-features", action="store_true", default=False,
                   help="Include vertex distance features (cols 2-5). ⚠️ USES MC TRUTH - for ablation only!")

    # --- Other ---
    p.add_argument("--use-checkpointing", dest="use_checkpointing", action="store_true", default=True)
    p.add_argument("--no-checkpointing", dest="use_checkpointing", action="store_false")
    p.add_argument("--resume-from", type=str, default=None)
    p.add_argument("--min-nu-hits", type=int, default=0)
    p.add_argument("--shuffle", type=str, default="weighted", choices=["random", "balance", "weighted"])
    p.add_argument("--balance-frac", type=float, default=0.10)

    p.add_argument("--limit-train-batches", type=float, default=1.0)
    p.add_argument("--limit-val-batches", type=float, default=1.0)
    p.add_argument("--limit-test-batches", type=float, default=1.0)
    p.add_argument("--train-fraction", type=float, default=1.0)

    args = p.parse_args()

    pl.seed_everything(1337, workers=True)
    main(args)
