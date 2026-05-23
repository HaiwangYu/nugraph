"""NuGraph3 model architecture"""
import argparse
import warnings
import os

import torch.cuda
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from pytorch_lightning import LightningModule

from .types import Data
from .transform import Transform
from .encoder import Encoder
from .core import NuGraphCore
from .decoders import (SemanticDecoder, FilterDecoder, EventDecoder, VertexDecoder, InstanceDecoder,
                       SpacepointDecoder)

from ...data import H5DataModule

# Optional RMM (RAPIDS) allocator: enable only if requested AND available
_use_rmm_env = os.environ.get("NUGRAPH_USE_RMM", "0").lower() in ("1","true","yes","y")
if torch.cuda.is_available() and _use_rmm_env:
    try:
        from rmm.allocators.torch import rmm_torch_allocator  # type: ignore
        torch.cuda.memory.change_current_allocator(rmm_torch_allocator)
        print("[nugraph] Using RMM CUDA allocator.")
    except Exception as _e:
        print(f"[nugraph] RMM not available ({_e}); using default CUDA allocator.")


class NuGraph3(LightningModule):
    """
    NuGraph3 model architecture.

    Args:
        in_features: Number of input node features
        hit_features: Number of hit node features
        nexus_features: Number of nexus node features
        interaction_features: Number of interaction node features
        instance_features: Number of instance features
        planes: Tuple of detector plane names
        semantic_classes: Tuple of semantic classes
        event_classes: Tuple of event classes
        num_iters: Number of message-passing iterations
        event_head: Whether to enable event decoder
        semantic_head: Whether to enable semantic decoder
        filter_head: Whether to enable filter decoder
        vertex_head: Whether to enable vertex decoder
        instance_head: Whether to enable instance decoder
        spacepoint_head: Whether to enable spacepoint decoder
        use_checkpointing: Whether to use checkpointing
        use_sp_features: Whether to use SP-level features in encoder
        use_vtx_features: Whether to include vertex features (cheating if True)
        lr: Learning rate
    """
    def __init__(self,
                 in_features: int = 4,
                 hit_features: int = 128,
                 nexus_features: int = 32,
                 interaction_features: int = 32,
                 instance_features: int = 8,
                 planes: tuple[str] = ("u","v","y"),
                 semantic_classes: tuple[str] = ('MIP','HIP','shower','michel','diffuse'),
                 semantic_class_weight: list[float] | None = None,
                 event_classes: tuple[str] = ('numu','nue','nc'),
                 num_iters: int = 5,
                 event_head: bool = False,
                 semantic_head: bool = True,
                 filter_head: bool = True,
                 vertex_head: bool = False,
                 instance_head: bool = False,
                 spacepoint_head: bool = False,
                 use_checkpointing: bool = False,
                 use_sp_features: bool = True,
                 use_vtx_features: bool = False,
                 lr: float = 0.001):
        super().__init__()

        warnings.filterwarnings("ignore", ".*NaN values found in confusion matrix.*")

        self.save_hyperparameters()

        self.hit_features = hit_features
        self.nexus_features = nexus_features
        self.interaction_features = interaction_features

        self.semantic_classes = semantic_classes
        self.event_classes = event_classes
        self.num_iters = num_iters
        self.lr = lr
        
        # Store feature flags for reference
        self.use_sp_features = use_sp_features
        self.use_vtx_features = use_vtx_features

        # Encoder with SP feature support
        self.encoder = Encoder(
            in_features, 
            hit_features,
            nexus_features, 
            interaction_features,
            use_sp_features=use_sp_features,
            use_vtx_features=use_vtx_features,
        )

        self.core_net = NuGraphCore(hit_features,
                                    nexus_features,
                                    interaction_features,
                                    use_checkpointing)

        self.decoders = []

        if event_head:
            self.event_decoder = EventDecoder(interaction_features, event_classes)
            self.decoders.append(self.event_decoder)

        if semantic_head:
            self.semantic_decoder = SemanticDecoder(hit_features, semantic_classes, class_weight=self.hparams.semantic_class_weight)
            self.decoders.append(self.semantic_decoder)
            

        if filter_head:
            self.filter_decoder = FilterDecoder(hit_features,)
            self.decoders.append(self.filter_decoder)

        if vertex_head:
            self.vertex_decoder = VertexDecoder(interaction_features)
            self.decoders.append(self.vertex_decoder)

        if instance_head:
            self.instance_decoder = InstanceDecoder(hit_features, instance_features)
            self.decoders.append(self.instance_decoder)

        if spacepoint_head:
            self.spacepoint_decoder = SpacepointDecoder(hit_features, len(planes))
            self.decoders.append(self.spacepoint_decoder)

        if not self.decoders:
            raise RuntimeError('At least one decoder head must be enabled!')
        
        # One-time logging of feature configuration
        if not hasattr(self, '_logged_feature_config'):
            print(f"[NuGraph3] Feature configuration:")
            print(f"  use_sp_features: {use_sp_features}")
            print(f"  use_vtx_features: {use_vtx_features}")
            if use_vtx_features:
                print(f"  ⚠️  WARNING: Vertex features enabled - using MC truth info!")
            self._logged_feature_config = True

    def forward(self, data: Data, stage: str = None): # pylint: disable=arguments-differ
        """
        NuGraph3 forward function

        This function runs the forward pass of the NuGraph3 architecture,
        and then loops over each decoder to compute the loss and calculate
        and log any performance metrics.

        Args:
            data: Graph data object
            stage: String tag defining the step type
        """
        self.encoder(data)
        for _ in range(self.num_iters):
            self.core_net(data)
        total_loss = 0.
        total_metrics = {}
        for decoder in self.decoders:
            loss, metrics = decoder(data, stage)
            total_loss += loss
            total_metrics.update(metrics)

        return total_loss, total_metrics

    def _sync_dist(self) -> bool:
        try:
            return int(getattr(self.trainer, "world_size", 1)) > 1
        except RuntimeError:
            return False

    def training_step(self,
                      batch: Data,
                      batch_idx: int) -> float:
        loss, metrics = self(batch, 'train')
        sync_dist = self._sync_dist()
        self.log('loss/train', loss, batch_size=batch.num_graphs, prog_bar=True, sync_dist=sync_dist)
        self.log_dict(metrics, batch_size=batch.num_graphs, sync_dist=sync_dist)
        return loss

    def on_train_epoch_end(self) -> None:
        # stop updating running average for feature norm
        self.encoder.input_norm.update = False

    def validation_step(self,
                        batch,
                        batch_idx: int) -> None:
        loss, metrics = self(batch, 'val')
        sync_dist = self._sync_dist()
        self.log('loss/val', loss, batch_size=batch.num_graphs, sync_dist=sync_dist)
        self.log_dict(metrics, batch_size=batch.num_graphs, sync_dist=sync_dist)

    def on_validation_epoch_end(self) -> None:
        epoch = self.trainer.current_epoch + 1
        for decoder in self.decoders:
            decoder.on_epoch_end(self.logger, 'val', epoch)

    def test_step(self,
                  batch,
                  batch_idx: int = 0) -> None:
        loss, metrics = self(batch, 'test')
        sync_dist = self._sync_dist()
        self.log('loss/test', loss, batch_size=batch.num_graphs, sync_dist=sync_dist)
        self.log_dict(metrics, batch_size=batch.num_graphs, sync_dist=sync_dist)

    def on_test_epoch_end(self) -> None:
        epoch = self.trainer.current_epoch + 1
        for decoder in self.decoders:
            decoder.on_epoch_end(self.logger, 'test', epoch)

    def predict_step(self,
                     batch: Data,
                     batch_idx: int = 0) -> Data:
        self(batch)
        return batch

    def configure_optimizers(self) -> tuple:
        optimizer = AdamW(self.parameters(),
                          lr=self.lr)
        onecycle = OneCycleLR(
                optimizer,
                max_lr=self.lr,
                total_steps=self.trainer.estimated_stepping_batches)
        return [optimizer], {'scheduler': onecycle, 'interval': 'step'}

    @staticmethod
    def transform(planes: tuple[str], in_features: int = 4) -> Transform:
        return Transform(planes, in_features=in_features)


    @staticmethod
    def add_model_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """
        Add argparse argument group for NuGraph3 model

        Args:
            parser: Argument parser to append argument group to
        """
        model = parser.add_argument_group('model', 'NuGraph3 model configuration')
        model.add_argument('--num-iters', type=int, default=5,
                           help='Number of message-passing iterations')
        model.add_argument('--in-feats', type=int, default=4,
                           help='Number of input node features')
        model.add_argument('--hit-feats', type=int, default=128,
                           help='Hidden dimensionality of hit convolutions')
        model.add_argument('--nexus-feats', type=int, default=32,
                           help='Hidden dimensionality of nexus convolutions')
        model.add_argument('--interaction-feats', type=int, default=32,
                           help='Hidden dimensionality of interaction layer')
        model.add_argument('--instance-feats', type=int, default=8,
                           help='Hidden dimensionality of object condensation')
        model.add_argument('--event', action='store_true',
                           help='Enable event classification head')
        model.add_argument('--semantic', action='store_true',
                           help='Enable semantic segmentation head')
        model.add_argument('--filter', action='store_true',
                           help='Enable background filter head')
        model.add_argument('--instance', action='store_true',
                           help='Enable instance segmentation head')
        model.add_argument('--vertex', action='store_true',
                           help='Enable vertex regression head')
        model.add_argument("--spacepoint", action="store_true",
                           help="Enable spacepoint prediction head")
        model.add_argument('--no-checkpointing', action='store_false',
                           dest="use_checkpointing",
                           help='Disable checkpointing during training')
        model.add_argument('--epochs', type=int, default=80,
                           help='Maximum number of epochs to train for')
        model.add_argument('--learning-rate', type=float, default=0.001,
                           help='Max learning rate during training')
        # NEW: SP feature control
        model.add_argument('--use-sp-features', action='store_true', default=True,
                           help='Use SP-level features (charge, hit count) in encoder')
        model.add_argument('--no-sp-features', action='store_false', dest='use_sp_features',
                           help='Disable SP-level features in encoder')
        model.add_argument('--use-vtx-features', action='store_true', default=False,
                           help='Include vertex distance features (USES MC TRUTH - for ablation only)')
        return parser

    @classmethod
    def from_args(cls, args: argparse.Namespace, nudata: H5DataModule) -> 'NuGraph3':
        """
        Construct model from arguments

        Args:
            args: Namespace containing parsed arguments
            nudata: Data module
        """
        return cls(
            in_features=args.in_feats,
            hit_features=args.hit_feats,
            nexus_features=args.nexus_feats,
            interaction_features=args.interaction_feats,
            instance_features=args.instance_feats,
            planes=nudata.planes,
            semantic_classes=nudata.semantic_classes,
            event_classes=nudata.event_classes,
            num_iters=args.num_iters,
            event_head=args.event,
            semantic_head=args.semantic,
            filter_head=args.filter,
            vertex_head=args.vertex,
            instance_head=args.instance,
            spacepoint_head=args.spacepoint,
            use_checkpointing=args.use_checkpointing,
            use_sp_features=getattr(args, 'use_sp_features', True),
            use_vtx_features=getattr(args, 'use_vtx_features', False),
            lr=args.learning_rate)
