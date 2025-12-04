from nugraph.data import H5DataModule
from nugraph.models import NuGraph4
import torch

dm = H5DataModule(
    data_path="/lus/eagle/projects/neutrinoGPU/abhat/sbnd/clustering/23334072_nug4_vertex.h5",
    model=NuGraph4,
    batch_size=1,
    num_workers=0,
)
dm.setup("fit")
batch = next(iter(dm.train_dataloader()))
print("hit.x shape:", batch["hit"].x.shape)
