#!/usr/bin/env python3
import argparse, os, random
import h5py, numpy as np
from sklearn.neighbors import NearestNeighbors

# ---- constants the loader expects ----
PLANES = ["u", "v", "y"]
SEMANTIC_CLASSES = ["background", "neutrino"]  # tweak if you change label semantics

# ---- helpers for metadata/splits ----
def _str_dtype():
    return h5py.string_dtype(encoding="utf-8")

def _write_file_metadata(h5: h5py.File):
    """Write minimal metadata in multiple places (root, /metadata, /dataset)."""
    dt = _str_dtype()

    def wds(parent, name, vals):
        if name in parent:
            del parent[name]
        parent.create_dataset(name, data=list(vals), dtype=dt)

    # root datasets + attrs
    wds(h5, "planes", PLANES)
    wds(h5, "semantic_classes", SEMANTIC_CLASSES)
    h5.attrs["planes"] = ",".join(PLANES)
    h5.attrs["semantic_classes"] = ",".join(SEMANTIC_CLASSES)

    # /metadata datasets
    mg = h5.require_group("metadata")
    wds(mg, "planes", PLANES)
    wds(mg, "semantic_classes", SEMANTIC_CLASSES)

    # /dataset datasets + attrs
    ds = h5.require_group("dataset")
    wds(ds, "planes", PLANES)
    wds(ds, "semantic_classes", SEMANTIC_CLASSES)
    ds.attrs["planes"] = ",".join(PLANES)
    ds.attrs["semantic_classes"] = ",".join(SEMANTIC_CLASSES)

def _write_splits(h5: h5py.File, events, seed=42, train_frac=0.8, val_frac=0.1):
    """Write flat split datasets to /samples/* and mirror to /splits/*."""
    dt = _str_dtype()
    evs = list(events)
    rng = random.Random(seed)
    rng.shuffle(evs)

    n = len(evs)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train = evs[:n_train]
    val   = evs[n_train:n_train + n_val]
    test  = evs[n_train + n_val:]

    def wds(group, name, items):
        if name in group:
            del group[name]
        group.create_dataset(name, data=list(items), dtype=dt)

    smp = h5.require_group("samples")
    wds(smp, "train", train)
    wds(smp, "validation",   val)
    wds(smp, "test",  test)

    spl = h5.require_group("splits")
    wds(spl, "train", train)
    wds(spl, "validation",   val)
    wds(spl, "test",  test)

    # dsz = h5.require_group("datasize")
    # dsz.attrs["train"] = len(train)
    # dsz.attrs["validation"]   = len(val)
    # dsz.attrs["test"]  = len(test)
    # return {"train": len(train), "val": len(val), "test": len(test)}

    dsz = h5.require_group("datasize")
    if "train" in dsz:
        del dsz["train"]

    # Create a placeholder dataset of zeros. Its length must match the
    # number of training samples. This satisfies the check in the data loader.
    placeholder_dsize = np.zeros(len(train))
    dsz.create_dataset("train", data=placeholder_dsize)

    # We can still write the counts as attributes for our own reference
    dsz.attrs["train"] = len(train)
    dsz.attrs["validation"] = len(val)
    dsz.attrs["test"] = len(test) 
    return {"train": len(train), "val": len(val), "test": len(test)}

# ---- your existing logic (kept) ----
def knn_edges(pos: np.ndarray, k: int) -> np.ndarray:
    if pos.shape[0] == 0:
        return np.zeros((2, 0), dtype=np.int64)
    nbrs = NearestNeighbors(
        n_neighbors=min(k + 1, max(1, pos.shape[0])), algorithm="kd_tree"
    ).fit(pos)
    _, idxs = nbrs.kneighbors(pos)
    # drop self neighbor at [:,0]
    k_eff = idxs.shape[1] - 1
    src = np.repeat(np.arange(pos.shape[0], dtype=np.int64), k_eff)
    dst = idxs[:, 1:].reshape(-1).astype(np.int64)
    return np.vstack([src, dst])

def write_event(dataset_group: h5py.Group, event_name: str, points: np.ndarray, labels: np.ndarray, k: int):
    # This final version creates a single "compound dataset" for each event,
    # and populates the "x" feature matrix with the "pos" data.

    # Process data into per-plane arrays
    pos_u = points[:, [0, 1]].astype(np.float32)
    pos_v = points[:, [2, 3]].astype(np.float32)
    pos_y = points[:, [4, 5]].astype(np.float32)
    lab = labels.astype(np.int64)

    ei_u = knn_edges(pos_u, k=k)
    ei_v = knn_edges(pos_v, k=k)
    ei_y = knn_edges(pos_y, k=k)

    # Get the variable shapes for this specific event
    Nu, Nv, Ny = pos_u.shape[0], pos_v.shape[0], pos_y.shape[0]
    Eu, Ev, Ey = ei_u.shape[1], ei_v.shape[1], ei_y.shape[1]

    # Define the exact, complex dtype for this event's data shapes
    event_dtype = np.dtype([
        ('u/pos',                'f4', (Nu, 2)),
        ('u/x',                  'f4', (Nu, 2)), # ADD THIS
        ('u/y_semantic',         'i8', (Nu,)),
        ('u_plane_u/edge_index', 'i8', (2, Eu)),
        ('v/pos',                'f4', (Nv, 2)),
        ('v/x',                  'f4', (Nv, 2)), # ADD THIS
        ('v/y_semantic',         'i8', (Nv,)),
        ('v_plane_v/edge_index', 'i8', (2, Ev)),
        ('y/pos',                'f4', (Ny, 2)),
        ('y/x',                  'f4', (Ny, 2)), # ADD THIS
        ('y/y_semantic',         'i8', (Ny,)),
        ('y_plane_y/edge_index', 'i8', (2, Ey))
    ])

    # Create a single structured record as a tuple of the data arrays
    record = (pos_u, pos_u, lab[:Nu], ei_u,          # ADD pos_u here
              pos_v, pos_v, lab[:Nv], ei_v,          # ADD pos_v here
              pos_y, pos_y, lab[:Ny], ei_y)          # ADD pos_y here
    
    # Create a scalar (shape=()) numpy array with this compound dtype
    event_data = np.array(record, dtype=event_dtype)
    
    # Save this single, complex object to the HDF5 file
    if event_name in dataset_group:
        del dataset_group[event_name]
    dataset_group.create_dataset(event_name, data=event_data)

def event_name_from_path(path: str) -> str:
    # Make a stable, readable name; include batch folder + file stem
    stem = os.path.splitext(os.path.basename(path))[0]  # rec-lab-apa0-7
    parent = os.path.basename(os.path.dirname(path))    # 23334072_176
    return f"{parent}_{stem}"

def main():
    ap = argparse.ArgumentParser(description="Convert DL-CLUS .npz list to NuGraph-style .gnn.h5")
    ap.add_argument("--list", required=True, help="Path to .lst (npz paths)")
    ap.add_argument("--out",  required=True, help="Output HDF5 path, e.g., our_min.gnn.h5")
    ap.add_argument("--k", type=int, default=10, help="k for per-plane kNN edges")
    ap.add_argument("--limit", type=int, default=0, help="Optional: convert only first N entries")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed for split shuffling")
    ap.add_argument("--train-frac", type=float, default=0.8, help="Train fraction")
    ap.add_argument("--val-frac", type=float, default=0.1, help="Val fraction (rest is test)")
    args = ap.parse_args()

    paths = []
    with open(args.list) as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                paths.append(s)
    if args.limit and args.limit > 0:
        paths = paths[:args.limit]

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Create a fresh, clean HDF5. libver='latest' + track_order tends to avoid FS quirks.
    events_written = []
    with h5py.File(args.out, "w", libver="latest", track_order=True) as h5:
        _write_file_metadata(h5)
        dataset_group = h5.require_group("dataset")

        for i, p in enumerate(paths, 1):
            if not os.path.isfile(p):
                print(f"[skip] not found: {p}")
                continue
            try:
                npz = np.load(p, allow_pickle=True)
                if "points" not in npz or "is_nu" not in npz:
                    print(f"[skip] missing keys in {p} (need 'points' and 'is_nu')")
                    continue
                pts = npz["points"]
                lab = npz["is_nu"]
                if pts.ndim != 2 or pts.shape[1] != 6:
                    print(f"[skip] points shape is {pts.shape}, expected (N,6) in {p}")
                    continue
                if lab.shape[0] != pts.shape[0]:
                    print(f"[skip] label size {lab.shape[0]} != N {pts.shape[0]} in {p}")
                    continue

                evname = event_name_from_path(p)
                write_event(dataset_group, evname, pts, lab, k=args.k)
                events_written.append(evname)
                print(f"[{i}/{len(paths)}] wrote event {evname} (N={pts.shape[0]})")
            except Exception as e:
                print(f"[error] {p}: {e}")

        counts = _write_splits(
            h5, events_written,
            seed=args.seed, train_frac=args.train_frac, val_frac=args.val_frac
        )
        print(f"Wrote splits: {counts} | total events: {len(events_written)}")

if __name__ == "__main__":
    main()
