#!/usr/bin/env python
# plot_h5_edges.py
import argparse, h5py, numpy as np, networkx as nx
import matplotlib
matplotlib.use("Agg")  # no display on login nodes
import matplotlib.pyplot as plt
from pathlib import Path

def draw_graph(coords, edge_index, out_png, title):
    G = nx.Graph()
    n = coords.shape[0]
    G.add_nodes_from(range(n))
    # edge_index is shape (2, E)
    src, dst = edge_index
    edges = list(zip(src.tolist(), dst.tolist()))
    G.add_edges_from(edges)

    # coords: (N,2)
    pos = {i: (coords[i,0], coords[i,1]) for i in range(n)}

    fig = plt.figure(figsize=(6,6), dpi=150)
    ax = plt.gca()
    nx.draw_networkx_nodes(G, pos, node_size=4, linewidths=0)
    nx.draw_networkx_edges(G, pos, width=0.3, alpha=0.5)
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x"); ax.set_ylabel("y/t")
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}  (nodes={n}, edges={len(edges)})")

def main():
    ap = argparse.ArgumentParser(description="Quick HDF5 edge visualizer")
    ap.add_argument("--file", required=True, help="Path to .h5 (ppedges or baseline)")
    ap.add_argument("--event", required=True, help="Event key under /dataset, e.g. 23334072_0_rec-lab-apa0-0")
    ap.add_argument("--plane", choices=["u","v","y"], help="If set, plot 2D plane edges")
    ap.add_argument("--sp3d", action="store_true", help="If set, plot 3D spacepoint ppedges (projected x–z)")
    ap.add_argument("--outdir", default="edge_plots", help="Output directory")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.file, "r") as f:
        grp = f[f"/dataset/{args.event}"]

        if args.plane:
            p = args.plane
            coords = grp[f"{p}/pos"][()]            # shape (N,2): (x,t)
            e = grp[f"{p}_plane_{p}/edge_index"][()]  # shape (2,E)
            if coords.ndim != 2 or coords.shape[1] != 2:
                raise RuntimeError(f"{p}/pos must be (N,2); got {coords.shape}")
            out = outdir / f"{args.event}_plane_{p}.png"
            draw_graph(coords, e, out, f"{args.event}  plane={p}  edges={e.shape[1]}")

        if args.sp3d:
            # 3D spacepoints: project to x–z for plotting
            sp_pos = grp["sp/pos"][()]   # (N,3): (x,y,z)
            e3d = grp["sp_nexus_sp/edge_index"][()]  # (2,E)
            if sp_pos.ndim != 2 or sp_pos.shape[1] != 3:
                raise RuntimeError(f"sp/pos must be (N,3); got {sp_pos.shape}")
            # x–z projection
            coords_xz = sp_pos[:, [0, 2]].astype(np.float32)
            out = outdir / f"{args.event}_sp3d_xz.png"
            draw_graph(coords_xz, e3d, out, f"{args.event}  sp↔sp (ppedges)  edges={e3d.shape[1]}")

if __name__ == "__main__":
    main()
