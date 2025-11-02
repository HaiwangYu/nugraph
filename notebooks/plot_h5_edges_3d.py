#!/usr/bin/env python
# plot_h5_edges_3d.py
import argparse, h5py, numpy as np, plotly.graph_objects as go
from pathlib import Path

def load_event(h5file, event):
    """
    Robust loader for NuGraph events stored either as:
      (A) scalar compound dataset under /dataset/<event>  [your case], or
      (B) a Group with subgroups/datasets (older layout).
    Returns: pos(N,3), y(N,) or None, ei(2,E)
    """
    import h5py, numpy as np

    obj = h5file[f"/dataset/{event}"]

    # ---- Case A: scalar compound dataset (current 23334072_3d_ppedges.h5) ----
    if isinstance(obj, h5py.Dataset) and obj.shape == () and obj.dtype.names:
        rec = obj[()]  # numpy.void with named fields like 'sp/pos', 'sp_nexus_sp/edge_index'

        def get_field(name, required=True):
            if name not in obj.dtype.names:
                if required:
                    raise SystemExit(f"Missing field '{name}' in /dataset/{event}")
                return None
            val = rec[name]
            # ensure numpy array for consistency
            return np.array(val)

        pos = get_field("sp/pos")                       # (N,3)
        y   = get_field("sp/y_semantic", required=False)  # optional (N,)
        ei  = get_field("sp_nexus_sp/edge_index")       # (2,E)

        # normalize shapes
        if y is not None and getattr(y, "ndim", 0) == 0:
            y = np.array([y])

        return pos, y, ei

    # ---- Case B: group layout (fallback for other files) ----
    elif isinstance(obj, h5py.Group):
        g = obj
        sp = g.get("sp")
        if sp is None:
            raise SystemExit(f"Missing 'sp' group under /dataset/{event}")
        pos = np.array(sp["pos"][()])
        y = np.array(sp["y_semantic"][()]) if "y_semantic" in sp else None
        ei = np.array(g["sp_nexus_sp/edge_index"][()])
        if y is not None and getattr(y, "ndim", 0) == 0:
            y = np.array([y])
        return pos, y, ei

    else:
        raise SystemExit(f"/dataset/{event} has unsupported type {type(obj)}")





def downsample_edges(ei, max_edges):
    E = ei.shape[1]
    if max_edges and E > max_edges:
        idx = np.random.default_rng(0).choice(E, size=max_edges, replace=False)
        ei = ei[:, idx]
    return ei

def build_edge_segments(pos, ei):
    # Build NaN-separated segments for Plotly lines
    src, dst = ei
    P = np.column_stack([pos[src,0], pos[dst,0], np.full_like(src, np.nan, dtype=float)])
    Q = np.column_stack([pos[src,1], pos[dst,1], np.full_like(src, np.nan, dtype=float)])
    R = np.column_stack([pos[src,2], pos[dst,2], np.full_like(src, np.nan, dtype=float)])
    x = P.reshape(-1); y = Q.reshape(-1); z = R.reshape(-1)
    return x, y, z

def main():
    ap = argparse.ArgumentParser(description="Interactive 3D ppedges visualizer (Plotly)")
    ap.add_argument("--file", required=True, help="Path to .h5 with ppedges")
    ap.add_argument("--event", required=True, help="Event key under /dataset (e.g. 23334072_0_rec-lab-apa0-0)")
    ap.add_argument("--max-edges", type=int, default=5000, help="Cap edges for speed (0 = no cap)")
    ap.add_argument("--point-size", type=float, default=2.0, help="Marker size for nodes")
    ap.add_argument("--out", default="sp3d_ppedges.html", help="Output HTML")
    args = ap.parse_args()

    out = Path(args.out)

    with h5py.File(args.file, "r") as f:
        pos, y, ei = load_event(f, args.event)

    ei = downsample_edges(ei, None if args.max_edges == 0 else args.max_edges)
    x_e, y_e, z_e = build_edge_segments(pos, ei)

    # Node coloring
    if y is not None:
        # assume 0=nu, 1=cosmic (adjust names if reversed)
        labels = np.array(["ν" if i==0 else "cosmic" for i in y])
        marker_color = (y.astype(float))  # 0/1 maps to default colorscale
        hovertext = [f"#{i}  label={labels[i]}" for i in range(len(labels))]
    else:
        marker_color = np.zeros(len(pos))
        hovertext = [f"#{i}" for i in range(len(pos))]

    node_trace = go.Scatter3d(
        x=pos[:,0], y=pos[:,1], z=pos[:,2],
        mode="markers",
        marker=dict(size=args.point_size, opacity=0.8),
        marker_color=marker_color,
        text=hovertext,
        hoverinfo="text",
        name="spacepoints",
    )

    edge_trace = go.Scatter3d(
        x=x_e, y=y_e, z=z_e,
        mode="lines",
        line=dict(width=1, color="rgba(50,50,50,0.25)"),
        hoverinfo="none",
        name=f"edges ({ei.shape[1]})",
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title=f"{args.event} — sp↔sp ppedges (E={ei.shape[1]})",
        showlegend=True,
        scene=dict(aspectmode="data"),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    fig.write_html(out, include_plotlyjs="cdn", full_html=True)
    print(f"Wrote {out.resolve()}")

if __name__ == "__main__":
    main()
