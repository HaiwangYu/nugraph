# filename: inspect_feature_weight_norms.py
import argparse
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to Lightning checkpoint (.ckpt)")
    ap.add_argument("--in-features", type=int, required=True, help="Input feature dimension (e.g. 18)")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    state_dict = ckpt["state_dict"]

    in_feats = args.in_features

    # 1) Find the first linear layer weight that consumes in_feats
    linear_name = None
    linear_weight = None

    for name, tensor in state_dict.items():
        if not name.endswith(".weight"):
            continue
        if tensor.ndim != 2:
            continue

        out_dim, in_dim = tensor.shape
        if in_dim == in_feats:
            linear_name = name
            linear_weight = tensor
            break

    if linear_weight is None:
        print(f"[Error] No Linear weight with in_features={in_feats} found in state_dict.")
        print("       Maybe double-check --in-features or inspect ckpt keys manually.")
        return 1

    print(f"[Info] Using linear layer: {linear_name} with weight shape {tuple(linear_weight.shape)}")
    w = linear_weight.detach().cpu()  # (out_dim, in_feats)

    # 2) Compute L2 norm of each input feature column
    col_norms = torch.norm(w, dim=0)  # (in_feats,)
    print("\nPer-input-feature weight L2 norms:")
    for i, n in enumerate(col_norms.tolist()):
        print(f"  feat[{i:2d}] norm = {n:.4f}")

    # 3) Try to print InputNorm running stats if present
    mean_key = None
    var_key = None
    for k in state_dict.keys():
        if "input_norm.norm.mean" in k:
            mean_key = k
        if "input_norm.norm.var" in k:
            var_key = k

    if mean_key is not None and var_key is not None:
        mean = state_dict[mean_key].detach().cpu()
        var = state_dict[var_key].detach().cpu()
        print(f"\n[Info] Found InputNorm stats: mean_key={mean_key}, var_key={var_key}")
        print("InputNorm running std per feature:")
        std = (var + 1e-5).sqrt()
        for i, s in enumerate(std.tolist()):
            print(f"  feat[{i:2d}] std ≈ {s:.4f}")
    else:
        print("\n[Warn] Could not find InputNorm running stats in state_dict; "
              "they might live under a different key name.")

    print("\nLegend (for in_features=18):")
    print("  0..2  : x, y, z")
    print("  3..7  : base WCML features (Qtot, mean_err, nhits, pitch_min, pitch_max)")
    print("  8..11 : vertex features    (vtx_dist, vtx_dx, vtx_dy, vtx_dz)")
    print(" 12..17 : sidecar features   (d_wall, d_top, linearity, sphericity, ty, tz)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
