#!/usr/bin/env python3

import argparse
from pathlib import Path
import re
import subprocess
import sys

SUBDIRS = ["img-clus", "celltree0", "celltree1"]
IDX_RE = re.compile(r"_(\d+)$")


def xrdfs_ls(host: str, path: str) -> list[str]:
    p = subprocess.run(
        ["xrdfs", host, "ls", path],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if p.returncode != 0:
        raise RuntimeError(f"xrdfs ls failed for {path}\n{p.stderr}")
    return [x.strip() for x in p.stdout.splitlines() if x.strip()]


def get_indexed_dirs(host: str, base: str, subdir: str) -> dict[int, str]:
    out = {}
    for path in xrdfs_ls(host, f"{base}/{subdir}"):
        name = Path(path.rstrip("/")).name
        m = IDX_RE.search(name)
        if not m:
            continue
        idx = int(m.group(1))
        if idx in out:
            raise RuntimeError(f"Duplicate job index {idx} under {subdir}")
        out[idx] = path.rstrip("/")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build aligned img-clus/celltree job lists for an SBND production."
    )
    parser.add_argument("--base", required=True, help="XRootD-visible production base")
    parser.add_argument("--xrd-host", default="fndcadoor.fnal.gov:1094")
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    maps = {
        subdir: get_indexed_dirs(args.xrd_host, args.base.rstrip("/"), subdir)
        for subdir in SUBDIRS
    }

    all_sets = {k: set(v) for k, v in maps.items()}
    common = set.intersection(*all_sets.values())
    union = set.union(*all_sets.values())

    missing_rows = []
    for idx in sorted(union):
        missing = [subdir for subdir in SUBDIRS if idx not in all_sets[subdir]]
        if missing:
            missing_rows.append((idx, ",".join(missing)))

    joblist = args.output_dir / "joblist_ncpi0.txt"
    triplet_joblist = args.output_dir / "joblist_ncpi0_triplets.txt"
    triplets = args.output_dir / "ncpi0_triplets.tsv"
    missing_path = args.output_dir / "ncpi0_missing_triplets.tsv"
    summary_path = args.output_dir / "ncpi0_mapping_summary.txt"

    with joblist.open("w") as fout:
        for idx in sorted(common):
            fout.write(maps["img-clus"][idx] + "\n")

    with triplet_joblist.open("w") as fout:
        for idx in sorted(common):
            fout.write(
                f"{maps['img-clus'][idx]} {maps['celltree0'][idx]} "
                f"{maps['celltree1'][idx]}\n"
            )

    with triplets.open("w") as fout:
        fout.write("index\timg_clus\tcelltree0\tcelltree1\n")
        for idx in sorted(common):
            fout.write(
                f"{idx}\t{maps['img-clus'][idx]}\t"
                f"{maps['celltree0'][idx]}\t{maps['celltree1'][idx]}\n"
            )

    with missing_path.open("w") as fout:
        fout.write("index\tmissing_subdirs\n")
        for idx, missing in missing_rows:
            fout.write(f"{idx}\t{missing}\n")

    summary = [
        f"BASE={args.base}",
        f"img-clus directories={len(maps['img-clus'])}",
        f"celltree0 directories={len(maps['celltree0'])}",
        f"celltree1 directories={len(maps['celltree1'])}",
        f"complete triplets={len(common)}",
        f"indices missing one or more input directories={len(missing_rows)}",
        f"joblist={joblist}",
        f"triplet_joblist={triplet_joblist}",
        f"mapping={triplets}",
        f"missing={missing_path}",
    ]

    text = "\n".join(summary) + "\n"
    print(text)
    summary_path.write_text(text)

    print("First five aligned triplets:")
    for idx in sorted(common)[:5]:
        print(
            idx,
            Path(maps["img-clus"][idx]).name,
            Path(maps["celltree0"][idx]).name,
            Path(maps["celltree1"][idx]).name,
        )

    if not common:
        sys.exit("[ERROR] No aligned input triplets found.")


if __name__ == "__main__":
    main()
