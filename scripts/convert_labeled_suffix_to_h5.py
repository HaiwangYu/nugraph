#!/usr/bin/env python3
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import h5py
import numpy as np

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

# Direct script execution places scripts/, not the repository root, on
# sys.path. Add only this checkout's root so the integrated pywcml is used.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from pywcml.converter import WCMLConverter
from pywcml.config import ConversionConfig


LAB_RE = re.compile(r"rec-lab-apa[01]-\d+\.npz$")

AUTH_PATTERNS = (
    "Auth failed",
    "No protocols left to try",
    "Could not get bearer token",
)


def looks_like_auth_failure(text: str) -> bool:
    text = text or ""
    return any(pat in text for pat in AUTH_PATTERNS)


def refresh_token() -> None:
    """
    Refresh SBND bearer token.

    After the first browser login, htgettoken normally uses the saved refresh
    token and succeeds non-interactively.
    """
    uid = os.getuid()
    os.environ["BEARER_TOKEN_FILE"] = f"/tmp/bt_u{uid}"

    print("[auth] refreshing SBND token...", flush=True)
    p = subprocess.run(
        ["htgettoken", "-i", "sbnd", "--vaultserver", "htvaultprod.fnal.gov"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=180,
    )
    msg = (p.stdout or "") + "\n" + (p.stderr or "")
    if p.returncode != 0:
        print("[auth] htgettoken failed:", msg[-1500:], flush=True)
        raise RuntimeError("htgettoken failed; cannot continue safely")
    print("[auth] token refresh OK", flush=True)


def run(cmd, check=True):
    p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if check and p.returncode != 0:
        raise RuntimeError(
            f"Command failed rc={p.returncode}\n"
            f"CMD={' '.join(cmd)}\n"
            f"STDOUT={p.stdout}\n"
            f"STDERR={p.stderr}"
        )
    return p


def xrdfs_ls(host, path):
    """
    List a PNFS/XRootD directory.

    Auth failures are never treated as empty directories. We refresh token and
    retry. Non-auth missing-directory errors return an empty list.
    """
    last = ""
    for attempt in range(1, 4):
        p = run(["xrdfs", host, "ls", path], check=False)
        combined = (p.stdout or "") + "\n" + (p.stderr or "")
        last = combined

        if p.returncode == 0:
            return [x.strip() for x in p.stdout.splitlines() if x.strip()]

        if looks_like_auth_failure(combined):
            print(f"[auth] xrdfs auth failure attempt {attempt}/3 for {path}", flush=True)
            refresh_token()
            continue

        return []

    raise RuntimeError(f"xrdfs auth failed after retries for {path}\nlast={last[-1500:]}")


def xrdcp_with_retries(host, remote_path, local_path, retries=5, sleep_s=10):
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)

    # If PNFS is locally mounted on this machine, avoid XRootD.
    src = Path(remote_path)
    if src.exists():
        shutil.copy2(src, local_path)
        if local_path.exists() and local_path.stat().st_size > 0:
            return True

    url = f"root://{host}{remote_path}"
    last = ""

    for attempt in range(1, retries + 1):
        p = subprocess.run(
            ["xrdcp", "-f", url, str(local_path)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        if p.returncode == 0 and local_path.exists() and local_path.stat().st_size > 0:
            return True

        last = (p.stdout or "") + "\n" + (p.stderr or "")
        print(f"[WARN] xrdcp failed attempt {attempt}/{retries}: {url}", flush=True)
        print(last[-1000:], flush=True)

        if looks_like_auth_failure(last):
            refresh_token()

        time.sleep(sleep_s * attempt)

    raise RuntimeError(f"xrdcp failed after {retries} attempts: {url}\nlast_err={last}")


def collect_remote_npzs(joblist, labeled_base, host):
    rows = [line.strip().split() for line in Path(joblist).read_text().splitlines() if line.strip()]
    out = []

    labeled_base_path = Path(labeled_base)
    use_local = labeled_base_path.exists()

    print(f"[collect] labeled_base local_exists={use_local}: {labeled_base}", flush=True)

    for i, row in enumerate(rows, 1):
        reco = row[0]
        job = Path(reco).name

        if use_local:
            outdir = labeled_base_path / job
            labs = sorted(outdir.glob("rec-lab-apa[01]-*.npz"))
            for local_path in labs:
                unique_name = f"{job}__{local_path.name}"
                out.append((job, str(local_path), unique_name))
        else:
            outdir = f"{labeled_base}/{job}"
            files = xrdfs_ls(host, outdir)
            labs = sorted([f for f in files if LAB_RE.search(Path(f).name)])
            for remote in labs:
                unique_name = f"{job}__{Path(remote).name}"
                out.append((job, remote, unique_name))

        if i % 500 == 0:
            print(f"[collect] checked {i}/{len(rows)} jobs, npz={len(out)}", flush=True)

    return out


def write_manifest(rows, path):
    with open(path, "w") as f:
        for job, remote, unique_name in rows:
            f.write(f"{job} {remote} {unique_name}\n")


def convert_chunk(rows, chunk_id, tmp_base, chunk_out, host, copy_retries, copy_sleep, workers, diagnostics):
    cache = tmp_base / "npz_cache" / f"chunk_{chunk_id:05d}"
    if cache.exists():
        shutil.rmtree(cache)
    cache.mkdir(parents=True, exist_ok=True)

    local_paths = []
    for job, remote, unique_name in rows:
        local = cache / unique_name
        xrdcp_with_retries(host, remote, local, retries=copy_retries, sleep_s=copy_sleep)
        local_paths.append(local)

    cfg = ConversionConfig(write_diagnostics=diagnostics)
    conv = WCMLConverter(cfg)

    print(f"[convert] chunk {chunk_id}: converting {len(local_paths)} files", flush=True)
    graphs = conv.convert_many(local_paths, workers=workers)

    out = chunk_out / f"chunk_{chunk_id:05d}.h5"
    if out.exists():
        out.unlink()
    conv.write_hdf5(graphs, out)

    print(f"[convert] chunk {chunk_id}: wrote {out}", flush=True)

    shutil.rmtree(cache)
    return out


def copy_metadata_once(fin, fout):
    for key in fin.keys():
        if key in {"dataset", "samples", "datasize"}:
            continue
        if key not in fout:
            fin.copy(key, fout, name=key)


def write_samples_and_datasize(fout, event_names, train_fraction=0.70, val_fraction=0.15, seed=12345):
    rng = np.random.default_rng(seed)

    names = np.array(sorted(event_names), dtype=object)
    idx = np.arange(len(names))
    rng.shuffle(idx)

    n = len(names)
    n_train = int(round(train_fraction * n))
    n_val = int(round(val_fraction * n))

    train = names[idx[:n_train]]
    val = names[idx[n_train:n_train + n_val]]
    test = names[idx[n_train + n_val:]]

    if "samples" in fout:
        del fout["samples"]

    samples = fout.create_group("samples")
    dt = h5py.string_dtype(encoding="utf-8")
    samples.create_dataset("train", data=train.astype(object), dtype=dt)
    samples.create_dataset("val", data=val.astype(object), dtype=dt)
    samples.create_dataset("test", data=test.astype(object), dtype=dt)

    if "datasize" in fout:
        del fout["datasize"]
    fout.create_dataset("datasize", data=np.array([len(train), len(val), len(test)], dtype=np.int64))

    print(f"[merge] samples: train={len(train)} val={len(val)} test={len(test)}", flush=True)


def merge_chunks(chunk_files, final_h5, overwrite=False):
    final_h5 = Path(final_h5)
    final_h5.parent.mkdir(parents=True, exist_ok=True)

    if final_h5.exists():
        if overwrite:
            final_h5.unlink()
        else:
            raise RuntimeError(f"Final H5 exists; use --overwrite: {final_h5}")

    event_names = []

    with h5py.File(final_h5, "w") as fout:
        dset_out = fout.create_group("dataset")

        first = True
        for cf in chunk_files:
            print(f"[merge] reading {cf}", flush=True)
            with h5py.File(cf, "r") as fin:
                if first:
                    copy_metadata_once(fin, fout)
                    first = False

                for ev in fin["dataset"].keys():
                    if ev in dset_out:
                        raise RuntimeError(f"Duplicate event name during merge: {ev}")
                    fin.copy(fin["dataset"][ev], dset_out, name=ev)
                    event_names.append(ev)

        write_samples_and_datasize(fout, event_names)

    print(f"[merge] wrote final H5: {final_h5}", flush=True)
    print(f"[merge] events: {len(event_names)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", required=True, help="Sample suffix, e.g. 1 or 2")
    ap.add_argument("--joblist", required=True)
    ap.add_argument("--labeled-base", required=True)
    ap.add_argument("--output-h5", required=True)
    ap.add_argument(
        "--tmp-base",
        type=Path,
        default=Path(tempfile.gettempdir()) / "nugraph_wcml_convert",
    )
    ap.add_argument("--xrd-host", default="fndcadoor.fnal.gov:1094")
    ap.add_argument("--files-per-chunk", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--copy-retries", type=int, default=5)
    ap.add_argument("--copy-sleep", type=int, default=10)
    ap.add_argument("--diagnostics", action="store_true")
    ap.add_argument("--max-files", type=int, default=None, help="For smoke tests only.")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--skip-existing-chunks", action="store_true")
    ap.add_argument("--min-files", type=int, default=0, help="Fail if fewer labeled NPZs are collected.")
    args = ap.parse_args()

    refresh_token()

    tmp_base = Path(args.tmp_base) / f"suffix_{args.suffix}"
    chunk_out = tmp_base / "chunk_out"
    manifest_dir = tmp_base / "manifests"
    chunk_out.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    print("[suffix]", args.suffix)
    print("[joblist]", args.joblist)
    print("[labeled-base]", args.labeled_base)
    print("[output-h5]", args.output_h5)
    print("[tmp-base]", tmp_base)
    print("[files-per-chunk]", args.files_per_chunk)

    rows = collect_remote_npzs(args.joblist, args.labeled_base, args.xrd_host)
    rows = sorted(rows, key=lambda x: (x[0], x[2]))

    if args.max_files is not None:
        rows = rows[:args.max_files]

    if not rows:
        raise RuntimeError("No rec-lab-apa*.npz files found.")

    if args.min_files and len(rows) < args.min_files:
        raise RuntimeError(
            f"Collected only {len(rows)} labeled NPZ files, below --min-files={args.min_files}. "
            "This usually means XRootD auth failed during collection or the labeled sample is incomplete."
        )

    manifest = manifest_dir / f"suffix_{args.suffix}_manifest.txt"
    write_manifest(rows, manifest)

    print(f"[collect] total labeled npz files: {len(rows)}")
    print(f"[collect] wrote manifest: {manifest}")

    chunk_files = []
    for chunk_id, start in enumerate(range(0, len(rows), args.files_per_chunk)):
        sub = rows[start:start + args.files_per_chunk]
        chunk_h5 = chunk_out / f"chunk_{chunk_id:05d}.h5"

        if args.skip_existing_chunks and chunk_h5.exists():
            print(f"[skip] existing chunk {chunk_h5}", flush=True)
            chunk_files.append(chunk_h5)
            continue

        out = convert_chunk(
            sub,
            chunk_id=chunk_id,
            tmp_base=tmp_base,
            chunk_out=chunk_out,
            host=args.xrd_host,
            copy_retries=args.copy_retries,
            copy_sleep=args.copy_sleep,
            workers=args.workers,
            diagnostics=args.diagnostics,
        )
        chunk_files.append(out)

    merge_chunks(chunk_files, args.output_h5, overwrite=args.overwrite)

    print("[done]")


if __name__ == "__main__":
    main()
