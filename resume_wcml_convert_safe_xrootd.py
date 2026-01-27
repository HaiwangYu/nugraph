#!/usr/bin/env python3
"""
resume_wcml_convert_safe_xrootd.py

WCML NPZ -> NuGraph HDF5 conversion with:
  - Inputs on PNFS scratch (read via XRootD)
  - Chunk outputs staged to PNFS scratch (write via XRootD)
  - Final merged H5 staged to PNFS scratch (write via XRootD)
  - Local scratch used ONLY as transient workspace; local chunk + local NPZ copies deleted after stage-out
  - Periodic + reactive htgettoken refresh (same pattern as your labeling batch script)

Assumptions about PNFS labeled NPZ layout:
  PNFS_INPUT_BASE/
    <job_name>/rec-lab-apa0-*.npz
    <job_name>/rec-lab-apa1-*.npz
    ...

We enumerate NPZs by:
  xrdfs ls PNFS_INPUT_BASE  -> job dirs
  xrdfs ls job_dir          -> rec-lab-apa*-*.npz files
"""

import os
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import sys, math, time, gc, shutil, re, threading, logging
from pathlib import Path
from typing import List, Tuple, Dict, Any

import h5py

# -----------------------------------------------------------------------------
# Repo imports
# -----------------------------------------------------------------------------
repo_path = "/home/abhat/Clustering/lynn_repo/nugraph"
if repo_path not in sys.path:
    sys.path.insert(0, repo_path)

import pywcml
from pywcml.converter import WCMLConverter
from pywcml.config import ConversionConfig

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
XRDFS_HOST = "fndcadoor.fnal.gov:1094"
XRDCP_PREFIX = f"root://{XRDFS_HOST}"

# PNFS paths (namespace paths; use with xrdfs/xrdcp root://...//pnfs/...)
PNFS_INPUT_BASE  = "/pnfs/fnal.gov/usr/sbnd/scratch/users/abhat/dl-light_samples/labeled_samples"
PNFS_CHUNK_DIR   = "/pnfs/fnal.gov/usr/sbnd/scratch/users/abhat/dl-light_samples/_wcml_chunks"
PNFS_FINAL_H5    = "/pnfs/fnal.gov/usr/sbnd/scratch/users/abhat/dl-light_samples/converted_nu_fix.h5"

# Local transient workspace (MUST be local POSIX FS)
LOCAL_WORK_BASE  = Path("/scratch/7DayLifetime/abhat/wirecell/clustering/wcml_convert_tmp")
LOCAL_NPZ_DIR     = LOCAL_WORK_BASE / "npz_cache"   # per-chunk subdirs
LOCAL_CHUNK_DIR   = LOCAL_WORK_BASE / "chunk_out"   # local chunk files
LOCAL_MERGE_DIR   = LOCAL_WORK_BASE / "merge_tmp"   # download chunks here for merge
LOCAL_FINAL_H5    = LOCAL_WORK_BASE / "converted_nu_fix.local.h5"

# Converter runtime
WORKERS = 16
CHUNK_FILES = 5000
MAX_RETRIES_PER_CHUNK = 3

# XRootD copy behavior
XRDCP_RETRIES = 3
XRDCP_RETRY_DELAY = 2.0

# Token refresh
HTGETTOKEN_CMD = ["htgettoken", "-i", "sbnd", "--vaultserver", "htvaultprod.fnal.gov"]
PROACTIVE_REFRESH = True
TOKEN_REFRESH_EVERY_XRD_COPIES = 1500
REACTIVE_REFRESH_ON_AUTH_FAIL = True
BACKGROUND_REFRESH = True
BACKGROUND_REFRESH_INTERVAL = 60 * 40  # 40 minutes

# Stageout behavior
DELETE_LOCAL_AFTER_STAGEOUT = True

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

_xrd_success_counter = 0
_xrd_counter_lock = threading.Lock()

# -----------------------------------------------------------------------------
# Utilities: subprocess
# -----------------------------------------------------------------------------
import subprocess

def run_command(cmd, cwd=None, env=None, capture_output=False):
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE if capture_output else None,
            stderr=subprocess.PIPE if capture_output else None,
            text=True,
        )
        out = proc.stdout if capture_output else ""
        err = proc.stderr if capture_output else ""
        return proc.returncode, out, err
    except FileNotFoundError as e:
        return 127, "", str(e)
    except Exception as e:
        return 1, "", str(e)

# -----------------------------------------------------------------------------
# Token refresh
# -----------------------------------------------------------------------------
def refresh_htgettoken():
    try:
        logging.info("[TOKEN] running htgettoken to refresh token...")
        rc, out, err = run_command(HTGETTOKEN_CMD, capture_output=True)
        if rc == 0:
            logging.info("[TOKEN] htgettoken succeeded.")
            return True
        logging.warning("[TOKEN] htgettoken failed rc=%d out=%s err=%s", rc, out.strip(), err.strip())
        return False
    except Exception as e:
        logging.exception("[TOKEN] exception running htgettoken: %s", e)
        return False

def _increment_xrd_success_counter():
    global _xrd_success_counter
    with _xrd_counter_lock:
        _xrd_success_counter += 1
        cnt = _xrd_success_counter
    if PROACTIVE_REFRESH and cnt and (cnt % TOKEN_REFRESH_EVERY_XRD_COPIES == 0):
        logging.info("[TOKEN] proactive refresh triggered after %d xrd copies", cnt)
        refresh_htgettoken()
    return cnt

def background_token_refresher(interval_seconds=BACKGROUND_REFRESH_INTERVAL):
    def _loop():
        while True:
            time.sleep(interval_seconds)
            logging.info("[TOKEN-BG] background token refresh triggered")
            refresh_htgettoken()
    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    return t

# -----------------------------------------------------------------------------
# XRootD helpers
# -----------------------------------------------------------------------------
def xrdfs_ls(pnfs_dir: str) -> List[str]:
    """Return raw entries for `xrdfs host ls <pnfs_dir>` (full paths)."""
    rc, out, err = run_command(["xrdfs", XRDFS_HOST, "ls", pnfs_dir], capture_output=True)
    if rc != 0:
        logging.warning("xrdfs ls failed for %s rc=%d err=%s", pnfs_dir, rc, err.strip())
        return []
    return [l.strip() for l in (out or "").splitlines() if l.strip()]

def xrdfs_mkdir_p(pnfs_dir: str) -> bool:
    rc, out, err = run_command(["xrdfs", XRDFS_HOST, "mkdir", "-p", pnfs_dir], capture_output=True)
    if rc != 0:
        logging.warning("xrdfs mkdir -p failed for %s rc=%d err=%s out=%s",
                        pnfs_dir, rc, err.strip(), out.strip())
        return False
    return True

def _is_auth_problem(rc: int, out: str, err: str) -> bool:
    msg = (out or "") + (err or "")
    return (rc == 52) or ("Auth failed" in msg) or ("authorization" in msg.lower())

def xrdcp_with_retries(src: str, dst: str, retries: int = XRDCP_RETRIES) -> Tuple[bool, str]:
    """
    Generic xrdcp (either direction). Use src/dst as strings.
      - remote paths should be root://HOST//pnfs/...
      - local paths should be /scratch/...
    """
    last_err = ""
    for attempt in range(1, retries + 1):
        rc, out, err = run_command(["xrdcp", "-f", src, dst], capture_output=True)
        last_err = (out or "") + (err or "")
        if rc == 0:
            _increment_xrd_success_counter()
            return True, out + err

        auth_problem = _is_auth_problem(rc, out, err)
        logging.warning("xrdcp failed attempt %d/%d rc=%d src=%s dst=%s short_err=%s",
                        attempt, retries, rc, src, dst, (err.strip() or out.strip()))
        if REACTIVE_REFRESH_ON_AUTH_FAIL and auth_problem:
            logging.info("[TOKEN] detected auth failure; attempting htgettoken and retry")
            ok = refresh_htgettoken()
            if ok:
                time.sleep(0.5)
                continue
        time.sleep(XRDCP_RETRY_DELAY)
    return False, last_err

def remote_url(pnfs_path: str) -> str:
    """Convert pnfs namespace path to xrootd URL."""
    # Ensure double slash after host
    if pnfs_path.startswith("/pnfs/"):
        return f"{XRDCP_PREFIX}//{pnfs_path}"
    raise ValueError(f"Expected pnfs namespace path, got: {pnfs_path}")

# -----------------------------------------------------------------------------
# Enumerate NPZs on PNFS via xrdfs
# -----------------------------------------------------------------------------
_npz_re = re.compile(r"rec-lab-apa[01]-\d+\.npz$")

def collect_npz_pnfs(pnfs_root: str) -> List[str]:
    """
    Return list of PNFS namespace paths (NOT URLs) for rec-lab-apa*-*.npz.
    Enumerates as: root -> job dirs -> files.
    """
    job_entries = xrdfs_ls(pnfs_root)
    if not job_entries:
        logging.error("No entries under PNFS_INPUT_BASE=%s (xrdfs ls returned empty)", pnfs_root)
        return []

    # Heuristic: job dirs are directories; xrdfs ls returns full paths.
    # We can just attempt to list each entry; if it errors, skip.
    npz_paths: List[str] = []
    for job in sorted(job_entries):
        files = xrdfs_ls(job)
        if not files:
            continue
        for f in files:
            if _npz_re.search(f):
                # filter out any truthTID variants just like your original
                if "truthTID" in Path(f).name:
                    continue
                npz_paths.append(f)

    npz_paths.sort()
    return npz_paths

# -----------------------------------------------------------------------------
# HDF5 open retry (LOCAL files only)
# -----------------------------------------------------------------------------
def open_h5_retry(path: Path, mode: str, tries: int = 20, sleep: float = 0.5):
    for i in range(tries):
        try:
            return h5py.File(str(path), mode)
        except (BlockingIOError, OSError):
            if i == tries - 1:
                raise
            time.sleep(sleep)

# -----------------------------------------------------------------------------
# Merge: local final file + download chunk files from PNFS in batches
# -----------------------------------------------------------------------------
def merge_chunks_from_pnfs(chunk_pnfs_paths: List[str], local_final_h5: Path, batch_size: int = 10):
    """
    For each batch of remote chunks:
      - download chunk to LOCAL_MERGE_DIR
      - copy events into local_final_h5 under /dataset
      - delete local downloaded chunk
    """
    if not chunk_pnfs_paths:
        return

    LOCAL_MERGE_DIR.mkdir(parents=True, exist_ok=True)
    logging.info("[Merge] Starting merge of %d chunks into %s", len(chunk_pnfs_paths), local_final_h5)

    t0 = time.time()
    for i in range(0, len(chunk_pnfs_paths), batch_size):
        batch = chunk_pnfs_paths[i:i + batch_size]
        logging.info("[Merge] Batch %d: %d chunks", i // batch_size + 1, len(batch))

        ff = open_h5_retry(local_final_h5, "a")
        try:
            if "dataset" not in ff:
                raise RuntimeError(f"Final file missing /dataset: {local_final_h5}")

            gdst = ff["dataset"]

            for chunk_pnfs in batch:
                chunk_name = Path(chunk_pnfs).name
                local_chunk = LOCAL_MERGE_DIR / chunk_name

                # download
                ok, msg = xrdcp_with_retries(remote_url(chunk_pnfs), str(local_chunk))
                if not ok:
                    logging.warning("[Merge] download failed for %s: %s", chunk_name, msg[:200])
                    continue

                try:
                    with h5py.File(str(local_chunk), "r") as fc:
                        if "dataset" not in fc:
                            logging.warning("[Merge] no /dataset in %s", chunk_name)
                            continue
                        gsrc = fc["dataset"]
                        n_copied = 0
                        for ev in gsrc.keys():
                            if ev not in gdst:
                                fc.copy(gsrc[ev], gdst, name=ev)
                                n_copied += 1
                        logging.info("[Merge] merged %s: %d new events", chunk_name, n_copied)
                    ff.flush()
                finally:
                    try:
                        local_chunk.unlink()
                    except Exception:
                        pass

        finally:
            ff.close()

    logging.info("[Merge] Completed in %.1fs", time.time() - t0)

# -----------------------------------------------------------------------------
# Chunk processing: download NPZ inputs locally -> convert -> write local chunk -> stageout chunk -> cleanup local
# -----------------------------------------------------------------------------
def download_npz_list_to_local(npz_pnfs_paths: List[str], local_dir: Path, pnfs_input_base: str) -> Tuple[bool, List[Path], List[str]]:
    """
    Download PNFS NPZs to local_dir while preserving job folder to avoid basename collisions.
    local layout:
      local_dir/<job_name>/rec-lab-apaX-N.npz
    Returns (ok_all, local_paths, errors).
    """
    local_dir.mkdir(parents=True, exist_ok=True)
    local_paths: List[Path] = []
    errors: List[str] = []

    base = pnfs_input_base.rstrip("/") + "/"

    for pnfs_path in npz_pnfs_paths:
        # pnfs_path example:
        # /pnfs/.../labeled_samples/67941055_0/rec-lab-apa0-0.npz
        # extract job_name = 67941055_0
        try:
            rel = pnfs_path[len(base):]  # "67941055_0/rec-lab-apa0-0.npz"
        except Exception:
            rel = pnfs_path.lstrip("/")

        job_name = rel.split("/", 1)[0]
        fname = Path(pnfs_path).name

        job_local_dir = local_dir / job_name
        job_local_dir.mkdir(parents=True, exist_ok=True)

        dst = job_local_dir / fname

        if dst.exists() and dst.stat().st_size > 0:
            local_paths.append(dst)
            continue

        ok, msg = xrdcp_with_retries(remote_url(pnfs_path), str(dst))
        if not ok:
            errors.append(f"download_failed:{job_name}/{fname}::{msg[:200]}")
        else:
            local_paths.append(dst)

    return (len(errors) == 0), local_paths, errors


def stageout_file_to_pnfs(local_file: Path, pnfs_dest_dir: str) -> Tuple[bool, str]:
    """
    Ensure pnfs_dest_dir exists, then xrdcp local_file -> pnfs_dest_dir/local_file.name
    """
    if not xrdfs_mkdir_p(pnfs_dest_dir):
        return False, f"mkdir_failed:{pnfs_dest_dir}"
    remote = f"{remote_url(pnfs_dest_dir)}/{local_file.name}"
    ok, msg = xrdcp_with_retries(str(local_file), remote)
    return ok, msg

def process_chunk_safe(ci: int, n_chunks: int, chunk_pnfs_npzs: List[str],
                       pnfs_chunk_dir: str, converter: WCMLConverter,
                       workers: int, max_retries: int = MAX_RETRIES_PER_CHUNK) -> Tuple[bool, str]:
    """
    For chunk index ci:
      - download NPZs to local cache dir
      - convert and write chunk locally
      - stage out chunk H5 to PNFS
      - delete local NPZ cache + local chunk file on success
    """
    chunk_name = f"chunk_{ci:05d}.h5"
    pnfs_chunk_path = f"{pnfs_chunk_dir}/{chunk_name}"
    local_npz_dir = LOCAL_NPZ_DIR / f"chunk_{ci:05d}"
    local_chunk_h5 = LOCAL_CHUNK_DIR / chunk_name

    for attempt in range(max_retries):
        try:
            logging.info("[Chunk %d/%d] attempt %d/%d: %d NPZs",
                         ci + 1, n_chunks, attempt + 1, max_retries, len(chunk_pnfs_npzs))
            t0 = time.time()

            # Download NPZs for this chunk
            ok_dl, local_npzs, dl_errs = download_npz_list_to_local(chunk_pnfs_npzs, local_npz_dir, PNFS_INPUT_BASE)
            if not ok_dl:
                raise RuntimeError(f"NPZ download errors: {dl_errs[:3]} ... (n={len(dl_errs)})")

            # Reduce workers on retry to avoid OOM
            current_workers = workers if attempt == 0 else max(8, workers // 2)

            # Convert
            graphs = converter.convert_many(local_npzs, workers=current_workers)

            # Write local chunk
            LOCAL_CHUNK_DIR.mkdir(parents=True, exist_ok=True)
            if local_chunk_h5.exists():
                local_chunk_h5.unlink()
            logging.info("[Chunk %d/%d] writing local %s", ci + 1, n_chunks, local_chunk_h5.name)
            converter.write_hdf5(graphs, local_chunk_h5)

            del graphs
            gc.collect()

            # Stage out chunk to PNFS
            logging.info("[Chunk %d/%d] stage-out to PNFS: %s", ci + 1, n_chunks, pnfs_chunk_path)
            ok_up, msg = stageout_file_to_pnfs(local_chunk_h5, pnfs_chunk_dir)
            if not ok_up:
                raise RuntimeError(f"chunk stageout failed: {msg[:200]}")

            # Cleanup local transient data
            if DELETE_LOCAL_AFTER_STAGEOUT:
                try:
                    shutil.rmtree(local_npz_dir, ignore_errors=True)
                except Exception:
                    pass
                try:
                    local_chunk_h5.unlink()
                except Exception:
                    pass

            elapsed = time.time() - t0
            logging.info("[Chunk %d/%d] ✓ done in %.1fs", ci + 1, n_chunks, elapsed)
            return True, pnfs_chunk_path

        except Exception as e:
            logging.error("[ERROR] Chunk %d attempt %d failed: %s", ci, attempt + 1, e)

            # Cleanup partial local chunk
            try:
                if local_chunk_h5.exists():
                    local_chunk_h5.unlink()
            except Exception:
                pass

            # Optional: keep NPZ cache for debugging? default delete to save scratch.
            try:
                shutil.rmtree(local_npz_dir, ignore_errors=True)
            except Exception:
                pass

            gc.collect()
            time.sleep(5)

            if attempt == max_retries - 1:
                logging.error("[FATAL] Chunk %d failed after %d attempts", ci, max_retries)
                return False, pnfs_chunk_path

    return False, pnfs_chunk_path

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    # Reduce thread contention aggressively
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(var, "1")

    # Tools check
    if not shutil.which("xrdcp"):
        logging.error("xrdcp not found in PATH.")
        return 1
    if not shutil.which("xrdfs"):
        logging.error("xrdfs not found in PATH.")
        return 1

    # Background token refresh
    if BACKGROUND_REFRESH:
        background_token_refresher(BACKGROUND_REFRESH_INTERVAL)

    # Ensure PNFS dirs exist
    if not xrdfs_mkdir_p(PNFS_CHUNK_DIR):
        logging.error("Could not create PNFS_CHUNK_DIR: %s", PNFS_CHUNK_DIR)
        return 1

    # Ensure local transient dirs exist
    LOCAL_WORK_BASE.mkdir(parents=True, exist_ok=True)
    LOCAL_NPZ_DIR.mkdir(parents=True, exist_ok=True)
    LOCAL_CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    LOCAL_MERGE_DIR.mkdir(parents=True, exist_ok=True)

    logging.info("[Config] workers=%d chunk_files=%d", WORKERS, CHUNK_FILES)
    logging.info("[Paths] PNFS_INPUT_BASE=%s", PNFS_INPUT_BASE)
    logging.info("[Paths] PNFS_CHUNK_DIR=%s", PNFS_CHUNK_DIR)
    logging.info("[Paths] PNFS_FINAL_H5=%s", PNFS_FINAL_H5)
    logging.info("[Paths] LOCAL_WORK_BASE=%s", LOCAL_WORK_BASE)

    # Enumerate NPZs on PNFS
    npz_pnfs_paths = collect_npz_pnfs(PNFS_INPUT_BASE)
    if not npz_pnfs_paths:
        logging.error("No NPZs found under %s", PNFS_INPUT_BASE)
        return 1

    n_chunks = math.ceil(len(npz_pnfs_paths) / CHUNK_FILES)
    logging.info("[Info] Total NPZ files: %d", len(npz_pnfs_paths))
    logging.info("[Info] Expected chunks: %d", n_chunks)

    # Determine which chunks already exist on PNFS
    existing_remote_chunks = set()
    for p in xrdfs_ls(PNFS_CHUNK_DIR):
        if re.search(r"/chunk_\d{5}\.h5$", p):
            existing_remote_chunks.add(Path(p).name)
    logging.info("[Info] Found %d existing remote chunks in %s", len(existing_remote_chunks), PNFS_CHUNK_DIR)

    # Initialize converter
    cfg = ConversionConfig()
    converter = WCMLConverter(cfg)

    # Track failures
    failed_chunks: List[int] = []

    # Build missing chunks sequentially (memory safe)
    for ci in range(n_chunks):
        chunk_name = f"chunk_{ci:05d}.h5"
        if chunk_name in existing_remote_chunks:
            continue

        start = ci * CHUNK_FILES
        stop  = min((ci + 1) * CHUNK_FILES, len(npz_pnfs_paths))
        chunk_npzs = npz_pnfs_paths[start:stop]

        ok, pnfs_chunk_path = process_chunk_safe(
            ci=ci,
            n_chunks=n_chunks,
            chunk_pnfs_npzs=chunk_npzs,
            pnfs_chunk_dir=PNFS_CHUNK_DIR,
            converter=converter,
            workers=WORKERS,
            max_retries=MAX_RETRIES_PER_CHUNK
        )
        if not ok:
            failed_chunks.append(ci)

    if failed_chunks:
        logging.warning("[WARNING] %d chunks failed: %s", len(failed_chunks), failed_chunks[:20])

    # Build list of all available chunks on PNFS for merge
    all_remote_chunk_paths = sorted(
        p for p in xrdfs_ls(PNFS_CHUNK_DIR)
        if re.search(r"/chunk_\d{5}\.h5$", p)
    )
    if not all_remote_chunk_paths:
        logging.error("No chunk files available on PNFS to merge.")
        return 2

    logging.info("[Info] %d chunks ready on PNFS. Starting local merge...", len(all_remote_chunk_paths))

    # Initialize local final from first chunk (download then copy)
    if LOCAL_FINAL_H5.exists():
        LOCAL_FINAL_H5.unlink()

    first = all_remote_chunk_paths[0]
    tmp_first = LOCAL_MERGE_DIR / Path(first).name
    ok, msg = xrdcp_with_retries(remote_url(first), str(tmp_first))
    if not ok:
        logging.error("Failed to download first chunk for init: %s", msg[:200])
        return 3
    shutil.copy2(tmp_first, LOCAL_FINAL_H5)
    try:
        tmp_first.unlink()
    except Exception:
        pass

    # Merge remaining chunks
    remaining = all_remote_chunk_paths[1:]
    if remaining:
        merge_chunks_from_pnfs(remaining, LOCAL_FINAL_H5, batch_size=10)

    # Report local final
    with h5py.File(str(LOCAL_FINAL_H5), "r") as f:
        n_events = len(f["dataset"].keys()) if "dataset" in f else -1
    logging.info("[DONE] Local final H5: %s", LOCAL_FINAL_H5)
    logging.info("[DONE] Local events: %d", n_events)

    # Stage out final to PNFS
    pnfs_final_dir = str(Path(PNFS_FINAL_H5).parent)
    xrdfs_mkdir_p(pnfs_final_dir)

    # Upload final (overwrite)
    logging.info("[Final] staging out final H5 to PNFS: %s", PNFS_FINAL_H5)
    ok, msg = xrdcp_with_retries(str(LOCAL_FINAL_H5), remote_url(PNFS_FINAL_H5))
    if not ok:
        logging.error("[Final] stage-out failed: %s", msg[:200])
        return 4

    logging.info("[Final] ✓ stage-out OK: %s", PNFS_FINAL_H5)

    # Optionally delete local final
    # (keep it by default; uncomment if you want it gone)
    # try:
    #     LOCAL_FINAL_H5.unlink()
    # except Exception:
    #     pass

    if failed_chunks:
        logging.warning("[INCOMPLETE] Failed chunks exist. Final H5 may be missing events from those chunks.")
        return 10

    return 0

if __name__ == "__main__":
    sys.exit(main())
