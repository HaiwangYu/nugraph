#!/usr/bin/env python3
"""
batch_run_labeling_with_truth_EAF.py

Streams reco (img-clus) + truth (celltree0/celltree1) from PNFS via XRootD
into per-job scratch folders, runs labeling_with_truth_fixed.py, and optionally cleans up.

Truth interface (required by labeling_with_truth_fixed.py):
  --celltree-apa0 <path/to/celltree_apa0.root>
  --celltree-apa1 <path/to/celltree_apa1.root>

Job pairing is done by matching suffix "_N" across stage folders:
    img-clus/<imgBatch>_N  <->  celltree0/<ct0Batch>_N  <->  celltree1/<ct1Batch>_N
even when <imgBatch> != <ct0Batch> != <ct1Batch>.
"""

import os

# Limit thread spawning to prevent resource exhaustion - MUST be before other imports
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import subprocess
import sys
import time
import glob
import shutil
import logging
import threading
import re

# Runtime configuration is supplied through environment variables so this
# production driver remains portable across EAF installations and users.
XRDFS_HOST = os.environ.get("XRD_HOST", "fndcadoor.fnal.gov:1094")
XRDCP_PREFIX = f"root://{XRDFS_HOST}"

PNFS_RECO_BASE = os.environ.get("NUGRAPH_RECO_BASE", "")
PNFS_TRUTH_APA0_BASE = os.environ.get("NUGRAPH_TRUTH_APA0_BASE", "")
PNFS_TRUTH_APA1_BASE = os.environ.get("NUGRAPH_TRUTH_APA1_BASE", "")
SCRIPT_DIR = str(Path(__file__).resolve().parent)
LOCAL_OUTPUT_BASE = Path(os.environ.get("NUGRAPH_LOCAL_OUTPUT_BASE", "labeled_samples"))
JOBLIST_FILE = os.environ.get("NUGRAPH_JOBLIST", "")
RECO_BATCHID = os.environ.get("NUGRAPH_RECO_BATCH_ID", "")
CT0_BATCHID = os.environ.get("NUGRAPH_CT0_BATCH_ID", "")
CT1_BATCHID = os.environ.get("NUGRAPH_CT1_BATCH_ID", "")
JOB_START = int(os.environ.get("NUGRAPH_JOB_START", "0"))
JOB_END = int(os.environ.get("NUGRAPH_JOB_END", "-1"))

# Concurrency / safety
MAX_WORKERS = int(os.environ.get("NUGRAPH_MAX_WORKERS", "48"))
XRDCP_RETRIES = 3
XRDCP_RETRY_DELAY = 2.0
BATCH_ENTRY_CHUNK = 16
PYTHON_BIN = sys.executable

CLEANUP_AFTER_LABEL = True  # removes streamed rec/tru/root after each chunk; keeps rec-lab-*.npz + logs

# ---------------------------
# Token refresh configuration
# ---------------------------
HTGETTOKEN_CMD = ["htgettoken", "-i", "sbnd", "--vaultserver", "htvaultprod.fnal.gov"]
PROACTIVE_REFRESH = True
TOKEN_REFRESH_EVERY_FILES = 1000
REACTIVE_REFRESH_ON_AUTH_FAIL = True

BACKGROUND_REFRESH = True
BACKGROUND_REFRESH_INTERVAL = 60 * 40  # 40 minutes

# ---------------------------
# Logging setup
# ---------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

_xrd_success_counter = 0
_xrd_counter_lock = threading.Lock()


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
    if PROACTIVE_REFRESH and cnt and (cnt % TOKEN_REFRESH_EVERY_FILES == 0):
        logging.info("[TOKEN] proactive refresh triggered after %d xrd copies", cnt)
        refresh_htgettoken()
    return cnt


def xrdcp_with_retries(remote_path: str, local_path: Path, retries=XRDCP_RETRIES):
    last_err = ""
    for attempt in range(1, retries + 1):
        cmd = ["xrdcp", "-f", remote_path, str(local_path)]
        rc, out, err = run_command(cmd, capture_output=True)
        last_err = (out or "") + (err or "")
        if rc == 0:
            _increment_xrd_success_counter()
            return True, out + err

        auth_problem = (rc == 52) or ("Auth failed" in (err or "") or "Auth failed" in (out or ""))
        logging.warning(
            "xrdcp failed attempt %d/%d for %s rc=%d. short error: %s",
            attempt, retries, remote_path, rc, (err.strip() or out.strip()),
        )
        if REACTIVE_REFRESH_ON_AUTH_FAIL and auth_problem:
            logging.info("[TOKEN] detected auth failure; attempting htgettoken and retry immediately")
            ok = refresh_htgettoken()
            if ok:
                time.sleep(0.5)
                continue
        time.sleep(XRDCP_RETRY_DELAY)
    return False, last_err


def ensure_symlink_labeling(out_folder: Path, script_dir: str):
    target = Path(script_dir) / "labeling_with_truth_fixed.py"
    link = out_folder / "labeling_with_truth_fixed.py"
    try:
        if link.exists():
            if link.is_symlink():
                if os.path.realpath(link) != str(target):
                    link.unlink()
                    link.symlink_to(target)
        else:
            link.symlink_to(target)
    except Exception as e:
        logging.warning("Symlink failed (%s). Copying instead.", e)
        shutil.copy2(str(target), str(link))


def cleanup_local_files(files_iterable):
    for f in list(files_iterable):
        try:
            Path(f).unlink()
        except Exception:
            pass


def list_remote_files_via_xrdfs(pnfs_folder: str):
    cmd = ["xrdfs", XRDFS_HOST, "ls", pnfs_folder]
    rc, out, err = run_command(cmd, capture_output=True)
    if rc == 0:
        return [l.strip() for l in (out or "").splitlines() if l.strip()]
    return []


def parse_max_indices_from_listing(listing_lines):
    max_rec_apa0 = -1
    max_rec_apa1 = -1
    for line in listing_lines:
        m0 = re.search(r"rec-apa0-([0-9]+)\.npz\b", line)
        m1 = re.search(r"rec-apa1-([0-9]+)\.npz\b", line)
        if m0:
            max_rec_apa0 = max(max_rec_apa0, int(m0.group(1)))
        if m1:
            max_rec_apa1 = max(max_rec_apa1, int(m1.group(1)))
    return max_rec_apa0, max_rec_apa1


def probe_max_index_in_reco_folder(reco_pnfs_folder: str):
    if shutil.which("xrdfs"):
        lines = list_remote_files_via_xrdfs(reco_pnfs_folder)
        if lines:
            return parse_max_indices_from_listing(lines)

    # fallback: shallow xrdcp probe
    max_probe = 256
    max0, max1 = -1, -1
    consecutive_misses_req = 8
    for apa in ("apa0", "apa1"):
        consecutive_misses = 0
        idx = 0
        while idx < max_probe and consecutive_misses < consecutive_misses_req:
            remote_rec = f"{XRDCP_PREFIX}{reco_pnfs_folder}/rec-{apa}-{idx}.npz"
            tmp_local = Path("/tmp") / f".probe_{os.path.basename(reco_pnfs_folder)}_{apa}_{idx}"
            rc, out, err = run_command(["xrdcp", "-f", remote_rec, str(tmp_local)], capture_output=True)
            if rc == 0:
                try:
                    tmp_local.unlink()
                except Exception:
                    pass
                consecutive_misses = 0
                if apa == "apa0":
                    max0 = max(max0, idx)
                else:
                    max1 = max(max1, idx)
            else:
                consecutive_misses += 1
            idx += 1
    return max0, max1


def folder_exists_remote(pnfs_folder: str) -> bool:
    if not shutil.which("xrdfs"):
        # If xrdfs missing, we can't cheaply check; assume yes and let xrdcp fail loudly
        return True
    parent = str(Path(pnfs_folder).parent)
    name = str(Path(pnfs_folder).name)
    entries = list_remote_files_via_xrdfs(parent)
    return any(e.endswith("/" + name) or e.endswith(name) for e in entries)


def stream_celltree_roots(ct0_folder: str, ct1_folder: str, out_folder: Path):
    """
    Download required ROOT files:
      ct0_folder/celltree_apa0.root
      ct1_folder/celltree_apa1.root
    Returns (ok, [local_paths], [errors], local_apa0_root, local_apa1_root)
    """
    local_files = []
    errors = []

    remote0 = f"{XRDCP_PREFIX}{ct0_folder}/celltree_apa0.root"
    remote1 = f"{XRDCP_PREFIX}{ct1_folder}/celltree_apa1.root"

    local0 = out_folder / "celltree_apa0.root"
    local1 = out_folder / "celltree_apa1.root"

    if not local0.exists():
        ok, msg = xrdcp_with_retries(remote0, local0)
        if not ok:
            errors.append(f"celltree_apa0.root xrdcp failed: {remote0} ; last_err={msg[:200]}")
        else:
            local_files.append(local0)
    else:
        local_files.append(local0)

    if not local1.exists():
        ok, msg = xrdcp_with_retries(remote1, local1)
        if not ok:
            errors.append(f"celltree_apa1.root xrdcp failed: {remote1} ; last_err={msg[:200]}")
        else:
            local_files.append(local1)
    else:
        local_files.append(local1)

    return (len(errors) == 0), local_files, errors, local0, local1


def stream_needed_files(reco_folder: str, truth_folder: str, out_folder: Path,
                        apa: str, start_idx: int, end_idx: int):
    """
    Stream rec + tru for a chunk:
      rec from reco_folder:  rec-apaX-idx.npz
      tru from truth_folder: tru-apaX-idx.json
    Returns (ok, local_files, errors)
    """
    local_files = []
    errors = []

    for idx in range(start_idx, end_idx + 1):
        rec_remote = f"{XRDCP_PREFIX}{reco_folder}/rec-{apa}-{idx}.npz"
        tru_remote = f"{XRDCP_PREFIX}{truth_folder}/tru-{apa}-{idx}.json"

        rec_local = out_folder / f"rec-{apa}-{idx}.npz"
        tru_local = out_folder / f"tru-{apa}-{idx}.json"

        if not rec_local.exists():
            ok, msg = xrdcp_with_retries(rec_remote, rec_local)
            if not ok:
                errors.append(f"rec xrdcp failed: {rec_remote} ; last_err={msg[:200]}")
            else:
                local_files.append(rec_local)
        else:
            local_files.append(rec_local)

        if not tru_local.exists():
            ok, msg = xrdcp_with_retries(tru_remote, tru_local)
            if not ok:
                errors.append(f"tru xrdcp failed: {tru_remote} ; last_err={msg[:200]}")
            else:
                local_files.append(tru_local)
        else:
            local_files.append(tru_local)

    return (len(errors) == 0), local_files, errors


def run_labeling_in_folder(out_folder: Path, apa: str, start_idx: int, end_idx: int,
                           celltree0_local: Path, celltree1_local: Path, script_dir_local: str):
    ensure_symlink_labeling(out_folder, script_dir_local)
    entries_arg = f"{start_idx}-{end_idx}"
    logname = out_folder / f"label_truth_{apa}_{start_idx}_{end_idx}.log"

    cmd = [
        PYTHON_BIN,
        "labeling_with_truth_fixed.py",
        "--tru-prefix", f"tru-{apa}",
        "--rec-prefix", f"rec-{apa}",
        "--out-prefix", f"rec-lab-{apa}",
        "--entries",    entries_arg,
        "--celltree-apa0", str(celltree0_local),
        "--celltree-apa1", str(celltree1_local),

        # --- semantics (TRU JSON -> is_nu) ---
        "--tagging-alg", "blob",
        "--max-distance", "15",            # cm
        "--vtx-gate-reco-cm", "500",       # cm
        "--blob-grow-cm", "30",

        # --- instance (SimIDE/ctpc -> tid) ---
        "--shift-p0", "2992",
        "--shift-p1", "2992",
        "--shift-p2", "2992",
        "--dtdc-win", "1",
        "--conf-thr", "0.7",
        "--purity-thr", "0.85",
        "--min-blob-support", "1",

        # THE HERO FLAG (Track B): direct ctpc->point matching radius
        "--max-dist-mm", "10.0",          # mm

        # --- edge supervision written into NPZ ---
        "--write-edge-sup",
        "--balance-edges",
        "--neg-radius-mm", "60",
        "--z-offset-cm", "0",
    ]


    env = os.environ.copy()

    # Add thread limits to prevent resource exhaustion in subprocesses
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"

    prev = env.get("PYTHONPATH", "")
    repo_root = str(Path(script_dir_local).parent)
    env["PYTHONPATH"] = f"{repo_root}:{script_dir_local}" + (":" + prev if prev else "")

    with open(str(logname), "w") as logf:
        logging.info("Running labeling_with_truth for %s entries %s in %s", apa, entries_arg, out_folder)
        proc = subprocess.run(
            cmd,
            cwd=str(out_folder),
            env=env,
            stdout=logf,
            stderr=subprocess.STDOUT,
            text=True,
        )
    return (proc.returncode == 0), str(logname)


def process_job(reco_folder: str, ct0_folder: str, ct1_folder: str):
    job_name = os.path.basename(reco_folder)
    logging.info("[JOB] Starting %s", job_name)

    if not folder_exists_remote(reco_folder):
        logging.warning("[JOB] SKIP (missing RECO folder): %s", reco_folder)
        return reco_folder, {"skipped": "missing_reco"}

    if not folder_exists_remote(ct0_folder):
        logging.warning("[JOB] SKIP (missing CT0 folder): %s", ct0_folder)
        return reco_folder, {"skipped": "missing_ct0"}

    if not folder_exists_remote(ct1_folder):
        logging.warning("[JOB] SKIP (missing CT1 folder): %s", ct1_folder)
        return reco_folder, {"skipped": "missing_ct1"}

    out_folder = LOCAL_OUTPUT_BASE / job_name
    out_folder.mkdir(parents=True, exist_ok=True)

    # stream required ROOT files once per job
    ok_root, root_locals, root_errs, local_ct0_root, local_ct1_root = stream_celltree_roots(ct0_folder, ct1_folder, out_folder)
    if not ok_root:
        logging.error("[JOB %s] failed to stream celltree roots: %s", job_name, root_errs)
        return reco_folder, {"root_failed": root_errs}

    # determine max reco indices
    max0, max1 = -1, -1
    for apa in ("apa0", "apa1"):
        local_matches = glob.glob(str(out_folder / f"rec-{apa}-*.npz"))
        if local_matches:
            idxs = []
            for f in local_matches:
                m = re.search(rf"rec-{apa}-([0-9]+)\.npz$", os.path.basename(f))
                if m:
                    idxs.append(int(m.group(1)))
            if idxs:
                if apa == "apa0":
                    max0 = max(idxs)
                else:
                    max1 = max(idxs)

    if max0 < 0 or max1 < 0:
        p0, p1 = probe_max_index_in_reco_folder(reco_folder)
        if max0 < 0:
            max0 = p0
        if max1 < 0:
            max1 = p1

    logging.info("[JOB] RECO=%s", reco_folder)
    logging.info("[JOB] CT0 =%s", ct0_folder)
    logging.info("[JOB] CT1 =%s", ct1_folder)
    logging.info("[JOB] Found max indices: APA0=%d, APA1=%d", max0, max1)

    results = {"apa0": [], "apa1": []}

    # apa1 then apa0
    for apa, max_event, truth_folder in (
        ("apa1", max1, ct1_folder),
        ("apa0", max0, ct0_folder),
    ):
        if max_event < 0:
            continue

        for start in range(0, max_event + 1, BATCH_ENTRY_CHUNK):
            end = min(max_event, start + BATCH_ENTRY_CHUNK - 1)

            ok_stream, local_files, errs = stream_needed_files(
                reco_folder=reco_folder,
                truth_folder=truth_folder,
                out_folder=out_folder,
                apa=apa,
                start_idx=start,
                end_idx=end,
            )
            if not ok_stream:
                results[apa].append(("stream_failed", start, end, errs))
                continue

            ok_run, logpath = run_labeling_in_folder(
                out_folder, apa, start, end,
                celltree0_local=local_ct0_root,
                celltree1_local=local_ct1_root,
                script_dir_local=SCRIPT_DIR,
            )
            if not ok_run:
                results[apa].append(("label_failed", start, end, logpath))
            else:
                results[apa].append(("label_ok", start, end, logpath))

            if CLEANUP_AFTER_LABEL:
                cleanup_local_files(local_files)

    # optional cleanup: remove ROOTs too (usually keep for reuse across chunks in this job; but job is done now)
    if CLEANUP_AFTER_LABEL:
        cleanup_local_files(root_locals)

    return reco_folder, results


def load_joblist(file_path: str):
    with open(file_path, "r") as f:
        return [l.strip() for l in f if l.strip() and not l.strip().startswith("#")]


def background_token_refresher(interval_seconds=BACKGROUND_REFRESH_INTERVAL):
    def _loop():
        while True:
            time.sleep(interval_seconds)
            logging.info("[TOKEN-BG] background token refresh triggered")
            refresh_htgettoken()
    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    return t


def main():
    if not shutil.which("xrdcp"):
        logging.error("xrdcp not found in PATH.")
        return 1

    if BACKGROUND_REFRESH:
        background_token_refresher(BACKGROUND_REFRESH_INTERVAL)

    LOCAL_OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    jobs = []
    if JOBLIST_FILE and Path(JOBLIST_FILE).exists():
        lines = load_joblist(JOBLIST_FILE)
        for line in lines:
            # Expect 3 columns: reco ct0 ct1 (space separated)
            parts = line.split()
            if len(parts) != 3:
                raise ValueError(f"joblist line must be 3 paths: <reco> <ct0> <ct1>. Got: {line}")
            jobs.append((parts[0], parts[1], parts[2]))
        logging.info("Using joblist with %d jobs", len(jobs))
    elif all((PNFS_RECO_BASE, PNFS_TRUTH_APA0_BASE, PNFS_TRUTH_APA1_BASE,
              RECO_BATCHID, CT0_BATCHID, CT1_BATCHID)) and JOB_END >= JOB_START:
        logging.info("Using range mode: N=%d..%d (inclusive)", JOB_START, JOB_END)
        for n in range(JOB_START, JOB_END + 1):
            reco = f"{PNFS_RECO_BASE}/{RECO_BATCHID}_{n}"
            ct0  = f"{PNFS_TRUTH_APA0_BASE}/{CT0_BATCHID}_{n}"
            ct1  = f"{PNFS_TRUTH_APA1_BASE}/{CT1_BATCHID}_{n}"
            jobs.append((reco, ct0, ct1))
    else:
        logging.error(
            "Set NUGRAPH_JOBLIST, or configure the three input bases, three "
            "batch IDs, and NUGRAPH_JOB_END for range mode."
        )
        return 2

    logging.info("Will process %d jobs with max_workers=%d, chunk=%d", len(jobs), MAX_WORKERS, BATCH_ENTRY_CHUNK)
    logging.info("First few jobs: %s", jobs[:5])

    futures = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for reco, ct0, ct1 in jobs:
            futures.append(ex.submit(process_job, reco, ct0, ct1))

        for fut in as_completed(futures):
            try:
                job_path, results = fut.result()
                logging.info("Job %s completed: %s", job_path, results)
            except Exception as e:
                logging.exception("Job raised exception: %s", e)

    logging.info("All done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
