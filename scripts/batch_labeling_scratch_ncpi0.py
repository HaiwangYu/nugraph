#!/usr/bin/env python3
"""
batch_run_labeling_with_truth_EAF.py

Streams reco (img-clus) + truth (celltree0/celltree1) from PNFS via XRootD
into per-job LOCAL scratch folders, runs labeling_with_truth_fixed.py, then
STAGES OUT the labeled outputs to PNFS scratch via XRootD and deletes the local
job folder so /scratch doesn't fill up.

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
from typing import List, Tuple, Dict, Any

# Runtime locations are environment-driven; NCPi0-specific handling and
# scientific labeling options remain unchanged.
XRDFS_HOST = os.environ.get("XRD_HOST", "fndcadoor.fnal.gov:1094")
XRDCP_PREFIX = f"root://{XRDFS_HOST}"

PNFS_RECO_BASE = os.environ.get("NUGRAPH_RECO_BASE", "")
PNFS_TRUTH_APA0_BASE = os.environ.get("NUGRAPH_TRUTH_APA0_BASE", "")
PNFS_TRUTH_APA1_BASE = os.environ.get("NUGRAPH_TRUTH_APA1_BASE", "")
SCRIPT_DIR = str(Path(__file__).resolve().parent)
LOCAL_OUTPUT_BASE = Path(os.environ.get("NUGRAPH_LOCAL_OUTPUT_BASE", "labeled_samples_ncpi0"))
PNFS_OUTPUT_BASE = os.environ.get("NUGRAPH_PNFS_OUTPUT_BASE", "")

# What to stage out (relative to local job dir)
STAGE_OUT_PATTERNS = [
    "rec-lab-apa0-*.npz",
    "rec-lab-apa1-*.npz",
    "label_truth_*.log",
    # keep the script copy/symlink for provenance (small)
    "labeling_with_truth_fixed.py",
    # If you want to keep celltree roots too (usually NO), uncomment:
    # "celltree_apa0.root",
    # "celltree_apa1.root",
]

JOBLIST_FILE = os.environ.get("NUGRAPH_JOBLIST", "")
RECO_BATCHID = os.environ.get("NUGRAPH_RECO_BATCH_ID", "")
CT0_BATCHID = os.environ.get("NUGRAPH_CT0_BATCH_ID", "")
CT1_BATCHID = os.environ.get("NUGRAPH_CT1_BATCH_ID", "")
JOB_START = int(os.environ.get("NUGRAPH_JOB_START", "0"))
JOB_END = int(os.environ.get("NUGRAPH_JOB_END", "-1"))

# Concurrency / safety
MAX_WORKERS = int(os.environ.get("NUGRAPH_MAX_WORKERS", "2"))
XRDCP_RETRIES = 3
XRDCP_RETRY_DELAY = 2.0
BATCH_ENTRY_CHUNK = 4
PYTHON_BIN = sys.executable

# Cleanup policy:
# - During processing: delete streamed rec/tru JSON/NPZ inputs after each chunk
# - After job done: stage out labeled outputs to PNFS then delete local job folder
CLEANUP_AFTER_LABEL = True

# Stage-out behavior
STAGE_OUT_ENABLED = bool(PNFS_OUTPUT_BASE)
DELETE_LOCAL_JOB_DIR_AFTER_STAGEOUT = True  # delete local job folder ONLY if stage-out fully succeeds

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


def xrdcp_remote_to_local_with_retries(remote_path: str, local_path: Path, retries=XRDCP_RETRIES):
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
            "xrdcp (remote->local) failed attempt %d/%d for %s rc=%d. short error: %s",
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


def xrdcp_local_to_remote_with_retries(local_path: Path, remote_path: str, retries=XRDCP_RETRIES):
    last_err = ""
    for attempt in range(1, retries + 1):
        cmd = ["xrdcp", "-f", str(local_path), remote_path]
        rc, out, err = run_command(cmd, capture_output=True)
        last_err = (out or "") + (err or "")
        if rc == 0:
            _increment_xrd_success_counter()
            return True, out + err

        auth_problem = (rc == 52) or ("Auth failed" in (err or "") or "Auth failed" in (out or ""))
        logging.warning(
            "xrdcp (local->remote) failed attempt %d/%d for %s rc=%d. short error: %s",
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


def xrdfs_mkdir_p(pnfs_dir: str) -> bool:
    """
    Create directory in dCache namespace via xrdfs.
    pnfs_dir is an XRootD-visible destination directory.
    """
    if not shutil.which("xrdfs"):
        logging.error("xrdfs not found in PATH; cannot create remote dir: %s", pnfs_dir)
        return False
    rc, out, err = run_command(["xrdfs", XRDFS_HOST, "mkdir", "-p", pnfs_dir], capture_output=True)
    if rc != 0:
        logging.warning("xrdfs mkdir failed for %s rc=%d err=%s out=%s", pnfs_dir, rc, err.strip(), out.strip())
        return False
    return True


def stage_out_job_dir(local_job_dir: Path, pnfs_job_dir: str) -> Tuple[bool, List[str], List[str]]:
    """
    Stage out selected files from local_job_dir to pnfs_job_dir.
    Copies CONTENTS (not the parent folder) to avoid double nesting.
    Returns (ok_all, copied_files, failed_files)
    """
    ok_mkdir = xrdfs_mkdir_p(pnfs_job_dir)
    if not ok_mkdir:
        return False, [], [f"mkdir_failed:{pnfs_job_dir}"]

    copied: List[str] = []
    failed: List[str] = []

    # Copy patterns
    for pat in STAGE_OUT_PATTERNS:
        for f in sorted(local_job_dir.glob(pat)):
            remote = f"{XRDCP_PREFIX}//{pnfs_job_dir}/{f.name}"
            ok, msg = xrdcp_local_to_remote_with_retries(f, remote)
            if ok:
                copied.append(f.name)
            else:
                failed.append(f"{f.name} :: {msg[:200]}")

    ok_all = (len(failed) == 0)
    return ok_all, copied, failed


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


def folder_exists_remote(path: str) -> bool:
    """
    Check remote PNFS/XRootD folder existence.

    Critical behavior:
      - real missing folder -> False
      - auth/XRootD failure -> refresh token and retry
      - persistent unexpected failure -> raise, not silently skip
    """
    import os
    import subprocess
    import time

    os.environ["BEARER_TOKEN_FILE"] = f"/tmp/bt_u{os.getuid()}"

    def refresh_token():
        logging.warning("[XROOTD] refreshing SBND token before retry")
        q = subprocess.run(
            ["htgettoken", "-i", "sbnd", "--vaultserver", "htvaultprod.fnal.gov"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=180,
        )
        msg = (q.stdout or "") + "\n" + (q.stderr or "")
        if q.returncode != 0:
            raise RuntimeError(f"htgettoken failed while checking {path}: {msg[-1000:]}")

    last_msg = ""
    for attempt in range(1, 4):
        q = subprocess.run(
            ["xrdfs", XRDFS_HOST, "ls", path],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        msg = (q.stdout or "") + "\n" + (q.stderr or "")
        last_msg = msg

        if q.returncode == 0:
            return True

        if "Auth failed" in msg or "No protocols left to try" in msg or "Could not get bearer token" in msg:
            logging.warning("[XROOTD] auth failure checking %s, attempt %d/3", path, attempt)
            refresh_token()
            time.sleep(2 * attempt)
            continue

        if "Unable to locate" in msg or "No such file" in msg or "not found" in msg:
            return False

        # Other XRootD transient issue: retry, but do not call it missing.
        logging.warning("[XROOTD] non-auth failure checking %s, attempt %d/3: %s", path, attempt, msg[-500:])
        time.sleep(2 * attempt)

    raise RuntimeError(f"xrdfs folder check failed after retries for {path}: {last_msg[-1000:]}")


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
        ok, msg = xrdcp_remote_to_local_with_retries(remote0, local0)
        if not ok:
            errors.append(f"celltree_apa0.root xrdcp failed: {remote0} ; last_err={msg[:200]}")
        else:
            local_files.append(local0)
    else:
        local_files.append(local0)

    if not local1.exists():
        ok, msg = xrdcp_remote_to_local_with_retries(remote1, local1)
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
            ok, msg = xrdcp_remote_to_local_with_retries(rec_remote, rec_local)
            if not ok:
                errors.append(f"rec xrdcp failed: {rec_remote} ; last_err={msg[:200]}")
            else:
                local_files.append(rec_local)
        else:
            local_files.append(rec_local)

        if not tru_local.exists():
            ok, msg = xrdcp_remote_to_local_with_retries(tru_remote, tru_local)
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



def existing_remote_rec_indices(reco_folder: str, apa: str) -> list[int]:
    """
    Return exactly the reco indices that exist remotely for one APA.

    This avoids assuming indices are contiguous from 0..max, which is false
    for some _4 folders and causes xrdcp "No such file" errors.
    """
    import re

    files = list_remote_files_via_xrdfs(reco_folder)
    rx = re.compile(rf"/rec-{apa}-(\d+)\.npz$")
    out = []
    for f in files:
        m = rx.search(f)
        if m:
            out.append(int(m.group(1)))
    return sorted(set(out))


def chunks_from_indices(indices: list[int], chunk_size: int) -> list[list[int]]:
    return [indices[i:i + chunk_size] for i in range(0, len(indices), chunk_size)]


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

    # Local working directory for this job
    out_folder = LOCAL_OUTPUT_BASE / job_name
    out_folder.mkdir(parents=True, exist_ok=True)

    # PNFS stage-out directory for this job
    pnfs_job_dir = f"{PNFS_OUTPUT_BASE}/{job_name}"

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

    results: Dict[str, Any] = {"apa0": [], "apa1": [], "stageout": None}

    def remote_indices(folder: str, prefix: str) -> set[int]:
        """
        Return indices for files like rec-apa0-3.npz or tru-apa0-3.json.
        This uses the current remote listing and does not assume contiguous files.
        """
        import re
        files = list_remote_files_via_xrdfs(folder)
        rx = re.compile(rf"/{prefix}-(\d+)\.(npz|json)$")
        out = set()
        for f in files:
            m = rx.search(f)
            if m:
                out.add(int(m.group(1)))
        return out

    def contiguous_chunks(indices, chunk_size):
        """
        Convert existing indices into contiguous [start,end] chunks.
        Example: [0,1,2,4,5,6,7,8,9] with chunk_size=4
             -> [(0,2), (4,7), (8,9)]
        """
        indices = sorted(set(indices))
        if not indices:
            return []

        runs = []
        run_start = indices[0]
        prev = indices[0]

        for x in indices[1:]:
            if x == prev + 1 and (x - run_start + 1) <= chunk_size:
                prev = x
            else:
                runs.append((run_start, prev))
                run_start = x
                prev = x
        runs.append((run_start, prev))
        return runs

    # apa1 then apa0. Process only entries where both reco NPZ and truth JSON exist.
    for apa, truth_folder in (
        ("apa1", ct1_folder),
        ("apa0", ct0_folder),
    ):
        rec_idx = remote_indices(reco_folder, f"rec-{apa}")
        tru_idx = remote_indices(truth_folder, f"tru-{apa}")

        valid_idx = sorted(rec_idx & tru_idx)
        missing_rec = sorted(tru_idx - rec_idx)
        missing_tru = sorted(rec_idx - tru_idx)

        logging.info(
            "[JOB %s] %s existing indices: rec=%d truth=%d valid=%d missing_rec=%d missing_truth=%d",
            job_name, apa, len(rec_idx), len(tru_idx), len(valid_idx), len(missing_rec), len(missing_tru)
        )

        if not valid_idx:
            results[apa].append(("no_valid_indices", None, None, {
                "rec_n": len(rec_idx),
                "truth_n": len(tru_idx),
                "missing_rec": missing_rec[:20],
                "missing_truth": missing_tru[:20],
            }))
            continue

        for start, end in contiguous_chunks(valid_idx, BATCH_ENTRY_CHUNK):
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

    # ---------------------------
    # STAGE OUT + DELETE LOCAL JOB DIR
    # ---------------------------
    if STAGE_OUT_ENABLED:
        logging.info("[JOB %s] staging out to PNFS: %s", job_name, pnfs_job_dir)
        ok_all, copied, failed = stage_out_job_dir(out_folder, pnfs_job_dir)
        results["stageout"] = {
            "pnfs_dir": pnfs_job_dir,
            "ok_all": ok_all,
            "copied_n": len(copied),
            "failed_n": len(failed),
            "failed": failed[:10],
        }
        if ok_all:
            logging.info("[JOB %s] stage-out OK (%d files).", job_name, len(copied))
            if DELETE_LOCAL_JOB_DIR_AFTER_STAGEOUT:
                try:
                    shutil.rmtree(out_folder, ignore_errors=True)
                    logging.info("[JOB %s] deleted local job dir: %s", job_name, out_folder)
                except Exception as e:
                    logging.warning("[JOB %s] failed to delete local job dir %s: %s", job_name, out_folder, e)
        else:
            logging.warning("[JOB %s] stage-out had failures (%d). Keeping local dir: %s", job_name, len(failed), out_folder)

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
    if not shutil.which("xrdfs"):
        logging.error("xrdfs not found in PATH.")
        return 1

    if BACKGROUND_REFRESH:
        background_token_refresher(BACKGROUND_REFRESH_INTERVAL)

    # Local base must exist; this is only the local workspace root.
    LOCAL_OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    # Ensure PNFS output base exists (best-effort)
    if STAGE_OUT_ENABLED:
        xrdfs_mkdir_p(PNFS_OUTPUT_BASE)

    jobs: List[Tuple[str, str, str]] = []
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
