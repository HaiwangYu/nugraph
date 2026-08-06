#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import re
import subprocess
import time
from pathlib import Path

REC_RE = {
    "apa0": re.compile(r"/rec-apa0-(\d+)\.npz$"),
    "apa1": re.compile(r"/rec-apa1-(\d+)\.npz$"),
}
LAB_RE = {
    "apa0": re.compile(r"/rec-lab-apa0-(\d+)\.npz$"),
    "apa1": re.compile(r"/rec-lab-apa1-(\d+)\.npz$"),
}

AUTH_PATTERNS = (
    "Auth failed",
    "No protocols left to try",
    "Could not get bearer token",
    "Operation expired",
)

last_refresh = 0.0


def refresh_token(force: bool = False) -> None:
    global last_refresh
    now = time.time()

    if not force and now - last_refresh < 35 * 60:
        return

    os.environ["BEARER_TOKEN_FILE"] = f"/tmp/bt_u{os.getuid()}"
    print("[auth] refreshing SBND token", flush=True)

    p = subprocess.run(
        ["htgettoken", "-i", "sbnd", "--vaultserver", "htvaultprod.fnal.gov"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=240,
    )

    text = (p.stdout or "") + "\n" + (p.stderr or "")
    if p.returncode != 0:
        raise RuntimeError(f"htgettoken failed:\n{text[-2000:]}")

    last_refresh = time.time()
    print("[auth] token refresh OK", flush=True)


def xrdfs_ls(host: str, path: str, allow_missing: bool = False) -> list[str]:
    for attempt in range(1, 5):
        refresh_token()

        p = subprocess.run(
            ["xrdfs", host, "ls", path],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        text = (p.stdout or "") + "\n" + (p.stderr or "")

        if p.returncode == 0:
            return [x.strip() for x in p.stdout.splitlines() if x.strip()]

        if allow_missing and "No such file" in text:
            return []

        if any(pattern in text for pattern in AUTH_PATTERNS):
            refresh_token(force=True)

        if attempt < 4:
            time.sleep(2 ** attempt)

    raise RuntimeError(f"xrdfs ls failed for {path}\n{text[-1000:]}")


def indices(paths: list[str], regex: re.Pattern[str]) -> set[int]:
    out = set()
    for path in paths:
        m = regex.search(path)
        if m:
            out.add(int(m.group(1)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit labeled NPZ completeness against aligned input triplets."
    )
    parser.add_argument("--joblist", required=True, type=Path)
    parser.add_argument("--outbase", required=True, help="XRootD labeled-output base")
    parser.add_argument("--xrd-host", default="fndcadoor.fnal.gov:1094")
    parser.add_argument("--report-dir", type=Path, default=Path("."))
    args = parser.parse_args()

    args.report_dir.mkdir(parents=True, exist_ok=True)
    refresh_token(force=True)

    total = 0
    complete = 0
    empty = 0
    incomplete_lines: list[str] = []
    report: list[str] = []

    for line in args.joblist.read_text().splitlines():
        line = line.strip()
        if not line:
            continue

        total += 1
        reco, ct0, ct1 = line.split()
        job_name = Path(reco).name
        outdir = f"{args.outbase.rstrip('/')}/{job_name}"

        reco_files = xrdfs_ls(args.xrd_host, reco)
        out_files = xrdfs_ls(args.xrd_host, outdir, allow_missing=True)

        missing_info = []
        expected_total = 0

        for apa in ["apa0", "apa1"]:
            expected = indices(reco_files, REC_RE[apa])
            got = indices(out_files, LAB_RE[apa])
            missing = sorted(expected - got)

            expected_total += len(expected)
            if missing:
                missing_info.append(
                    f"{apa}: expected={len(expected)} got={len(got)} missing={missing}"
                )

        if expected_total == 0:
            empty += 1
            report.append(f"{job_name}\tempty/no rec-apa NPZ inputs")
        elif missing_info:
            incomplete_lines.append(line)
            report.append(f"{job_name}\t" + " ; ".join(missing_info))
        else:
            complete += 1

        if total % 100 == 0:
            print(
                f"[audit] checked {total} complete={complete} "
                f"incomplete={len(incomplete_lines)} empty={empty}",
                flush=True,
            )

    incomplete_path = args.report_dir / "joblist_ncpi0_incomplete_triplets.txt"
    report_path = args.report_dir / "audit_labeled_sample_ncpi0_report.txt"
    incomplete_path.write_text(
        "\n".join(incomplete_lines) + ("\n" if incomplete_lines else "")
    )
    report_path.write_text(
        "\n".join(report) + ("\n" if report else "")
    )

    print("=" * 80)
    print(f"total aligned jobs: {total}")
    print(f"complete: {complete}")
    print(f"empty/no reco input: {empty}")
    print(f"incomplete: {len(incomplete_lines)}")
    print(f"wrote: {incomplete_path}")
    print(f"wrote: {report_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
