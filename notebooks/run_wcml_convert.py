# filename: run_wcml_convert.py
#!/usr/bin/env python

import os
import multiprocessing as mp
from pathlib import Path

from pywcml.converter import convert_npz_directory
from pywcml.config import ConversionConfig


def main() -> None:
    # --- Optional: tame BLAS/OpenMP threading so workers don't oversubscribe ---
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("BLIS_NUM_THREADS", "1")

    # --- Paths (adapt as needed) ---
    input_dir = Path("/scratch/7DayLifetime/abhat/wirecell/clustering/labeled_samples_vertex")
    output_h5 = Path("/scratch/7DayLifetime/abhat/wirecell/clustering/23334072_nug4_vertex.h5")

    config = ConversionConfig()

    # Use a sensible number of workers (don’t use all 62)
    n_cpu = mp.cpu_count()
    workers = 4
    print(f"[Info] Detected {n_cpu} CPUs, using workers={workers}")

    convert_npz_directory(
        input_dir,
        output_h5,
        config=config,
        workers=workers,
    )


if __name__ == "__main__":
    # On Python 3.13 + HPC, being explicit is safer
    mp.set_start_method("spawn", force=True)
    main()
