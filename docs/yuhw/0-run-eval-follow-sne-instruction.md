# Running the NuGraph evaluation following Snehadri's runbook

*2026-07-27, yuhw — reproducing the Bee event display from
`/lus/eagle/projects/neutrinoGPU/snehadri/sbnd/docs` runbook
("Running the NuGraph Semantic Evaluation on Sophia", copy at
`docs/sophia_eval_guide.html`).*

## Goal

Follow the runbook to run the NuGraph4 semantic evaluation (stage 3, the Bee
event display) on Sophia and upload the result to the BNL Bee gallery.
First as a 1-event smoke test, then all 10 events.

## What the runbook provides, and what actually exists

All referenced pieces exist and are readable:

| Piece | Path | Status |
|---|---|---|
| Stage scripts + PBS driver | `/lus/eagle/projects/neutrinoGPU/snehadri/sbnd/` | present |
| Python env | `/lus/eagle/projects/neutrinoGPU/abhat/conda/envs/nugraph-a/bin/python` | present |
| NuGraph repo (import path) | `/lus/eagle/projects/neutrinoGPU/snehadri/nugraph` | present |
| Input events (10 balanced, 20 APA graphs) | `snehadri/sbnd/data/abhat_events.h5` | present, 61 MB |
| Checkpoint | see below | present |

**Checkpoint discrepancy:** the runbook's §02 table points at *abhat's* log dir
(`…/abhat/sbnd/clustering/nugraph/notebooks/log/N4_…_merged_350k_sophia`), but the
actual PBS driver (`nugraph_perevent_eval.pbs`) uses **snehadri's copy** of the
`ledg0p03_epw1p0` variant:

```
/lus/eagle/projects/neutrinoGPU/snehadri/clustering/nugraph/notebooks/log/
  N4_nw0_bs_4_lr3e4_nuhits_0_bf0p1_tf1p0_if4_hf256_nf64_intf32_nit10_shuffle_random_
  ledg0p03_epw1p0_lemb0p3_lcoh0_converted_labeled_samples_merged_350k_sophia/
  checkpoints/best-f1.ckpt
```

Use what the driver uses. (Consistent with the runbook's note that
`best-f1` / `best-instance-f1` / `best-joint-f1` are byte-identical.)

## Deviations needed to run as yuhw

1. **Working directory.** snehadri's dirs are `r-x` for the group — not writable.
   Set up `/lus/eagle/projects/neutrinoGPU/yuhw/sbnd_eval/` with a copy of
   `bee_nugraph_events.py` → `bee_one_event.py`, modified only to honor
   `BEE_MAX_EVENTS` (env var, default 1) and break out of the event loop early.

2. **Allocation.** The runbook's `#PBS -A neutrinogpu::wirecell` is a *restricted*
   suballocation (users: abhat, snehadri) — qsub rejects yuhw. Plain
   `-A neutrinoGPU` is also rejected ("Multiple suballocation"). The only
   unrestricted suballocation is `debug3`, so:

   ```
   #PBS -A neutrinoGPU::debug3
   ```

   Check with: `sbank allocations -p neutrinoGPU -f "+subname +restricted +users_list"`

3. **Bee upload.** The gallery upload is a CSRF-protected Django endpoint; a bare
   `curl -F file=@…` gets a 403. Also the BNL cert chain fails verification from
   Sophia, so `-k` is needed. Working recipe (from a Sophia login node):

   ```bash
   curl -sk -c /tmp/bee_cookies.txt https://www.phy.bnl.gov/twister/bee/ -o /dev/null
   TOKEN=$(awk '$6=="csrftoken"{print $7}' /tmp/bee_cookies.txt)
   curl -sk -b /tmp/bee_cookies.txt \
     -H "Referer: https://www.phy.bnl.gov/twister/bee/" \
     -H "X-CSRFToken: $TOKEN" \
     -F "csrfmiddlewaretoken=$TOKEN" \
     -F "file=@bee-nugraph.zip" \
     https://www.phy.bnl.gov/twister/bee/upload/
   ```

   The response body is the bare set UUID. URLs:
   - event list: `https://www.phy.bnl.gov/twister/bee/set/<uuid>/event/list/`
   - display: `https://www.phy.bnl.gov/twister/bee/set/<uuid>/event/0/`
     (may 500 briefly right after upload while the set unpacks — reload)

## Run record

### 1-event smoke test

- PBS job **169515** (`single-gpu` queue, `neutrinoGPU::debug3`), wall ~2 min,
  CPU-only as the runbook says (model construction dominates).
- Output: `yuhw/sbnd_eval/results/bee_event1/bee-nugraph.zip` (0.2 MB), 1 event,
  4 point-sets (`truth-semantic`, `nugraph-semantic`, `truth-instance`,
  `nugraph-instance`).
- Sanity: event 0 has **4518 blobs**, matching the runbook's results table row
  (eff 1.000 / purity 0.927).
- Bee set: `92899ae6-9d85-4538-a6d3-18b223130c8b`
  <https://www.phy.bnl.gov/twister/bee/set/92899ae6-9d85-4538-a6d3-18b223130c8b/event/0/>

### All 10 events

- PBS job **169532**, same setup, `BEE_MAX_EVENTS=10`, wall ~3 min.
- Output: `yuhw/sbnd_eval/results/bee_events10/bee-nugraph.zip` (2.0 MB),
  10 events × 4 point-sets = 40 JSON files.
- Bee set: `2aefd394-a9b2-4082-8da0-fd121ee88e60`
  - display: <https://www.phy.bnl.gov/twister/bee/set/2aefd394-a9b2-4082-8da0-fd121ee88e60/event/0/>
  - event list: <https://www.phy.bnl.gov/twister/bee/set/2aefd394-a9b2-4082-8da0-fd121ee88e60/event/list/>

## Beyond the runbook: other samples

The runbook's scripts assume the H5 test list is **apa0/apa1-consecutive** (both
the metrics script and the Bee script pool graphs `(2i, 2i+1)` as one event).
That holds for `abhat_events.h5` only — it was materialized that way on purpose.
Other samples in `snehadri/sbnd/data/` are not:

- `wct_all_truth.h5` (382 events / 764 graphs): all graphs are in `samples/test`,
  but ordered by APA then scrambled — only 1 of 382 adjacent pairs is a real
  event. Consecutive pooling would silently mix events.
- `v2_data_test.h5` / `v2_mc_test.h5` (10 events each): **one graph per event**
  (apa0 only), names like `18253_1_172230__rec-lab-apa0-1`. Consecutive pooling
  would merge two unrelated events.

So I wrote by-name variants that parse `<prefix>-apa<A>-<entry>` from each graph
name and pool per `(prefix, entry)`; the test dataloader iterates in
`samples/test` order with `shuffle=False`, so names can be zipped with the
loader:

- `perevent_semantic_blob_byname.py` — stage-2 metrics with name grouping
- `bee_byname.py` — stage-3 Bee display with name grouping; also handles
  events with 1 graph and skips truth point-sets when nothing is labeled

### v2 samples run (2026-07-27)

- `v2_data_test.h5` — **real detector data, zero truth labels**
  (`sp/y_semantic` all −1), so efficiency/purity are undefined. Job runs the
  Bee display only; output has `nugraph-semantic` / `nugraph-instance` sets,
  no truth sets.
- `v2_mc_test.h5` — MC with truth (~97% of blobs labeled). Job runs both the
  per-event metrics CSV and the 4-set Bee display.

Driver: `v2_eval.pbs` (generic, parameterized via
`qsub -N <name> -v TAG=…,H5=…,RUN_METRICS=0|1`). Final jobs **169543** (v2_data,
Bee only) and **169544** (v2_mc, metrics + Bee); an intermediate attempt on the
`single-node` queue failed to schedule (only 10 nodes carry the `prod`
queue-tag, all busy, and `single-node` needs a whole one — `single-gpu` can
backfill a single GPU slot, so it is the better queue even when it queues).

**Bee sets:**

- v2_data (predictions only, no truth sets):
  <https://www.phy.bnl.gov/twister/bee/set/aac9f9af-8337-40a6-bfd3-5175097ce520/event/0/>
- v2_mc (4 point-sets):
  <https://www.phy.bnl.gov/twister/bee/set/a8cfb0ca-fa33-4fba-8576-5024cb84babd/event/0/>

**v2_mc per-event metrics** (`results/v2_mc_perevent_semantic_blob.csv`,
per-blob, thr 0.5): mean eff **0.377** / pur **0.284**, median eff 0.099 /
pur 0.114 (one event has zero true-nu blobs). Far below the runbook's balanced
sample (mean 0.938 / 0.912) — these v2 MC events are strongly cosmic-dominated
(nu_frac 0–16%, vs 30–68% in the balanced selection), and the sample comes from
a different (v2) production than the `merged_350k` training set, so poor
transfer is not unexpected. Per-event spread is large: three events score well
(eff 0.67–1.0, pur 0.4–0.85), the rest near zero.

## Interactive login-node workflow (fast iteration)

For quick turnarounds the queue is overhead — during a busy period the v2 jobs
waited over an hour for a 2-minute run. Since the eval is CPU-only,
`run_eval.sh` runs it directly on a login node, politely: `nice -n 19`, all
BLAS/OMP backends pinned to 1 thread, no GPU. Keep inputs small (10–20 graphs,
a few minutes); large samples still belong in PBS jobs.

```bash
cd /lus/eagle/projects/neutrinoGPU/yuhw/sbnd_eval
./run_eval.sh <h5> <tag> [--metrics] [--thr 0.5] [--upload]
# outputs: results/bee_<tag>/bee-nugraph.zip (+ results/<tag>_perevent_semantic_blob.csv)
# --upload posts the zip to the Bee gallery (CSRF flow) and prints the URL
```

It wraps the by-name scripts, so it works on any sample regardless of test-list
ordering (APA-paired, scrambled, or single-graph events).

**Validation** (2026-07-27): `v2_mc_test.h5` run on the login node vs PBS job
169544 — metrics CSV and all 40 Bee JSON files **byte-identical** (CPU inference
with pinned threads is deterministic). Login-node wall time ~4 min for 10
graphs (~2× the compute node, from nice + shared load).

`abhat_events.h5` (10 events / 20 graphs) run through this workflow
(`abhat10_login`, ~5 min on the login node) reproduces the runbook's results
exactly — mean eff **0.938** / pur **0.912**, median 1.000 / 0.929, and every
per-event row matches the runbook table. By-name grouping reproduces the APA
pairing, with events ordered by name rather than file order, so Bee event
indices differ from the `bee_events10` set. Bee set:
`65f6460e-9c79-4585-9eb9-6b8270ffec3a`
<https://www.phy.bnl.gov/twister/bee/set/65f6460e-9c79-4585-9eb9-6b8270ffec3a/event/0/>

## Files

```
/lus/eagle/projects/neutrinoGPU/yuhw/sbnd_eval/
├── bee_one_event.py                   # snehadri's bee_nugraph_events.py + BEE_MAX_EVENTS
├── bee_one_event.pbs                  # 1-event smoke-test job
├── bee_all_events.pbs                 # 10-event job
├── perevent_semantic_blob_byname.py   # stage-2 metrics, by-name event pooling
├── bee_byname.py                      # stage-3 Bee display, by-name event pooling
├── wct_eval.pbs                       # wct_all_truth.h5 metrics job (killed, unused)
├── v2_eval.pbs                        # generic driver for v2_data / v2_mc
├── run_eval.sh                        # interactive login-node workflow (nice, 1 thread, optional --upload)
└── results/
    ├── bee_event1/bee-nugraph.zip
    ├── bee_events10/bee-nugraph.zip
    ├── bee_v2_data/bee-nugraph.zip
    ├── bee_v2_mc/bee-nugraph.zip
    └── v2_mc_perevent_semantic_blob.csv
```
