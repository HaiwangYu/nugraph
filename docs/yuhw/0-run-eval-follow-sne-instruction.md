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

## Files

```
/lus/eagle/projects/neutrinoGPU/yuhw/sbnd_eval/
├── bee_one_event.py       # snehadri's bee_nugraph_events.py + BEE_MAX_EVENTS
├── bee_one_event.pbs      # 1-event smoke-test job
├── bee_all_events.pbs     # 10-event job
└── results/
    ├── bee_event1/bee-nugraph.zip
    └── bee_events10/bee-nugraph.zip
```
