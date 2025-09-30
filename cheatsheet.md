

```bash
export NUGRAPH_LOG=/exp/sbnd/app/users/yuhw/nugraph/log
export LD_LIBRARY_PATH=/usr/lib64/mpich/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
```
## WCML → NuGraph conversion

```bash
python scripts/wcml_to_h5.py pywcml/sample pywcml/sample/rec-lab-apa0-0.h5
```

Adjust `--tolerance` (drift match) and `--projection-tolerance` (wire match) if needed.


rec-lab-apa0-0.h5, 23334072_0.h5

```bash
python scripts/explore_dataset.py \
/exp/sbnd/app/users/yuhw/nugraph/pywcml/sample/23334072_0.h5 \
--split test --limit 3 --outdir plots_demo
```

```bash
python scripts/plot_ctpc.py pywcml/sample/23334072_0/rec-lab-apa1-0.npz
python scripts/plot_blob_projections.py pywcml/sample/23334072_0/rec-lab-apa1-0.npz
```