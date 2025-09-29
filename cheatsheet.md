

```bash
export NUGRAPH_LOG=/exp/sbnd/app/users/yuhw/nugraph/log
```
## WCML → NuGraph conversion

```bash
python scripts/wcml_to_h5.py pywcml/sample pywcml/sample/rec-lab-apa0-0.h5
```

Adjust `--tolerance` (drift match) and `--projection-tolerance` (wire match) if needed.


```bash
python scripts/explore_dataset.py /exp/sbnd/app/users/yuhw/nugraph/pywcml/sample/rec-lab-apa0-0.h5        --split test --limit 3 --outdir plots_demo
```

```bash
python scripts/plot_ctpc.py pywcml/sample/rec-lab-apa0-0.npz --outdir plots_ctpc_test
```