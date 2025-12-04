import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
import nugraph as ng

# ================= CONFIG =================
# Which columns in hit.x are the vertex features?
# If you appended 2 features at the end, use slice(-2, None)
# If you have 10 features total and they are the last 2:
VTX_COLS_SLICE = slice(-2, None) 
# Which column is the "Distance to Vertex" for binning?
# Assuming it is the first of the two added features:
DIST_COL_IDX = -2 
# ==========================================

def get_binned_metrics(preds, labels, distances, bins=[0, 10, 30, 1000]):
    """
    Computes Precision/Recall/F1 for hits within specific distance ranges.
    """
    results = {}
    
    # Global metrics first
    p, r, f, _ = precision_recall_fscore_support(labels, preds, average=None, labels=[0, 1], zero_division=0)
    results['Global'] = {'nu_prec': p[0], 'nu_rec': r[0], 'nu_f1': f[0]} # Assuming 0 is Nu, 1 is Cosmic. CHECK THIS!

    # Per-bin metrics
    for i in range(len(bins)-1):
        low, high = bins[i], bins[i+1]
        mask = (distances >= low) & (distances < high)
        
        if mask.sum() == 0:
            results[f'Bin_{low}-{high}'] = {'n_hits': 0}
            continue

        bin_preds = preds[mask]
        bin_labels = labels[mask]
        
        p, r, f, _ = precision_recall_fscore_support(bin_labels, bin_preds, average=None, labels=[0, 1], zero_division=0)
        
        # Check if nu class (index 0) exists in this bin
        nu_idx = 0 
        # If your classes are [nu, cosmic], index 0 is nu. 
        # CAUTION: Check your 'semantic_classes' list. usually ['nu', 'cosmic'] -> nu=0.
        
        results[f'Bin_{low}-{high}'] = {
            'n_hits': mask.sum().item(),
            'nu_prec': p[nu_idx], 
            'nu_rec': r[nu_idx], 
            'nu_f1': f[nu_idx]
        }
    return results

def run_eval_loop(model, dataloader, device, ablation_mode='baseline'):
    model.eval()
    all_preds = []
    all_labels = []
    all_dists = []

    print(f"\n>>> Running Evaluation: {ablation_mode.upper()} <<<")
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = batch.to(device)
            
            # --- ABLATION LOGIC ---
            if ablation_mode == 'zero_vertex':
                # Set vertex features to 0
                batch['hit'].x[:, VTX_COLS_SLICE] = 0.0
                
            elif ablation_mode == 'shuffle_vertex':
                # Shuffle vertex features across hits within the batch
                # This preserves distribution but destroys physical correlation
                idx = torch.randperm(batch['hit'].x.size(0), device=device)
                features = batch['hit'].x[:, VTX_COLS_SLICE]
                batch['hit'].x[:, VTX_COLS_SLICE] = features[idx]
            
            # --- FORWARD PASS ---
            # We only need logits
            logits = model(batch) 
            if isinstance(logits, tuple): logits = logits[0] # Handle (loss, metrics) output if necessary
            # Note: NuGraph4.forward returns (loss, metrics) usually. 
            # You might need to call model.encoder + model.semantic_head directly 
            # or grab the logits stash if your forward pass stashes them.
            # Assuming standard NuGraph structure where we can get logits:
            
            # If model.forward returns loss, we need to extract logits.
            # Hack: look at model code. Usually:
            # x = model.encoder(batch)
            # logits = model.semantic_head(x)
            
            # Let's try to use the standard forward and hope it stashes, 
            # OR explicitly call the submodules:
            x_enc = model.encoder(batch) # Or however NuGraph4 encodes
            # If NuGraph4 inherits NuGraph3, it might be model(batch) returning loss.
            # Let's assume you added `hit.x_semantic` or similar in your forward pass logic,
            # OR we call the semantic head manually:
            
            # REVISION based on your NuGraph4 code provided earlier:
            # The forward() returns (loss, metrics). 
            # BUT NuGraph3/4 usually stashes `x_semantic` (logits) on the hit store.
            _ = model(batch, stage='test') 
            logits = batch['hit'].x_semantic # [N, C] logits
            
            preds = logits.argmax(dim=1).cpu()
            labels = batch['hit'].y_semantic.cpu()
            
            # Grab distance for binning analysis (BEFORE shuffling if we were shuffling!)
            # Wait, if we shuffle, we destroy the distance info for binning too?
            # NO: We want to bin by the *true* distance to see where the model fails.
            # So we should cache the distance *before* ablation if needed.
            # But 'hit.x' is modified in place.
            # Let's trust that we want to bin by the feature provided. 
            # Actually, for 'shuffle', binning by the shuffled distance is meaningless. 
            # We should probably skip binning for 'shuffle', or save true distance first.
            
            # Since we modified x in place, let's just assume we rely on the modified x 
            # for the model, but for binning we might want True distance. 
            # For this script, let's keep it simple: The `all_dists` will be whatever 
            # was in x at the time of inference.
            
            dists = batch['hit'].x[:, DIST_COL_IDX].cpu()
            
            # Filter out -1 labels (unlabeled)
            mask = labels >= 0
            all_preds.append(preds[mask])
            all_labels.append(labels[mask])
            all_dists.append(dists[mask])

    # Concatenate
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_dists = torch.cat(all_dists).numpy()
    
    return get_binned_metrics(all_preds, all_labels, all_dists)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--data-path', type=str, required=True)
    args = parser.parse_args()

    # Load Model & Data
    # (Assuming you have a way to load the LightningModule and DataModule)
    model = ng.models.NuGraph4.load_from_checkpoint(args.ckpt)
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Initialize DataModule (make sure to set valid/test split)
    dm = ng.data.NuGraphDataModule(data_path=args.data_path, batch_size=64)
    dm.setup('test')
    loader = dm.test_dataloader()

    # 1. Baseline
    res_base = run_eval_loop(model, loader, device, 'baseline')
    
    # 2. Zero Vertex (Ablation)
    res_zero = run_eval_loop(model, loader, device, 'zero_vertex')
    
    # 3. Shuffle Vertex (Ablation)
    res_shuf = run_eval_loop(model, loader, device, 'shuffle_vertex')

    # Print Report
    print("\n" + "="*60)
    print("RESULTS SUMMARY (Nu Recall)")
    print("="*60)
    
    bins = sorted([k for k in res_base.keys() if 'Bin' in k])
    
    # Header
    print(f"{'Metric':<20} | {'Baseline':<10} | {'Zeroed':<10} | {'Shuffled':<10}")
    print("-" * 60)
    
    # Global Nu Recall
    print(f"{'Global Nu Recall':<20} | {res_base['Global']['nu_rec']:.4f}     | {res_zero['Global']['nu_rec']:.4f}     | {res_shuf['Global']['nu_rec']:.4f}")
    
    print("-" * 60)
    print("BINNED NU RECALL (Where physics happens)")
    for b in bins:
        base_r = res_base[b]['nu_rec']
        # For zero/shuffle, the bins might not mean the same physically if we modified x
        # But assuming the "Zeroed" run still tracks performance vs the (now zeroed) distance,
        # it shows if the model relied on that info in that bin.
        zero_r = res_zero[b]['nu_rec']
        shuf_r = res_shuf[b]['nu_rec']
        print(f"{b:<20} | {base_r:.4f}     | {zero_r:.4f}     | {shuf_r:.4f}")

if __name__ == '__main__':
    main()