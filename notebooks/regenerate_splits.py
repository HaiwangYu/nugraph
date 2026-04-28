#!/usr/bin/env python
"""
Regenerate sample splits (train/validation/test) for an existing HDF5 file.
Use this when dataset/ has more events than samples/ references.
"""

import h5py
import numpy as np
import argparse

def regenerate_splits(filepath, train_frac=0.80, val_frac=0.10, test_frac=0.10, seed=42):
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, "Fractions must sum to 1.0"
    
    print(f"Opening: {filepath}")
    
    with h5py.File(filepath, 'r+') as f:
        # Get all event keys from dataset/
        all_keys = list(f['dataset'].keys())
        n_events = len(all_keys)
        print(f"Total events in dataset/: {n_events}")
        
        # Shuffle deterministically
        rng = np.random.default_rng(seed)
        rng.shuffle(all_keys)
        
        # Calculate split sizes
        n_train = int(n_events * train_frac)
        n_val = int(n_events * val_frac)
        n_test = n_events - n_train - n_val  # Remainder goes to test
        
        train_keys = all_keys[:n_train]
        val_keys = all_keys[n_train:n_train + n_val]
        test_keys = all_keys[n_train + n_val:]
        
        print(f"\nNew split sizes:")
        print(f"  train:      {len(train_keys):>8} ({100*len(train_keys)/n_events:.1f}%)")
        print(f"  validation: {len(val_keys):>8} ({100*len(val_keys)/n_events:.1f}%)")
        print(f"  test:       {len(test_keys):>8} ({100*len(test_keys)/n_events:.1f}%)")
        print(f"  ─────────────────────")
        print(f"  TOTAL:      {n_events:>8}")
        
        # Delete old samples group and recreate
        if 'samples' in f:
            del f['samples']
            print("\nDeleted old samples/ group")
        
        samples_grp = f.create_group('samples')
        dt = h5py.special_dtype(vlen=str)
        
        samples_grp.create_dataset('train', data=np.array(train_keys, dtype=object), dtype=dt)
        samples_grp.create_dataset('validation', data=np.array(val_keys, dtype=object), dtype=dt)
        samples_grp.create_dataset('test', data=np.array(test_keys, dtype=object), dtype=dt)
        print("Created new samples/ with train, validation, test")
        
        # Update datasize group
        if 'datasize' in f:
            del f['datasize']
        
        ds_grp = f.create_group('datasize')
        ds_grp.create_dataset('train', data=len(train_keys))
        ds_grp.create_dataset('validation', data=len(val_keys))
        ds_grp.create_dataset('test', data=len(test_keys))
        print("Updated datasize/ group")
    
    print(f"\n✅ Done! Splits regenerated for {n_events} events.")
    
    # Verify
    print("\nVerification:")
    with h5py.File(filepath, 'r') as f:
        print(f"  samples/train:      {len(f['samples/train'][:])}")
        print(f"  samples/validation: {len(f['samples/validation'][:])}")
        print(f"  samples/test:       {len(f['samples/test'][:])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regenerate HDF5 sample splits")
    parser.add_argument("filepath", help="Path to HDF5 file")
    parser.add_argument("--train-frac", type=float, default=0.80, help="Train fraction (default: 0.80)")
    parser.add_argument("--val-frac", type=float, default=0.10, help="Validation fraction (default: 0.10)")
    parser.add_argument("--test-frac", type=float, default=0.10, help="Test fraction (default: 0.10)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without modifying")
    args = parser.parse_args()
    
    if args.dry_run:
        print("DRY RUN - showing what would be done:\n")
        with h5py.File(args.filepath, 'r') as f:
            n = len(f['dataset'].keys())
            n_train = int(n * args.train_frac)
            n_val = int(n * args.val_frac)
            n_test = n - n_train - n_val
            print(f"Would create splits for {n} events:")
            print(f"  train:      {n_train}")
            print(f"  validation: {n_val}")
            print(f"  test:       {n_test}")
    else:
        regenerate_splits(
            args.filepath,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
            seed=args.seed
        )