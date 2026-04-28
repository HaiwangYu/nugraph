#!/usr/bin/env python
"""
Verify HDF5 file structure and sample splits for NuGraph training.
Checks for correct naming (validation vs val) and split consistency.
"""

import h5py
import argparse
import sys

def verify_h5(filepath):
    print(f"Verifying: {filepath}\n")
    
    with h5py.File(filepath, 'r') as f:
        # 1. Check top-level structure
        print("=" * 60)
        print("TOP-LEVEL STRUCTURE")
        print("=" * 60)
        for key in f.keys():
            if isinstance(f[key], h5py.Group):
                print(f"  {key}/ (group, {len(f[key])} items)")
            else:
                print(f"  {key} (dataset, shape={f[key].shape})")
        
        # 2. Check dataset/ group
        print("\n" + "=" * 60)
        print("DATASET GROUP")
        print("=" * 60)
        if 'dataset' in f:
            n_events = len(f['dataset'].keys())
            print(f"  Total events in dataset/: {n_events}")
            # Show first few keys
            keys = list(f['dataset'].keys())[:5]
            print(f"  First 5 keys: {keys}")
        else:
            print("  ERROR: 'dataset' group not found!")
            return False
        
        # 3. Check samples/ group - this is critical
        print("\n" + "=" * 60)
        print("SAMPLES GROUP (SPLITS)")
        print("=" * 60)
        if 'samples' not in f:
            print("  ERROR: 'samples' group not found!")
            return False
        
        samples = f['samples']
        print(f"  Available splits: {list(samples.keys())}")
        
        # Check for correct naming
        has_validation = 'validation' in samples
        has_val = 'val' in samples
        has_train = 'train' in samples
        has_test = 'test' in samples
        
        if has_val and not has_validation:
            print("\n  ⚠️  WARNING: Found 'val' but NuGraph expects 'validation'!")
            print("     Run the fix script to rename 'val' -> 'validation'")
        
        if not has_validation and not has_val:
            print("\n  ❌ ERROR: Neither 'validation' nor 'val' found!")
            return False
        
        if not has_train:
            print("\n  ❌ ERROR: 'train' split not found!")
            return False
        
        if not has_test:
            print("\n  ❌ ERROR: 'test' split not found!")
            return False
        
        # Count samples in each split
        train_count = len(samples['train'][:])
        val_key = 'validation' if has_validation else 'val'
        val_count = len(samples[val_key][:])
        test_count = len(samples['test'][:])
        total_samples = train_count + val_count + test_count
        
        print(f"\n  Split counts:")
        print(f"    train:      {train_count:>8} ({100*train_count/total_samples:.1f}%)")
        print(f"    {val_key}:{'  ' if has_validation else ''} {val_count:>8} ({100*val_count/total_samples:.1f}%)")
        print(f"    test:       {test_count:>8} ({100*test_count/total_samples:.1f}%)")
        print(f"    ─────────────────────")
        print(f"    TOTAL:      {total_samples:>8}")
        
        # Check consistency with dataset/
        if total_samples != n_events:
            print(f"\n  ⚠️  WARNING: Split total ({total_samples}) != dataset events ({n_events})")
        else:
            print(f"\n  ✅ Split total matches dataset events")
        
        # 4. Check datasize/ group
        print("\n" + "=" * 60)
        print("DATASIZE GROUP")
        print("=" * 60)
        if 'datasize' in f:
            ds = f['datasize']
            print(f"  Available: {list(ds.keys())}")
            for key in ds.keys():
                print(f"    {key}: {ds[key][()]}")
        else:
            print("  WARNING: 'datasize' group not found (optional)")
        
        # 5. Check for other required groups
        print("\n" + "=" * 60)
        print("OTHER GROUPS")
        print("=" * 60)
        for grp in ['planes', 'semantic_classes', 'gen']:
            if grp in f:
                if isinstance(f[grp], h5py.Group):
                    print(f"  ✅ {grp}/ (group)")
                else:
                    data = f[grp][:]
                    print(f"  ✅ {grp}: {data if len(data) < 10 else f'{len(data)} items'}")
            else:
                print(f"  ⚠️  {grp} not found")
        
        # 6. Final verdict
        print("\n" + "=" * 60)
        print("VERDICT")
        print("=" * 60)
        
        if has_validation and has_train and has_test:
            print("  ✅ File structure is CORRECT for NuGraph training")
            return True
        elif has_val and has_train and has_test:
            print("  ⚠️  File needs 'val' renamed to 'validation'")
            print("\n  Run this to fix:")
            print(f'''
python -c "
import h5py
f = h5py.File('{filepath}', 'r+')
if 'samples/val' in f:
    data = f['samples/val'][:]
    del f['samples/val']
    f.create_dataset('samples/validation', data=data)
    print('Renamed samples/val -> samples/validation')
if 'datasize/val' in f:
    data = f['datasize/val'][()]
    del f['datasize/val']
    f.create_dataset('datasize/validation', data=data)
    print('Renamed datasize/val -> datasize/validation')
f.close()
print('Done!')
"
''')
            return False
        else:
            print("  ❌ File structure has ERRORS")
            return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify HDF5 file for NuGraph")
    parser.add_argument("filepath", help="Path to HDF5 file")
    args = parser.parse_args()
    
    success = verify_h5(args.filepath)
    sys.exit(0 if success else 1)