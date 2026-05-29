#!/usr/bin/env python3
"""Quick analysis of a TF_MultiProtoDG experiment directory."""
import os, glob, csv, sys, json
from collections import defaultdict

def analyze_experiment(exp_path):
    """Extract test accuracies and key metrics from experiment results."""
    if not os.path.isdir(exp_path):
        print(f"ERROR: {exp_path} not found")
        return None

    results = []
    for f in sorted(glob.glob(os.path.join(exp_path, '**/test_result_*.csv'), recursive=True)):
        with open(f) as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                acc_col = [k for k in row.keys() if 'acc' in k.lower() and 'test' in k.lower()
                          and 'decision' not in k.lower() and 'anchor' not in k.lower()
                          and 'residual' not in k.lower()]
                if acc_col:
                    results.append({
                        'file': os.path.basename(f),
                        'acc': float(row[acc_col[0]]),
                    })

    if not results:
        print(f"NO RESULTS in {exp_path}")
        return None

    accs = [r['acc'] for r in results]
    print(f"Seeds: {len(accs)}")
    print(f"Mean:  {sum(accs)/len(accs):.4f}")
    print(f"Min:   {min(accs):.4f}")
    print(f"Max:   {max(accs):.4f}")
    print(f"Std:   {(sum((a - sum(accs)/len(accs))**2 for a in accs)/len(accs))**0.5:.4f}")
    print(f"Seeds: {[f'{a:.4f}' for a in accs]}")

    return {'mean': sum(accs)/len(accs), 'min': min(accs), 'max': max(accs), 'accs': accs}

if __name__ == '__main__':
    if len(sys.argv) > 1:
        for path in sys.argv[1:]:
            full = os.path.join('/home/xuanli/work/PHM-Vibench/results/tmp', path)
            print(f"\n=== {path} ===")
            analyze_experiment(full)
    else:
        print("Usage: python tfmpdg_analyze_experiment.py <exp_dir> [<exp_dir> ...]")
