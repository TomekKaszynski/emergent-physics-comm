"""
Phase 54c Full Comparison: Original IL=40 vs Aggressive IL=20 (5 seeds each)
============================================================================
Runs two experiments back-to-back:
  Experiment 1: IL with receiver reset every 40 epochs (original Phase 54c)
  Experiment 2: IL with receiver reset every 20 epochs (aggressive)
Both use the original _phase54c_iterated_learning.py code path via subprocess.
5 seeds each: [42, 123, 456, 789, 1337].

Run:
  PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=1 python3 _phase54c_full_comparison.py
"""

import subprocess
import sys
import json
import time
import numpy as np
from pathlib import Path

RESULTS_DIR = Path("results")
SEEDS = [42, 123, 456, 789, 1337]

# Previous multi-seed results (IL=40, from _phase54c_multiseed.py — different code path)
PREV_MULTISEED = {
    'holdout_both': [0.736, 0.780, 0.735, 0.843, 0.790],
    'pos_dis': [0.050, 0.153, 0.307, 0.098, 0.380],
    'topsim': [0.623, 0.631, 0.607, 0.597, 0.633],
    'mi_e': [0.877, 0.760, 0.842, 0.851, 0.761],
    'mi_f': [0.811, 0.854, 0.904, 0.895, 0.930],
}


def run_original_script(seed, reset_interval=40):
    """Run _phase54c_iterated_learning.py with given seed and reset interval.
    Returns parsed results dict or None on failure."""
    env_vars = "PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=1"

    # We need to temporarily patch RECEIVER_RESET_INTERVAL if != 40
    if reset_interval != 40:
        wrapper = (f"import sys; sys.argv = ['_phase54c_iterated_learning.py', '{seed}']; "
                   f"import _phase54c_iterated_learning as m; "
                   f"m.RECEIVER_RESET_INTERVAL = {reset_interval}; "
                   f"m.main()")
        cmd = f'{env_vars} python3 -c "{wrapper}"'
    else:
        cmd = f"{env_vars} python3 _phase54c_iterated_learning.py {seed}"

    print(f"    Running: seed={seed}, IL={reset_interval}...", flush=True)
    t0 = time.time()

    proc = subprocess.run(
        cmd, shell=True, capture_output=True, text=True, timeout=300)

    dt = time.time() - t0

    if proc.returncode != 0:
        print(f"    FAILED (exit {proc.returncode})", flush=True)
        # Print last few lines of stderr
        for line in proc.stderr.strip().split('\n')[-5:]:
            print(f"      {line}", flush=True)
        return None

    # Parse results from saved JSON
    suffix = f"_seed{seed}"
    results_path = RESULTS_DIR / f"phase54c{suffix}_results.json"
    if not results_path.exists():
        print(f"    No results file: {results_path}", flush=True)
        return None

    with open(results_path) as f:
        results = json.load(f)

    # Extract key metrics
    mi_matrix = None
    # Read MI from the stdout (it's also in the compositionality printout)
    # Better: recompute from the saved model. But simplest: parse from results JSON
    # The results JSON has pos_dis and topsim but not raw MI matrix.
    # Let's extract from stdout
    best_mi_e = 0.0
    best_mi_f = 0.0
    for line in proc.stdout.split('\n'):
        if 'Pos 0:' in line and '[' in line:
            try:
                vals = line.split('[')[1].split(']')[0].split(',')
                mi_e0, mi_f0 = float(vals[0]), float(vals[1])
            except (IndexError, ValueError):
                mi_e0, mi_f0 = 0, 0
        if 'Pos 1:' in line and '[' in line:
            try:
                vals = line.split('[')[1].split(']')[0].split(',')
                mi_e1, mi_f1 = float(vals[0]), float(vals[1])
            except (IndexError, ValueError):
                mi_e1, mi_f1 = 0, 0
            best_mi_e = max(mi_e0, mi_e1)
            best_mi_f = max(mi_f0, mi_f1)

    return {
        'seed': seed,
        'holdout_both': results.get('2x8_holdout_both', 0),
        'holdout_e': results.get('2x8_holdout_e', 0),
        'holdout_f': results.get('2x8_holdout_f', 0),
        'train_both': results.get('2x8_train_both', 0),
        'pos_dis': results.get('2x8_posdis', 0),
        'topsim': results.get('2x8_topsim', 0),
        'best_mi_e': best_mi_e,
        'best_mi_f': best_mi_f,
        'time_sec': dt,
        'reset_interval': reset_interval,
    }


def print_table(label, results):
    """Print formatted results table."""
    print(f"\n  {label}", flush=True)
    print(f"  {'Seed':>6} | {'Holdout Both':>12} | {'PosDis':>7} | "
          f"{'TopSim':>7} | {'MI→e':>7} | {'MI→f':>7}", flush=True)
    print(f"  {'─'*6}-+-{'─'*12}-+-{'─'*7}-+-{'─'*7}-+-{'─'*7}-+-{'─'*7}",
          flush=True)

    for r in results:
        print(f"  {r['seed']:>6} | {r['holdout_both']:>11.1%} | "
              f"{r['pos_dis']:>7.3f} | {r['topsim']:>7.3f} | "
              f"{r['best_mi_e']:>7.3f} | {r['best_mi_f']:>7.3f}", flush=True)

    hb = [r['holdout_both'] for r in results]
    pd = [r['pos_dis'] for r in results]
    ts = [r['topsim'] for r in results]
    me = [r['best_mi_e'] for r in results]
    mf = [r['best_mi_f'] for r in results]

    print(f"  {'─'*6}-+-{'─'*12}-+-{'─'*7}-+-{'─'*7}-+-{'─'*7}-+-{'─'*7}",
          flush=True)
    print(f"  {'Mean':>6} | {np.mean(hb):>5.1%}±{np.std(hb):>4.1%} | "
          f"{np.mean(pd):>7.3f} | {np.mean(ts):>7.3f} | "
          f"{np.mean(me):>7.3f} | {np.mean(mf):>7.3f}", flush=True)

    return hb, pd, ts, me, mf


def main():
    print("=" * 70, flush=True)
    print("Phase 54c Full Comparison", flush=True)
    print("=" * 70, flush=True)
    print(f"  Seeds: {SEEDS}", flush=True)
    print(f"  Experiment 1: Original IL (reset every 40 epochs)", flush=True)
    print(f"  Experiment 2: Aggressive IL (reset every 20 epochs)", flush=True)

    t_total = time.time()

    # ════════════════════════════════════════════════════════════
    # EXPERIMENT 1: Original script, IL=40, 5 seeds
    # ════════════════════════════════════════════════════════════
    print(f"\n{'='*70}", flush=True)
    print("EXPERIMENT 1: Original _phase54c_iterated_learning.py × 5 seeds (IL=40)",
          flush=True)
    print(f"{'='*70}", flush=True)

    results_il40 = []
    for seed in SEEDS:
        r = run_original_script(seed, reset_interval=40)
        if r is not None:
            results_il40.append(r)
            print(f"    → holdout={r['holdout_both']:.1%}  "
                  f"PosDis={r['pos_dis']:.3f}  "
                  f"MI→e={r['best_mi_e']:.3f}  MI→f={r['best_mi_f']:.3f}  "
                  f"({r['time_sec']:.0f}s)", flush=True)

    # ════════════════════════════════════════════════════════════
    # EXPERIMENT 2: Aggressive IL=20, 5 seeds
    # ════════════════════════════════════════════════════════════
    print(f"\n{'='*70}", flush=True)
    print("EXPERIMENT 2: Aggressive IL (reset every 20 epochs) × 5 seeds",
          flush=True)
    print(f"{'='*70}", flush=True)

    results_il20 = []
    for seed in SEEDS:
        r = run_original_script(seed, reset_interval=20)
        if r is not None:
            results_il20.append(r)
            print(f"    → holdout={r['holdout_both']:.1%}  "
                  f"PosDis={r['pos_dis']:.3f}  "
                  f"MI→e={r['best_mi_e']:.3f}  MI→f={r['best_mi_f']:.3f}  "
                  f"({r['time_sec']:.0f}s)", flush=True)

    # ════════════════════════════════════════════════════════════
    # RESULTS TABLES
    # ════════════════════════════════════════════════════════════
    print(f"\n\n{'='*70}", flush=True)
    print("DETAILED RESULTS", flush=True)
    print(f"{'='*70}", flush=True)

    hb40, pd40, ts40, me40, mf40 = print_table(
        "Experiment 1: Original IL=40", results_il40)
    hb20, pd20, ts20, me20, mf20 = print_table(
        "Experiment 2: Aggressive IL=20", results_il20)

    # ════════════════════════════════════════════════════════════
    # COMPARISON TABLE
    # ════════════════════════════════════════════════════════════
    print(f"\n\n{'='*70}", flush=True)
    print("COMPARISON: Original Script (IL=40) vs Multiseed (IL=40) vs Aggressive (IL=20)",
          flush=True)
    print(f"{'='*70}", flush=True)

    prev_hb = PREV_MULTISEED['holdout_both']
    prev_pd = PREV_MULTISEED['pos_dis']
    prev_me = PREV_MULTISEED['mi_e']
    prev_mf = PREV_MULTISEED['mi_f']

    def fmt(vals, pct=False):
        m, s = np.mean(vals), np.std(vals)
        if pct:
            return f"{m:.1%} ± {s:.1%}"
        return f"{m:.3f} ± {s:.3f}"

    print(f"\n  {'Condition':<25} | {'Holdout Both':<16} | {'PosDis':<16} | "
          f"{'MI→e':<16} | {'MI→f':<16}", flush=True)
    print(f"  {'─'*25}-+-{'─'*16}-+-{'─'*16}-+-{'─'*16}-+-{'─'*16}", flush=True)

    if results_il40:
        print(f"  {'Original IL=40 ×5':<25} | {fmt(hb40, pct=True):<16} | "
              f"{fmt(pd40):<16} | {fmt(me40):<16} | {fmt(mf40):<16}", flush=True)

    print(f"  {'Multiseed IL=40 ×5':<25} | {fmt(prev_hb, pct=True):<16} | "
          f"{fmt(prev_pd):<16} | {fmt(prev_me):<16} | {fmt(prev_mf):<16}",
          flush=True)

    if results_il20:
        print(f"  {'Aggressive IL=20 ×5':<25} | {fmt(hb20, pct=True):<16} | "
              f"{fmt(pd20):<16} | {fmt(me20):<16} | {fmt(mf20):<16}", flush=True)

    # ════════════════════════════════════════════════════════════
    # Save all results
    # ════════════════════════════════════════════════════════════
    save_data = {
        'experiment_1_il40': {
            'per_seed': results_il40,
            'summary': {
                'holdout_both': {'mean': float(np.mean(hb40)),
                                 'std': float(np.std(hb40))},
                'pos_dis': {'mean': float(np.mean(pd40)),
                            'std': float(np.std(pd40))},
                'topsim': {'mean': float(np.mean(ts40)),
                           'std': float(np.std(ts40))},
                'best_mi_e': {'mean': float(np.mean(me40)),
                              'std': float(np.std(me40))},
                'best_mi_f': {'mean': float(np.mean(mf40)),
                              'std': float(np.std(mf40))},
            } if results_il40 else {},
        },
        'experiment_2_il20': {
            'per_seed': results_il20,
            'summary': {
                'holdout_both': {'mean': float(np.mean(hb20)),
                                 'std': float(np.std(hb20))},
                'pos_dis': {'mean': float(np.mean(pd20)),
                            'std': float(np.std(pd20))},
                'topsim': {'mean': float(np.mean(ts20)),
                           'std': float(np.std(ts20))},
                'best_mi_e': {'mean': float(np.mean(me20)),
                              'std': float(np.std(me20))},
                'best_mi_f': {'mean': float(np.mean(mf20)),
                              'std': float(np.std(mf20))},
            } if results_il20 else {},
        },
        'prev_multiseed_il40': PREV_MULTISEED,
    }

    save_path = RESULTS_DIR / "phase54c_full_comparison.json"
    with open(save_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Saved to {save_path}", flush=True)

    dt = time.time() - t_total
    print(f"\n{'='*70}", flush=True)
    print(f"Total time: {dt/60:.1f} min", flush=True)
    print(f"{'='*70}", flush=True)


if __name__ == "__main__":
    main()
