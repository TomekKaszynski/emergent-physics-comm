#!/usr/bin/env python3
"""Train oracle probes to measure information accessibility in features.

Usage:
    python scripts/run_oracle.py --features results/collision_dinov2_features.pt --n-seeds 20
"""

import argparse
import json
import time
import numpy as np
import torch
import torch.nn.functional as F

from src.oracle import Oracle, train_oracle
from src.datasets import load_features, create_splits, sample_pairs


def evaluate_oracle(model, data, p1_bins, p2_bins, ids, device,
                    batch_size=32, n_rounds=20):
    model.eval()
    rng = np.random.RandomState(999)
    correct_1, correct_2, correct_both, total = 0, 0, 0, 0
    with torch.no_grad():
        for _ in range(n_rounds):
            ia, ib = sample_pairs(ids, batch_size, rng)
            da = data[ia].to(device)
            db = data[ib].to(device)
            l1 = (p1_bins[ia] > p1_bins[ib]).astype(np.float32)
            l2 = (p2_bins[ia] > p2_bins[ib]).astype(np.float32)
            p1, p2 = model(da, db)
            p1 = (p1.cpu().numpy() > 0).astype(np.float32)
            p2 = (p2.cpu().numpy() > 0).astype(np.float32)
            correct_1 += (p1 == l1).sum()
            correct_2 += (p2 == l2).sum()
            correct_both += ((p1 == l1) & (p2 == l2)).sum()
            total += len(ia)
    return correct_1/total, correct_2/total, correct_both/total


def main():
    parser = argparse.ArgumentParser(description="Oracle probe training")
    parser.add_argument("--features", required=True, help="Path to feature .pt file")
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available()
                              else "cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    try:
        data, p1_bins, p2_bins = load_features(args.features, weights_only=True)
    except Exception:
        data, p1_bins, p2_bins = load_features(args.features, weights_only=False)

    input_dim = data.shape[-1]
    train_ids, holdout_ids = create_splits(p1_bins, p2_bins)

    print(f"Data: {data.shape}, input_dim={input_dim}", flush=True)
    print(f"Train: {len(train_ids)}, Holdout: {len(holdout_ids)}", flush=True)
    print(f"Device: {device}", flush=True)

    results = []
    t0_all = time.time()

    for seed in range(args.n_seeds):
        t0 = time.time()
        print(f"\n--- Seed {seed} ---", flush=True)
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = Oracle(args.hidden_dim, input_dim).to(device)
        model = train_oracle(model, data, p1_bins, p2_bins, train_ids, device,
                             epochs=args.epochs, seed=seed)

        he, hf, hb = evaluate_oracle(model, data, p1_bins, p2_bins, holdout_ids, device)
        te, tf, tb = evaluate_oracle(model, data, p1_bins, p2_bins, train_ids, device)
        elapsed = time.time() - t0
        print(f"  -> holdout={hb*100:.1f}%  train={tb*100:.1f}%  ({elapsed:.0f}s)", flush=True)

        results.append({
            'seed': seed,
            'holdout_both': float(hb), 'holdout_p1': float(he), 'holdout_p2': float(hf),
            'train_both': float(tb),
        })

    holdouts = [r['holdout_both'] for r in results]
    print(f"\n{'='*60}", flush=True)
    print(f"Oracle Summary ({args.n_seeds} seeds):", flush=True)
    print(f"  Holdout: {np.mean(holdouts)*100:.1f}% +/- {np.std(holdouts)*100:.1f}%", flush=True)
    print(f"  Total time: {(time.time()-t0_all)/60:.1f}min", flush=True)

    output = {
        'config': {'features': args.features, 'epochs': args.epochs,
                   'n_seeds': args.n_seeds, 'input_dim': input_dim},
        'per_seed': results,
        'summary': {
            'holdout_both_mean': float(np.mean(holdouts)),
            'holdout_both_std': float(np.std(holdouts)),
        }
    }
    out_path = args.output or args.features.replace('.pt', '_oracle_results.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Saved {out_path}", flush=True)


if __name__ == '__main__':
    main()
