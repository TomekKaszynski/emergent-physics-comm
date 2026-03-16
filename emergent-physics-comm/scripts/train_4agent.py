#!/usr/bin/env python3
"""Train 4-agent compositional communication with iterated learning.

Each agent observes a temporal slice of the scene and produces an independent
discrete message. Joint messages are decoded by a shared receiver population.

Usage:
    python scripts/train_4agent.py --features results/dino_ramp_features.pt --n-agents 4
    python scripts/train_4agent.py --features results/collision_dinov2_features.pt --n-agents 4
"""

import argparse
import json
import time
from functools import partial
import numpy as np
import torch

from src.models import (TemporalEncoder, CompositionalSender,
                        MultiAgentSender, CompositionalReceiver)
from src.datasets import load_features, create_splits
from src.training import train_communication, evaluate_4agent
from src.metrics import compute_compositionality_multiagent


def main():
    parser = argparse.ArgumentParser(description="Multi-agent communication training")
    parser.add_argument("--features", required=True, help="Path to feature .pt file")
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--n-agents", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=5)
    parser.add_argument("--n-heads", type=int, default=2)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available()
                              else "cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Load data
    try:
        data, p1_bins, p2_bins = load_features(args.features, weights_only=True)
    except Exception:
        data, p1_bins, p2_bins = load_features(args.features, weights_only=False)

    input_dim = data.shape[-1]
    total_frames = data.shape[1]
    frames_per_agent = total_frames // args.n_agents
    train_ids, holdout_ids = create_splits(p1_bins, p2_bins)
    msg_dim = args.n_agents * args.vocab_size * args.n_heads

    print(f"Data: {data.shape}, input_dim={input_dim}", flush=True)
    print(f"Agents: {args.n_agents}, frames/agent: {frames_per_agent}", flush=True)
    print(f"Train: {len(train_ids)}, Holdout: {len(holdout_ids)}", flush=True)
    print(f"Message: {args.n_agents}x{args.n_heads}x{args.vocab_size} = {msg_dim} dim", flush=True)
    print(f"Device: {device}", flush=True)

    # Evaluation function with agent config bound
    eval_fn = partial(evaluate_4agent, n_agents=args.n_agents,
                      frames_per_agent=frames_per_agent)

    results = []
    t0_all = time.time()

    for seed in range(args.n_seeds):
        t0 = time.time()
        print(f"\n--- Seed {seed} ---", flush=True)
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Build multi-agent sender
        senders = []
        for _ in range(args.n_agents):
            enc = TemporalEncoder(args.hidden_dim, input_dim)
            s = CompositionalSender(enc, args.hidden_dim, args.vocab_size, args.n_heads)
            senders.append(s)
        multi_sender = MultiAgentSender(senders).to(device)

        receivers = [CompositionalReceiver(msg_dim, args.hidden_dim).to(device) for _ in range(3)]

        nan_count = train_communication(
            multi_sender, receivers, data, p1_bins, p2_bins,
            train_ids, holdout_ids, device, msg_dim,
            epochs=args.epochs, seed=seed,
            eval_fn=eval_fn,
        )

        # Final evaluation
        te, tf, tb = eval_fn(multi_sender, receivers, data, p1_bins, p2_bins, train_ids, device)
        he, hf, hb = eval_fn(multi_sender, receivers, data, p1_bins, p2_bins, holdout_ids, device)
        comp = compute_compositionality_multiagent(
            multi_sender, data, p1_bins, p2_bins, device,
            n_agents=args.n_agents, frames_per_agent=frames_per_agent,
            n_heads=args.n_heads, vocab_size=args.vocab_size,
        )
        elapsed = time.time() - t0
        print(f"  -> holdout={hb*100:.1f}%  PosDis={comp['pos_dis_global']:.3f}  "
              f"best_agent={comp['best_agent_posdis']:.3f}  NaN={nan_count}  ({elapsed:.0f}s)",
              flush=True)

        results.append({
            'seed': seed,
            'holdout_both': float(hb), 'holdout_p1': float(he), 'holdout_p2': float(hf),
            'train_both': float(tb),
            'pos_dis_global': comp['pos_dis_global'],
            'best_agent_posdis': comp['best_agent_posdis'],
            'per_agent_posdis': comp['per_agent_posdis'],
            'topsim': comp['topsim'],
            'nan_count': nan_count,
        })

    # Summary
    holdouts = [r['holdout_both'] for r in results]
    posdis_vals = [r['pos_dis_global'] for r in results]
    n_comp = sum(1 for r in results if r['pos_dis_global'] >= 0.4)

    print(f"\n{'='*60}", flush=True)
    print(f"{args.n_agents}-Agent Summary ({args.n_seeds} seeds):", flush=True)
    print(f"  Holdout: {np.mean(holdouts)*100:.1f}% +/- {np.std(holdouts)*100:.1f}%", flush=True)
    print(f"  PosDis:  {np.mean(posdis_vals):.3f} +/- {np.std(posdis_vals):.3f}", flush=True)
    print(f"  Compositional: {n_comp}/{args.n_seeds}", flush=True)
    print(f"  Total time: {(time.time()-t0_all)/60:.1f}min", flush=True)

    output = {
        'config': {
            'features': args.features, 'n_agents': args.n_agents, 'n_seeds': args.n_seeds,
            'epochs': args.epochs, 'vocab_size': args.vocab_size, 'n_heads': args.n_heads,
            'input_dim': input_dim, 'msg_dim': msg_dim,
            'frames_per_agent': frames_per_agent,
        },
        'per_seed': results,
        'summary': {
            'holdout_both_mean': float(np.mean(holdouts)),
            'holdout_both_std': float(np.std(holdouts)),
            'pos_dis_mean': float(np.mean(posdis_vals)),
            'pos_dis_std': float(np.std(posdis_vals)),
            'n_compositional': n_comp, 'n_seeds': args.n_seeds,
        }
    }
    out_path = args.output or args.features.replace('.pt', f'_{args.n_agents}agent_results.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Saved {out_path}", flush=True)


if __name__ == '__main__':
    main()
