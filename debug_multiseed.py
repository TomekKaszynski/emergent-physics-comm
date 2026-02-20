#!/usr/bin/env python3
"""Test SA across multiple random seeds to measure reliability.

The original standalone run broke through at epoch 30 but was a lucky seed.
This test runs 5 seeds for 60 epochs each and checks which break through.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import sys

sys.path.insert(0, '/Users/tomek/AI')
from test_reference_sa import SlotAttentionAE, generate_clevr_images

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Device: {device}", flush=True)

n_seeds = 5
n_epochs = 60
batch_size = 32
n_slots = 7
results = []

for seed in range(n_seeds):
    print(f"\n{'='*50}", flush=True)
    print(f"SEED {seed}", flush=True)
    print(f"{'='*50}", flush=True)

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Generate data with this seed
    images, _ = generate_clevr_images(n_images=2000, img_size=64, max_objects=3)
    n_train = 1600
    train_imgs = images[:n_train].to(device)

    # Fresh model
    model = SlotAttentionAE(n_slots=n_slots, slot_dim=64, img_size=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=4e-4)

    warmup_epochs = 30
    steps_done = 0
    t0 = time.time()
    best_entropy = 1.0
    breakthrough_epoch = None

    for epoch in range(1, n_epochs + 1):
        model.train()
        perm = torch.randperm(n_train)
        epoch_loss, n_batches = 0, 0

        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]
            batch = train_imgs[idx]

            if epoch <= warmup_epochs:
                lr = 4e-4 * min(1.0, steps_done / max(1, warmup_epochs * (n_train // batch_size)))
            else:
                lr = 4e-4 * (0.98 ** (epoch - warmup_epochs))
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            loss, _, _, _ = model(batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
            steps_done += 1

        avg_loss = epoch_loss / n_batches

        if epoch % 10 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                _, _, _, alpha = model(train_imgs[:32])
                pixel_entropy = -(alpha * (alpha + 1e-8).log()).sum(dim=1).mean()
                norm_entropy = pixel_entropy.item() / np.log(n_slots)
                coverage = alpha.mean(dim=(0, 2, 3))
                max_cov = coverage.max().item()

            elapsed = time.time() - t0
            print(f"  Ep {epoch:3d}: loss={avg_loss:.4f} entropy={norm_entropy:.3f} "
                  f"max_cov={max_cov*100:.1f}% [{elapsed:.0f}s]", flush=True)

            if norm_entropy < best_entropy:
                best_entropy = norm_entropy
            if norm_entropy < 0.9 and breakthrough_epoch is None:
                breakthrough_epoch = epoch
                print(f"  >>> BREAKTHROUGH at epoch {epoch}!", flush=True)

    result = {
        'seed': seed,
        'best_entropy': best_entropy,
        'breakthrough': breakthrough_epoch,
    }
    results.append(result)
    print(f"\n  Seed {seed}: best_entropy={best_entropy:.3f}, "
          f"breakthrough={'epoch '+str(breakthrough_epoch) if breakthrough_epoch else 'NONE'}", flush=True)

print(f"\n{'='*50}", flush=True)
print("SUMMARY", flush=True)
print(f"{'='*50}", flush=True)
for r in results:
    status = f"BREAK at ep {r['breakthrough']}" if r['breakthrough'] else "FAIL"
    print(f"  Seed {r['seed']}: entropy={r['best_entropy']:.3f} — {status}", flush=True)

n_success = sum(1 for r in results if r['breakthrough'] is not None)
print(f"\nSuccess rate: {n_success}/{n_seeds} ({100*n_success/n_seeds:.0f}%)", flush=True)
