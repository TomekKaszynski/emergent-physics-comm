#!/usr/bin/env python3
"""Train AEv5 (from world_model.py) using standalone's training loop.

If this works → training loop in run_all.py is the problem.
If this fails → something environmental or import-related.
"""
import torch
import torch.nn.functional as F
import numpy as np
import time
import sys

sys.path.insert(0, '/Users/tomek/AI')
from world_model import SlotAttentionAEv5
from test_reference_sa import generate_clevr_images

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Device: {device}", flush=True)

# Same data as standalone
print("Generating data...", flush=True)
images, masks_gt = generate_clevr_images(n_images=2000, img_size=64, max_objects=3)
n_train = 1600
train_imgs = images[:n_train].to(device)
print(f"Data: {images.shape}, train={n_train}", flush=True)

# AEv5 model from world_model.py
n_slots = 7
model = SlotAttentionAEv5(n_slots=n_slots, slot_dim=64, img_size=64).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model: SlotAttentionAEv5 from world_model.py, {n_params:,} params", flush=True)

# Same optimizer as standalone
optimizer = torch.optim.Adam(model.parameters(), lr=4e-4)

n_epochs = 100  # enough to see if symmetry breaks
batch_size = 32
warmup_epochs = 30
steps_done = 0
t0 = time.time()

print(f"\nTraining for {n_epochs} epochs, batch={batch_size}", flush=True)
print(f"Using standalone's step-level warmup + 0.98 decay", flush=True)

for epoch in range(1, n_epochs + 1):
    model.train()
    perm = torch.randperm(n_train)
    epoch_loss = 0
    n_batches = 0

    for start in range(0, n_train, batch_size):
        idx = perm[start:start + batch_size]
        batch = train_imgs[idx]

        # Same LR schedule as standalone (step-level warmup + 0.98 decay)
        if epoch <= warmup_epochs:
            lr = 4e-4 * min(1.0, steps_done / max(1, warmup_epochs * (n_train // batch_size)))
        else:
            lr = 4e-4 * (0.98 ** (epoch - warmup_epochs))
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        # AEv5 forward returns: total_loss, recon_loss, entropy_reg, recon, slots, alpha
        total_loss, recon_loss, entropy_reg, recon, slots, alpha = model(batch, training=True)
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        epoch_loss += recon_loss.item()
        n_batches += 1
        steps_done += 1

    avg_loss = epoch_loss / n_batches

    if epoch % 10 == 0 or epoch == 1:
        model.eval()
        with torch.no_grad():
            val_batch = train_imgs[:32]  # quick check on train data
            _, _, _, _, _, alpha = model(val_batch, training=False)
            coverage = alpha.mean(dim=(0, 2, 3))
            max_cov = coverage.max().item()
            active = (coverage > 0.01).sum().item()
            pixel_entropy = -(alpha * (alpha + 1e-8).log()).sum(dim=1).mean()
            norm_entropy = pixel_entropy.item() / np.log(n_slots)

        elapsed = time.time() - t0
        print(f"  Ep {epoch:4d}/{n_epochs}: loss={avg_loss:.4f} "
              f"active={active}/{n_slots} max_cov={max_cov*100:.1f}% "
              f"entropy={norm_entropy:.3f} lr={lr:.1e} [{elapsed:.0f}s]", flush=True)

        if norm_entropy < 0.3:
            print(f"\n  BREAKTHROUGH at epoch {epoch}!", flush=True)

print("\nDone.", flush=True)
