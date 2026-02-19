#!/usr/bin/env python3
"""Standalone reference-faithful Slot Attention test.

This is a MINIMAL, self-contained implementation that matches the Google Research
reference as closely as possible (adapted from TF to PyTorch). The goal is to
determine if our SlotAttentionAEv5 has a bug or if we just need more training.

Key reference details:
- slots_mu: glorot_uniform (xavier), learnable
- slots_log_sigma: glorot_uniform (xavier), learnable → sigma = exp(log_sigma) ≈ 1.0
- Encoder: 4 Conv2d layers, ALL stride=1, NO downsampling
- Position embed: 4-channel [x, 1-x, y, 1-y] → Linear → add to features
- Encoder MLP: Linear(D, D) + ReLU + Linear(D, D) after pos embed
- LayerNorm before SA
- SA: 3 iterations, softmax over slots, GRU update, MLP residual
- Decoder: MLP spatial broadcast (slot_dim + 2 → 4 layers → RGB+alpha)
- Training: Adam lr=4e-4, warmup 10K steps, decay 0.5 at 100K steps
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import sys


# ─── Data Generation ──────────────────────────────────────────────────

def generate_clevr_images(n_images=2000, img_size=64, max_objects=3, min_objects=2):
    """Generate simple CLEVR-like images: colored circles on gray background."""
    colors = [
        (1.0, 0.0, 0.0),  # red
        (0.0, 1.0, 0.0),  # green
        (0.0, 0.0, 1.0),  # blue
        (1.0, 1.0, 0.0),  # yellow
        (0.0, 1.0, 1.0),  # cyan
        (1.0, 0.0, 1.0),  # magenta
    ]
    images = torch.full((n_images, 3, img_size, img_size), 0.5)
    masks = torch.zeros(n_images, max_objects + 1, img_size, img_size)
    masks[:, 0] = 1.0  # background mask

    yy, xx = torch.meshgrid(torch.arange(img_size), torch.arange(img_size), indexing='ij')

    for i in range(n_images):
        n_obj = np.random.randint(min_objects, max_objects + 1)
        placed = []
        for j in range(n_obj):
            r = np.random.randint(6, 12)
            for _ in range(100):
                cx = np.random.randint(r + 2, img_size - r - 2)
                cy = np.random.randint(r + 2, img_size - r - 2)
                ok = True
                for (px, py, pr) in placed:
                    if (cx - px)**2 + (cy - py)**2 < (r + pr + 3)**2:
                        ok = False
                        break
                if ok:
                    break
            placed.append((cx, cy, r))
            dist = ((xx - cx).float()**2 + (yy - cy).float()**2).sqrt()
            mask = (dist <= r).float()
            color = colors[j % len(colors)]
            for c in range(3):
                images[i, c] = images[i, c] * (1 - mask) + color[c] * mask
            masks[i, 0] -= mask
            masks[i, j + 1] = mask

        if i % 500 == 0:
            print(f"  Generated {i}/{n_images}", flush=True)

    masks[:, 0] = masks[:, 0].clamp(min=0)
    print(f"  Generated {n_images}/{n_images}", flush=True)
    return images, masks


# ─── Model ────────────────────────────────────────────────────────────

class SoftPositionEmbed(nn.Module):
    """Exactly matching reference: 4-channel grid [x, 1-x, y, 1-y]."""
    def __init__(self, hidden_dim, resolution):
        super().__init__()
        H, W = resolution
        xs = torch.linspace(0, 1, W)
        ys = torch.linspace(0, 1, H)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        grid = torch.stack([xx, 1 - xx, yy, 1 - yy], dim=-1)  # [H, W, 4]
        self.register_buffer('grid', grid)
        self.proj = nn.Linear(4, hidden_dim)

    def forward(self, x):
        return x + self.proj(self.grid)


class SlotAttention(nn.Module):
    """Reference-faithful slot attention module."""

    def __init__(self, n_slots, slot_dim, n_iters=3, feature_dim=64, eps=1e-8):
        super().__init__()
        self.n_slots = n_slots
        self.slot_dim = slot_dim
        self.n_iters = n_iters
        self.eps = eps

        # Reference init: glorot_uniform for both, exp(log_sigma) for noise
        self.slots_mu = nn.Parameter(torch.empty(1, 1, slot_dim))
        self.slots_log_sigma = nn.Parameter(torch.empty(1, 1, slot_dim))
        nn.init.xavier_uniform_(self.slots_mu)
        nn.init.xavier_uniform_(self.slots_log_sigma)

        self.project_q = nn.Linear(slot_dim, slot_dim, bias=False)
        self.project_k = nn.Linear(feature_dim, slot_dim, bias=False)
        self.project_v = nn.Linear(feature_dim, slot_dim, bias=False)

        self.gru = nn.GRUCell(slot_dim, slot_dim)

        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, slot_dim * 2),
            nn.ReLU(),
            nn.Linear(slot_dim * 2, slot_dim),
        )

        self.norm_inputs = nn.LayerNorm(feature_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.norm_mlp = nn.LayerNorm(slot_dim)

    def forward(self, inputs):
        B, N, _ = inputs.shape
        inputs = self.norm_inputs(inputs)
        k = self.project_k(inputs)
        v = self.project_v(inputs)

        # Reference init: mu + exp(log_sigma) * N(0,1)
        mu = self.slots_mu.expand(B, self.n_slots, -1)
        sigma = self.slots_log_sigma.exp().expand(B, self.n_slots, -1)
        slots = mu + sigma * torch.randn_like(mu)

        for _ in range(self.n_iters):
            slots_prev = slots
            slots = self.norm_slots(slots)
            q = self.project_q(slots)

            scale = self.slot_dim ** -0.5
            # [B, N, K] - each position distributes attention across slots
            attn_logits = torch.einsum('bnd,bkd->bnk', k, q) * scale
            attn = F.softmax(attn_logits, dim=-1)  # softmax over slots (competition)

            # Weighted mean: normalize over spatial positions
            attn = attn + self.eps
            attn = attn / attn.sum(dim=1, keepdim=True)

            updates = torch.einsum('bnk,bnd->bkd', attn, v)

            slots = self.gru(
                updates.reshape(-1, self.slot_dim),
                slots_prev.reshape(-1, self.slot_dim),
            ).reshape(B, self.n_slots, self.slot_dim)

            slots = slots + self.mlp(self.norm_mlp(slots))

        return slots


class SlotAttentionAE(nn.Module):
    """Reference-faithful Slot Attention Autoencoder.

    Key: NO STRIDE in encoder. Full resolution features.
    """

    def __init__(self, n_slots=7, slot_dim=64, img_size=64):
        super().__init__()
        self.n_slots = n_slots
        self.slot_dim = slot_dim
        self.img_size = img_size
        resolution = img_size  # no downsampling

        # Encoder: 4 conv layers, ALL stride=1 (reference-faithful)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 5, padding=2), nn.ReLU(),
            nn.Conv2d(64, 64, 5, padding=2), nn.ReLU(),
            nn.Conv2d(64, 64, 5, padding=2), nn.ReLU(),
            nn.Conv2d(64, 64, 5, padding=2), nn.ReLU(),
        )

        self.encoder_pos = SoftPositionEmbed(64, (resolution, resolution))
        self.encoder_mlp = nn.Sequential(
            nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 64),
        )
        self.encoder_norm = nn.LayerNorm(64)

        self.slot_attention = SlotAttention(
            n_slots=n_slots, slot_dim=slot_dim, n_iters=3, feature_dim=64)

        # MLP spatial broadcast decoder
        self.decoder = nn.Sequential(
            nn.Linear(slot_dim + 2, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 4),
        )

        # Decoder position grid
        xs = torch.linspace(-1, 1, img_size)
        ys = torch.linspace(-1, 1, img_size)
        gy, gx = torch.meshgrid(ys, xs, indexing='ij')
        self.register_buffer('dec_grid', torch.stack([gx, gy], dim=-1).reshape(-1, 2))

    def encode(self, image):
        x = self.encoder(image)  # [B, 64, H, W] - FULL resolution
        B = x.shape[0]
        H = W = self.img_size
        x = x.permute(0, 2, 3, 1)  # [B, H, W, 64]
        x = self.encoder_pos(x)
        x = x.reshape(B, H * W, 64)
        x = self.encoder_mlp(x)
        x = self.encoder_norm(x)
        return self.slot_attention(x)

    def decode(self, slots):
        B, K, D = slots.shape
        N = self.img_size ** 2

        # Spatial broadcast
        slots_bc = slots.unsqueeze(2).expand(B, K, N, D)
        grid = self.dec_grid.unsqueeze(0).unsqueeze(0).expand(B, K, N, 2)
        dec_in = torch.cat([slots_bc, grid], dim=-1)

        decoded = self.decoder(dec_in)
        rgb = decoded[..., :3]
        alpha_logits = decoded[..., 3:]

        alpha = F.softmax(alpha_logits, dim=1)
        recon = (alpha * rgb).sum(dim=1)
        recon = recon.reshape(B, self.img_size, self.img_size, 3).permute(0, 3, 1, 2)
        alpha = alpha.squeeze(-1).reshape(B, K, self.img_size, self.img_size)
        return recon, alpha

    def forward(self, image):
        slots = self.encode(image)
        recon, alpha = self.decode(slots)
        loss = F.mse_loss(recon, image)
        return loss, recon, slots, alpha


# ─── Evaluation ───────────────────────────────────────────────────────

def compute_ari(pred_masks, gt_masks):
    """Compute Adjusted Rand Index between predicted and ground-truth masks."""
    from sklearn.metrics import adjusted_rand_score
    B = pred_masks.shape[0]
    aris = []
    for i in range(B):
        pred = pred_masks[i].argmax(dim=0).cpu().numpy().flatten()
        gt = gt_masks[i].argmax(dim=0).cpu().numpy().flatten()
        # Only evaluate on non-background pixels
        fg = gt_masks[i][1:].sum(dim=0).cpu().numpy().flatten() > 0.5
        if fg.sum() < 10:
            continue
        aris.append(adjusted_rand_score(gt[fg], pred[fg]))
    return np.mean(aris) if aris else 0.0


# ─── Training ─────────────────────────────────────────────────────────

def train():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)

    # Generate data
    print("Generating data...", flush=True)
    images, masks_gt = generate_clevr_images(n_images=2000, img_size=64, max_objects=3)
    n_train = 1600
    train_imgs = images[:n_train].to(device)
    val_imgs = images[n_train:].to(device)
    val_masks = masks_gt[n_train:]
    print(f"Data: {images.shape}, train={n_train}", flush=True)

    # Model
    n_slots = 7
    model = SlotAttentionAE(n_slots=n_slots, slot_dim=64, img_size=64).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params", flush=True)

    # Optimizer with warmup
    optimizer = torch.optim.Adam(model.parameters(), lr=4e-4)

    # Training
    n_epochs = 300
    batch_size = 32  # smaller batch for MPS memory with 4096 tokens
    warmup_epochs = 30
    steps_done = 0
    t0 = time.time()

    print(f"\nTraining for {n_epochs} epochs, batch={batch_size}", flush=True)
    print(f"Tokens per image: {64*64}=4096 (no stride)", flush=True)
    print(f"Steps/epoch: {n_train // batch_size}", flush=True)
    print(f"Total steps: ~{n_epochs * (n_train // batch_size)}", flush=True)
    print(flush=True)

    for epoch in range(1, n_epochs + 1):
        model.train()
        perm = torch.randperm(n_train)
        epoch_loss = 0
        n_batches = 0

        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]
            batch = train_imgs[idx]

            # Learning rate warmup
            if epoch <= warmup_epochs:
                lr = 4e-4 * min(1.0, steps_done / max(1, warmup_epochs * (n_train // batch_size)))
            else:
                lr = 4e-4 * (0.98 ** (epoch - warmup_epochs))
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            loss, _, _, _ = model(batch)
            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping (reference uses max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            steps_done += 1

        avg_loss = epoch_loss / n_batches

        # Log every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            # Quick eval: check slot usage on a few val images
            model.eval()
            with torch.no_grad():
                val_batch = val_imgs[:32]
                _, _, _, alpha = model(val_batch)
                # Per-slot pixel coverage
                coverage = alpha.mean(dim=(0, 2, 3))  # [K]
                max_cov = coverage.max().item()
                active = (coverage > 0.01).sum().item()
                # Entropy
                pixel_entropy = -(alpha * (alpha + 1e-8).log()).sum(dim=1).mean()
                max_entropy = np.log(n_slots)
                norm_entropy = pixel_entropy.item() / max_entropy

            elapsed = time.time() - t0
            print(f"  Ep {epoch:4d}/{n_epochs}: loss={avg_loss:.4f} "
                  f"active={active}/{n_slots} max_cov={max_cov*100:.1f}% "
                  f"entropy={norm_entropy:.3f} lr={lr:.1e} [{elapsed:.0f}s]", flush=True)

    # Final evaluation
    print(f"\n{'='*50}", flush=True)
    print("Final Evaluation", flush=True)
    print(f"{'='*50}", flush=True)

    model.eval()
    aris = []
    with torch.no_grad():
        for start in range(0, len(val_imgs), 32):
            batch = val_imgs[start:start + 32]
            batch_masks = val_masks[start:start + 32]
            _, _, _, alpha = model(batch)
            ari = compute_ari(alpha, batch_masks)
            aris.append(ari)

    mean_ari = np.mean(aris)
    print(f"ARI: {mean_ari:.3f}", flush=True)

    # Check slot usage
    with torch.no_grad():
        _, _, _, alpha = model(val_imgs[:32])
        coverage = alpha.mean(dim=(0, 2, 3))
        active = (coverage > 0.01).sum().item()
        max_cov = coverage.max().item()

    print(f"Active slots: {active}/{n_slots}", flush=True)
    print(f"Max coverage: {max_cov*100:.1f}%", flush=True)

    passed = mean_ari > 0.5 and max_cov < 0.4 and active >= 3
    print(f"\nVERDICT: {'PASS' if passed else 'FAIL'}", flush=True)

    # Save visualization
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        model.eval()
        with torch.no_grad():
            batch = val_imgs[:4]
            _, recon, _, alpha = model(batch)

        fig, axes = plt.subplots(4, 2 + n_slots, figsize=(2 * (2 + n_slots), 8))
        for i in range(4):
            axes[i, 0].imshow(batch[i].cpu().permute(1, 2, 0).clamp(0, 1))
            axes[i, 0].set_title('Input' if i == 0 else '')
            axes[i, 1].imshow(recon[i].cpu().permute(1, 2, 0).clamp(0, 1))
            axes[i, 1].set_title('Recon' if i == 0 else '')
            for k in range(n_slots):
                axes[i, 2 + k].imshow(alpha[i, k].cpu(), vmin=0, vmax=1, cmap='viridis')
                axes[i, 2 + k].set_title(f'Slot {k}' if i == 0 else '')
            for ax in axes[i]:
                ax.axis('off')

        plt.tight_layout()
        plt.savefig('/Users/tomek/AI/results/test_reference_sa.png', dpi=100)
        plt.close()
        print(f"Saved: results/test_reference_sa.png", flush=True)
    except Exception as e:
        print(f"Viz error: {e}", flush=True)

    return mean_ari, passed


if __name__ == '__main__':
    train()
