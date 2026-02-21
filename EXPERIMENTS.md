# Experiment Log

All visualizations saved in results/ with naming convention phaseXX_description.png. Model checkpoints: phaseXX_model.pt or phaseXX_autoencoder.pt.

## Pre-Phase 26 Summary (Phases 1-25d)

Phases 1-25d developed the multi-agent visual world model incrementally:
- **Phases 1-5**: Direct MLP world model, latent JEPA, basic physics sim
- **Phases 6-9**: Communication bottleneck, two-agent setup, VICReg
- **Phases 10-14**: Hierarchical models, action conditioning, multi-step prediction
- **Phases 15-19**: Visual encoder (CNN), split-view cameras, image-based JEPA
- **Phases 20-24**: Slot-based representations (state-level, not pixel-level)
- **Phase 25-25d**: Slot attention on state vectors — worked well (R²>0.9 binding)

Key insight from 25d: Slot attention works perfectly on structured state vectors but has never been tested on raw pixels. Phase 26 begins pixel-level slot attention.

---

## Phase 26: Object Discovery via Slot Attention
**Date:** Feb 18 ~15:06 | **Duration:** ~2h

- **Config:** `ObjectCentricJEPA`, 6 slots x 64-dim, 300 epochs, batch=48, lr=3e-4
- **Dataset:** 300 episodes x 40 steps, 5 objects, 64x64 images
- **Loss:** prediction + β*KL + VICReg (β annealed 0→0.001)
- **Result:** Recon OK, slot binding R²≈0 for all slot/object pairs
- **Verdict:** FAIL — slots identical, no object binding

## Phase 26b: Full Capacity, No KL
**Date:** Feb 18 ~17:08 | **Duration:** ~2h

- **Change:** Removed KL penalty entirely (β=0), deterministic communication
- **Config:** `ObjectCentricJEPAv2`, 6 slots x 64-dim, 300 epochs
- **Result:** R²≈0 for all pairs
- **Verdict:** FAIL — removing KL didn't help

## Phase 26c: Two-Stage (AE then JEPA)
**Date:** Feb 18 ~19:36 | **Duration:** ~2h

- **Change:** Separated autoencoder from prediction. Stage 1: train AE on pixel recon. Stage 2: freeze slots, train predictor.
- **Config:** `SlotAttentionAutoencoder`, 6 slots x 64-dim, 50 AE epochs + prediction
- **Result:** R²≈0, slots still uniform
- **Verdict:** FAIL — two-stage didn't break symmetry

## Phase 26d: Constrained Decoder + Diversity Loss
**Date:** Feb 18 ~21:34 | **Duration:** ~2h

- **Change:** Added slot dropout, diversity regularization, constrained decoder
- **Config:** `SlotAttentionAEv2`, 6 slots x 64-dim, 200 epochs
- **Result:** R²≈0 despite diversity loss
- **Verdict:** FAIL — diversity loss insufficient for symmetry breaking

## Phase 26e: Tiny Slots + Rich Scenes
**Date:** Feb 19 ~00:47 | **Duration:** ~3h

- **Change:** 10 slots x 12-dim (tiny), richer scenes (8 objects, 3 shapes, checkerboard floor)
- **Config:** `SlotAttentionAEv3`, batch=48, 50 AE epochs
- **Result:** Slots still uniform
- **Verdict:** FAIL — tiny slots didn't force specialization

## Phase 26f: Match Original Slot Attention (Research-Informed)
**Date:** Feb 19 ~04:13 | **Duration:** ~2h

- **Change:** CNN ConvTranspose2d decoder (locality bias), dense encoder (stride-2-once → 32x32=1024 tokens), gradient background, LR warmup + decay
- **Config:** `SlotAttentionAEv5`, 10 slots x 64-dim, 677K params, 100 epochs (early stopped at 50)
- **Result:** recon=0.0005 (excellent), entropy=1.000 (completely uniform), 0/8 bound
- **Verdict:** FAIL — recon perfect but all slots identical

## Phase 26f CLEVR Diagnostic (6 sub-runs)
**Date:** Feb 19 ~06:00-15:16 | **Duration:** ~9h total

Simplified test: colored circles on gray background (64x64). Goal: isolate architecture bugs from scene complexity.

| Sub-run | Change | Epochs | Entropy | ARI | Verdict |
|---------|--------|--------|---------|-----|---------|
| CNN decoder, per-slot init | ConvTranspose2d decoder | 50 | 0.016 | 0.042 | FAIL (Voronoi tiles) |
| MLP decoder, per-slot init | Spatial broadcast decoder | 50 | 0.003 | 0.003 | FAIL |
| Per-slot init, 5 iters, 200ep | Long training | 200 | 0.014 | 0.000 | FAIL |
| Shared init, 5 iters, 200ep | Reference-style shared init | 200 | 0.999 | 0.025 | FAIL |
| 1-iteration SA | Minimal iterations | 50 | 1.000 | 0.030 | FAIL |
| Gumbel softmax | Different attention | 50 | 1.000 | 0.016 | FAIL |

**Root causes found during diagnosis:**
1. **Encoder dying features:** PyTorch default `kaiming_uniform_(a=sqrt(5))` shrinks features 24x over 4 conv layers → fixed with `kaiming_normal_`
2. **Missing encoder MLP:** Reference has CNN→PosEmbed→MLP→SA, we had CNN→PosEmbed→SA → fixed
3. **Sigma init 7x too small:** Direct xavier sigma≈±0.15, reference uses `exp(log_sigma)`≈1.0
4. **Insufficient spatial tokens:** stride-2 gave 1024 tokens, reference uses no stride (4096+)
5. **Symmetric equilibrium trap:** All slots identical → symmetric gradients → zero differentiation signal. Training makes slots MORE similar (cosine 0.055→0.794 after 3 SA iters, worsens to 0.749→0.992 after training).

## Phase 26f Standalone Reference Test (BREAKTHROUGH)
**Date:** Feb 19 ~15:16-18:51 | **Duration:** ~3.5h (killed at epoch 150)

- **Script:** `test_reference_sa.py` (standalone, reference-faithful)
- **Three critical fixes:**
  1. `sigma = exp(log_sigma)` — 7x more initial noise (sigma≈1.0 vs 0.15)
  2. No encoder stride — 4096 tokens vs 1024 (4x spatial resolution)
  3. Reference attention normalization order: `softmax + eps` then `/ sum`
- **Config:** 2000 images, 7 slots, 64-dim, batch=32, lr=4e-4 with warmup+exp decay
- **Result:**
  ```
  Ep   1: loss=0.1975 entropy=1.000
  Ep  30: loss=0.0236 entropy=0.861  ← FIRST BREAK
  Ep  40: loss=0.0062 entropy=0.552  ← MAJOR DROP
  Ep 100: loss=0.0014 entropy=0.530  ← plateau (LR decayed too fast)
  Ep 150: loss=0.0011 entropy=0.540  ← still plateau
  ```
- **Verdict:** BREAKTHROUGH — first run to ever break entropy below 1.000. Plateaued at 0.53 due to aggressive LR decay (0.98/epoch → lr=1e-5 by epoch 200).

---

## Phase 26g: Port Reference Fixes + Constant LR (FAIL)
**Date:** Feb 19 ~20:45 | **Duration:** killed at epoch 40

- **Changes from 26f standalone:** Ported 3 fixes (log_sigma, no stride, attn eps order) into `SlotAttentionAEv5`. Changed LR to constant (warmup 30ep → constant 4e-4 → halve at 250). 300 epochs, batch=32, 5000 CLEVR images.
- **Result:**
  ```
  Ep  10: recon=0.0379 entropy=1.000 [380s]
  Ep  20: recon=0.0301 entropy=1.000 [1715s]
  Ep  30: recon=0.0179 entropy=1.000 [3194s]
  Ep  40: recon=0.0142 entropy=1.000 [4415s]  ← killed
  ```
- **Verdict:** FAIL — entropy stuck at 1.000 through epoch 40, when standalone test was at 0.552

### Root Cause: Encoder CNN Init

Side-by-side comparison found ONE remaining difference between the standalone (works) and ported AEv5 (fails):

| Aspect | Standalone (entropy=0.53) | AEv5 (entropy=1.00) |
|--------|---------------------------|---------------------|
| Encoder CNN init | **PyTorch default** (kaiming_uniform, a=sqrt(5)) | **kaiming_normal_(fan_out, relu)** |
| Everything else | identical | identical |

The `kaiming_normal_` init was added in Phase 26f to "fix dying features" (features shrank 24x across 4 conv layers with default init). But the standalone test WORKED with default init.

**Why the "fix" broke it:** With default init, encoder features are small → position embedding has MORE relative influence → spatial positions are distinguishable → SA can differentiate positions → symmetry breaking. With kaiming_normal_, features start large → position embedding is relatively weak → all positions look similar → SA can't differentiate → slots stay uniform.

**Fix for Phase 26h:** Remove the `kaiming_normal_` init block (lines 2009-2016 in world_model.py). Use PyTorch default init, matching the standalone test exactly.

---

## Phase 26h: Remove kaiming_normal_ Init (Match Standalone)
**Date:** Feb 20

- **Change:** Removed `kaiming_normal_` encoder CNN init (was lines 2009-2016). Now uses PyTorch default `kaiming_uniform_(a=sqrt(5))`, matching the standalone test exactly.
- **Config:** Same as 26g (300 epochs, batch=32, constant LR 4e-4, 4096 tokens, 7 slots)
- **Result:** entropy=1.000 through epoch 30 (killed). Same failure as 26g.
- **Verdict:** FAIL — removing kaiming_normal_ was not the root cause

### Diagnosis: Architecture Equivalence Proven, Original Run Was Lucky Seed

**Forward pass comparison test:** Wrote `debug_compare.py` — copied weights from standalone `SlotAttentionAE` to `SlotAttentionAEv5`, fed same input with same random seed. Results: **bit-for-bit identical** outputs (loss, recon, slots, alpha all match). Architectures are provably equivalent.

**AEv5 with standalone training loop:** Ran `debug_train_aev5.py` — imported AEv5 from world_model.py, used standalone's data (2000 images), standalone's step-level LR warmup + 0.98 decay. Result: entropy=1.000 through epoch 40. Same failure.

**Standalone re-run:** Re-ran `test_reference_sa.py` (unchanged file). Result: entropy=1.000 through epoch 50. **The standalone itself no longer reproduces the breakthrough.** The original successful run (entropy=0.53) was a lucky random seed.

### Multi-Seed Reliability Test

Ran `debug_multiseed.py`: 5 seeds × 60 epochs each, standalone SA model, 2000 images, 2-3 circles.

| Seed | Best Entropy | Breakthrough | Verdict |
|------|-------------|-------------|---------|
| 0 | 1.000 | NONE | FAIL |
| 1 | 0.999 | NONE | FAIL |
| 2 | 1.000 | NONE | FAIL |
| 3-4 | (not reached, killed) | — | — |

**Success rate: 0/3 (0%) in 60 epochs.** SA is unreliable with current training scale.

### Root Cause: Insufficient Training

The original successful run was 1 lucky seed out of many attempts. With 2000 images × 32 batch × 60 epochs = 3000 gradient steps, SA doesn't reliably break symmetry. The reference paper trains for **500K steps** with batch 64 on 128×128 CLEVR. We're at ~3K steps — **167x fewer** than the paper.

**Fix for next session:** Scale up training:
- More steps: 500+ epochs (25K+ steps) or larger dataset
- Larger batch: 64 (if MPS memory allows with 4096 tokens)
- Longer warmup: match paper's 10K step warmup
- Consider 128×128 images (paper resolution)

---

## Phase 26i: Extended Training (500 Epochs)
**Date:** Feb 20

- **Change:** ae_epochs 300→500, LR halve at 450 (not 250), print every 5 epochs for first 25 then every 10, failure exit at epoch 100 if entropy>0.99
- **Config:** 5000 CLEVR images, batch=32, 7 slots, constant LR 4e-4 after 30-epoch warmup
- **Result:**
  ```
  Ep   1: recon=0.3259 entropy=1.000
  Ep  25: recon=0.0249 entropy=0.999
  Ep  50: recon=0.0124 entropy=0.982
  Ep  70: recon=0.0070 entropy=0.935
  Ep 100: recon=0.0054 entropy=0.869  ← passed failure exit
  Ep 140: recon=0.0032 entropy=0.854  ← best
  Ep 250: recon=0.0017 entropy=0.872  ← plateau, killed
  ```
- **Verdict:** PARTIAL — proved symmetry breaking is possible with extended training but shared Gaussian init cannot achieve sharp binding. Plateaus at ~0.85 entropy. Reconstruction excellent (0.002). Never reached 0.2 target.

### Analysis

Shared init `N(μ, σ)` relies on random noise to differentiate slots. With 7 slots and 64-dim, the random vectors have cosine similarity ~0, giving initial differentiation. But as training progresses, the learned μ dominates and all slots converge toward it. The random noise becomes insufficient to maintain slot diversity — hence the plateau at ~0.85 entropy.

## Phase 26j: Per-Slot Learnable Init (BO-QSA Style)
**Date:** Feb 20

- **Change:** Replaced shared Gaussian init `N(μ, σ)` with per-slot learnable vectors: `nn.Parameter(torch.randn(1, n_slots, slot_dim) * 0.02)`. No random sampling — deterministic init breaking symmetry by design.
- **Config:** Same as 26i (500 epochs, batch=32, 7 slots, constant LR 4e-4, 5000 CLEVR images)
- **Result:**
  ```
  Ep   1: recon=0.2391 entropy=1.000
  Ep  25: recon=0.0154 entropy=0.994
  Ep  50: recon=0.0089 entropy=0.958
  Ep  80: recon=0.0036 entropy=0.746
  Ep 100: recon=0.0026 entropy=0.596
  Ep 140: recon=0.0023 entropy=0.574
  Ep 170: recon=0.0017 entropy=0.562  ← best
  Ep 180: recon=0.0016 entropy=0.561  ← plateau, killed
  ```
- **Verdict:** PARTIAL — entropy 1.000→0.562, significantly better than 26i (0.85) but far from 0.2 target. Background slot claims ~90% coverage, object slots remain blurry. Diagnosis: decoder reconstructs well without sharp masks — the loss doesn't force sharp slot boundaries. Next: need to constrain the decoder or change the loss to require sharp masks.

---

## Phase 27: DINOv2 Feature Reconstruction
**Date:** Feb 20-21

- **Change:** Replaced pixel reconstruction with DINOv2 feature reconstruction. Frozen DINOv2-Small encoder (dinov2_vits14, 22M params frozen) extracts 256 patch tokens (16×16, 384-dim) from 224×224 resized images. SA groups patch features, MLP decoder reconstructs DINOv2 features (not pixels). Loss: MSE on features.
- **Architecture:** `SlotAttentionDINO` — 640K trainable params + 22M frozen DINOv2. SA feature_dim=384, slot_dim=64, per-slot learnable init.
- **Config:** Same training as 26j (500 epochs, batch=32, 7 slots, constant LR 4e-4, warmup 30, halve at 450)
- **Note:** Required patching DINOv2 cached files with `from __future__ import annotations` for Python 3.9 compatibility (latest DINOv2 uses `float | None` syntax requiring Python 3.10+).
- **Result:**
  ```
  Ep   1: recon=6.6785 entropy=1.000 active=5/7
  Ep  15: recon=1.6855 entropy=0.767 active=6/7  ← first major break
  Ep  20: recon=1.3404 entropy=0.578 active=7/7
  Ep  25: recon=1.1428 entropy=0.406 active=7/7  ← already better than 26j's best
  Ep  50: recon=0.8326 entropy=0.365 active=7/7
  Ep 100: recon=0.6247 entropy=0.368 active=7/7  ← plateau established
  Ep 150: recon=0.5276 entropy=0.370 active=7/7
  Ep 200: recon=0.4743 entropy=0.366 active=7/7  ← killed, plateau confirmed
  ```
- **Verdict:** SUCCESS — entropy 1.000→0.37 (best 0.362 at epoch 120). 7/7 slots active with even coverage (~33-37% max). No background domination (26j had 90% max_cov). Slots reliably differentiate. DINOv2's semantic features make SA's job dramatically easier: reached 0.406 entropy in 25 epochs vs 26j's 0.562 after 170 epochs with pixel reconstruction. 2x faster per epoch (30s vs 75s) due to 256 patches vs 4096 pixels.

### Comparison Across Phases

| Phase | Init | Target | Best Entropy | Epochs to Best | Slots Active |
|-------|------|--------|-------------|----------------|--------------|
| 26i | Shared Gaussian | Pixels | 0.854 | 140 | 6/7 (90% bg) |
| 26j | Per-slot learnable | Pixels | 0.562 | 170 | 5-6/7 (90% bg) |
| **27** | **Per-slot learnable** | **DINOv2 features** | **0.362** | **120** | **7/7 (33% max)** |

### Phase 27b Test 0: SA iterations ablation (5 iters)
**Date:** Feb 21

- **Change:** SA iterations 3→5 in SlotAttentionDINO. Everything else identical.
- **Config:** 100 epochs, batch=32, 7 slots, constant LR 4e-4, warmup 30
- **Result:**
  ```
  Ep   1: recon=6.6501 entropy=1.000 active=6/7
  Ep  15: recon=1.6827 entropy=0.606 active=7/7
  Ep  25: recon=1.4652 entropy=0.416 active=6/7
  Ep  50: recon=1.2220 entropy=0.255 active=7/7
  Ep  70: recon=1.1210 entropy=0.236 active=7/7
  Ep 100: recon=1.0179 entropy=0.234 active=7/7 max_cov=29.3%
  ```
  Eval: entropy=0.232, 7/7 active, max_cov=29.6%
- **Verdict:** 5 iters significantly better than 3 (0.232 vs 0.362). Improvement of 0.130 — well above 0.03 threshold. Near-perfect even distribution (29% max_cov with 7 slots = 1/7 = 14% ideal). Still dropping at epoch 100, not fully plateaued. Next: try 7 iters.

### Phase 27b Test 1: SA iterations ablation (7 iters)
**Date:** Feb 21

- **Change:** SA iterations 5→7 in SlotAttentionDINO. Everything else identical.
- **Config:** 100 epochs, batch=32, 7 slots, constant LR 4e-4, warmup 30
- **Result:**
  ```
  Ep   1: recon=6.6824 entropy=1.000 active=5/7
  Ep   5: recon=2.7099 entropy=1.000 active=1/7 max_cov=99.2%  ← winner-take-all
  Ep  10: recon=2.5086 entropy=1.000 active=2/7 max_cov=95.3%
  Ep  15: recon=1.7693 entropy=0.967 active=7/7 max_cov=19.3%  ← recovered
  Ep  25: recon=1.5159 entropy=0.559 active=7/7
  Ep  50: recon=1.2849 entropy=0.323 active=7/7
  Ep 100: recon=1.0752 entropy=0.268 active=7/7 max_cov=30.0%
  ```
  Eval: entropy=0.266, 7/7 active, max_cov=30.0%
- **Verdict:** 7 iters worse than 5 (0.266 vs 0.232). Early winner-take-all collapse (epochs 5-10, 1-2 slots grabbed 95-99% coverage) before recovering at epoch 15. Too many iterations over-sharpens attention early, causing one slot to dominate before others can differentiate.

### Phase 27b Test 2: SA iterations ablation (6 iters)
**Date:** Feb 21

- **Change:** SA iterations 7→6 in SlotAttentionDINO. Only 50 epochs (enough to compare trajectory).
- **Config:** 50 epochs, batch=32, 7 slots, constant LR 4e-4, warmup 30
- **Result:**
  ```
  Ep   1: recon=6.6766 entropy=1.000 active=5/7
  Ep   5: recon=2.6107 entropy=1.000 active=2/7 max_cov=95.3%  ← similar early collapse
  Ep  10: recon=2.2732 entropy=0.996 active=7/7 max_cov=27.4%
  Ep  15: recon=1.6965 entropy=0.753 active=7/7
  Ep  20: recon=1.5407 entropy=0.717 active=7/7
  Ep  25: recon=1.4697 entropy=0.676 active=7/7
  Ep  30: recon=1.4021 entropy=0.546 active=7/7
  Ep  40: recon=1.2737 entropy=0.452 active=7/7
  Ep  50: recon=1.2116 entropy=0.402 active=7/7 max_cov=31.7%
  ```
  Eval: entropy=0.405, 7/7 active, max_cov=33.0%
- **Verdict:** 6 iters at epoch 50 (0.402) is worse than 5 iters at epoch 50 (0.255). Also shows early winner-take-all tendency (epoch 5: 95.3% max_cov), similar to 7 iters. Confirms 5 iters is the sweet spot.

### SA Iterations Comparison Summary

| Iters | Entropy@50ep | Entropy@100ep | Early collapse? | Winner-take-all? |
|-------|-------------|---------------|-----------------|------------------|
| 3     | 0.365       | 0.368         | No              | No               |
| **5** | **0.255**   | **0.232**     | **No**          | **No**           |
| 6     | 0.402       | —             | Yes (ep 5)      | Mild             |
| 7     | 0.323       | 0.266         | Yes (ep 5)      | Severe           |

**Finding:** 5 iterations is optimal. More iterations (6, 7) cause early winner-take-all collapse where 1-2 slots grab >95% coverage before recovering. While 7 iters eventually reaches 0.266, it never catches up to 5 iters (0.232). The over-sharpening from extra iterations is counterproductive with per-slot learnable init.

### Phase 27b Test 0b: Extended training (5 iters, 200 epochs)
**Date:** Feb 21

- **Change:** ae_epochs 100→200 to push entropy below 0.2. n_iters=5, everything else identical.
- **Config:** 200 epochs (max), batch=32, 7 slots, constant LR 4e-4, warmup 30, 5 SA iters
- **Result:**
  ```
  Ep   1: recon=6.6766 entropy=1.000 active=4/7
  Ep   5: recon=2.6098 entropy=1.000 active=3/7
  Ep  10: recon=2.1788 entropy=0.809 active=3/7
  Ep  15: recon=1.9038 entropy=0.836 active=3/7
  Ep  20: recon=1.7055 entropy=0.886 active=4/7
  Ep  60: recon=0.9828 entropy=0.527 active=7/7 max_cov=36.4%
  Ep  70: recon=0.8989 entropy=0.173 active=7/7 max_cov=62.8%  ← SUCCESS, early stop
  ```
  Eval: entropy=0.170, 7/7 active, max_cov=63.8%
- **Verdict:** SUCCESS — entropy broke below 0.2 target at epoch 70 (0.170 vs 0.232 at 100ep in Test 0). Sharp drop from 0.527→0.173 between epochs 60-70 suggests phase transition in slot specialization. Max coverage rose to 63.8% (one dominant slot, likely background), but all 7 slots active. Best entropy yet.

## Current State (Feb 21)

DINOv2 + 5 SA iters: entropy 0.170 at epoch 70 (early stop). Sub-0.2 target achieved. 7/7 slots active, max_cov=63.8%. Best result yet. Ready for next phase.
