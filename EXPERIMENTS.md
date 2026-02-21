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

### Phase 27b Test 1: Complex CLEVR stress test
**Date:** Feb 21

- **Change:** Harder dataset — overlapping objects, varied radius 4-15, 2-6 objects/scene, random background colors. Locked config: DINOv2 + 5 SA iters + per-slot init.
- **Config:** 200 epochs, batch=32, 7 slots, constant LR 4e-4, warmup 30, 5 SA iters
- **Dataset:** `generate_clevr_images_complex()` — overlapping circles, random pastel backgrounds, max_objects=6
- **Result:**
  ```
  Ep   1: recon=6.5058 entropy=1.000 active=6/7
  Ep  10: recon=2.5472 entropy=0.963 active=4/7
  Ep  25: recon=1.9387 entropy=0.949 active=7/7
  Ep  40: recon=1.5262 entropy=0.564 active=7/7
  Ep  50: recon=1.3786 entropy=0.431 active=7/7  ← below 0.5 target
  Ep  70: recon=1.2530 entropy=0.398 active=7/7
  Ep  90: recon=1.1750 entropy=0.374 active=7/7  ← best
  Ep 100: recon=1.1211 entropy=0.378 active=7/7
  Ep 150: recon=0.9428 entropy=0.410 active=7/7
  Ep 200: recon=0.8523 entropy=0.426 active=7/7 max_cov=24.6%
  ```
  Eval: entropy=0.429, 7/7 active, max_cov=24.6%
- **Verdict:** SUCCESS — entropy < 0.5 target met at epoch 50 (0.431). Best entropy 0.374 at epoch 90, slight regression to 0.429 by 200. Very even slot coverage (22-25% max_cov, nearly ideal 14% for 7 slots). Harder dataset converges slower (ep 50 vs ep 25 for simple CLEVR) but slots still differentiate reliably. Mild entropy regression after ep 90 suggests cosine LR schedule might help, or just train 100 epochs.

### Phase 27b Test 2: Video frame consistency (inference only)
**Date:** Feb 21

- **Change:** Inference-only test. Load Test 1 model (complex CLEVR trained). Generate 10 bouncing-circle sequences (20 frames, 2-4 objects, linear motion + wall bounce). Encode every frame, track slot-object assignments via Hungarian matching on attention masks.
- **Config:** Frozen model from Test 1. 10 sequences × 20 frames = 200 frames total.
- **Result:**
  ```
  Seq 0: 100.0% (3 obj)   Seq 5:  75.9% (2 obj)
  Seq 1:  86.5% (2 obj)   Seq 6:  78.9% (3 obj)
  Seq 2:  98.5% (2 obj)   Seq 7:  91.0% (2 obj)
  Seq 3: 100.0% (2 obj)   Seq 8:  75.9% (3 obj)
  Seq 4:  95.5% (4 obj)   Seq 9: 100.0% (4 obj)

  Average consistency: 90.2%  (target: 80%+)
  ```
- **Verdict:** SUCCESS — 90.2% average consistency, well above 80% target. Slots track objects across frames despite no temporal training. 4/10 sequences hit 100%. Lowest was 75.9% (2 sequences). Model trained on static images generalizes to temporal tracking — DINOv2 features provide enough spatial coherence for consistent slot binding.

### Phase 27b Test 3: Real images (Flowers102)
**Date:** Feb 21

- **Change:** Train on real photos (Oxford Flowers102, 2040 images). VOC server down, used Flowers102 as substitute — diverse real photos with varied backgrounds, foreground objects, lighting. Locked config unchanged.
- **Config:** 200 epochs, batch=16, 7 slots, constant LR 4e-4, warmup 30, 5 SA iters, 224×224 input
- **Dataset:** Flowers102 train+val splits (1020+1020 = 2040 images), 80/20 train/val
- **Result:**
  ```
  Ep   1: recon=6.0687 entropy=1.000 active=7/7 max_cov=43.6%
  Ep  20: recon=4.0562 entropy=0.960 active=6/7 max_cov=36.1%
  Ep  40: recon=3.5021 entropy=0.515 active=7/7 max_cov=23.8%
  Ep  60: recon=3.2172 entropy=0.354 active=7/7 max_cov=18.5%
  Ep  80: recon=3.0791 entropy=0.332 active=7/7 max_cov=17.3%
  Ep 100: recon=2.9829 entropy=0.326 active=7/7 max_cov=17.6%
  Ep 120: recon=2.9083 entropy=0.305 active=7/7 max_cov=17.8%
  Ep 140: recon=2.8648 entropy=0.297 active=7/7 max_cov=17.1%
  Ep 160: recon=2.8174 entropy=0.299 active=7/7 max_cov=18.4%
  Ep 180: recon=2.7667 entropy=0.291 active=7/7 max_cov=17.6%
  Ep 200: recon=2.7593 entropy=0.293 active=7/7 max_cov=17.0%
  ```
  Eval: entropy=0.313, 7/7 active, max_cov=18.2%
- **Verdict:** SUCCESS — entropy 0.293 on real photos, well below any reasonable threshold. Nearly ideal even coverage (17-18% max_cov vs 14.3% ideal for 7 slots). All 7 slots active. Recon loss much higher than CLEVR (2.76 vs 0.85) as expected — real images have far more complex features — but slot specialization is strong. DINOv2+SA locked config works on real-world images.

### Phase 28: JEPA Dynamics in Slot Space
**Date:** Feb 21

- **Goal:** Train predictor: slots(t) → slots(t+1) in frozen slot representation space.
- **Architecture:** SlotPredictor MLP — flatten all 7 slots [B, 448] → Linear(448,256)+LN+ReLU → Linear(256,256)+ReLU → Linear(256,448) → reshape [B, 7, 64]. 296K params.
- **Data:** 1000 sequences × 20 frames (bouncing circles, 2-4 objects). Frozen SlotAttentionDINO encodes all frames → cached [1000, 20, 7, 64]. 15,200 train pairs, 3,800 val pairs.
- **Config:** 200 epochs, batch=64, Adam lr=1e-3, cosine schedule to 1e-5
- **Baseline:** Copy-previous MSE = 0.02265 (consecutive frames are very similar)
- **Training result:**
  ```
  Ep   1: train=0.4078 val=0.0830
  Ep  50: train=0.0261 val=0.0274
  Ep 100: train=0.0231 val=0.0258
  Ep 150: train=0.0209 val=0.0248
  Ep 200: train=0.0200 val=0.0247  ← best val=0.0247 at ep 190
  ```
  Slot variance: 5.54, slot mean L2 norm: 18.84
- **Autoregressive rollout (predict frames 6-10 from context 1-5):**
  ```
  Step 1 (f6):  MSE=0.0241  cosine=0.9975
  Step 2 (f7):  MSE=0.0430  cosine=0.9955
  Step 3 (f8):  MSE=0.0610  cosine=0.9936
  Step 4 (f9):  MSE=0.0854  cosine=0.9910
  Step 5 (f10): MSE=0.1146  cosine=0.9879
  ```
- **Success criteria:**
  - 1-step < 0.1×variance (0.554): **YES** (0.024 ≪ 0.554)
  - 5-step < 2× 1-step: **NO** (4.76×, error accumulates)
  - Better than copy baseline: **NO** (0.024 vs 0.023, 6% worse)
- **Verdict:** PARTIAL — predictor learns meaningful dynamics (0.024 MSE, cosine 0.998), but doesn't beat copy-previous baseline because consecutive frames are nearly identical (objects move ~1.5 px/frame). The predictor is essentially correct but the task is too easy for copy to be a strong baseline. Error accumulates during autoregressive rollout (4.76× at step 5). **Next steps:** (1) increase object velocities to widen copy baseline gap, (2) try predicting multiple steps ahead (Δ=3 or Δ=5), (3) add residual connection (predict delta instead of absolute).

### Phase 28b: Faster Physics + Δ=3 Prediction
**Date:** Feb 21

- **Goal:** Beat copy-previous baseline by making prediction harder: faster objects + skip-frame prediction.
- **Changes:** Object velocity ±1.5 → ±5.0 (3.3× faster). Prediction delta Δ=1 → Δ=3 (predict 3 frames ahead). Same predictor architecture (296K params).
- **Data:** 1000 sequences × 20 frames, 13,600 train pairs (Δ=3), 3,400 val pairs.
- **Baselines:** Copy MSE Δ=3: 0.279 (was 0.023 at Δ=1 with slow motion — 12× harder). Copy MSE Δ=1: 0.076.
- **Training result:**
  ```
  Ep   1: train=0.5224 val=0.2363 vs_copy=15.4% better
  Ep  10: train=0.2066 val=0.2108 vs_copy=24.6% better
  Ep  20: train=0.2000 val=0.2099 vs_copy=24.9% better  ← best
  Ep  50: train=0.1821 val=0.2158 vs_copy=22.8% better
  Early stop at epoch 50 (no val improvement for 30 epochs)
  ```
- **Autoregressive rollout (Δ=3 per step, from frame 2):**
  ```
  Step 1 (f5):  MSE=0.2222  cosine=0.9764  (copy: 0.279, 20.5% better)
  Step 2 (f8):  MSE=0.3743  cosine=0.9603
  Step 3 (f11): MSE=0.4573  cosine=0.9506
  Step 4 (f14): MSE=0.5107  cosine=0.9444
  Step 5 (f17): MSE=0.5409  cosine=0.9416
  ```
- **Success criteria:**
  - 1-step < 0.1×variance (0.555): **YES** (0.222 < 0.555)
  - 5-step < 2× 1-step: **NO** (2.43×)
  - Better than copy: **YES** (20.5% better!)
- **Verdict:** SUCCESS — predictor beats copy baseline by 20.5% on 1-step and maintains cosine >0.94 through 5-step rollout. Early stopping at epoch 50 suggests limited model capacity or need for more data. Error accumulation (2.43×) is better than Phase 28 (4.76×). The slot JEPA framework works: learned dynamics in slot space outperform naive copy when objects move fast enough.

### Phase 28 vs 28b Comparison

| Metric | Phase 28 (slow, Δ=1) | Phase 28b (fast, Δ=3) |
|--------|----------------------|----------------------|
| Velocity | ±1.5 px/frame | ±5.0 px/frame |
| Copy baseline | 0.023 | 0.279 |
| Predictor MSE | 0.025 | 0.210 |
| vs copy | -6% (worse) | **+25% (better)** |
| 1-step cosine | 0.998 | 0.976 |
| 5-step rollout ratio | 4.76× | 2.43× |

### Phase 29: Mass Inference from Dynamics
**Date:** Feb 21

- **Goal:** Infer object mass (light=1.0 vs heavy=3.0) from slot trajectories during elastic collisions. Target: >80% classification accuracy.
- **Architecture:** MassClassifier MLP — temporal features (mean + variance + mean_abs_delta of slot trajectory over 20 frames) → [3×64=192] → Linear(192,64)+ReLU → Linear(64,64)+ReLU → Linear(64,1). 16.6K params.
- **Physics:** Elastic collisions with mass: `v1_new = ((m1-m2)*v1 + 2*m2*v2) / (m1+m2)`. 2-4 circles per scene, each randomly light (1.0) or heavy (3.0).
- **Data:** 1000 sequences × 20 frames. Frozen SlotAttentionDINO encodes → [1000, 20, 7, 64]. Slot-object matching via centroid distance (avg match dist: 0.074). 2490 samples (1264 light, 1226 heavy). Train: 1992, Val: 498.
- **v1 attempt (mean-only features, temporal-variance matching):** 55.3% val accuracy — barely above chance. Poor slot-object matching and mean-only pooling discarded temporal info.
- **v2 attempt (temporal features, centroid matching):**
  ```
  Ep   1: train_acc=49.6% val_acc=52.6%
  Ep  20: train_acc=57.2% val_acc=53.2%  ← best val
  Ep  40: train_acc=63.0% val_acc=49.8%
  Ep 100: train_acc=77.4% val_acc=49.8%
  Ep 200: train_acc=85.7% val_acc=49.0%
  ```
  Best val: 53.2% at epoch 20.
- **Diagnosis:** Extreme overfitting — train 85.7% but val ~50%. The classifier memorizes training sequences but doesn't generalize. Slot vectors encode object appearance (position, shape, color) but mass information lives in collision dynamics (how velocities change on impact). The temporal summary features (mean, var, delta) over slot space don't isolate collision events or velocity changes.
- **Verdict:** FAIL — 53.2% val accuracy vs 80% target. Slot representations don't encode mass-distinguishing dynamics. Possible fixes: (1) explicitly compute velocity from slot centroids rather than using raw slot features, (2) detect collision frames and compare pre/post velocity ratios, (3) use a sequence model (LSTM/transformer) instead of pooled features, (4) train the slot attention end-to-end with mass classification loss.

### Phase 29b: Centroid Trajectory Features
**Date:** Feb 21

- **Goal:** Fix Phase 29 by using 2D centroid positions instead of 64-dim slot vectors. Mass info lives in how objects move, not what they look like.
- **Change:** Extract centroid positions from slot attention alpha masks at all 20 frames (not just frame 0). Compute velocity [T-1, 2] and acceleration [T-2, 2] from position deltas. Flatten all → 114-dim input to MLP classifier. Same physics, encoding, and matching as Phase 29.
- **Architecture:** MassClassifier MLP — positions(40) + velocities(38) + accelerations(36) = 114 dims → Linear(114,64)+ReLU → Linear(64,64)+ReLU → Linear(64,1). 11.6K params.
- **Data:** Same as Phase 29: 1000 sequences, 2490 samples, centroid cache [1000, 20, 7, 2]. Avg match distance 0.074.
- **Result:**
  ```
  Ep   1: train_acc=51.2% val_acc=48.6%
  Ep  40: train_acc=65.6% val_acc=51.2%  ← best val
  Ep 100: train_acc=81.2% val_acc=48.8%
  Ep 200: train_acc=86.7% val_acc=50.4%
  ```
  Best val: 51.2% at epoch 40.
- **Diagnosis:** Same overfitting pattern as Phase 29 (train 87%, val ~50%). Centroid positions from 16×16 DINOv2 patch grid alpha masks are noisy approximations — the spatial resolution is too coarse to capture subtle velocity changes during collisions. The mass signal (how much velocity changes on collision) is buried in centroid extraction noise. The classifier memorizes noisy training trajectories but can't generalize.
- **Verdict:** FAIL — 51.2% val accuracy vs 80% target. Neither slot vectors nor centroid trajectories extracted from frozen slot attention carry enough mass information. Root cause: the 16×16 patch grid gives only ~4px resolution for centroid positions on 64×64 images, while mass-dependent velocity differences during collisions are sub-pixel at this scale.
- **Possible fixes:** (1) Use ground-truth positions to verify the task is solvable at all, (2) higher-resolution encoder (not DINOv2 14×14 patches), (3) end-to-end training with mass classification loss, (4) longer sequences with more collisions, (5) larger mass ratio (1:10 instead of 1:3).

### Phase 29c: Ground-Truth Position Diagnostic
**Date:** Feb 21

- **Goal:** Diagnostic — classify mass from ground-truth (x,y) positions. No DINOv2, no slot attention. Tests whether mass signal exists in trajectory data at all.
- **Architecture:** Same MassClassifier as 29b — positions(40) + velocities(38) + accelerations(36) = 114 dims → MLP → logit. 11.6K params.
- **Collision statistics:**
  - Sequences with ≥1 collision: 767/1000 (76.7%)
  - Avg collisions per sequence: 1.7
  - Distribution: 0 collisions=233, 1=278, 2=222, 3+=267
- **Result:**
  ```
  Ep   1: train_acc=50.7% val_acc=49.8%
  Ep  20: train_acc=64.3% val_acc=61.2%
  Ep  80: train_acc=75.8% val_acc=64.9%
  Ep 140: train_acc=80.8% val_acc=66.7%
  Ep 180: train_acc=81.7% val_acc=67.7%  ← best val
  Ep 200: train_acc=82.0% val_acc=66.5%
  ```
  Best val: 67.7% at epoch 180. Light: 63.2%, Heavy: 72.2%.
- **Diagnosis:** Signal exists (67.7% >> 50%) but is weak. Even with perfect ground-truth positions, the MLP only reaches 67.7%. Root causes: (1) only 1.7 collisions per sequence on avg — many objects never collide and carry zero mass info, (2) 20 frames is too short for enough collision events, (3) wall bounces don't reveal mass (all masses bounce identically off walls). The overfitting gap is much smaller (82% vs 68%) than Phase 29/29b, confirming ground-truth positions are cleaner signal.
- **Verdict:** PARTIAL — confirms mass signal exists but is weak. The task itself is hard with current physics config (sparse collisions, short sequences). To reach >80%, need: more collisions (faster objects, smaller arena, more objects) or longer sequences.

### Phase 29d: Dense Collision Physics — GT Positions
**Date:** Feb 21

- **Goal:** Force mass signal with denser collisions. GT positions, no DINOv2. Target: >85% val accuracy, 5+ avg collisions.
- **Physics changes vs 29c:**
  | Parameter | 29c | 29d |
  |-----------|-----|-----|
  | Frames | 20 | 40 |
  | Objects | 2-3 | 3-4 |
  | Radius | 6-9 | 7-10 |
  | Velocity | ±4 | ±7 |
- **Collision statistics:**
  - Sequences with ≥1 collision: 1000/1000 (100%)
  - Avg collisions per sequence: 16.0 (target was 5+)
  - Distribution: 0=0, 1=0, 2=2, 3-5=34, 6+=964
- **Data:** 3480 samples (1730 light, 1750 heavy). Train: 2784, Val: 696.
- **Result:**
  ```
  Ep   1: train_acc=50.1% val_acc=50.7%
  Ep  20: train_acc=86.9% val_acc=76.4%  ← best val
  Ep  40: train_acc=94.0% val_acc=76.4%
  Ep 100: train_acc=99.6% val_acc=74.3%
  Ep 200: train_acc=99.8% val_acc=74.0%
  ```
  Best val: 76.4% at epoch 20. Light: 65.8%, Heavy: 86.6%.
- **Diagnosis:** Collision density is excellent (16 avg). Signal is stronger than 29c (76.4% vs 67.7%). But massive overfitting: train 99.8% vs val 76.4%. The MLP with 19K params memorizes the 2784 training trajectories (each a unique path through 234-dim space) but can't generalize to unseen trajectories. The flattened position/velocity/acceleration features don't give the MLP an inductive bias for isolating collision events — it sees raw coordinates, not physical interactions. Needs: (1) regularization (dropout, weight decay), (2) collision-aware features (detect collision frames, compute velocity ratio), or (3) more training data.
- **Verdict:** BELOW TARGET — 76.4% vs 85%. Collisions are dense enough but the classifier architecture can't generalize. Did not proceed to slot centroids.

### Phase 29e: Collision-Aware Features — GT Positions
**Date:** Feb 21

- **Goal:** Replace 234 raw coordinates with 5 physics-informed features. GT positions. Target: >85%.
- **Features per object** (computed from GT position trajectory):
  1. `avg_dv_coll` — average |Δvelocity| at collision frames (|Δv| > 0.02 threshold)
  2. `max_dv` — peak velocity change
  3. `avg_speed` — mean speed across all frames
  4. `speed_var` — variance of speed over time
  5. `n_coll_norm` — fraction of frames with detected collisions
- **Feature separation (light vs heavy):**
  ```
  avg_dv_coll:  light=0.130  heavy=0.080  ratio=1.63
  max_dv:       light=0.301  heavy=0.173  ratio=1.75
  avg_speed:    light=0.102  heavy=0.067  ratio=1.52
  speed_var:    light=0.0022 heavy=0.0009 ratio=2.52  ← best separator
  n_coll_norm:  light=0.572  heavy=0.460  ratio=1.24
  ```
- **Classifier:** Linear(5,32)+ReLU+Linear(32,1) — **225 params** (vs 19K in 29d).
- **Result:**
  ```
  Ep   1: train_acc=46.6% val_acc=53.0%
  Ep  20: train_acc=81.6% val_acc=79.5%
  Ep  60: train_acc=83.3% val_acc=81.5%
  Ep 140: train_acc=83.7% val_acc=81.9%  ← best val
  Ep 200: train_acc=83.5% val_acc=81.8%
  ```
  Best val: 81.9% at epoch 140. Light: 76.3%, Heavy: 87.7%.
- **Diagnosis:** No overfitting (train 83.5% ≈ val 81.9%). Physics features generalize perfectly — 225 params can't memorize. The 5 features all separate mass (ratios 1.24–2.52×), with `speed_var` being the strongest discriminator. Below 85% target because feature distributions overlap (mass 1:3 ratio isn't extreme enough for clean separation). All features confirm heavy objects deflect less and move slower after collisions, consistent with conservation of momentum.
- **Verdict:** PARTIAL SUCCESS — 81.9% with zero overfitting proves the approach works. Collision-aware features are the right abstraction. Gap to 85% could be closed with: larger mass ratio, more data, or slightly more features.

### Phase 29f: Pairwise Collision Features — Newton's Third Law
**Date:** Feb 21

- **Goal:** Add pairwise Δv ratio feature. By Newton's third law, |Δv_A|/|Δv_B| = m_B/m_A. Light objects deflect more, heavy deflect less. GT positions. Target: >85%.
- **New feature:** `avg_dv_ratio` — mean(|my_Δv| / |partner_Δv|) across all collisions an object participates in. Logged during physics sim by recording both objects' |Δv| at each collision event.
- **All 6 features (light vs heavy):**
  ```
  avg_dv_ratio:  light=2.2154  heavy=0.5674  ratio=3.90  ← KEY
  avg_dv_coll:   light=0.1298  heavy=0.0798  ratio=1.63
  max_dv:        light=0.3013  heavy=0.1726  ratio=1.75
  avg_speed:     light=0.1024  heavy=0.0671  ratio=1.52
  speed_var:     light=0.0022  heavy=0.0009  ratio=2.52
  n_coll_norm:   light=0.5721  heavy=0.4604  ratio=1.24
  ```
- **Classifier:** Linear(6,32)+ReLU+Linear(32,1) — **257 params**.
- **Result:**
  ```
  Ep   1: train_acc=49.4% val_acc=48.9%
  Ep  20: train_acc=99.2% val_acc=98.1%
  Ep 120: train_acc=99.4% val_acc=98.6%  ← best val
  Ep 200: train_acc=99.4% val_acc=98.6%
  ```
  Best val: **98.6%** at epoch 120. Light: 97.1%, Heavy: 100.0%.
- **Diagnosis:** The `dv_ratio` feature achieves 3.9× separation between light (2.22) and heavy (0.57) — near-perfect discrimination. This is Newton's third law directly encoded: momentum conservation means |Δv| is inversely proportional to mass. With 257 params and 6 physics features, no overfitting possible. The remaining 1.4% errors are likely objects with very few collisions where the ratio estimate is noisy.
- **Verdict:** SUCCESS — 98.6% val accuracy with 257 params. Mass inference from dynamics is solved with the right features. Next: bridge to slot centroids (can we extract dv_ratio from visual observations?).

### Phase 30: Emergent Communication about Mass
**Date:** Feb 21

- **Goal:** Two agents discover mass from raw trajectories and communicate via discrete messages. No hand-coded features. Sender: transformer on raw [3, 40, 2] GT positions → Gumbel-softmax message (vocab=8). Receiver: message → predict which object is heavy (3-way). Target: >80%.
- **Architecture:** MassSender (26K params): Linear(2,32) + sinusoidal PE + 2-layer TransformerEncoder(d=32, nhead=2, ff=64) per object → cross-object attention (1-layer) → Linear(32,8) → Gumbel-softmax. MassReceiver (387 params): Linear(8,32)+ReLU+Linear(32,3).
- **Data:** 2000 sequences, 3 objects (exactly 1 heavy), 40 frames, dense physics (v=±7, r=7-10). 10.5 avg collisions/seq. Train: 1600, Val: 400.
- **Training:** Adam lr=1e-3, cosine to 1e-5, 300 epochs. Gumbel τ: 2.0→0.5.
- **Result:**
  ```
  Ep   1: train=34.9% val=33.2% msgs=1/8
  Ep 100: train=33.8% val=33.0% msgs=1/8
  Ep 200: train=33.0% val=33.0% msgs=1/8
  Ep 300: train=34.3% val=33.0% msgs=1/8
  ```
  Best val: 33.2% (= random chance). Sender collapsed to single message (token 6) for ALL inputs.
- **Diagnosis:** Classic mode collapse in emergent communication. The sender never learned to differentiate inputs — Gumbel-softmax `hard=True` through a 26K-param transformer provides too noisy a gradient signal. The loss stayed at log(3)≈1.099 throughout, indicating zero learning. The receiver learned to always predict the majority class for the single message.
- **Verdict:** FAIL — mode collapse. Possible fixes: (1) add message entropy bonus to loss to prevent single-message collapse, (2) start with continuous communication then discretize, (3) simpler sender (MLP) to verify learnability, (4) warm up sender with supervised pre-training on mass labels before adding discrete bottleneck.

### Phase 30b: Continuous Warmup → Discrete Communication (Feb 21)
- **Goal:** Fix Phase 30 mode collapse. Phase A (epochs 1-150): continuous messages (raw 8-dim logits, no Gumbel-softmax). Phase B (epochs 151-300): discretize with Gumbel-softmax (τ=1.0→0.3). Auto-retry with MLP sender if Phase A val < 50%.
- **Architecture:** Same MassSender (26K params) + MassReceiver (387 params). Fallback: MLPSender (40K params, flatten 240→128→64→8).
- **Training:** Phase A: Adam lr=3e-4, continuous messages. Phase B: Adam lr=1e-4 (reset), Gumbel-softmax τ annealing. Grad clip=1.0.
- **Result — Transformer sender:**
  ```
  Phase A (continuous): train 80.1%, val 35.7% at epoch 150
  → FAILED Phase A (<50%), auto-retry triggered
  ```
  Massive overfitting. Messages not informative — all labels have identical token distributions (token 5 dominant for all). The 8-dim continuous vector encodes sequence identity, not mass features.
- **Result — MLP sender (auto-retry):**
  ```
  Phase A (continuous): train 100%, val 41.0% at epoch 20 (peak)
  → FAILED Phase A (<50%), both attempts failed
  ```
  Even worse overfitting. MLP memorized training data by epoch 40 but no generalization. Same token distribution across all labels.
- **Best overall:** MLP at epoch 20, val=41.0% (barely above 33% chance).
- **Diagnosis:** The fundamental issue is NOT mode collapse — it's that the bottleneck is too narrow for the sender to communicate mass-relevant features, while being wide enough (8 continuous dims) for memorization. The sender learns to encode sequence-specific features that don't generalize, rather than extracting collision dynamics. The receiver cannot decode mass from these arbitrary encodings.
- **Key insight:** Emergent communication requires the sender to independently discover that collision velocity ratios matter — which took us 6 phases of careful feature engineering (Phase 29→29f). Expecting a neural network to discover Newton's third law AND encode it in 8 dims from 240 raw coordinates is too much. The sender needs either (a) physics-informed input features, or (b) a much stronger inductive bias toward collision detection.
- **Verdict:** FAIL — overfitting without generalization (val 41%, target >60%).

### Phase 30c: Staged Communication — Physics Features → Discrete Message (Feb 21)
- **Goal:** Separate perception from communication. Pre-compute 6 physics features per object (same as 29f), then learn emergent language for "which object is heavy." Input: [3, 6] = 18 numbers instead of 240 raw coordinates. Target: >75%.
- **Architecture:** FeatureSender (872 params): Linear(18,32)+ReLU+Linear(32,8)+Gumbel-softmax. MassReceiver (387 params): Linear(8,32)+ReLU+Linear(32,3). Total: 1259 params.
- **Training:** 300 epochs (early stopped at 20), Adam lr=3e-4, τ: 2.0→0.3 cosine.
- **Feature separation:** avg_dv_ratio light=1.93 vs heavy=0.33 (5.8× ratio). All 6 features separate light from heavy.
- **Result:**
  ```
  Ep   1: train=31.8% val=34.7% tokens=5
  Ep  10: train=41.4% val=65.2% tokens=5
  Ep  20: train=69.6% val=98.5% tokens=3  ← EARLY STOP
  ```
  **98.5% val accuracy in 20 epochs.** Only 6 errors in 400 val samples.
- **Emergent language:** 3 tokens used, one per heavy-object index:
  ```
  Token 7 → "object 0 is heavy" (n=150, purity=96%)
  Token 5 → "object 1 is heavy" (n=115, purity=100%)
  Token 1 → "object 2 is heavy" (n=135, purity=100%)
  ```
  The sender discovered that avg_dv_ratio separates heavy from light, identified WHICH object has the lowest ratio, and encoded the answer as a discrete token. The receiver learned the token→label mapping. Nobody told them which token means what — the language emerged end-to-end through Gumbel-softmax.
- **Verdict:** SUCCESS — 98.5% val acc (target was >75%). Emergent communication works when perception is handled. The sender only needs to compare 18 numbers and pick one of 3 objects, not discover physics from 240 raw coordinates.

### Phase 29g: Slot Centroid Collision Features (Feb 21)
- **Goal:** Bridge 29f (GT features, 98.6%) with 29b (slot centroids, 51%). Use GT collision timestamps but measure Δv from slot centroid velocities. Tests whether slot centroid tracking is precise enough for mass inference. Target: >80%.
- **Architecture:** Same 257-param classifier (Linear(6,32)+ReLU+Linear(32,1)). Frozen SlotAttentionDINO (phase27_model.pt). Greedy slot-object matching on frame 0.
- **Side-by-side feature comparison (the key result):**
  ```
  Feature          GT light  GT heavy  GT sep   Slot light  Slot heavy  Slot sep
  avg_dv_ratio       2.22      0.57     3.90       2.74        2.70      1.01
  avg_dv_coll        0.13      0.08     1.63       0.09        0.08      1.03
  max_dv             0.30      0.17     1.75       0.26        0.25      1.03
  avg_speed          0.10      0.07     1.52       0.06        0.06      1.05
  speed_var          0.002     0.001    2.52       0.003       0.002     1.07
  n_coll_norm        0.57      0.46     1.24       0.79        0.77      1.02
  ```
  **Slot centroid features have ZERO separation.** GT avg_dv_ratio separates 3.9×, slot version: 1.01×. The 16×16 patch grid (~4px resolution) adds so much noise to centroid positions that velocity differences during collisions are completely buried. All 6 features collapse to separation ≈1.0.
- **Result:**
  ```
  GT features:    99.1% val (epoch 80)
  Slot centroids: 54.9% val (epoch 80) — barely above chance
  Gap: 44.3pp
  ```
- **Match distance:** avg 0.136 in [0,1] space = ~8.7 pixels on 64×64. This is the slot centroid noise floor — comparable to object radii (7-10px). Velocity changes during collisions are ~2-5 pixels/frame, well below the centroid noise.
- **Diagnosis:** The slot attention model tracks objects adequately for visual binding (90% consistency) but the centroid positions are too noisy for physics-level measurements. The 16×16 DINOv2 patch grid gives ~4px resolution — velocity changes during collisions (Δv ≈ 0.02-0.1 in normalized space = 1-6px) are at or below the noise floor. This is a fundamental resolution limit, not a feature engineering problem.
- **Verdict:** FAIL — 54.9% val (target >80%). Slot centroids are NOT accurate enough for pairwise collision features. The vision→physics pipeline needs either: (a) higher resolution encoder, (b) end-to-end training with physics loss, or (c) larger visual effects of mass (bigger objects, higher velocities, more extreme mass ratios).

### Phase 29h: Prediction Error as Mass Signal — JEPA Surprise (Feb 21)
- **Goal:** Use frozen SlotPredictor (phase28, Δ=1, trained on v=±1.5) prediction errors as mass signal. Heavy objects deflect less in collisions → smaller prediction error spikes. Operates in 64-dim slot space, bypassing centroid noise. Target: >80%.
- **Architecture:** Frozen SlotAttentionDINO + frozen SlotPredictor. 4 features per object from prediction error time series: avg_error, max_error, error_var, spike_frac. Classifier: Linear(4,32)+ReLU+Linear(32,1), 193 params.
- **Feature separation (light vs heavy):**
  ```
  avg_error:   light=0.2239  heavy=0.2166  ratio=1.03
  max_error:   light=0.5861  heavy=0.5720  ratio=1.02
  error_var:   light=0.0222  heavy=0.0211  ratio=1.05
  spike_frac:  light=0.1334  heavy=0.1223  ratio=1.09
  ```
  **All features ≈ 1.0× separation.** No mass signal in prediction error.
- **Result:** Best val 50.6% (= chance). Classifier learns nothing.
- **Diagnosis:** Prediction error is dominated by general slot representation noise (mean=0.17, std=0.21), not collision-specific physics. The predictor trained on v=±1.5 has high baseline error on v=±7 data — ALL frames are surprising, not just collision frames. The domain mismatch amplifies noise uniformly rather than selectively amplifying collision events. Additionally, the joint predictor (all 7 slots → all 7 slots) distributes prediction error across all slots, diluting per-object collision signals. Slot vector changes during collisions are not distinguishable from general frame-to-frame slot variation.
- **Verdict:** FAIL — 50.6% val (chance). Prediction error does not encode mass. The "surprise" hypothesis fails because slot representations are too noisy and collision-specific changes are not separable from background variation.

### Phase 29i: Slot Delta LSTM — Learned Collision Signatures (Feb 21)
- **Goal:** Feed full 64-dim slot change vectors (Δslot = slot[t+1] - slot[t]) into LSTM to learn collision signatures across multiple dimensions. Avoids collapsing to 2D centroids (29g) or scalar error (29h). Target: >65%.
- **Architecture:** LSTM(input=64, hidden=32, 1 layer) over [39, 64] delta sequence → final hidden [32] → Linear(32,1). 12.6K params.
- **Slot delta stats:** Avg delta norm — light: 2.97, heavy: 2.87, ratio: **1.03×**. Same negligible separation as 29h.
- **Result:**
  ```
  Ep   1: train=48.7% val=47.8%
  Ep  30: train=81.9% val=51.0%
  Ep  50: train=90.6% val=52.3%  ← best val
  Ep 100: train=99.5% val=48.9%
  Ep 200: train=100%  val=48.3%
  ```
  Complete overfitting. LSTM memorized 2784 training sequences perfectly but extracted zero generalizable mass signal. Val accuracy never exceeded 52.3%.
- **Diagnosis:** The fundamental problem is confirmed across three approaches: frozen DINOv2+SA slot representations do NOT encode mass-related dynamics differently for heavy vs light objects. The avg delta norm ratio (1.03×) matches 29h's prediction error ratio (1.03×) and 29g's feature separation (~1.01×). The slot vectors encode appearance and spatial position — they change when objects move — but the *magnitude* of change does not reflect mass-dependent collision physics. DINOv2 features are optimized for visual similarity, not physical dynamics. Even an LSTM with 12.6K params and access to all 64 dimensions cannot find a signal that isn't there.
- **Verdict:** FAIL — 52.3% val (chance). Slot delta sequences do not contain mass information extractable by any classifier.

### Phase 29j: Communication-Driven Physics Perception (Feb 21)
- **Goal:** End-to-end trainable CNN → LSTM → Gumbel-softmax → receiver. No frozen DINOv2. The CNN learns what visual features matter for mass through communication loss. Target: >50% (chance=33%).
- **Architecture:** PhysicsCNN (RGB+coord→128 per frame, 4 conv layers stride-2) → subsample 8 of 40 frames → LSTM(128,64) → Linear(64,8) → Gumbel-softmax → CommReceiver(embed 8→32→3). Sender: 496K params, Receiver: 387 params.
- **Data:** 478 three-object sequences (382 train, 96 val). Filtered from 1000 sequences — 4-object sequences excluded for consistent 3-way classification.
- **Training:** 200 epochs, batch=16. τ=1.0 (ep 1-100), anneal 1.0→0.1 (ep 101-200). Loss = CE - 0.1*entropy. Adam lr=3e-4 cosine.
- **Result:**
  ```
  Ep   1: train=33.0% val=29.2% tokens=1 (mode collapse)
  Ep  50: train=36.4% val=37.5% tokens=4
  Ep 100: train=42.1% val=28.1% tokens=6
  Ep 150: train=89.8% val=27.1% tokens=3
  Ep 200: train=92.9% val=30.2% tokens=3
  ```
  Best val: 37.5% at epoch 10 (= chance). Mode collapse throughout — single token dominates. Training accuracy rises to 93% by memorizing 382 samples with 496K params.
- **Diagnosis:** Two compounding failures: (1) Severe data scarcity — 382 training sequences for 496K params = 1300× overparameterized. (2) Mode collapse — Gumbel-softmax with CNN gives same problem as Phase 30. The entropy bonus (0.1) wasn't enough to prevent collapse. Token distributions identical across all labels.
- **Verdict:** FAIL — 37.5% val (chance). Insufficient data for end-to-end CNN training, plus mode collapse.

### Phase 29j-v2: Staged Communication-Driven Perception (Feb 21)
- **Goal:** Fix 29j's data scarcity and mode collapse. 2000 sequences (all 3-object), smaller model (~55K params vs 496K), three training stages: see→talk→refine.
- **Architecture:** PhysicsCNNv2 (42K params, 3 conv layers 5→16→32→64, k=4, s=2), CommSenderv2 (LSTM 64→32 + message head, ~13K params), CommReceiver (387 params). Total: 55K params on 1600 train. Ratio: 29×.
- **Stage 1 (see):** AE pretraining, 50 epochs. Val recon loss: 0.012. CNN learns basic spatial features.
- **Stage 2 (talk):** Frozen CNN + train LSTM/receiver, 100 epochs, τ=1.0.
  ```
  All 100 epochs: train=35.3%, val=29.0%, loss=1.098 (= log(3))
  ```
  **Zero learning.** Loss never moved from cross-entropy of random guessing. Token distributions uniform across all labels.
- **Stage 3 (refine):** Unfreeze CNN, end-to-end, 100 epochs, τ 1.0→0.1.
  ```
  All 100 epochs: train=35.3%, val=29.0%, loss=1.098
  ```
  **Still zero learning.** Fine-tuning didn't help — gradients through Gumbel-softmax couldn't reshape CNN features.
- **Delta S3-S2:** -1.0pp (no improvement from fine-tuning).
- **Diagnosis:** The autoencoder CNN learns to reconstruct appearance (recon=0.012) but these features carry zero information about mass. The LSTM+receiver cannot extract mass signal from appearance-only features, and the Gumbel-softmax gradient pathway is too noisy to reshape CNN features toward physics-relevant representations. The fundamental issue: mass is invisible in individual frames — it only manifests through temporal dynamics (how objects move differently during collisions). A reconstruction loss doesn't incentivize learning dynamic features.
- **Verdict:** FAIL — 29.0% val (below chance). Staged training doesn't help when the base features are physics-blind.

### Phase 29k — Amplified physics signal (Feb 21)
- **Goal:** Test if stronger physics signal (extreme mass ratio) or higher resolution (224×224) helps slot centroid features. Same 29g pipeline: DINOv2+SA → centroids → pairwise collision features → 257-param classifier.
- **Config A (10:1 mass, 64×64):** mass_light=1.0, mass_heavy=10.0, canvas=64, v=±7, r=7-10. 1000 sequences.
- **Config B (3:1 mass, 224×224):** mass_light=1.0, mass_heavy=3.0, canvas=224, v=±24, r=24-35. 1000 sequences.
- **Results:**
  ```
  Experiment                       GT val%  Slot val%  dv_ratio GT sep  dv_ratio Slot sep
  29g baseline (3:1, 64x64)         96.0%    54.9%         3.90            1.01
  29k-A (10:1, 64x64)               99.7%    51.9%        17.13            1.08
  29k-B (3:1, 224x224)              99.3%    53.1%         3.85            1.01
  ```
- **Key finding:** Even with 17× GT separation (10:1 mass ratio), slot centroid dv_ratio separation is only 1.08×. The 224×224 canvas gives exactly the same 1.01× as 64×64. The problem is NOT insufficient physics signal or canvas resolution — it's that **slot attention centroids are fundamentally too noisy to track velocity changes**. The centroid extraction process (attention-weighted average over 16×16 patches) introduces ~8.7% normalized noise, completely drowning the 2-5% collision Δv signal.
- **Verdict:** FAIL — 51.9% (A) and 53.1% (B). Neither amplified mass nor higher resolution helps. The centroid noise floor is structural.

### Phase 29l — Motion energy via slot masks (Feb 21)
- **Goal:** Use slot masks to identify which pixels belong to each object, then frame differencing at full 64×64 to measure motion. Avoids centroid quantization.
- **Features (5):** avg_motion, max_motion, motion_var, n_spikes, avg_spike_ratio.
- **Results:**
  ```
  Feature              Light      Heavy   Separation
  avg_motion           0.1199     0.1131       1.06x
  max_motion           0.2360     0.2244       1.05x
  motion_var           0.0028     0.0026       1.09x
  n_spikes             0.0487     0.0484       1.01x
  avg_spike_ratio      5.4588     4.6157       1.18x
  ```
  Val accuracy: **52.7%** (epoch 120). Classifier: 225 params.
- **Diagnosis:** avg_spike_ratio shows the most separation (1.18×) — light objects do show slightly more motion at collision frames. But the slot masks are too soft/overlapping — motion energy leaks between objects. A heavy object moving fast looks the same as a light object moving fast when measured through soft masks. The masks identify object regions but don't cleanly separate overlapping motion at collision time.
- **Verdict:** FAIL — 52.7% (target >65%). Motion energy through slot masks is still too noisy.

### Phase 29m — Optical flow + hard slot masks (Feb 21)
- **Goal:** Farneback optical flow for sub-pixel velocity estimation, hard argmax masks to prevent cross-object leakage. Same 6 pairwise features as 29f. Side-by-side GT comparison.
- **Results:**
  ```
  Velocity accuracy (flow vs GT):
    Mean error:   4.813 px (GT range: 0-10 px)
    Median error: 4.085 px
    90th pctile:  9.453 px

  Feature comparison — GT vs Optical Flow:
  Feature         GT sep   Flow sep
  avg_dv_ratio     3.90     1.00    ← completely destroyed
  avg_speed        1.52     1.04
  speed_var        2.52     1.09
  ```
  GT classifier: **99.3%**. Flow classifier: **50.0%** (pure chance).
- **Diagnosis:** Farneback optical flow fails catastrophically on uniform colored circles — there's no texture/gradient for it to track within each object. The aperture problem: a solid circle moving right looks identical pixel-by-pixel to the same circle — only the edges provide signal, and at 64×64 those edges are ~2px wide. Mean velocity error is 4.8px against a 0-10px range (~50% noise), completely destroying all collision features. The dv_ratio, which had 3.9× GT separation, collapses to 1.00× with flow.
- **Verdict:** FAIL — 50.0% (chance). Optical flow is the wrong tool for textureless objects at 64×64.

### Phase 29 Diagnostics — Comprehensive gap analysis (Feb 21)
- **D1: Noise tolerance sweep** — GT positions + Gaussian noise σ:
  ```
  σ (px)   Val acc   dv_sep
    0.0     91.4%    2.55x
    0.5     91.1%    2.49x
    1.0     87.1%    2.24x
    2.0     81.8%    1.76x    ← need <2px noise for >80%
    3.0     76.9%    1.44x
    4.0     68.0%    1.34x    ← need <4px for >65%
    5.0     63.6%    1.24x
    8.0     60.1%    1.11x
   10.0     52.2%    1.07x
  ```
- **D2: Position error by method** (pixels, 64×64 space):
  ```
  Method             Mean    Median   90th    95th
  GT                 0.000   0.000    0.000   0.000
  centroid_16x16    20.402  19.303   37.012  41.158
  hard_com_64x64    21.999  20.808   39.817  44.511
  soft_com_64x64    20.100  19.062   36.285  40.319
  ```
- **D3: Velocity error** (pixels/frame):
  ```
  Method             Mean    Median   90th
  centroid_16x16     6.564   5.629   12.568
  hard_com_64x64     7.212   5.867   14.163
  soft_com_64x64     6.458   5.559   12.325
  ```
- **D4: Feature separation + classifier**:
  ```
  Method             Pos err  Vel err  dv_ratio sep  Val acc
  GT                   0.0      0.0       2.55x      91.7%
  centroid_16x16      20.4      6.6       1.01x      51.3%
  hard_com_64x64      22.0      7.2       1.02x      50.9%
  soft_com_64x64      20.1      6.5       1.02x      52.3%
  ```
- **Key findings:**
  1. **Precision target: <2px position error needed for >80% accuracy, <4px for >65%.** All SA methods have 20px mean error — 5× too high.
  2. **Hard masks don't help.** hard_com_64x64 is actually slightly *worse* than soft centroid_16x16 (22.0 vs 20.4px position error). Upsampling a 16×16 grid to 64×64 doesn't add real spatial precision.
  3. **The bottleneck is the 16×16 patch grid.** DINOv2 produces 16×16=256 patch tokens. Each patch covers 4×4 pixels. Centroid precision is fundamentally limited to ~2px at best (half a patch), but slot attention introduces additional noise through soft competition → 20px actual error.
  4. **All three SA position methods are equivalent.** The position error is dominated by slot assignment noise (which object does each slot track?), not the centroid extraction method.
- **Verdict:** The vision→physics gap is quantitatively understood. SA position error (20px) is 5-10× above the noise tolerance threshold (2-4px). No downstream feature engineering can overcome this.

### Phase 29 Position Error Over Time (Feb 21)
- **Goal:** Does SA tracking drift over time, or is error constant?
- **Method:** soft_com_64x64 positions vs GT, per timestep t=0..39.
- **Results:** Error starts at **8.6px** (t=0), rises to **~20px** by t=10, then plateaus at **~22px** for t=10-39.
- **Interpretation:** Even at t=0 (the matching frame), error is 8.6px — already above the 4px threshold for >65% accuracy. This is the baseline slot-to-object assignment error: the initial centroid match is imprecise because slot attention masks don't tightly wrap single objects. The additional drift from 8.6→22px (frames 0-10) reflects slot swaps and tracking failures as objects move. After t=10 the error saturates — it's essentially random which slot tracks which object at that point.
- **Key insight:** The 8.6px t=0 error means even perfect temporal tracking wouldn't be enough — the initial object localization through slot attention is already 2× above the 4px threshold. The problem is slot attention's spatial precision, not temporal drift.

### Phase 29n — Slot position refinement (Feb 22)
- **Goal:** SlotRefineNet (~15K params) takes 16×16 crop centered on coarse centroid + slot mask, predicts (dx, dy) offset to refine position. Target: <2px error, >80% mass acc.
- **Results:**
  ```
  Method          Pos err(px)  dv_ratio sep  Val acc
  GT                    0.00        2.55x     92.2%
  Coarse (SA)          20.10        1.02x     52.9%
  Refined (29n)        17.92        1.08x     50.9%
  ```
  RefineNet offset error: 18.33px (barely below coarse error of 20.1px).
- **Diagnosis:** When the coarse centroid is 20px off, the 16×16 crop (covering 25% of the 64×64 canvas) often doesn't contain the actual object center. The refinement net can't find what isn't in its field of view. The 10.8% position improvement (20.1→17.9px) doesn't meaningfully change the dv_ratio separation (1.02→1.08×). The fundamental issue is that slot attention assigns the wrong region to objects — refinement within that wrong region can't recover the true position.
- **Verdict:** FAIL — 50.9% val, 17.9px position error (target <2px). Local refinement can't fix global slot assignment errors.

## Current State (Feb 22)

**Validated pipeline:**
- DINOv2 + SlotAttention (5 iters, 7 slots, 64-dim): entropy 0.170 on CLEVR, 0.429 complex CLEVR, 0.293 real photos, 90.2% video consistency
- Slot JEPA predictor: beats copy baseline by 20.5% (v=±5.0, Δ=3)
- **Mass inference from dynamics: 98.6%** with pairwise collision features on GT positions (Phase 29f)
- **Emergent communication: 98.5%** with physics features → Gumbel-softmax → receiver (Phase 30c)

**Mass inference from vision (Phase 29g-n + diagnostics):**
- 29g-m: Eight approaches FAILED (54.9% → 50.0%). See individual entries above.
- 29n: position refinement (SlotRefineNet) → **50.9%** (FAIL). Local crop refinement can't fix global slot assignment errors (20→17.9px, still 5× above threshold).
- **29 Diagnostics: Root cause identified.** SA position error is 20px mean. Need <4px for >65% accuracy, <2px for >80%. The 16×16 DINOv2 patch grid fundamentally limits precision to ~2px/patch, and slot attention noise adds another ~18px on top. No position extraction method or local refinement can fix this.
- **The gap is structural:** 20px error vs 2-4px required = 5-10× too imprecise.

**Emergent communication (Phase 30 series):**
- Phase 30: mode collapse (33%). Phase 30b: overfitting (41%).
- Phase 30c: **98.5%** — separated perception from communication. 3-token emergent language.

**Next steps:** (1) Accept GT positions and build the full multi-agent communication pipeline (the interesting research is in communication, not perception), (2) If vision is needed: train a dedicated object tracker (not slot attention) on this specific task, or use higher-resolution backbone (32×32 patches).
