#!/usr/bin/env python3
"""Compare standalone SA vs AEv5 to find remaining differences."""
import torch
import torch.nn as nn
import sys
sys.path.insert(0, '/Users/tomek/AI')

# Import both implementations
from test_reference_sa import SlotAttentionAE as StandaloneAE
from world_model import SlotAttentionAEv5 as AEv5

torch.manual_seed(42)

# Create both models
standalone = StandaloneAE(n_slots=7, slot_dim=64, img_size=64)
aev5 = AEv5(n_slots=7, slot_dim=64, img_size=64)

# Print param counts
s_params = sum(p.numel() for p in standalone.parameters())
a_params = sum(p.numel() for p in aev5.parameters())
print(f"Standalone params: {s_params:,}", flush=True)
print(f"AEv5 params:       {a_params:,}", flush=True)
print(f"Match: {s_params == a_params}", flush=True)

# Map weights from standalone to AEv5
print("\n--- Weight mapping ---", flush=True)
s_state = standalone.state_dict()
a_state = aev5.state_dict()

# Print all parameter names side by side
print("\nStandalone keys:", flush=True)
for k, v in s_state.items():
    print(f"  {k}: {v.shape}", flush=True)

print("\nAEv5 keys:", flush=True)
for k, v in a_state.items():
    print(f"  {k}: {v.shape}", flush=True)

# Build mapping
mapping = {
    # Encoder CNN
    'encoder.0.weight': 'encoder_cnn.0.weight',
    'encoder.0.bias': 'encoder_cnn.0.bias',
    'encoder.2.weight': 'encoder_cnn.2.weight',
    'encoder.2.bias': 'encoder_cnn.2.bias',
    'encoder.4.weight': 'encoder_cnn.4.weight',
    'encoder.4.bias': 'encoder_cnn.4.bias',
    'encoder.6.weight': 'encoder_cnn.6.weight',
    'encoder.6.bias': 'encoder_cnn.6.bias',
    # Pos embed
    'encoder_pos.grid': 'encoder_pos.pos',
    'encoder_pos.proj.weight': 'encoder_pos.proj.weight',
    'encoder_pos.proj.bias': 'encoder_pos.proj.bias',
    # Encoder MLP
    'encoder_mlp.0.weight': 'encoder_mlp.0.weight',
    'encoder_mlp.0.bias': 'encoder_mlp.0.bias',
    'encoder_mlp.2.weight': 'encoder_mlp.2.weight',
    'encoder_mlp.2.bias': 'encoder_mlp.2.bias',
    # Encoder norm
    'encoder_norm.weight': 'encoder_norm.weight',
    'encoder_norm.bias': 'encoder_norm.bias',
    # SA
    'slot_attention.slots_mu': 'slot_attention.slots_mu',
    'slot_attention.slots_log_sigma': 'slot_attention.slots_log_sigma',
    'slot_attention.project_q.weight': 'slot_attention.project_q.weight',
    'slot_attention.project_k.weight': 'slot_attention.project_k.weight',
    'slot_attention.project_v.weight': 'slot_attention.project_v.weight',
    'slot_attention.gru.weight_ih': 'slot_attention.gru.weight_ih',
    'slot_attention.gru.weight_hh': 'slot_attention.gru.weight_hh',
    'slot_attention.gru.bias_ih': 'slot_attention.gru.bias_ih',
    'slot_attention.gru.bias_hh': 'slot_attention.gru.bias_hh',
    'slot_attention.mlp.0.weight': 'slot_attention.mlp.0.weight',
    'slot_attention.mlp.0.bias': 'slot_attention.mlp.0.bias',
    'slot_attention.mlp.2.weight': 'slot_attention.mlp.2.weight',
    'slot_attention.mlp.2.bias': 'slot_attention.mlp.2.bias',
    'slot_attention.norm_inputs.weight': 'slot_attention.norm_inputs.weight',
    'slot_attention.norm_inputs.bias': 'slot_attention.norm_inputs.bias',
    'slot_attention.norm_slots.weight': 'slot_attention.norm_slots.weight',
    'slot_attention.norm_slots.bias': 'slot_attention.norm_slots.bias',
    'slot_attention.norm_mlp.weight': 'slot_attention.norm_mlp.weight',
    'slot_attention.norm_mlp.bias': 'slot_attention.norm_mlp.bias',
    # Decoder
    'decoder.0.weight': 'decoder.0.weight',
    'decoder.0.bias': 'decoder.0.bias',
    'decoder.2.weight': 'decoder.2.weight',
    'decoder.2.bias': 'decoder.2.bias',
    'decoder.4.weight': 'decoder.4.weight',
    'decoder.4.bias': 'decoder.4.bias',
    'decoder.6.weight': 'decoder.6.weight',
    'decoder.6.bias': 'decoder.6.bias',
    # Decoder grid (buffer)
    'dec_grid': 'grid',
}

# Copy weights
print("\n--- Copying weights ---", flush=True)
new_a_state = {}
for s_key, a_key in mapping.items():
    if s_key in s_state and a_key in a_state:
        if s_state[s_key].shape == a_state[a_key].shape:
            new_a_state[a_key] = s_state[s_key].clone()
            print(f"  ✓ {s_key} -> {a_key} ({s_state[s_key].shape})", flush=True)
        else:
            print(f"  ✗ SHAPE MISMATCH: {s_key} {s_state[s_key].shape} vs {a_key} {a_state[a_key].shape}", flush=True)
    else:
        if s_key not in s_state:
            print(f"  ✗ Missing in standalone: {s_key}", flush=True)
        if a_key not in a_state:
            print(f"  ✗ Missing in AEv5: {a_key}", flush=True)

# Check for unmapped keys
s_mapped = set(mapping.keys())
a_mapped = set(mapping.values())
s_unmapped = set(s_state.keys()) - s_mapped
a_unmapped = set(a_state.keys()) - a_mapped
if s_unmapped:
    print(f"\n  Unmapped standalone keys: {s_unmapped}", flush=True)
if a_unmapped:
    print(f"\n  Unmapped AEv5 keys: {a_unmapped}", flush=True)

# Load mapped weights
aev5.load_state_dict(new_a_state, strict=True)
print("\n✓ All weights copied successfully", flush=True)

# Test with same input and same random seed
print("\n--- Forward pass comparison ---", flush=True)
torch.manual_seed(123)
x = torch.randn(2, 3, 64, 64)

# Need same random noise in SA, so set seed before each forward
torch.manual_seed(999)
s_loss, s_recon, s_slots, s_alpha = standalone(x)

torch.manual_seed(999)
a_loss, a_recon_loss, a_entropy, a_recon, a_slots, a_alpha = aev5(x)

print(f"Standalone loss: {s_loss.item():.6f}", flush=True)
print(f"AEv5 loss:       {a_loss.item():.6f}", flush=True)
print(f"Loss match: {torch.allclose(s_loss, a_loss, atol=1e-5)}", flush=True)
print(f"Recon match: {torch.allclose(s_recon, a_recon, atol=1e-5)}", flush=True)
print(f"Slots match: {torch.allclose(s_slots, a_slots, atol=1e-5)}", flush=True)
print(f"Alpha match: {torch.allclose(s_alpha, a_alpha, atol=1e-5)}", flush=True)

if not torch.allclose(s_slots, a_slots, atol=1e-5):
    print(f"\nSlot diff max: {(s_slots - a_slots).abs().max().item():.8f}", flush=True)
    print(f"Slot diff mean: {(s_slots - a_slots).abs().mean().item():.8f}", flush=True)

    # Check intermediate: encoder output
    print("\n--- Checking encoder outputs ---", flush=True)
    torch.manual_seed(123)
    x = torch.randn(2, 3, 64, 64)

    # Standalone encode steps
    sx = standalone.encoder(x)
    ax = aev5.encoder_cnn(x)
    print(f"CNN output match: {torch.allclose(sx, ax, atol=1e-5)}", flush=True)

    B = sx.shape[0]
    sx = sx.permute(0, 2, 3, 1)
    ax = ax.permute(0, 2, 3, 1)
    sx = standalone.encoder_pos(sx)
    ax = aev5.encoder_pos(ax)
    print(f"After pos embed match: {torch.allclose(sx, ax, atol=1e-5)}", flush=True)

    sx = sx.reshape(B, 64*64, 64)
    ax = ax.reshape(B, 64*64, 64)
    sx = standalone.encoder_mlp(sx)
    ax = aev5.encoder_mlp(ax)
    print(f"After MLP match: {torch.allclose(sx, ax, atol=1e-5)}", flush=True)

    sx = standalone.encoder_norm(sx)
    ax = aev5.encoder_norm(ax)
    print(f"After norm match: {torch.allclose(sx, ax, atol=1e-5)}", flush=True)

print("\nDone.", flush=True)
