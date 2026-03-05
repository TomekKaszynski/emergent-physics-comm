"""
Generate all paper figures from existing experiment results.
No training — just load JSONs and plot.

Run: python3 generate_figures.py
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

# Style
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

# Colorblind-friendly palette
CB_BLUE = '#0072B2'
CB_ORANGE = '#E69F00'
CB_GREEN = '#009E73'
CB_RED = '#D55E00'
CB_PURPLE = '#CC79A7'
CB_GRAY = '#999999'
CB_CYAN = '#56B4E9'


# ══════════════════════════════════════════════════════════════════
# FIGURE 1: MI Heatmap (compositional vs holistic)
# ══════════════════════════════════════════════════════════════════

def figure1():
    print("Generating Figure 1: MI Heatmap...", flush=True)
    d = json.load(open('results/phase54f_extended.json'))
    dh = json.load(open('results/phase54g_control.json'))

    # Best compositional seed
    best = max(d['per_seed'], key=lambda x: x['pos_dis'])
    mi = np.array(best['mi_matrix'])  # (2, 2): rows=positions, cols=[elasticity, friction]
    posdis = best['pos_dis']

    # Average holistic MI (1 position, 2 properties)
    hol_mis = np.array([s['mi_matrix'] for s in dh['per_seed']])  # (20, 1, 2)
    hol_avg = hol_mis.mean(axis=0)  # (1, 2)

    fig, axes = plt.subplots(1, 2, figsize=(7, 2.5),
                              gridspec_kw={'width_ratios': [1, 0.6]})

    # Panel A: Compositional 2×2
    ax = axes[0]
    im = ax.imshow(mi, cmap='Blues', aspect='auto', vmin=0, vmax=1.4)
    for i in range(2):
        for j in range(2):
            color = 'white' if mi[i, j] > 0.7 else 'black'
            ax.text(j, i, f'{mi[i, j]:.2f}', ha='center', va='center',
                    fontsize=12, fontweight='bold', color=color)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Elasticity', 'Friction'])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Position 0', 'Position 1'])
    ax.set_title(f'Compositional (PosDis = {posdis:.2f})', fontweight='bold')
    plt.colorbar(im, ax=ax, label='Mutual Information (nats)', shrink=0.8)

    # Panel B: Holistic 1×2
    ax = axes[1]
    im2 = ax.imshow(hol_avg, cmap='Blues', aspect='auto', vmin=0, vmax=1.4)
    for j in range(2):
        color = 'white' if hol_avg[0, j] > 0.7 else 'black'
        ax.text(j, 0, f'{hol_avg[0, j]:.2f}', ha='center', va='center',
                fontsize=12, fontweight='bold', color=color)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Elasticity', 'Friction'])
    ax.set_yticks([0])
    ax.set_yticklabels(['Position 0'])
    ax.set_title('Holistic (1×25)', fontweight='bold')

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'fig1_mi_heatmap.pdf')
    plt.close(fig)
    print("  Saved fig1_mi_heatmap.pdf", flush=True)


# ══════════════════════════════════════════════════════════════════
# FIGURE 2: PosDis Distribution
# ══════════════════════════════════════════════════════════════════

def figure2():
    print("Generating Figure 2: PosDis Distribution...", flush=True)
    d = json.load(open('results/phase54f_extended.json'))

    posdis_vals = [s['pos_dis'] for s in d['per_seed']]

    fig, ax = plt.subplots(figsize=(5, 3))

    bins = np.arange(0, 1.0, 0.1)
    n, bins_out, patches = ax.hist(posdis_vals, bins=bins, edgecolor='black',
                                    linewidth=0.8, color=CB_BLUE, alpha=0.8)

    # Threshold line
    ax.axvline(x=0.4, color=CB_RED, linestyle='--', linewidth=1.5, label='Threshold')

    # Count compositional vs holistic
    n_comp = sum(1 for v in posdis_vals if v >= 0.4)
    n_hol = sum(1 for v in posdis_vals if v < 0.4)

    ax.annotate(f'{n_comp}/20 compositional',
                xy=(0.65, max(n) * 0.85), fontsize=9, color=CB_BLUE, fontweight='bold')
    ax.annotate(f'{n_hol}/20 holistic',
                xy=(0.05, max(n) * 0.85), fontsize=9, color=CB_GRAY, fontweight='bold')

    ax.set_xlabel('Positional Disentanglement (PosDis)')
    ax.set_ylabel('Number of Seeds')
    ax.set_title('PosDis Distribution Across 20 Seeds')
    ax.set_xlim(0, 0.9)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'fig2_posdis.pdf')
    plt.close(fig)
    print("  Saved fig2_posdis.pdf", flush=True)


# ══════════════════════════════════════════════════════════════════
# FIGURE 3: Surgical Ablation Bar Chart (THE MONEY PLOT)
# ══════════════════════════════════════════════════════════════════

def figure3():
    print("Generating Figure 3: Surgical Ablation...", flush=True)

    # From Phase 58b ablation_summary.transfer (pre-trained compositional sender)
    d = json.load(open('results/phase58b_cross_property.json'))
    abl = d['ablation_summary']['transfer']

    labels = [
        'Agent A pos 0\n(elasticity)',
        'Agent A pos 1\n(friction)',
        'Agent B pos 0\n(elasticity)',
        'Agent B pos 1\n(friction)',
    ]
    keys = ['A_pos0', 'A_pos1', 'B_pos0', 'B_pos1']
    means = [abl[k]['mean'] * 100 for k in keys]  # convert to %
    stds = [abl[k]['std'] * 100 for k in keys]

    # Relevant = large drop (>5%), Irrelevant = small drop
    colors = [CB_RED, CB_GRAY, CB_GRAY, CB_RED]

    fig, ax = plt.subplots(figsize=(4.5, 3.2))

    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors,
                  edgecolor='black', linewidth=0.8, width=0.6,
                  error_kw={'linewidth': 1.2})

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('Accuracy Drop (%)')
    ax.set_title('Surgical Position Ablation', fontweight='bold')

    # Annotate values
    for i, (m, s) in enumerate(zip(means, stds)):
        y_pos = m + s + 0.8
        ax.text(i, y_pos, f'{m:.1f}%', ha='center', va='bottom', fontsize=8,
                fontweight='bold')

    ax.set_ylim(-2, max(means) + max(stds) + 4)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=CB_RED, edgecolor='black', label='Relevant position'),
        Patch(facecolor=CB_GRAY, edgecolor='black', label='Irrelevant position'),
    ]
    ax.legend(handles=legend_elements, loc='upper center', frameon=True)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'fig3_ablation.pdf')
    plt.close(fig)
    print("  Saved fig3_ablation.pdf", flush=True)


# ══════════════════════════════════════════════════════════════════
# FIGURE 4: Bandwidth Allocation Scatter
# ══════════════════════════════════════════════════════════════════

def figure4():
    print("Generating Figure 4: Bandwidth Allocation...", flush=True)

    # Vision (Phase 68): 6 properties
    d68 = json.load(open('results/phase68_visual_multiattribute.json'))
    prop_names = d68['properties']['names']
    orc = d68['conditions']['oracle']
    sp0 = d68['conditions']['comp_6pos']['sender_specs']['sender_0']
    sp1 = d68['conditions']['comp_6pos']['sender_specs']['sender_1']
    mi0 = np.array(sp0['avg_total_mi_per_prop'])
    mi1 = np.array(sp1['avg_total_mi_per_prop'])
    vision_mi = mi0 + mi1
    vision_acc = np.array([orc[f'holdout_{p}_mean'] * 100 for p in prop_names])

    # Physics (Phase 55): 3 properties
    d55 = json.load(open('results/phase55_results.json'))
    phys_props = ['elasticity', 'friction', 'damping']
    phys_acc_keys = ['e', 'f', 'd']
    phys_oracle_acc = np.array([
        np.mean([s['oracle'][k] for s in d55['per_seed']]) * 100
        for k in phys_acc_keys
    ])
    mi_mats = np.array([s['mi_matrix'] for s in d55['per_seed']])
    avg_mi = mi_mats.mean(axis=0)
    phys_mi = avg_mi.sum(axis=0)  # sum across 3 positions

    fig, ax = plt.subplots(figsize=(4.5, 3.2))

    # Vision points
    ax.scatter(vision_acc, vision_mi, s=60, c=CB_BLUE, zorder=5,
               label='Vision (6 properties)', edgecolors='black', linewidth=0.5)
    short_names = {
        'brightness': 'bright.', 'saturation': 'satur.',
        'hue_conc': 'hue', 'edge_density': 'edge',
        'spatial_freq': 'spat.freq', 'color_diversity': 'col.div',
    }
    for i, pname in enumerate(prop_names):
        label = short_names.get(pname, pname)
        offset = (2, 0.08)
        if pname == 'saturation':
            offset = (-2, 0.1)
        elif pname == 'brightness':
            offset = (2, -0.15)
        ax.annotate(label, (vision_acc[i], vision_mi[i]),
                    textcoords='offset points', xytext=offset,
                    fontsize=7, color=CB_BLUE)

    # Physics points
    ax.scatter(phys_oracle_acc, phys_mi, s=60, c=CB_ORANGE, zorder=5,
               marker='s', label='Physics (3 properties)',
               edgecolors='black', linewidth=0.5)
    phys_offsets = {'elasticity': (-45, -8), 'friction': (-38, 8), 'damping': (3, -8)}
    for i, pname in enumerate(phys_props):
        ax.annotate(pname, (phys_oracle_acc[i], phys_mi[i]),
                    textcoords='offset points', xytext=phys_offsets[pname],
                    fontsize=7, color=CB_ORANGE)

    # Linear fits
    # Vision
    z_v = np.polyfit(vision_acc, vision_mi, 1)
    x_fit_v = np.linspace(vision_acc.min() - 2, vision_acc.max() + 2, 100)
    ax.plot(x_fit_v, np.polyval(z_v, x_fit_v), '--', color=CB_BLUE,
            alpha=0.5, linewidth=1)
    r_v = np.corrcoef(vision_acc, vision_mi)[0, 1]

    # Physics
    z_p = np.polyfit(phys_oracle_acc, phys_mi, 1)
    x_fit_p = np.linspace(phys_oracle_acc.min() - 2, phys_oracle_acc.max() + 2, 100)
    ax.plot(x_fit_p, np.polyval(z_p, x_fit_p), '--', color=CB_ORANGE,
            alpha=0.5, linewidth=1)
    r_p = np.corrcoef(phys_oracle_acc, phys_mi)[0, 1]

    ax.set_xlabel('Oracle Per-Property Accuracy (%)')
    ax.set_ylabel('Total MI Allocated (nats)')
    ax.set_title('Bandwidth Allocation', fontweight='bold')

    # Annotate r values
    ax.text(0.05, 0.95, f'Vision r = {r_v:.3f}\nPhysics r = {r_p:.3f}',
            transform=ax.transAxes, fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='gray', alpha=0.8))

    ax.legend(loc='lower right', frameon=True)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'fig4_bandwidth.pdf')
    plt.close(fig)
    print("  Saved fig4_bandwidth.pdf", flush=True)


# ══════════════════════════════════════════════════════════════════
# FIGURE 5: Protocol Visualization (Symbol Semantics)
# ══════════════════════════════════════════════════════════════════

def figure5():
    print("Generating Figure 5: Protocol Visualization...", flush=True)

    d = json.load(open('results/phase54f_extended.json'))

    # Use best seed (highest PosDis) to show clean specialization
    best = max(d['per_seed'], key=lambda x: x['pos_dis'])
    best_mi = np.array(best['mi_matrix'])  # (2, 2)

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3))

    # Panel A: Best seed - grouped bar showing MI per position
    ax = axes[0]
    positions = ['Position 0', 'Position 1']
    x = np.arange(len(positions))
    width = 0.35

    bars1 = ax.bar(x - width/2, best_mi[:, 0], width, label='Elasticity',
                   color=CB_BLUE, edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, best_mi[:, 1], width, label='Friction',
                   color=CB_ORANGE, edgecolor='black', linewidth=0.5)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)

    ax.set_ylabel('Mutual Information (nats)')
    ax.set_xticks(x)
    ax.set_xticklabels(positions)
    ax.set_title(f'Best Seed (PosDis = {best["pos_dis"]:.2f})', fontweight='bold')
    ax.legend(frameon=True)
    ax.set_ylim(0, 1.5)

    # Panel B: Per-seed PosDis vs holdout accuracy scatter
    ax = axes[1]
    posdis = [s['pos_dis'] for s in d['per_seed']]
    holdout = [s['holdout_both'] * 100 for s in d['per_seed']]

    ax.scatter(posdis, holdout, s=40, c=CB_BLUE,
               edgecolors='black', linewidth=0.5, zorder=5)

    z = np.polyfit(posdis, holdout, 1)
    x_fit = np.linspace(0, 0.9, 100)
    ax.plot(x_fit, np.polyval(z, x_fit), '--', color=CB_GRAY, linewidth=1)

    r = np.corrcoef(posdis, holdout)[0, 1]
    ax.text(0.05, 0.05, f'r = {r:.2f}', transform=ax.transAxes,
            fontsize=9, bbox=dict(boxstyle='round', facecolor='white',
                                   edgecolor='gray', alpha=0.8))

    ax.set_xlabel('Positional Disentanglement (PosDis)')
    ax.set_ylabel('Holdout Both-Correct (%)')
    ax.set_title('PosDis vs Accuracy', fontweight='bold')

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'fig5_protocol.pdf')
    plt.close(fig)
    print("  Saved fig5_protocol.pdf", flush=True)


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 60, flush=True)
    print("Generating paper figures", flush=True)
    print("=" * 60, flush=True)

    figure1()
    figure2()
    figure3()
    figure4()
    figure5()

    print("\nAll figures saved to figures/", flush=True)
