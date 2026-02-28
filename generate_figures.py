"""
Generate publication-quality figures for the paper.
Loads results from Phase 54f, 54g, and 55.

Usage:
  python3 generate_figures.py
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'font.family': 'serif',
})

# Colorblind-friendly palette (Okabe-Ito)
C_BLUE = '#0072B2'
C_ORANGE = '#E69F00'
C_GREEN = '#009E73'
C_RED = '#D55E00'
C_PURPLE = '#CC79A7'
C_GREY = '#999999'
C_CYAN = '#56B4E9'

# ── Load data ──────────────────────────────────────────────────────
with open('results/phase54f_extended.json') as f:
    d54f = json.load(f)
with open('results/phase54g_control.json') as f:
    d54g = json.load(f)
with open('results/phase55_results.json') as f:
    d55 = json.load(f)


# ══════════════════════════════════════════════════════════════════
# Figure 1: Compositionality (7 x 2.5 in, 3 panels)
# ══════════════════════════════════════════════════════════════════

def fig1():
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.5),
                             gridspec_kw={'width_ratios': [1, 0.8, 1.2]})

    # ── Panel (a): PosDis histogram ────────────────────────────────
    ax = axes[0]
    pd_2x5 = [s['pos_dis'] for s in d54f['per_seed']]
    pd_1x25 = [s['pos_dis'] for s in d54g['per_seed']]

    bins = np.arange(0, 0.85, 0.1)
    ax.hist(pd_2x5, bins=bins, alpha=0.75, color=C_BLUE, label='2$\\times$5',
            edgecolor='white', linewidth=0.5)
    ax.hist(pd_1x25, bins=bins, alpha=0.75, color=C_ORANGE, label='1$\\times$25',
            edgecolor='white', linewidth=0.5)
    ax.axvline(0.4, color=C_RED, linestyle='--', linewidth=1, label='Threshold')
    ax.set_xlabel('PosDis')
    ax.set_ylabel('Seeds')
    ax.set_title('(a) Compositionality')
    ax.legend(loc='upper left', frameon=False)
    ax.set_xlim(-0.05, 0.85)
    ax.set_ylim(0, 12)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # ── Panel (b): MI matrix heatmap for best compositional seed ──
    ax = axes[1]
    # Find best compositional 2x5 seed (highest PosDis)
    best_seed = max(d54f['per_seed'], key=lambda s: s['pos_dis'])
    mi = np.array(best_seed['mi_matrix'])  # (2, 2)

    im = ax.imshow(mi, cmap='Blues', vmin=0, aspect='auto')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Elast.', 'Frict.'])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Pos 0', 'Pos 1'])
    ax.set_title(f'(b) MI matrix (seed {best_seed["seed"]})')
    ax.set_xlabel('Property')
    ax.set_ylabel('Position')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{mi[i, j]:.2f}', ha='center', va='center',
                    fontsize=9, fontweight='bold',
                    color='white' if mi[i, j] > mi.max() * 0.6 else 'black')

    # ── Panel (c): Grouped bar chart ──────────────────────────────
    ax = axes[2]
    # 54f: 2x5 compositional
    train_e_2x5 = np.mean([s['train_e'] for s in d54f['per_seed']])
    train_f_2x5 = np.mean([s['train_f'] for s in d54f['per_seed']])
    hold_e_2x5 = np.mean([s['holdout_e'] for s in d54f['per_seed']])
    hold_f_2x5 = np.mean([s['holdout_f'] for s in d54f['per_seed']])
    train_both_2x5 = np.mean([s['train_both'] for s in d54f['per_seed']])
    hold_both_2x5 = np.mean([s['holdout_both'] for s in d54f['per_seed']])

    # 54g: 1x25 control
    train_e_1x25 = np.mean([s['train_e'] for s in d54g['per_seed']])
    train_f_1x25 = np.mean([s['train_f'] for s in d54g['per_seed']])
    hold_e_1x25 = np.mean([s['holdout_e'] for s in d54g['per_seed']])
    hold_f_1x25 = np.mean([s['holdout_f'] for s in d54g['per_seed']])
    train_both_1x25 = np.mean([s['train_both'] for s in d54g['per_seed']])
    hold_both_1x25 = np.mean([s['holdout_both'] for s in d54g['per_seed']])

    labels = ['Elast.', 'Frict.', 'Both']
    x = np.arange(len(labels))
    w = 0.18

    bars_train_2x5 = [train_e_2x5, train_f_2x5, train_both_2x5]
    bars_hold_2x5 = [hold_e_2x5, hold_f_2x5, hold_both_2x5]
    bars_train_1x25 = [train_e_1x25, train_f_1x25, train_both_1x25]
    bars_hold_1x25 = [hold_e_1x25, hold_f_1x25, hold_both_1x25]

    ax.bar(x - 1.5*w, bars_train_2x5, w, label='2$\\times$5 train', color=C_BLUE, alpha=0.6)
    ax.bar(x - 0.5*w, bars_hold_2x5, w, label='2$\\times$5 holdout', color=C_BLUE)
    ax.bar(x + 0.5*w, bars_train_1x25, w, label='1$\\times$25 train', color=C_ORANGE, alpha=0.6)
    ax.bar(x + 1.5*w, bars_hold_1x25, w, label='1$\\times$25 holdout', color=C_ORANGE)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Accuracy')
    ax.set_title('(c) Train vs. holdout')
    ax.set_ylim(0.5, 1.0)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, 0))
    ax.legend(loc='lower left', frameon=False, ncol=2, fontsize=7)

    fig.tight_layout(w_pad=1.5)
    fig.savefig('figures/fig1_compositionality.pdf')
    plt.close()
    print('Saved figures/fig1_compositionality.pdf', flush=True)


# ══════════════════════════════════════════════════════════════════
# Figure 2: Ablation (5 x 3 in)
# ══════════════════════════════════════════════════════════════════

def fig2():
    fig, ax1 = plt.subplots(figsize=(5, 3))

    conditions = [
        'No IL',
        'IL only',
        'IL+pop\nstaggered',
        'IL+pop\nsimult. 200ep',
        'IL+pop\nextended 400ep',
    ]
    holdout = [0.829, 0.771, 0.806, 0.745, 0.767]
    posdis = [0.048, 0.291, 0.272, 0.295, 0.486]

    x = np.arange(len(conditions))

    # Left axis: holdout bars
    bars = ax1.bar(x, holdout, 0.55, color=C_BLUE, alpha=0.75,
                   edgecolor='white', linewidth=0.5, label='Holdout accuracy')
    ax1.set_ylabel('Holdout accuracy', color=C_BLUE)
    ax1.set_ylim(0.6, 0.9)
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, 0))
    ax1.tick_params(axis='y', colors=C_BLUE)
    ax1.set_xticks(x)
    ax1.set_xticklabels(conditions, fontsize=7.5)

    # Bar value labels
    for bar, val in zip(bars, holdout):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.008,
                 f'{val:.1%}', ha='center', va='bottom', fontsize=7, color=C_BLUE)

    # Right axis: PosDis line
    ax2 = ax1.twinx()
    ax2.plot(x, posdis, 'o-', color=C_RED, linewidth=2, markersize=6,
             label='PosDis', zorder=5)
    ax2.set_ylabel('PosDis', color=C_RED)
    ax2.set_ylim(0, 0.6)
    ax2.tick_params(axis='y', colors=C_RED)
    ax2.axhline(0.4, color=C_RED, linestyle=':', linewidth=0.8, alpha=0.5)
    ax2.spines['right'].set_visible(True)
    ax2.spines['right'].set_color(C_RED)

    # PosDis value labels
    for xi, val in zip(x, posdis):
        offset = -0.035 if val > 0.4 else 0.025
        ax2.text(xi, val + offset, f'{val:.3f}', ha='center', va='bottom',
                 fontsize=7, color=C_RED)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left',
               frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig('figures/fig2_ablation.pdf')
    plt.close()
    print('Saved figures/fig2_ablation.pdf', flush=True)


# ══════════════════════════════════════════════════════════════════
# Figure 3: Three-property (6 x 2.5 in, 2 panels)
# ══════════════════════════════════════════════════════════════════

def fig3():
    fig, axes = plt.subplots(1, 2, figsize=(6, 2.5))

    # ── Panel (a): Average 3x3 MI matrix ──────────────────────────
    ax = axes[0]
    mean_mi = np.array(d55['summary']['mean_mi_matrix'])  # (3, 3)

    im = ax.imshow(mean_mi, cmap='Blues', vmin=0, aspect='auto')
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['Elast.', 'Frict.', 'Damp.'])
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['Pos 0', 'Pos 1', 'Pos 2'])
    ax.set_xlabel('Property')
    ax.set_ylabel('Position')
    ax.set_title('(a) Mean MI matrix (3$\\times$5)')
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f'{mean_mi[i, j]:.2f}', ha='center', va='center',
                    fontsize=9, fontweight='bold',
                    color='white' if mean_mi[i, j] > mean_mi.max() * 0.6 else 'black')

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label('MI (nats)', fontsize=8)

    # ── Panel (b): PosDis histogram for 3x5 ───────────────────────
    ax = axes[1]
    pd_3x5 = [s['pos_dis'] for s in d55['per_seed']]

    bins = np.arange(0, 0.75, 0.05)
    ax.hist(pd_3x5, bins=bins, alpha=0.75, color=C_GREEN,
            edgecolor='white', linewidth=0.5, label='3$\\times$5')
    ax.axvline(0.4, color=C_RED, linestyle='--', linewidth=1, label='Threshold')
    ax.axvline(0.486, color=C_BLUE, linestyle=':', linewidth=1.5,
               label='2-prop mean (0.486)')
    ax.set_xlabel('PosDis')
    ax.set_ylabel('Seeds')
    ax.set_title('(b) 3-property compositionality')
    ax.set_xlim(0.15, 0.65)
    ax.set_ylim(0, 8)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.legend(loc='upper right', frameon=False, fontsize=7)

    fig.tight_layout(w_pad=2)
    fig.savefig('figures/fig3_three_property.pdf')
    plt.close()
    print('Saved figures/fig3_three_property.pdf', flush=True)


# ── Main ───────────────────────────────────────────────────────────
if __name__ == '__main__':
    fig1()
    fig2()
    fig3()
    print('All figures generated.', flush=True)
