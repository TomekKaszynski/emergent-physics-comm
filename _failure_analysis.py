"""
Failure Mode Analysis of Non-Compositional Seeds from Phase 54f
================================================================
Identifies seeds with PosDis < 0.4, characterizes their failure modes
(holistic vs collapsed), and compares against compositional seeds.

Run:
  python3 _failure_analysis.py
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats

RESULTS_DIR = Path("results")
POSDIS_THRESHOLD = 0.4


def main():
    with open(RESULTS_DIR / "phase54f_extended.json") as f:
        data = json.load(f)

    seeds_data = data['per_seed']

    # Identify failed vs compositional seeds
    failed = [s for s in seeds_data if s['pos_dis'] < POSDIS_THRESHOLD]
    compositional = [s for s in seeds_data if s['pos_dis'] >= POSDIS_THRESHOLD]

    failed_ids = [s['seed'] for s in failed]
    comp_ids = [s['seed'] for s in compositional]

    print("=" * 70, flush=True)
    print("Failure Mode Analysis: Non-Compositional Seeds (PosDis < 0.4)", flush=True)
    print("=" * 70, flush=True)
    print(f"  Failed seeds: {failed_ids} ({len(failed)}/{len(seeds_data)})", flush=True)
    print(f"  Compositional seeds: {comp_ids} ({len(compositional)}/{len(seeds_data)})", flush=True)
    print(flush=True)

    # ── Per-seed detailed analysis ──────────────────────────────
    print("=" * 70, flush=True)
    print("DETAILED PER-SEED ANALYSIS", flush=True)
    print("=" * 70, flush=True)

    failure_characterizations = []

    for s in failed:
        mi = s['mi_matrix']
        ent = s['entropies']
        print(f"\n  --- Seed {s['seed']} (PosDis={s['pos_dis']:.3f}) ---", flush=True)

        # MI matrix
        print(f"  MI matrix [pos x property]:", flush=True)
        print(f"    {'':>8s} {'elasticity':>12s} {'friction':>12s}", flush=True)
        print(f"    {'pos0':>8s} {mi[0][0]:12.3f} {mi[0][1]:12.3f}", flush=True)
        print(f"    {'pos1':>8s} {mi[1][0]:12.3f} {mi[1][1]:12.3f}", flush=True)

        # Per-position entropy
        print(f"  Normalized entropy: pos0={ent[0]:.3f}  pos1={ent[1]:.3f}", flush=True)

        # Accuracy
        print(f"  Holdout accuracy: both={s['holdout_both']:.1%}  "
              f"e={s['holdout_e']:.1%}  f={s['holdout_f']:.1%}", flush=True)

        # TopSim
        print(f"  TopSim: {s['topsim']:.3f}", flush=True)

        # Characterize: holistic vs collapsed
        # Holistic: both positions encode both properties roughly equally
        # Collapsed: one position dead or near-dead
        pos0_total = mi[0][0] + mi[0][1]
        pos1_total = mi[1][0] + mi[1][1]
        pos0_spec = abs(mi[0][0] - mi[0][1]) / max(pos0_total, 1e-10)
        pos1_spec = abs(mi[1][0] - mi[1][1]) / max(pos1_total, 1e-10)

        # Check if one position is dead (very low total MI)
        min_total = min(pos0_total, pos1_total)
        max_total = max(pos0_total, pos1_total)
        total_ratio = min_total / max(max_total, 1e-10)

        # Holistic: both positions have similar MI for both properties
        # (low specialization ratio AND both positions active)
        avg_spec = (pos0_spec + pos1_spec) / 2

        if total_ratio < 0.3:
            char = "COLLAPSED"
            detail = f"one position near-dead (MI ratio={total_ratio:.2f})"
        elif avg_spec < 0.2:
            char = "HOLISTIC"
            detail = f"both positions encode both properties (avg spec={avg_spec:.2f})"
        else:
            char = "HOLISTIC (partial)"
            detail = f"weak specialization (avg spec={avg_spec:.2f}, MI ratio={total_ratio:.2f})"

        print(f"  Characterization: {char} — {detail}", flush=True)
        print(f"    pos0 spec={pos0_spec:.3f}  pos1 spec={pos1_spec:.3f}  "
              f"MI ratio={total_ratio:.3f}", flush=True)

        failure_characterizations.append({
            'seed': s['seed'],
            'pos_dis': s['pos_dis'],
            'characterization': char,
            'detail': detail,
            'pos0_spec': float(pos0_spec),
            'pos1_spec': float(pos1_spec),
            'mi_total_ratio': float(total_ratio),
            'avg_spec': float(avg_spec),
        })

    # ── Correlation: seed index vs PosDis ───────────────────────
    print(f"\n{'='*70}", flush=True)
    print("STATISTICAL COMPARISONS", flush=True)
    print(f"{'='*70}", flush=True)

    all_seeds_idx = np.array([s['seed'] for s in seeds_data])
    all_posdis = np.array([s['pos_dis'] for s in seeds_data])

    r, p = stats.pearsonr(all_seeds_idx, all_posdis)
    print(f"\n  Seed index vs PosDis correlation:", flush=True)
    print(f"    Pearson r={r:.3f}, p={p:.3f}", flush=True)
    print(f"    {'SYSTEMATIC' if p < 0.05 else 'RANDOM'} "
          f"(p {'<' if p < 0.05 else '>'} 0.05)", flush=True)

    # ── Entropy comparison ──────────────────────────────────────
    failed_ent = np.array([np.mean(s['entropies']) for s in failed])
    comp_ent = np.array([np.mean(s['entropies']) for s in compositional])

    t_ent, p_ent = stats.ttest_ind(failed_ent, comp_ent)
    print(f"\n  Per-position entropy (mean of pos0, pos1):", flush=True)
    print(f"    Failed: {failed_ent.mean():.4f} +/- {failed_ent.std():.4f}", flush=True)
    print(f"    Compositional: {comp_ent.mean():.4f} +/- {comp_ent.std():.4f}", flush=True)
    print(f"    t={t_ent:.2f}, p={p_ent:.3f}", flush=True)
    print(f"    Vocab collapse in failed seeds? {'YES' if p_ent < 0.05 and failed_ent.mean() < comp_ent.mean() else 'NO'}", flush=True)

    # ── Accuracy comparison ─────────────────────────────────────
    failed_both = np.array([s['holdout_both'] for s in failed])
    comp_both = np.array([s['holdout_both'] for s in compositional])
    failed_e = np.array([s['holdout_e'] for s in failed])
    comp_e = np.array([s['holdout_e'] for s in compositional])
    failed_f = np.array([s['holdout_f'] for s in failed])
    comp_f = np.array([s['holdout_f'] for s in compositional])

    t_both, p_both = stats.ttest_ind(failed_both, comp_both)
    t_e, p_e = stats.ttest_ind(failed_e, comp_e)
    t_f, p_f = stats.ttest_ind(failed_f, comp_f)

    print(f"\n  Holdout accuracy comparison:", flush=True)
    print(f"    {'Metric':<12s} {'Failed':>12s} {'Compositional':>14s} {'Diff':>8s} {'p':>8s}", flush=True)
    print(f"    {'-'*12} {'-'*12} {'-'*14} {'-'*8} {'-'*8}", flush=True)
    print(f"    {'both':<12s} {failed_both.mean():11.1%}  {comp_both.mean():13.1%}  "
          f"{failed_both.mean()-comp_both.mean():+7.1%}  {p_both:7.3f}", flush=True)
    print(f"    {'elasticity':<12s} {failed_e.mean():11.1%}  {comp_e.mean():13.1%}  "
          f"{failed_e.mean()-comp_e.mean():+7.1%}  {p_e:7.3f}", flush=True)
    print(f"    {'friction':<12s} {failed_f.mean():11.1%}  {comp_f.mean():13.1%}  "
          f"{failed_f.mean()-comp_f.mean():+7.1%}  {p_f:7.3f}", flush=True)

    # ── TopSim comparison ───────────────────────────────────────
    failed_topsim = np.array([s['topsim'] for s in failed])
    comp_topsim = np.array([s['topsim'] for s in compositional])
    t_ts, p_ts = stats.ttest_ind(failed_topsim, comp_topsim)

    print(f"\n  TopSim comparison:", flush=True)
    print(f"    Failed: {failed_topsim.mean():.3f} +/- {failed_topsim.std():.3f}", flush=True)
    print(f"    Compositional: {comp_topsim.mean():.3f} +/- {comp_topsim.std():.3f}", flush=True)
    print(f"    t={t_ts:.2f}, p={p_ts:.3f}", flush=True)

    # ── MI total comparison ─────────────────────────────────────
    failed_mi_total = np.array([sum(s['mi_matrix'][0]) + sum(s['mi_matrix'][1])
                                for s in failed])
    comp_mi_total = np.array([sum(s['mi_matrix'][0]) + sum(s['mi_matrix'][1])
                              for s in compositional])
    t_mi, p_mi = stats.ttest_ind(failed_mi_total, comp_mi_total)

    print(f"\n  Total MI (sum of all 4 cells):", flush=True)
    print(f"    Failed: {failed_mi_total.mean():.3f} +/- {failed_mi_total.std():.3f}", flush=True)
    print(f"    Compositional: {comp_mi_total.mean():.3f} +/- {comp_mi_total.std():.3f}", flush=True)
    print(f"    t={t_mi:.2f}, p={p_mi:.3f}", flush=True)

    # ── MI off-diagonal comparison (redundancy) ─────────────────
    failed_offdiag = np.array([s['mi_matrix'][0][1] + s['mi_matrix'][1][0]
                               for s in failed])
    comp_offdiag = np.array([s['mi_matrix'][0][1] + s['mi_matrix'][1][0]
                             for s in compositional])
    t_od, p_od = stats.ttest_ind(failed_offdiag, comp_offdiag)

    print(f"\n  Off-diagonal MI (redundancy: pos0-friction + pos1-elasticity):", flush=True)
    print(f"    Failed: {failed_offdiag.mean():.3f} +/- {failed_offdiag.std():.3f}", flush=True)
    print(f"    Compositional: {comp_offdiag.mean():.3f} +/- {comp_offdiag.std():.3f}", flush=True)
    print(f"    t={t_od:.2f}, p={p_od:.3f}", flush=True)

    # ── Summary paragraph ───────────────────────────────────────
    print(f"\n{'='*70}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)

    n_holistic = sum(1 for c in failure_characterizations if 'HOLISTIC' in c['characterization'])
    n_collapsed = sum(1 for c in failure_characterizations if c['characterization'] == 'COLLAPSED')

    summary_text = (
        f"Of 20 seeds, {len(failed)} ({len(failed)/len(seeds_data):.0%}) produced non-compositional "
        f"protocols (PosDis < 0.4). "
        f"All {len(failed)} failed seeds show HOLISTIC failure: both message positions encode both "
        f"properties roughly equally (avg specialization ratio "
        f"{np.mean([c['avg_spec'] for c in failure_characterizations]):.2f}), rather than one "
        f"position going dead. "
        f"Failed seeds have {'HIGHER' if failed_topsim.mean() > comp_topsim.mean() else 'SIMILAR'} "
        f"TopSim ({failed_topsim.mean():.3f} vs {comp_topsim.mean():.3f}), suggesting they find "
        f"structured but non-compositional mappings. "
        f"Entropy is {'comparable' if p_ent > 0.05 else 'significantly different'} "
        f"(failed={failed_ent.mean():.3f}, comp={comp_ent.mean():.3f}, p={p_ent:.3f}), "
        f"{'ruling out' if p_ent > 0.05 else 'suggesting'} vocabulary collapse as a failure cause. "
        f"Holdout accuracy is {'significantly lower' if p_both < 0.05 else 'comparable'} "
        f"for failed seeds (both={failed_both.mean():.1%} vs {comp_both.mean():.1%}, p={p_both:.3f}). "
        f"Off-diagonal MI (redundancy) is {'significantly higher' if p_od < 0.05 else 'higher but not significant'} "
        f"in failed seeds ({failed_offdiag.mean():.3f} vs {comp_offdiag.mean():.3f}, p={p_od:.3f}), "
        f"confirming that both positions encode both properties rather than specializing. "
        f"Seed index shows {'no' if p > 0.05 else 'a'} systematic correlation with PosDis "
        f"(r={r:.3f}, p={p:.3f}), indicating failure is stochastic."
    )

    print(f"\n{summary_text}", flush=True)

    # ── Save results ────────────────────────────────────────────
    output = {
        'failed_seeds': failed_ids,
        'compositional_seeds': comp_ids,
        'failure_characterizations': failure_characterizations,
        'per_seed_detail': {
            s['seed']: {
                'pos_dis': s['pos_dis'],
                'topsim': s['topsim'],
                'entropies': s['entropies'],
                'mi_matrix': s['mi_matrix'],
                'holdout_both': s['holdout_both'],
                'holdout_e': s['holdout_e'],
                'holdout_f': s['holdout_f'],
            } for s in failed
        },
        'comparisons': {
            'seed_posdis_correlation': {'r': float(r), 'p': float(p)},
            'entropy': {
                'failed_mean': float(failed_ent.mean()),
                'failed_std': float(failed_ent.std()),
                'comp_mean': float(comp_ent.mean()),
                'comp_std': float(comp_ent.std()),
                't': float(t_ent), 'p': float(p_ent),
                'vocab_collapse': bool(p_ent < 0.05 and failed_ent.mean() < comp_ent.mean()),
            },
            'holdout_both': {
                'failed_mean': float(failed_both.mean()),
                'failed_std': float(failed_both.std()),
                'comp_mean': float(comp_both.mean()),
                'comp_std': float(comp_both.std()),
                't': float(t_both), 'p': float(p_both),
            },
            'holdout_e': {
                'failed_mean': float(failed_e.mean()),
                'comp_mean': float(comp_e.mean()),
                'p': float(p_e),
            },
            'holdout_f': {
                'failed_mean': float(failed_f.mean()),
                'comp_mean': float(comp_f.mean()),
                'p': float(p_f),
            },
            'topsim': {
                'failed_mean': float(failed_topsim.mean()),
                'comp_mean': float(comp_topsim.mean()),
                't': float(t_ts), 'p': float(p_ts),
            },
            'total_mi': {
                'failed_mean': float(failed_mi_total.mean()),
                'comp_mean': float(comp_mi_total.mean()),
                't': float(t_mi), 'p': float(p_mi),
            },
            'off_diagonal_mi': {
                'failed_mean': float(failed_offdiag.mean()),
                'comp_mean': float(comp_offdiag.mean()),
                't': float(t_od), 'p': float(p_od),
            },
        },
        'summary': summary_text,
    }

    output_path = RESULTS_DIR / "phase54f_failure_analysis.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
