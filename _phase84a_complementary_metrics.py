"""
Phase 84a: Complementary Compositionality Metrics
===================================================
Compute TopSim and BosDis for all existing trained conditions.
Uses pre-computed MI matrices and token data from result JSONs.

Run:
  PYTHONUNBUFFERED=1 python3 _phase84a_complementary_metrics.py
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats

RESULTS_DIR = Path("results")
VOCAB_SIZE = 5


def compute_bosdis_from_mi(mi_matrix, n_attributes=2):
    """
    Bag-of-Symbols Disentanglement (BosDis).
    For each symbol position, measure how much MI is concentrated
    on a single attribute (like PosDis), but computed at the
    symbol-attribute level rather than position-attribute level.

    Since we have MI(position, attribute) already, BosDis is:
    For each position p, for each symbol value s in that position:
      MI(symbol_p == s; attribute_j) for each j

    But we don't have per-symbol MI in the JSONs. Instead, compute
    a variant: for each position, measure the gap between the
    best-attributed MI and second-best, normalized by best.
    This IS PosDis. So BosDis here uses an alternative:
    the residual entropy approach.

    Alternative BosDis: For each position p, compute the proportion
    of total MI that goes to the best attribute. Average across positions.
    """
    mi = np.array(mi_matrix)
    n_pos = mi.shape[0]

    if n_pos == 0:
        return 0.0

    # BosDis variant: for each position, what fraction of MI goes to best attribute?
    # Higher = more specialized
    bosdis = 0.0
    active_pos = 0
    for p in range(n_pos):
        total_mi = mi[p].sum()
        if total_mi > 1e-10:
            best_mi = mi[p].max()
            bosdis += best_mi / total_mi
            active_pos += 1

    return bosdis / max(active_pos, 1)


def compute_conflict_score(mi_matrix):
    """
    Conflict score: how much do positions disagree about attributes?
    Low conflict = each position maps cleanly to one attribute.
    """
    mi = np.array(mi_matrix)
    n_pos = mi.shape[0]

    if n_pos == 0:
        return 0.0

    conflicts = []
    for p in range(n_pos):
        sorted_mi = np.sort(mi[p])[::-1]
        if sorted_mi[0] > 1e-10:
            # Ratio of second-best to best (0 = perfect, 1 = ambiguous)
            conflict = sorted_mi[1] / sorted_mi[0] if len(sorted_mi) > 1 else 0.0
            conflicts.append(conflict)

    return np.mean(conflicts) if conflicts else 0.0


def analyze_condition(name, path, is_4agent=False):
    """Load results and compute metrics for one condition."""
    with open(path) as f:
        data = json.load(f)

    seeds = data.get('per_seed', [])
    n_seeds = len(seeds)

    results = {
        'name': name,
        'n_seeds': n_seeds,
        'topsim': [],
        'posdis': [],
        'holdout': [],
        'bosdis': [],
        'conflict': [],
    }

    for s in seeds:
        ts = s.get('topsim', None)
        pd = s.get('pos_dis', None)
        hb = s.get('holdout_both', None)
        mi = s.get('mi_matrix', None)

        if ts is not None:
            results['topsim'].append(float(ts))
        if pd is not None:
            results['posdis'].append(float(pd))
        if hb is not None:
            results['holdout'].append(float(hb))
        if mi is not None:
            mi_arr = np.array(mi)
            results['bosdis'].append(compute_bosdis_from_mi(mi_arr))
            results['conflict'].append(compute_conflict_score(mi_arr))

    return results


def print_summary(results):
    """Print formatted summary for one condition."""
    name = results['name']
    n = results['n_seeds']

    def fmt(arr):
        if not arr:
            return "N/A"
        return f"{np.mean(arr):.3f} ± {np.std(arr):.3f}"

    print(f"\n  {name} ({n} seeds):", flush=True)
    print(f"    TopSim:   {fmt(results['topsim'])}", flush=True)
    print(f"    PosDis:   {fmt(results['posdis'])}", flush=True)
    print(f"    BosDis:   {fmt(results['bosdis'])}", flush=True)
    print(f"    Conflict: {fmt(results['conflict'])}", flush=True)
    print(f"    Holdout:  {fmt(results['holdout'])}", flush=True)


if __name__ == "__main__":
    print("Phase 84a: Complementary Compositionality Metrics", flush=True)
    print("=" * 70, flush=True)

    all_results = {}

    # ─── 1. 4-agent ramp DINOv2 (80 seeds) ───
    print("\n━━━ RAMP DATASET ━━━", flush=True)

    r72 = analyze_condition("4-Agent DINOv2 Ramp (80 seeds)",
                            "results/phase72_4agent_80seeds.json", is_4agent=True)
    print_summary(r72)
    all_results['ramp_4agent_dinov2_80'] = r72

    # ─── 2. 2-agent ramp DINOv2 (20 seeds) ───
    r54f = analyze_condition("2-Agent DINOv2 Ramp (20 seeds)",
                             "results/phase54f_extended.json")
    print_summary(r54f)
    all_results['ramp_2agent_dinov2_20'] = r54f

    # ─── 3. Collision DINOv2 ViT-S ───
    print("\n━━━ COLLISION DATASET ━━━", flush=True)

    r79_4 = analyze_condition("4-Agent DINOv2 ViT-S Collision (20 seeds)",
                              "results/phase79_dinov2_4agent_collision.json", is_4agent=True)
    print_summary(r79_4)
    all_results['collision_4agent_dinov2_vits'] = r79_4

    r79_2 = analyze_condition("2-Agent DINOv2 ViT-S Collision (20 seeds)",
                              "results/phase79_dinov2_collision.json")
    print_summary(r79_2)
    all_results['collision_2agent_dinov2_vits'] = r79_2

    # ─── 4. Collision V-JEPA 2 ───
    r79b_4 = analyze_condition("4-Agent V-JEPA 2 Collision (20 seeds)",
                               "results/phase79b_vjepa2_4agent_collision.json", is_4agent=True)
    print_summary(r79b_4)
    all_results['collision_4agent_vjepa2'] = r79b_4

    r79b_2 = analyze_condition("2-Agent V-JEPA 2 Collision (20 seeds)",
                               "results/phase79b_vjepa2_collision.json")
    print_summary(r79b_2)
    all_results['collision_2agent_vjepa2'] = r79b_2

    # ─── 5. Collision DINOv2 ViT-L ───
    r82_4 = analyze_condition("4-Agent DINOv2 ViT-L Collision (20 seeds)",
                              "results/phase82_dinov2_vitl_collision_4agent.json", is_4agent=True)
    print_summary(r82_4)
    all_results['collision_4agent_dinov2_vitl'] = r82_4

    r82_2 = analyze_condition("2-Agent DINOv2 ViT-L Collision (20 seeds)",
                              "results/phase82_dinov2_vitl_collision_2agent.json")
    print_summary(r82_2)
    all_results['collision_2agent_dinov2_vitl'] = r82_2

    # ═══════════════════════════════════════════════════════════════
    # CORRELATIONS (80-seed 4-agent ramp)
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("CORRELATIONS (80-seed 4-agent DINOv2 ramp)", flush=True)
    print("=" * 70, flush=True)

    ts = np.array(r72['topsim'])
    pd = np.array(r72['posdis'])
    hb = np.array(r72['holdout'])
    bd = np.array(r72['bosdis'])

    def corr_str(x, y, xname, yname):
        r, p = stats.pearsonr(x, y)
        return f"  {xname:12s} vs {yname:12s}: r={r:+.3f} (p={p:.4f})"

    print(corr_str(ts, pd, "TopSim", "PosDis"), flush=True)
    print(corr_str(ts, hb, "TopSim", "Holdout"), flush=True)
    print(corr_str(pd, hb, "PosDis", "Holdout"), flush=True)
    print(corr_str(bd, hb, "BosDis", "Holdout"), flush=True)
    print(corr_str(bd, pd, "BosDis", "PosDis"), flush=True)
    print(corr_str(bd, ts, "BosDis", "TopSim"), flush=True)

    # ═══════════════════════════════════════════════════════════════
    # TopSim for compositional vs holistic (2-agent ramp)
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("COMPOSITIONAL vs HOLISTIC (2-agent DINOv2 ramp)", flush=True)
    print("=" * 70, flush=True)

    # Use the broader 2-agent data — check multiple files for more seeds
    all_2agent_seeds = []
    for path in ["results/phase54f_extended.json", "results/phase54e_20seeds.json",
                  "results/phase54g_control.json"]:
        try:
            with open(path) as f:
                d = json.load(f)
            for s in d.get('per_seed', []):
                if 'topsim' in s and 'pos_dis' in s:
                    all_2agent_seeds.append(s)
        except Exception:
            pass

    comp_ts = [s['topsim'] for s in all_2agent_seeds if s.get('pos_dis', 0) > 0.4]
    hol_ts = [s['topsim'] for s in all_2agent_seeds if s.get('pos_dis', 0) <= 0.4]

    print(f"  Total 2-agent seeds with TopSim: {len(all_2agent_seeds)}", flush=True)
    print(f"  Compositional (PosDis>0.4): {len(comp_ts)} seeds, TopSim={np.mean(comp_ts):.3f} ± {np.std(comp_ts):.3f}" if comp_ts else "  No compositional seeds", flush=True)
    print(f"  Holistic (PosDis≤0.4):      {len(hol_ts)} seeds, TopSim={np.mean(hol_ts):.3f} ± {np.std(hol_ts):.3f}" if hol_ts else "  No holistic seeds", flush=True)

    if comp_ts and hol_ts:
        t, p = stats.ttest_ind(comp_ts, hol_ts)
        d = (np.mean(comp_ts) - np.mean(hol_ts)) / np.sqrt(
            (np.std(comp_ts)**2 + np.std(hol_ts)**2) / 2)
        print(f"  t={t:.3f}, p={p:.4f}, Cohen's d={d:.3f}", flush=True)

    # Also check 2-agent ramp correlations
    ts_2ag = np.array([s['topsim'] for s in all_2agent_seeds if 'topsim' in s and 'holdout_both' in s])
    pd_2ag = np.array([s['pos_dis'] for s in all_2agent_seeds if 'topsim' in s and 'holdout_both' in s])
    hb_2ag = np.array([s['holdout_both'] for s in all_2agent_seeds if 'topsim' in s and 'holdout_both' in s])

    if len(ts_2ag) > 5:
        print(f"\n  2-agent correlations ({len(ts_2ag)} seeds):", flush=True)
        r1, p1 = stats.pearsonr(ts_2ag, pd_2ag)
        r2, p2 = stats.pearsonr(ts_2ag, hb_2ag)
        r3, p3 = stats.pearsonr(pd_2ag, hb_2ag)
        print(f"    TopSim-PosDis:  r={r1:+.3f} (p={p1:.4f})", flush=True)
        print(f"    TopSim-Holdout: r={r2:+.3f} (p={p2:.4f})", flush=True)
        print(f"    PosDis-Holdout: r={r3:+.3f} (p={p3:.4f})", flush=True)

    # ═══════════════════════════════════════════════════════════════
    # CROSS-CONDITION COMPARISON TABLE
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("CROSS-CONDITION COMPARISON", flush=True)
    print("=" * 70, flush=True)

    header = f"  {'Condition':<45s} {'TopSim':>8s} {'PosDis':>8s} {'BosDis':>8s} {'Holdout':>8s}"
    print(header, flush=True)
    print(f"  {'-'*79}", flush=True)

    order = [
        ('ramp_4agent_dinov2_80', '4-Ag DINOv2 Ramp'),
        ('ramp_2agent_dinov2_20', '2-Ag DINOv2 Ramp'),
        ('collision_4agent_vjepa2', '4-Ag V-JEPA2 Collision'),
        ('collision_2agent_vjepa2', '2-Ag V-JEPA2 Collision'),
        ('collision_4agent_dinov2_vits', '4-Ag DINOv2-S Collision'),
        ('collision_2agent_dinov2_vits', '2-Ag DINOv2-S Collision'),
        ('collision_4agent_dinov2_vitl', '4-Ag DINOv2-L Collision'),
        ('collision_2agent_dinov2_vitl', '2-Ag DINOv2-L Collision'),
    ]

    for key, label in order:
        r = all_results[key]
        def m(arr):
            return f"{np.mean(arr):.3f}" if arr else "N/A"
        print(f"  {label:<45s} {m(r['topsim']):>8s} {m(r['posdis']):>8s} "
              f"{m(r['bosdis']):>8s} {m(r['holdout']):>8s}", flush=True)

    # ═══════════════════════════════════════════════════════════════
    # SAVE
    # ═══════════════════════════════════════════════════════════════

    save_data = {}
    for key, r in all_results.items():
        save_data[key] = {
            'name': r['name'],
            'n_seeds': r['n_seeds'],
            'topsim_mean': float(np.mean(r['topsim'])) if r['topsim'] else None,
            'topsim_std': float(np.std(r['topsim'])) if r['topsim'] else None,
            'posdis_mean': float(np.mean(r['posdis'])) if r['posdis'] else None,
            'posdis_std': float(np.std(r['posdis'])) if r['posdis'] else None,
            'bosdis_mean': float(np.mean(r['bosdis'])) if r['bosdis'] else None,
            'bosdis_std': float(np.std(r['bosdis'])) if r['bosdis'] else None,
            'conflict_mean': float(np.mean(r['conflict'])) if r['conflict'] else None,
            'conflict_std': float(np.std(r['conflict'])) if r['conflict'] else None,
            'holdout_mean': float(np.mean(r['holdout'])) if r['holdout'] else None,
            'holdout_std': float(np.std(r['holdout'])) if r['holdout'] else None,
        }

    # Add correlations
    save_data['correlations_80seed_4agent'] = {
        'topsim_posdis': {'r': float(stats.pearsonr(ts, pd)[0]), 'p': float(stats.pearsonr(ts, pd)[1])},
        'topsim_holdout': {'r': float(stats.pearsonr(ts, hb)[0]), 'p': float(stats.pearsonr(ts, hb)[1])},
        'posdis_holdout': {'r': float(stats.pearsonr(pd, hb)[0]), 'p': float(stats.pearsonr(pd, hb)[1])},
        'bosdis_holdout': {'r': float(stats.pearsonr(bd, hb)[0]), 'p': float(stats.pearsonr(bd, hb)[1])},
        'bosdis_posdis': {'r': float(stats.pearsonr(bd, pd)[0]), 'p': float(stats.pearsonr(bd, pd)[1])},
        'bosdis_topsim': {'r': float(stats.pearsonr(bd, ts)[0]), 'p': float(stats.pearsonr(bd, ts)[1])},
    }

    if len(ts_2ag) > 5:
        save_data['correlations_2agent_ramp'] = {
            'n_seeds': len(ts_2ag),
            'topsim_posdis': {'r': float(stats.pearsonr(ts_2ag, pd_2ag)[0]), 'p': float(stats.pearsonr(ts_2ag, pd_2ag)[1])},
            'topsim_holdout': {'r': float(stats.pearsonr(ts_2ag, hb_2ag)[0]), 'p': float(stats.pearsonr(ts_2ag, hb_2ag)[1])},
            'posdis_holdout': {'r': float(stats.pearsonr(pd_2ag, hb_2ag)[0]), 'p': float(stats.pearsonr(pd_2ag, hb_2ag)[1])},
        }

    if comp_ts and hol_ts:
        save_data['compositional_vs_holistic'] = {
            'n_compositional': len(comp_ts),
            'n_holistic': len(hol_ts),
            'comp_topsim_mean': float(np.mean(comp_ts)),
            'comp_topsim_std': float(np.std(comp_ts)),
            'hol_topsim_mean': float(np.mean(hol_ts)),
            'hol_topsim_std': float(np.std(hol_ts)),
            'ttest_t': float(t),
            'ttest_p': float(p),
            'cohens_d': float(d),
        }

    save_path = RESULTS_DIR / 'phase84a_complementary_metrics.json'
    with open(save_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Saved {save_path}", flush=True)
    print("\nPhase 84a complete.", flush=True)
