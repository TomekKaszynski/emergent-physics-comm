"""
Phase 94 resume: Finish fall sweep (from run 420+) then run ramp.

Run:
  PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=1 python3 _phase94_resume.py
"""
import time, json, sys, os, traceback
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from _phase94_sweep import (
    load_features, build_run_queue, make_agent_configs, single_run,
    generate_summary_tables, generate_heatmaps, generate_advantage_plot,
    _save_checkpoint, RESULTS_DIR
)


def resume_fall():
    """Resume the fall sweep from the last checkpoint."""
    print("═══ Resuming FALL sweep ═══", flush=True)

    save_path = RESULTS_DIR / "phase94_fall_sweep.json"

    # Load existing checkpoint
    with open(save_path) as f:
        data = json.load(f)

    existing_results = data["results"]
    completed_keys = set()
    for r in existing_results:
        key = (r["pairing"], r["K"], r["n_agents"], r["seed"])
        completed_keys.add(key)

    print(f"  Loaded checkpoint: {len(existing_results)} existing runs", flush=True)

    # Build full queue and find remaining
    queue = build_run_queue("fall")
    remaining = []
    for pairing, K, n_agents, scenario, seed, priority in queue:
        key = (pairing, K, n_agents, seed)
        if key not in completed_keys:
            remaining.append((pairing, K, n_agents, scenario, seed, priority))

    print(f"  Remaining: {len(remaining)} runs", flush=True)

    if not remaining:
        print("  Fall already complete!", flush=True)
        return existing_results

    # Load features
    vjepa_feat, dino_temporal, obj_names, mass_values = load_features("fall")

    results_list = list(existing_results)
    completed = sum(1 for r in results_list if r.get("status") == "success")
    failed = sum(1 for r in results_list if r.get("status") == "failed")
    start_time = datetime.now()
    t0 = time.time()

    for run_idx, (pairing, K, n_agents, scenario, seed, priority) in enumerate(remaining):
        try:
            configs = make_agent_configs(pairing, n_agents, vjepa_feat, dino_temporal)
            result = single_run(configs, mass_values, obj_names, K, seed)
            result.update({
                "pairing": pairing, "K": K, "n_agents": n_agents,
                "scenario": scenario, "seed": seed, "priority": priority,
            })
            results_list.append(result)
            if result["status"] == "success":
                completed += 1
            else:
                failed += 1
        except Exception as e:
            failed += 1
            results_list.append({
                "pairing": pairing, "K": K, "n_agents": n_agents,
                "scenario": scenario, "seed": seed, "priority": priority,
                "status": "failed", "error": str(e),
            })

        if (run_idx + 1) % 10 == 0 or run_idx == len(remaining) - 1:
            elapsed = time.time() - t0
            print(f"  Resume: {run_idx+1}/{len(remaining)} done "
                  f"(total {completed}/{len(results_list)}). "
                  f"Elapsed: {elapsed/60:.1f}min", flush=True)

    # Final save with end_time
    _save_checkpoint(results_list, completed, failed,
                     datetime.fromisoformat(data["metadata"]["start_time"]),
                     save_path, end_time=datetime.now())

    # Generate outputs
    suffix = "_fall"
    tables_md = generate_summary_tables(results_list)
    with open(RESULTS_DIR / f"phase94{suffix}_tables.md", "w") as f:
        f.write(tables_md)
    generate_heatmaps(results_list, "fall")
    generate_advantage_plot(results_list, "fall")
    print(f"  Fall sweep complete: {completed} runs", flush=True)

    torch.mps.empty_cache()
    return results_list


def run_ramp():
    """Run the full ramp sweep."""
    print("\n═══ Starting RAMP sweep ═══", flush=True)
    from _phase94_sweep import run_phase94
    return run_phase94(scenario="ramp")


if __name__ == "__main__":
    resume_fall()
    run_ramp()
    print("\n" + "="*60, flush=True)
    print("ALL SCENARIOS COMPLETE", flush=True)
    print("="*60, flush=True)
