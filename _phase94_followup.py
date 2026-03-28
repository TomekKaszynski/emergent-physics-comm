"""
Phase 94 follow-up: Run fall and ramp sweeps after spring completes.
Waits for results/phase94_full_sweep.json to contain end_time, then launches.

Run:
  PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=1 python3 _phase94_followup.py
"""
import time, json, sys, os
from pathlib import Path

RESULTS_DIR = Path("results")


def wait_for_spring():
    """Wait for the spring sweep to finish."""
    spring_path = RESULTS_DIR / "phase94_full_sweep.json"
    print("Waiting for spring sweep to complete...", flush=True)

    while True:
        if spring_path.exists():
            try:
                with open(spring_path) as f:
                    data = json.load(f)
                if data.get("metadata", {}).get("end_time") is not None:
                    n = data["metadata"]["completed_runs"]
                    print(f"Spring sweep complete: {n} runs. Proceeding.", flush=True)
                    return
            except (json.JSONDecodeError, KeyError):
                pass
        time.sleep(60)  # Check every minute


def main():
    wait_for_spring()

    from _phase94_sweep import run_phase94

    # Run fall
    print("\n" + "="*70, flush=True)
    print("Starting FALL sweep", flush=True)
    print("="*70, flush=True)
    try:
        run_phase94(scenario="fall")
    except Exception as e:
        print(f"FALL sweep failed: {e}", flush=True)
        import traceback
        traceback.print_exc()

    # Clear MPS cache
    import torch
    torch.mps.empty_cache()

    # Run ramp
    print("\n" + "="*70, flush=True)
    print("Starting RAMP sweep", flush=True)
    print("="*70, flush=True)
    try:
        run_phase94(scenario="ramp")
    except Exception as e:
        print(f"RAMP sweep failed: {e}", flush=True)
        import traceback
        traceback.print_exc()

    print("\n" + "="*70, flush=True)
    print("ALL SCENARIOS COMPLETE", flush=True)
    print("="*70, flush=True)


if __name__ == "__main__":
    main()
