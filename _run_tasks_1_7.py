"""
Tasks 1-7 runner: Protocol spec integration, demos, phases 107-109, README overhaul.
Run sequentially.

Usage:
  PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=1 python3 _run_tasks_1_7.py
"""

import time, os, sys, subprocess, traceback

def run_task(name, cmd):
    print(f"\n{'='*70}", flush=True)
    print(f"  TASK: {name}", flush=True)
    print(f"{'='*70}", flush=True)
    t0 = time.time()
    try:
        result = subprocess.run(cmd, shell=True, capture_output=False,
                                env={**os.environ, "PYTHONUNBUFFERED": "1",
                                     "PYTORCH_ENABLE_MPS_FALLBACK": "1"})
        elapsed = (time.time() - t0) / 60
        status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
        print(f"\n  {name}: {status} ({elapsed:.1f}min)", flush=True)
        return result.returncode == 0
    except Exception as e:
        print(f"\n  {name}: EXCEPTION: {e}", flush=True)
        return False


if __name__ == "__main__":
    t_total = time.time()
    results = {}

    # Task 1: Compliance test
    results["T1_compliance"] = run_task(
        "Task 1: Spec Compliance Test",
        "python3 protocol-spec/tests/test_spec_compliance.py")

    # Task 2: Onboarding demo (dry run for speed, full already validated in Phase 104)
    results["T2_onboarding"] = run_task(
        "Task 2: Onboarding Demo",
        "python3 protocol-spec/examples/onboard_new_encoder.py")

    # Task 3: Pub-sub demo
    results["T3_pubsub"] = run_task(
        "Task 3: Pub-Sub Demo",
        "python3 protocol-spec/examples/pubsub_demo.py")

    # Task 4-6: Phases 107-109
    results["T4_T5_T6_phases"] = run_task(
        "Tasks 4-6: Phases 107-109",
        "python3 -c 'from _phase107_109 import run_all; run_all()'")

    # Task 7: README overhaul (done by the calling process after this script)
    print(f"\n  Task 7 (README overhaul) will be done by the calling process.", flush=True)

    total = (time.time() - t_total) / 60
    print(f"\n{'='*70}", flush=True)
    print(f"  ALL TASKS COMPLETE. Total: {total:.1f} minutes", flush=True)
    for name, ok in results.items():
        print(f"  {name}: {'OK' if ok else 'FAILED'}", flush=True)
    print(f"{'='*70}", flush=True)
