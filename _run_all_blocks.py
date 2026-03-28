"""Master runner: Block 1 tests + Block 2 experiments + Block 3 infrastructure."""
import time, os, sys, subprocess

def run(name, cmd):
    print(f"\n{'='*70}\n  {name}\n{'='*70}", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, shell=True, env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTORCH_ENABLE_MPS_FALLBACK": "1"})
    m = (time.time() - t0) / 60
    s = "OK" if r.returncode == 0 else "FAILED"
    print(f"  {name}: {s} ({m:.1f}min)", flush=True)
    return r.returncode == 0

if __name__ == "__main__":
    t = time.time()
    results = {}
    # Block 1: Package tests
    results["B1_tests"] = run("Block 1: pytest", "python3 -m pytest wmcp/tests/ -v --tb=short")
    results["B1_cli"] = run("Block 1: CLI", "python3 -m wmcp.cli info")
    # Block 2: Experiments
    results["B2_phases"] = run("Block 2: Phases 110-115",
                                "python3 -c 'from _phase110_115 import run_all; run_all()'")
    # Block 3: Infrastructure
    results["B3_infra"] = run("Block 3: Dashboard + Figures + Compat",
                               "python3 -c 'from _block3_infra import run_all; run_all()'")
    total = (time.time() - t) / 60
    print(f"\n{'='*70}\n  ALL BLOCKS COMPLETE. Total: {total:.1f}min\n{'='*70}", flush=True)
    for k, v in results.items():
        print(f"  {k}: {'OK' if v else 'FAILED'}", flush=True)
