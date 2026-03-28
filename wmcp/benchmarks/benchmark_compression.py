"""Bandwidth compression benchmark."""

import json
import math
from wmcp.adaptive import compute_bandwidth


def run(output_path: str = None) -> dict:
    """Compute compression ratios across configurations."""
    results = []
    for K in [2, 3, 5, 8, 16, 32]:
        for n_agents in [2, 4]:
            bw = compute_bandwidth(K, L=2, n_agents=n_agents)
            results.append(bw)

    result = {"benchmark": "compression", "results": results}
    if output_path:
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
    return result


if __name__ == "__main__":
    r = run()
    for entry in r["results"]:
        if entry["n_agents"] == 2:
            print(f"  K={entry['K']:2d} n={entry['n_agents']}: "
                  f"{entry['total_bits']:.1f} bits, "
                  f"V-JEPA compression={entry['compression_ratios']['V-JEPA 2']:.0f}×")
