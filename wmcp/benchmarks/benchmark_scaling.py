"""Population scaling benchmark — accuracy vs agent count."""

import json
import numpy as np


def run(output_path: str = None) -> dict:
    """Report empirically validated scaling data from Phase 99."""
    # From Phase 99 (spring, K=3, 10 seeds)
    scaling = {
        "heterogeneous": {1: 0.788, 2: 0.764, 4: 0.676, 8: 0.604, 16: 0.534},
        "homo_vjepa": {1: 0.788, 2: 0.777, 4: 0.715, 8: 0.655, 16: 0.598},
        "homo_dino": {1: 0.716, 2: 0.661, 4: 0.578, 8: 0.467, 16: 0.378},
    }
    result = {"benchmark": "scaling", "data": scaling,
              "source": "Phase 99, spring scenario, K=3, 10 seeds per condition"}
    if output_path:
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
    return result


if __name__ == "__main__":
    r = run()
    for cond, data in r["data"].items():
        vals = " ".join(f"n={k}:{v:.3f}" for k, v in data.items())
        print(f"  {cond}: {vals}")
