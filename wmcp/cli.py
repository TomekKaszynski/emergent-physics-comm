"""WMCP command-line interface."""

import argparse
import sys
import json
import time
import wmcp


def cmd_info(args):
    """Print protocol info."""
    print(f"WMCP — World Model Communication Protocol")
    print(f"Version: {wmcp.__version__}")
    print(f"Spec: https://github.com/TomekKaszynski/emergent-physics-comm/tree/main/protocol-spec")
    print(f"Paper: https://doi.org/10.5281/zenodo.19197757")
    print(f"\nDefault configuration:")
    print(f"  Vocabulary (K): 3")
    print(f"  Positions (L):  2 per agent")
    print(f"  Hidden dim:     128")
    print(f"\nValidated encoders:")
    print(f"  V-JEPA 2 ViT-L  (1024-dim, temporal SSL)")
    print(f"  DINOv2 ViT-S/14 (384-dim, spatial SSL)")
    print(f"  CLIP ViT-L/14   (768-dim, language-supervised)")


def cmd_validate(args):
    """Run compliance validation."""
    import torch
    from wmcp.compliance import validate_protocol

    if not args.model_path:
        print("Error: --model-path required")
        sys.exit(1)

    print(f"Loading protocol from {args.model_path}...")
    data = torch.load(args.model_path, weights_only=False)

    if "protocol" not in data or "agent_views" not in data:
        print("Error: model file must contain 'protocol' and 'agent_views' keys")
        sys.exit(1)

    result = validate_protocol(
        data["protocol"], data["agent_views"],
        data["mass_values"], data["obj_names"])

    for test in result["tests"]:
        status = "PASS" if test["passed"] else "FAIL"
        print(f"  [{status}] {test['name']}: {test['detail']}")

    print(f"\n{result['n_pass']}/{result['n_total']} tests passed")
    sys.exit(0 if result["all_pass"] else 1)


def cmd_benchmark(args):
    """Run latency benchmark."""
    import torch
    import numpy as np
    from wmcp.protocol import Protocol

    print("Building test protocol (2-agent, K=3)...")
    protocol = Protocol([(1024, 4), (384, 4)], vocab_size=3)
    protocol.eval()

    n = 1000
    views_a = [torch.randn(1, 4, 1024), torch.randn(1, 4, 384)]
    views_b = [torch.randn(1, 4, 1024), torch.randn(1, 4, 384)]

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            protocol.communicate(views_a, views_b)

    latencies = []
    for _ in range(n):
        t = time.perf_counter()
        with torch.no_grad():
            protocol.communicate(views_a, views_b)
        latencies.append((time.perf_counter() - t) * 1000)

    lats = np.array(latencies)
    print(f"\nLatency ({n} iterations):")
    print(f"  Mean:   {np.mean(lats):.2f}ms")
    print(f"  Median: {np.median(lats):.2f}ms")
    print(f"  P95:    {np.percentile(lats, 95):.2f}ms")
    print(f"  P99:    {np.percentile(lats, 99):.2f}ms")
    print(f"  Throughput: {1000/np.mean(lats):.0f} comms/s")


def cmd_onboard(args):
    """Onboard new encoder."""
    print(f"Onboarding: --encoder={args.encoder} --protocol={args.protocol} "
          f"--steps={args.steps}")
    print("This command requires pre-extracted features.")
    print("See: protocol-spec/ONBOARDING.md for the full procedure.")


def main():
    parser = argparse.ArgumentParser(prog="wmcp",
                                      description="WMCP CLI")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("info", help="Print protocol info")

    p_val = sub.add_parser("validate", help="Run compliance suite")
    p_val.add_argument("--model-path", type=str, help="Path to saved protocol")

    p_bench = sub.add_parser("benchmark", help="Run latency benchmark")
    p_bench.add_argument("--protocol", type=str, help="Path to protocol (optional)")

    p_onb = sub.add_parser("onboard", help="Onboard new encoder")
    p_onb.add_argument("--encoder", type=str, required=True)
    p_onb.add_argument("--protocol", type=str, required=True)
    p_onb.add_argument("--steps", type=int, default=50)

    p_export = sub.add_parser("export", help="Export protocol to .wmcp file")
    p_export.add_argument("--protocol", type=str, required=True)
    p_export.add_argument("--domain", type=str, default="physics_spring")
    p_export.add_argument("--output", type=str, default="protocol.wmcp")

    p_load = sub.add_parser("load", help="Inspect a .wmcp file")
    p_load.add_argument("--file", type=str, required=True)

    args = parser.parse_args()

    if args.command == "info":
        cmd_info(args)
    elif args.command == "validate":
        cmd_validate(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)
    elif args.command == "onboard":
        cmd_onboard(args)
    elif args.command == "export":
        print(f"Export: --protocol={args.protocol} --domain={args.domain} "
              f"--output={args.output}")
        print("See: wmcp.offline.export_protocol() for programmatic use.")
    elif args.command == "load":
        from wmcp.offline import inspect_wmcp
        info = inspect_wmcp(args.file)
        import json
        print(json.dumps(info, indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
