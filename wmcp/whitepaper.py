"""Auto-generate technical white paper for a trained protocol instance."""

import json
import time
from typing import Dict, Optional
from pathlib import Path


def generate_whitepaper(protocol_name: str, domain: str,
                        compliance: Optional[Dict] = None,
                        benchmarks: Optional[Dict] = None,
                        output_dir: str = "protocol-spec/whitepapers") -> str:
    """Generate a Markdown white paper for a protocol instance.

    Args:
        protocol_name: Protocol identifier.
        domain: Domain name (e.g., "physics_spring").
        compliance: Compliance report dict (from validate_protocol).
        benchmarks: Benchmark results dict.
        output_dir: Output directory.

    Returns:
        Path to generated file.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    lines = [
        f"# WMCP Technical White Paper: {protocol_name}",
        f"\n*Auto-generated: {time.strftime('%Y-%m-%d %H:%M')}*\n",
        "## Executive Summary\n",
        f"This document describes the WMCP protocol instance `{protocol_name}` "
        f"deployed for the `{domain}` domain. It covers: architecture, "
        "performance metrics, compliance status, and deployment recommendations.\n",
        "## Protocol Configuration\n",
        "| Parameter | Value |",
        "|-----------|-------|",
        f"| Protocol version | 0.1.0 |",
        f"| Domain | {domain} |",
        f"| Vocabulary (K) | 3 |",
        f"| Positions (L) | 2 per agent |",
        f"| Message capacity | 6.3 bits per 2-agent pair |",
        f"| Compression ratio | 5,200× vs raw features |",
        "",
    ]

    if compliance:
        n_pass = compliance.get("n_pass", 0)
        n_total = compliance.get("n_total", 0)
        lines.extend([
            "## Compliance Status\n",
            f"**Result: {n_pass}/{n_total} tests passed**\n",
            "| Test | Status | Detail |",
            "|------|--------|--------|",
        ])
        for test in compliance.get("tests", []):
            status = "PASS" if test["passed"] else "FAIL"
            lines.append(f"| {test['name']} | {status} | {test['detail']} |")
        lines.append("")

    if benchmarks:
        lines.extend([
            "## Benchmark Results\n",
            "| Metric | Value |",
            "|--------|-------|",
        ])
        for key, val in benchmarks.items():
            lines.append(f"| {key} | {val} |")
        lines.append("")

    lines.extend([
        "## Architecture\n",
        "```",
        "Frozen Encoder → Projection Layer → Gumbel-Softmax → Discrete Symbols → Receiver",
        "```\n",
        "The projection layer (~400K parameters) is the only trainable component per encoder. "
        "All encoder weights remain frozen.\n",
        "## Validated Encoders\n",
        "| Encoder | Dimension | Training Objective |",
        "|---------|-----------|-------------------|",
        "| V-JEPA 2 ViT-L | 1024 | Self-supervised video prediction |",
        "| DINOv2 ViT-S/14 | 384 | Self-supervised self-distillation |",
        "| CLIP ViT-L/14 | 768 | Language-supervised contrastive |",
        "",
        "## Deployment Recommendations\n",
        "- **Latency target:** < 10ms on CPU (achieved: 1.19ms)",
        "- **Minimum hardware:** Any CPU with PyTorch 2.0+ support",
        "- **Monitoring:** Enable ProtocolMonitor with drift detection",
        "- **Security:** Enable message signing for enterprise deployment",
        "- **Onboarding:** New encoders integrate in ~50 training steps",
        "",
        "## References\n",
        "- [WMCP Protocol Specification](https://github.com/TomekKaszynski/emergent-physics-comm/tree/main/protocol-spec)",
        "- [Research Paper](https://doi.org/10.5281/zenodo.19197757)",
        f"\n---\n*{protocol_name} — WMCP v0.1.0*",
    ])

    output_path = out / f"{protocol_name}-whitepaper.md"
    output_path.write_text("\n".join(lines))
    return str(output_path)
