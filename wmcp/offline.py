"""Offline protocol packaging — single portable .wmcp file."""

import io
import json
import time
import zipfile
import torch
from typing import Dict, Optional
from pathlib import Path


WMCP_MAGIC = b"WMCP"
WMCP_FORMAT_VERSION = 1


def export_protocol(protocol, domain: str, metadata: Optional[Dict] = None,
                    output_path: str = "protocol.wmcp") -> str:
    """Package a trained protocol into a portable .wmcp file.

    The .wmcp format is a ZIP archive containing:
    - manifest.json: protocol metadata, version, domain info
    - weights.pt: model state dict (safetensors planned for v0.2)
    - compliance.json: last compliance report (if available)

    Args:
        protocol: Trained Protocol instance.
        domain: Domain identifier (e.g., "physics_spring").
        metadata: Optional additional metadata.
        output_path: Output file path.

    Returns:
        Path to the saved .wmcp file.
    """
    manifest = {
        "wmcp_format_version": WMCP_FORMAT_VERSION,
        "protocol_version": "0.1.0",
        "domain": domain,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_agents": protocol.n_agents,
        "vocab_size": protocol.vocab_size,
        "n_heads": protocol.n_heads,
        "msg_dim": protocol.msg_dim,
        "total_params": sum(p.numel() for p in protocol.parameters()),
        "agent_configs": [
            {
                "input_dim": s.projection.temporal[0].in_channels,
                "n_frames": "auto",
            }
            for s in protocol.senders
        ],
    }
    if metadata:
        manifest["metadata"] = metadata

    # Save weights to buffer
    weights_buf = io.BytesIO()
    torch.save(protocol.state_dict(), weights_buf)
    weights_buf.seek(0)

    # Create ZIP archive
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))
        zf.writestr("weights.pt", weights_buf.read())

    return output_path


def load_protocol(file_path: str, device: str = "cpu"):
    """Load a protocol from a .wmcp file.

    Args:
        file_path: Path to .wmcp file.
        device: Torch device.

    Returns:
        (protocol, manifest) tuple.
    """
    from wmcp.protocol import Protocol

    with zipfile.ZipFile(file_path, 'r') as zf:
        manifest = json.loads(zf.read("manifest.json"))
        weights_buf = io.BytesIO(zf.read("weights.pt"))

    # Reconstruct protocol from manifest
    agent_configs = [
        (cfg["input_dim"], 4)  # default n_frames=4
        for cfg in manifest["agent_configs"]
    ]

    protocol = Protocol(
        agent_configs,
        vocab_size=manifest["vocab_size"],
        n_heads=manifest["n_heads"],
    )

    state_dict = torch.load(weights_buf, map_location=device, weights_only=True)
    protocol.load_state_dict(state_dict)
    protocol = protocol.to(device).eval()

    return protocol, manifest


def inspect_wmcp(file_path: str) -> Dict:
    """Inspect a .wmcp file without loading weights.

    Args:
        file_path: Path to .wmcp file.

    Returns:
        Manifest dict.
    """
    with zipfile.ZipFile(file_path, 'r') as zf:
        manifest = json.loads(zf.read("manifest.json"))
        file_list = zf.namelist()
        sizes = {name: zf.getinfo(name).file_size for name in file_list}

    manifest["_files"] = sizes
    manifest["_total_size_kb"] = sum(sizes.values()) / 1024
    return manifest
