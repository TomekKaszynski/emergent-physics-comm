"""Protocol diff tool — compare two protocol versions or instances."""

import json
import numpy as np
import torch
from typing import Dict, Optional
from pathlib import Path
from wmcp.offline import inspect_wmcp, load_protocol


def diff_protocols(old_path: str, new_path: str,
                   test_features: Optional[Dict] = None) -> Dict:
    """Compare two .wmcp protocol files.

    Args:
        old_path: Path to old protocol .wmcp file.
        new_path: Path to new protocol .wmcp file.
        test_features: Optional dict with 'views' for accuracy comparison.

    Returns:
        Dict with structural and semantic differences.
    """
    old_info = inspect_wmcp(old_path)
    new_info = inspect_wmcp(new_path)

    result = {
        "old_version": old_info.get("protocol_version", "unknown"),
        "new_version": new_info.get("protocol_version", "unknown"),
        "structural_changes": [],
        "semantic_changes": [],
        "compatible": True,
    }

    # Check structural compatibility
    if old_info.get("vocab_size") != new_info.get("vocab_size"):
        result["structural_changes"].append(
            f"vocab_size changed: {old_info.get('vocab_size')} → {new_info.get('vocab_size')}")
        result["compatible"] = False

    if old_info.get("n_heads") != new_info.get("n_heads"):
        result["structural_changes"].append(
            f"n_heads changed: {old_info.get('n_heads')} → {new_info.get('n_heads')}")
        result["compatible"] = False

    if old_info.get("n_agents") != new_info.get("n_agents"):
        result["structural_changes"].append(
            f"n_agents changed: {old_info.get('n_agents')} → {new_info.get('n_agents')}")

    # Check agent configs
    old_agents = old_info.get("agent_configs", [])
    new_agents = new_info.get("agent_configs", [])

    added = len(new_agents) - len(old_agents)
    if added > 0:
        result["structural_changes"].append(f"{added} agent(s) added")
    elif added < 0:
        result["structural_changes"].append(f"{-added} agent(s) removed")

    for i, (old_a, new_a) in enumerate(zip(old_agents, new_agents)):
        if old_a.get("input_dim") != new_a.get("input_dim"):
            result["structural_changes"].append(
                f"agent {i} input_dim changed: {old_a.get('input_dim')} → {new_a.get('input_dim')}")

    # Size comparison
    old_size = old_info.get("_total_size_kb", 0)
    new_size = new_info.get("_total_size_kb", 0)
    result["size_change_kb"] = new_size - old_size

    # Domain check
    if old_info.get("domain") != new_info.get("domain"):
        result["semantic_changes"].append(
            f"domain changed: {old_info.get('domain')} → {new_info.get('domain')}")

    if not result["structural_changes"]:
        result["structural_changes"].append("No structural changes")
    if not result["semantic_changes"]:
        result["semantic_changes"].append("No semantic changes detected (weight comparison requires test data)")

    return result


def diff_manifests(old_manifest: Dict, new_manifest: Dict) -> Dict:
    """Compare two protocol manifests (without loading weights)."""
    changes = []
    for key in set(list(old_manifest.keys()) + list(new_manifest.keys())):
        if key.startswith("_"):
            continue
        old_val = old_manifest.get(key)
        new_val = new_manifest.get(key)
        if old_val != new_val:
            changes.append({"field": key, "old": old_val, "new": new_val})
    return {"changes": changes, "n_changes": len(changes)}
