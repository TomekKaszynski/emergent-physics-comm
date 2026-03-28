"""Multi-domain protocol router — classify and route to correct protocol."""

import time
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from wmcp.offline import load_protocol


class ProtocolRouter:
    """Routes input features to the correct domain protocol.

    Loads multiple .wmcp domain packs and classifies inputs by
    comparing feature statistics to domain signatures.

    Usage:
        router = ProtocolRouter()
        router.add_domain("spring", "spring.wmcp", signature_features)
        router.add_domain("ramp", "ramp.wmcp", signature_features)
        domain, protocol, confidence = router.route(input_features)
    """

    def __init__(self):
        self._domains: Dict[str, Dict] = {}

    def add_domain(self, name: str, wmcp_path: Optional[str] = None,
                   signature: Optional[np.ndarray] = None,
                   protocol=None):
        """Register a domain protocol.

        Args:
            name: Domain identifier (e.g., "spring", "ramp").
            wmcp_path: Path to .wmcp file (lazy-loaded).
            signature: (D,) mean feature vector for domain classification.
            protocol: Pre-loaded Protocol instance (alternative to wmcp_path).
        """
        self._domains[name] = {
            "path": wmcp_path,
            "signature": signature,
            "protocol": protocol,
            "loaded": protocol is not None,
        }

    def route(self, features: torch.Tensor) -> Tuple[str, object, float]:
        """Route input features to the best domain protocol.

        Args:
            features: (1, T, D) or (D,) input features.

        Returns:
            (domain_name, protocol, confidence) tuple.
        """
        if not self._domains:
            raise ValueError("No domains registered")

        # Compute mean feature for classification
        if features.dim() == 3:
            feat_mean = features.mean(dim=(0, 1)).numpy()
        elif features.dim() == 2:
            feat_mean = features.mean(dim=0).numpy()
        else:
            feat_mean = features.numpy()

        # Find closest domain by cosine similarity
        best_domain = None
        best_sim = -1

        for name, domain in self._domains.items():
            if domain["signature"] is None:
                continue
            sig = domain["signature"]
            # Truncate to min dimension
            min_d = min(len(feat_mean), len(sig))
            cos_sim = np.dot(feat_mean[:min_d], sig[:min_d]) / (
                np.linalg.norm(feat_mean[:min_d]) * np.linalg.norm(sig[:min_d]) + 1e-10)
            if cos_sim > best_sim:
                best_sim = cos_sim
                best_domain = name

        if best_domain is None:
            best_domain = list(self._domains.keys())[0]
            best_sim = 0.0

        # Lazy load protocol
        domain = self._domains[best_domain]
        if not domain["loaded"] and domain["path"]:
            protocol, _ = load_protocol(domain["path"])
            domain["protocol"] = protocol
            domain["loaded"] = True

        return best_domain, domain["protocol"], float(best_sim)

    @property
    def domains(self) -> List[str]:
        return list(self._domains.keys())

    @property
    def info(self) -> Dict:
        return {
            "n_domains": len(self._domains),
            "domains": {name: {"loaded": d["loaded"], "has_signature": d["signature"] is not None}
                        for name, d in self._domains.items()},
        }
