"""
WMCP — World Model Communication Protocol
==========================================
Discrete compositional communication between heterogeneous vision foundation models.

Quick start:
    from wmcp import Protocol, ProjectionLayer, GumbelSoftmaxBottleneck
    from wmcp.metrics import compute_posdis, compute_topsim, compute_bosdis
    from wmcp.compliance import validate_protocol
"""

__version__ = "0.1.0"

from wmcp.protocol import Protocol
from wmcp.projection import ProjectionLayer
from wmcp.bottleneck import GumbelSoftmaxBottleneck

__all__ = ["Protocol", "ProjectionLayer", "GumbelSoftmaxBottleneck"]
