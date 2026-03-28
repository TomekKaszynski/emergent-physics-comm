"""
WMCP Publisher Node — Encodes scenes and publishes discrete messages.

ROS 2 compatible interface. Uses rclpy if available, otherwise
pure Python pub-sub simulation with identical API.

Usage:
    python wmcp_ros2/publisher_node.py
"""

import time
import threading
import queue
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable

from wmcp_ros2.wmcp_msg import WMCPMessage

# Check for ROS 2
try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
    HAS_ROS2 = True
except ImportError:
    HAS_ROS2 = False


class WMCPProjection(nn.Module):
    """Minimal projection for the publisher node."""
    def __init__(self, input_dim, hidden_dim=128, n_frames=4, vocab_size=3, n_heads=2):
        super().__init__()
        ks = min(3, max(1, n_frames))
        self.temporal = nn.Sequential(
            nn.Conv1d(input_dim, 256, ks, padding=ks//2), nn.ReLU(),
            nn.Conv1d(256, 128, ks, padding=ks//2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1))
        self.fc = nn.Sequential(nn.Linear(128, hidden_dim), nn.ReLU())
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, vocab_size) for _ in range(n_heads)])
        self.vocab_size = vocab_size

    def encode(self, features: torch.Tensor) -> list:
        """Encode features to integer tokens."""
        self.eval()
        with torch.no_grad():
            h = self.fc(self.temporal(features.permute(0, 2, 1)).squeeze(-1))
            return [head(h).argmax(-1).item() for head in self.heads]


class WMCPPublisher:
    """Publishes WMCP messages to a topic.

    If ROS 2 is available, publishes to /wmcp/messages as String.
    Otherwise, uses a thread-safe queue as the message bus.
    """

    def __init__(self, agent_id: int, encoder_type: str, input_dim: int,
                 domain: str = "physics_spring", bus: Optional[queue.Queue] = None):
        self.agent_id = agent_id
        self.encoder_type = encoder_type
        self.domain = domain
        self.projection = WMCPProjection(input_dim)
        self.seq = 0
        self._bus = bus

        if HAS_ROS2 and bus is None:
            rclpy.init()
            self._node = rclpy.create_node(f'wmcp_publisher_{agent_id}')
            self._pub = self._node.create_publisher(String, '/wmcp/messages', 10)
        else:
            self._node = None
            self._pub = None

    def publish(self, features: torch.Tensor) -> WMCPMessage:
        """Encode features and publish a message."""
        tokens = self.projection.encode(features)
        self.seq += 1

        msg = WMCPMessage(
            agent_id=self.agent_id,
            sequence_id=self.seq,
            tokens=tokens,
            encoder_type=self.encoder_type,
            domain=self.domain,
        )

        if self._pub is not None:
            ros_msg = String()
            ros_msg.data = msg.to_json()
            self._pub.publish(ros_msg)
        elif self._bus is not None:
            self._bus.put(msg.to_json())

        return msg

    def shutdown(self):
        if self._node:
            self._node.destroy_node()
            rclpy.shutdown()
