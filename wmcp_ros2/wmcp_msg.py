"""
WMCP ROS 2 Message Definition (Pure Python)
=============================================
Defines the WMCPMessage format compatible with ROS 2 conventions.
If rclpy is available, wraps as proper ROS 2 messages.
Otherwise, provides a pure Python equivalent that follows the same interface.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import time
import json


@dataclass
class WMCPMessage:
    """WMCP protocol message — ROS 2 compatible format.

    Equivalent to the following .msg definition:
        # WMCPMessage.msg
        std_msgs/Header header
        uint8 protocol_version_major
        uint8 protocol_version_minor
        uint8 protocol_version_patch
        uint8 vocab_size          # K
        uint8 n_positions         # L
        uint8 agent_id
        string encoder_type       # "vjepa2", "dinov2", "clip"
        uint8[] tokens            # L integer tokens, each in [0, K-1]
        float32 timestamp
        string domain             # "physics_spring", "physics_ramp", etc.
    """
    # Header
    agent_id: int = 0
    sequence_id: int = 0
    timestamp: float = field(default_factory=time.time)

    # Protocol version
    version_major: int = 0
    version_minor: int = 1
    version_patch: int = 0

    # Message content
    vocab_size: int = 3       # K
    n_positions: int = 2      # L
    tokens: List[int] = field(default_factory=list)  # L integers in [0, K-1]

    # Metadata
    encoder_type: str = ""    # "vjepa2", "dinov2", "clip"
    domain: str = ""          # "physics_spring", etc.

    @property
    def version_string(self) -> str:
        return f"{self.version_major}.{self.version_minor}.{self.version_patch}"

    def to_json(self) -> str:
        """Serialize to JSON (wire format)."""
        return json.dumps({
            "agent_id": self.agent_id,
            "seq": self.sequence_id,
            "ts": self.timestamp,
            "version": self.version_string,
            "K": self.vocab_size,
            "L": self.n_positions,
            "tokens": self.tokens,
            "encoder": self.encoder_type,
            "domain": self.domain,
        })

    @classmethod
    def from_json(cls, data: str) -> "WMCPMessage":
        """Deserialize from JSON."""
        d = json.loads(data)
        v = d.get("version", "0.1.0").split(".")
        return cls(
            agent_id=d["agent_id"],
            sequence_id=d["seq"],
            timestamp=d["ts"],
            version_major=int(v[0]),
            version_minor=int(v[1]),
            version_patch=int(v[2]),
            vocab_size=d["K"],
            n_positions=d["L"],
            tokens=d["tokens"],
            encoder_type=d.get("encoder", ""),
            domain=d.get("domain", ""),
        )

    def validate(self) -> bool:
        """Check message is well-formed."""
        if len(self.tokens) != self.n_positions:
            return False
        if any(t < 0 or t >= self.vocab_size for t in self.tokens):
            return False
        return True


# ROS 2 .msg file content (for actual ROS 2 integration)
ROS2_MSG_DEFINITION = """
# wmcp_msgs/msg/WMCPMessage.msg
# WMCP protocol message for inter-agent communication

std_msgs/Header header

# Protocol version
uint8 version_major
uint8 version_minor
uint8 version_patch

# Message parameters
uint8 vocab_size          # K — symbols per position
uint8 n_positions         # L — message positions per agent

# Agent info
uint8 agent_id
string encoder_type       # "vjepa2", "dinov2", "clip"

# Payload
uint8[] tokens            # L integers, each in [0, K-1]

# Metadata
string domain             # "physics_spring", etc.
"""
