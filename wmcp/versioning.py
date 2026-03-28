"""Protocol versioning and compatibility negotiation."""

from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class ProtocolVersion:
    """Semantic version for WMCP protocol instances."""
    major: int = 0
    minor: int = 1
    patch: int = 0

    @property
    def string(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    @classmethod
    def parse(cls, version_str: str) -> "ProtocolVersion":
        parts = version_str.split(".")
        return cls(int(parts[0]), int(parts[1]), int(parts[2]) if len(parts) > 2 else 0)

    def is_compatible(self, other: "ProtocolVersion") -> bool:
        """Check if two versions can communicate.
        Same major version = compatible. Different major = breaking change."""
        return self.major == other.major

    def __str__(self) -> str:
        return self.string


def negotiate_version(agent_a_version: str, agent_b_version: str
                      ) -> Tuple[bool, str]:
    """Negotiate protocol version between two agents.

    Args:
        agent_a_version: Version string of agent A.
        agent_b_version: Version string of agent B.

    Returns:
        (compatible, negotiated_version) — the lower version is used.
    """
    va = ProtocolVersion.parse(agent_a_version)
    vb = ProtocolVersion.parse(agent_b_version)

    compatible = va.is_compatible(vb)
    # Use the lower version for communication
    if (va.major, va.minor, va.patch) <= (vb.major, vb.minor, vb.patch):
        negotiated = va.string
    else:
        negotiated = vb.string

    return compatible, negotiated


@dataclass
class MessageHeader:
    """Protocol message header with version information."""
    version: str = "0.1.0"
    agent_id: int = 0
    sequence: int = 0
    vocab_size: int = 3
    n_positions: int = 2
    domain: str = ""

    def validate_against(self, other: "MessageHeader") -> bool:
        """Check if two message headers are compatible."""
        va = ProtocolVersion.parse(self.version)
        vb = ProtocolVersion.parse(other.version)
        if not va.is_compatible(vb):
            return False
        if self.vocab_size != other.vocab_size:
            return False
        if self.n_positions != other.n_positions:
            return False
        return True


def check_migration_path(from_version: str, to_version: str) -> dict:
    """Check if a protocol can be migrated between versions.

    Args:
        from_version: Current version string.
        to_version: Target version string.

    Returns:
        Dict with migration info: compatible, breaking_changes, migration_steps.
    """
    vf = ProtocolVersion.parse(from_version)
    vt = ProtocolVersion.parse(to_version)

    if vf.major == vt.major:
        return {
            "compatible": True,
            "breaking_changes": [],
            "migration_steps": ["Update version metadata"],
            "retrain_required": False,
        }
    else:
        return {
            "compatible": False,
            "breaking_changes": [
                f"Major version change ({vf.major} → {vt.major})",
                "Message format may be incompatible",
            ],
            "migration_steps": [
                "Export projection layer weights",
                "Retrain with new protocol version",
                "Validate compositionality (PosDis > 0.5)",
            ],
            "retrain_required": True,
        }
