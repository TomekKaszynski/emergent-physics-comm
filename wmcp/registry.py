"""Agent identity and registry — track who's on the network."""

import json
import uuid
import time
from typing import Dict, List, Optional
from pathlib import Path
from dataclasses import dataclass, field, asdict


@dataclass
class AgentRecord:
    """Registry entry for a protocol agent."""
    agent_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    name: str = ""
    encoder_type: str = ""  # "vjepa2", "dinov2", "clip"
    input_dim: int = 0
    certification_level: str = "none"  # "bronze", "silver", "gold"
    onboarding_date: str = field(default_factory=lambda: time.strftime("%Y-%m-%d"))
    last_heartbeat: float = field(default_factory=time.time)
    status: str = "active"  # "active", "quarantined", "offline"
    metadata: Dict = field(default_factory=dict)


class AgentRegistry:
    """Registry for tracking protocol agents.

    Usage:
        registry = AgentRegistry()
        registry.add("factory-arm-01", "dinov2", 384)
        registry.heartbeat("factory-arm-01")
        agents = registry.list()
    """

    def __init__(self, persist_path: Optional[str] = None):
        self._agents: Dict[str, AgentRecord] = {}
        self._persist_path = persist_path

        if persist_path and Path(persist_path).exists():
            self.load(persist_path)

    def add(self, name: str, encoder_type: str, input_dim: int,
            certification: str = "none", **metadata) -> AgentRecord:
        """Register a new agent."""
        record = AgentRecord(
            name=name,
            encoder_type=encoder_type,
            input_dim=input_dim,
            certification_level=certification,
            metadata=metadata,
        )
        self._agents[name] = record
        self._save()
        return record

    def remove(self, name: str):
        """Remove an agent from the registry."""
        self._agents.pop(name, None)
        self._save()

    def heartbeat(self, name: str):
        """Update agent's last heartbeat."""
        if name in self._agents:
            self._agents[name].last_heartbeat = time.time()
            self._agents[name].status = "active"

    def quarantine(self, name: str):
        """Mark agent as quarantined."""
        if name in self._agents:
            self._agents[name].status = "quarantined"

    def get(self, name: str) -> Optional[AgentRecord]:
        """Get agent record by name."""
        return self._agents.get(name)

    def list(self, status: Optional[str] = None) -> List[AgentRecord]:
        """List all agents, optionally filtered by status."""
        agents = list(self._agents.values())
        if status:
            agents = [a for a in agents if a.status == status]
        return agents

    def export_json(self) -> str:
        """Export registry as JSON."""
        return json.dumps(
            {name: asdict(record) for name, record in self._agents.items()},
            indent=2, default=str)

    def _save(self):
        if self._persist_path:
            Path(self._persist_path).write_text(self.export_json())

    def load(self, path: str):
        """Load registry from JSON file."""
        data = json.loads(Path(path).read_text())
        for name, record_dict in data.items():
            self._agents[name] = AgentRecord(**record_dict)

    @property
    def summary(self) -> Dict:
        """Registry summary."""
        agents = list(self._agents.values())
        return {
            "total": len(agents),
            "active": sum(1 for a in agents if a.status == "active"),
            "quarantined": sum(1 for a in agents if a.status == "quarantined"),
            "offline": sum(1 for a in agents if a.status == "offline"),
            "by_encoder": {
                enc: sum(1 for a in agents if a.encoder_type == enc)
                for enc in set(a.encoder_type for a in agents)
            },
        }
