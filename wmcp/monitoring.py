"""Observability and drift detection for deployed protocols."""

import time
import json
import math
import numpy as np
from typing import List, Dict, Optional
from collections import deque
from dataclasses import dataclass, field


@dataclass
class AgentBaseline:
    """Enrolled baseline statistics for an agent."""
    agent_id: int
    mean_entropy: float
    std_entropy: float
    symbol_frequencies: List[float]
    enrolled_at: float = field(default_factory=time.time)


class ProtocolMonitor:
    """Real-time monitoring for WMCP protocol health.

    Tracks message entropy, symbol distributions, and detects drift
    from enrollment baselines.

    Usage:
        monitor = ProtocolMonitor(vocab_size=3, n_positions=2)
        monitor.enroll_agent(0, baseline_tokens)
        monitor.record_message(agent_id=0, tokens=[1, 2])
        alerts = monitor.check_alerts()
    """

    def __init__(self, vocab_size: int = 3, n_positions: int = 2,
                 window_size: int = 100, entropy_threshold: float = 0.3,
                 drift_threshold: float = 0.2):
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.window_size = window_size
        self.entropy_threshold = entropy_threshold
        self.drift_threshold = drift_threshold

        self._baselines: Dict[int, AgentBaseline] = {}
        self._windows: Dict[int, deque] = {}
        self._message_count = 0
        self._alerts: List[Dict] = []
        self._log_buffer: List[str] = []

    def enroll_agent(self, agent_id: int, baseline_tokens: np.ndarray) -> None:
        """Register an agent's baseline statistics from training.

        Args:
            agent_id: Agent identifier.
            baseline_tokens: (N, n_positions) token array from training.
        """
        entropies = []
        for p in range(baseline_tokens.shape[1]):
            counts = np.bincount(baseline_tokens[:, p], minlength=self.vocab_size)
            probs = counts / counts.sum()
            probs = probs[probs > 0]
            ent = -np.sum(probs * np.log(probs)) / np.log(self.vocab_size)
            entropies.append(ent)

        freqs = []
        for p in range(baseline_tokens.shape[1]):
            counts = np.bincount(baseline_tokens[:, p], minlength=self.vocab_size)
            freqs.extend((counts / counts.sum()).tolist())

        self._baselines[agent_id] = AgentBaseline(
            agent_id=agent_id,
            mean_entropy=float(np.mean(entropies)),
            std_entropy=float(np.std(entropies)),
            symbol_frequencies=freqs,
        )
        self._windows[agent_id] = deque(maxlen=self.window_size)

    def record_message(self, agent_id: int, tokens: List[int]) -> None:
        """Record a message from an agent."""
        self._message_count += 1
        if agent_id not in self._windows:
            self._windows[agent_id] = deque(maxlen=self.window_size)
        self._windows[agent_id].append(tokens)

        # JSON line log
        self._log_buffer.append(json.dumps({
            "ts": time.time(),
            "agent": agent_id,
            "tokens": tokens,
            "msg_id": self._message_count,
        }))

    def check_alerts(self) -> List[Dict]:
        """Check for entropy drops and distribution drift."""
        alerts = []

        for agent_id, window in self._windows.items():
            if len(window) < 10:
                continue

            recent = np.array(list(window))

            # Entropy check
            for p in range(min(recent.shape[1], self.n_positions)):
                counts = np.bincount(recent[:, p], minlength=self.vocab_size)
                probs = counts / counts.sum()
                probs = probs[probs > 0]
                ent = -np.sum(probs * np.log(probs)) / np.log(self.vocab_size)

                if ent < self.entropy_threshold:
                    alerts.append({
                        "type": "LOW_ENTROPY",
                        "agent_id": agent_id,
                        "position": p,
                        "entropy": float(ent),
                        "threshold": self.entropy_threshold,
                        "severity": "warning" if ent > 0.1 else "critical",
                    })

            # Drift check against baseline
            if agent_id in self._baselines:
                baseline = self._baselines[agent_id]
                current_freqs = []
                for p in range(min(recent.shape[1], self.n_positions)):
                    counts = np.bincount(recent[:, p], minlength=self.vocab_size)
                    current_freqs.extend((counts / counts.sum()).tolist())

                # KL divergence between current and baseline
                kl = 0.0
                for cf, bf in zip(current_freqs, baseline.symbol_frequencies):
                    if cf > 0 and bf > 0:
                        kl += cf * np.log(cf / bf)

                if kl > self.drift_threshold:
                    alerts.append({
                        "type": "DISTRIBUTION_DRIFT",
                        "agent_id": agent_id,
                        "kl_divergence": float(kl),
                        "threshold": self.drift_threshold,
                        "severity": "warning",
                    })

        self._alerts.extend(alerts)
        return alerts

    @property
    def health(self) -> Dict:
        """Current protocol health summary."""
        return {
            "total_messages": self._message_count,
            "active_agents": len(self._windows),
            "enrolled_agents": len(self._baselines),
            "active_alerts": len([a for a in self._alerts[-10:]
                                  if time.time() - a.get("ts", 0) < 60]),
            "status": "healthy" if not self._alerts[-5:] else "degraded",
        }

    def get_logs(self, n: int = 100) -> List[str]:
        """Get recent log lines (JSON Lines format)."""
        return self._log_buffer[-n:]

    def serve_dashboard_data(self) -> Dict:
        """Data for the monitoring dashboard."""
        return {
            "health": self.health,
            "recent_alerts": self._alerts[-20:],
            "agent_stats": {
                aid: {
                    "messages": len(w),
                    "enrolled": aid in self._baselines,
                } for aid, w in self._windows.items()
            },
        }
