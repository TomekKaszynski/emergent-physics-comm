"""Threaded pub-sub message bus for protocol communication."""

import time
import queue
import threading
from typing import Callable, Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class Message:
    """A protocol message transmitted through the bus."""
    sender_id: int
    payload: object
    timestamp: float = field(default_factory=time.perf_counter)
    metadata: Dict = field(default_factory=dict)


class MessageBus:
    """Threaded message bus for multi-agent protocol communication.

    Agents publish and subscribe to messages through a shared queue.
    Thread-safe for concurrent senders and receivers.

    Example:
        bus = MessageBus()
        bus.start()
        bus.publish(Message(sender_id=0, payload=encoded_msg))
        msg = bus.subscribe(timeout=1.0)
        bus.stop()
    """

    def __init__(self, max_size: int = 0):
        self._queue: queue.Queue = queue.Queue(maxsize=max_size)
        self._stats = {"published": 0, "received": 0, "dropped": 0}
        self._latencies: List[float] = []
        self._running = False

    def publish(self, message: Message) -> bool:
        """Publish a message to the bus.

        Args:
            message: Message to publish.

        Returns:
            True if published, False if bus is full.
        """
        try:
            self._queue.put_nowait(message)
            self._stats["published"] += 1
            return True
        except queue.Full:
            self._stats["dropped"] += 1
            return False

    def subscribe(self, timeout: float = 1.0) -> Optional[Message]:
        """Subscribe to the next message.

        Args:
            timeout: Seconds to wait before returning None.

        Returns:
            Next message, or None if timeout.
        """
        try:
            msg = self._queue.get(timeout=timeout)
            latency = (time.perf_counter() - msg.timestamp) * 1000
            self._latencies.append(latency)
            self._stats["received"] += 1
            return msg
        except queue.Empty:
            return None

    def send_sentinel(self) -> None:
        """Send a stop sentinel to signal the subscriber to exit."""
        self._queue.put(None)

    @property
    def stats(self) -> Dict:
        """Bus statistics."""
        import numpy as np
        lats = self._latencies if self._latencies else [0]
        return {
            **self._stats,
            "latency_mean_ms": float(np.mean(lats)),
            "latency_p95_ms": float(np.percentile(lats, 95)),
            "throughput_per_s": (
                float(self._stats["received"] / (sum(lats) / 1000))
                if sum(lats) > 0 else 0),
        }
