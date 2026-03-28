"""
WMCP Subscriber Node — Receives messages and decodes property predictions.
"""

import queue
import numpy as np
from typing import Optional, List, Callable
from wmcp_ros2.wmcp_msg import WMCPMessage

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
    HAS_ROS2 = True
except ImportError:
    HAS_ROS2 = False


class WMCPSubscriber:
    """Subscribes to WMCP messages from a topic.

    Collects messages from all agents, and when a complete round is received,
    calls the decode callback.
    """

    def __init__(self, n_agents: int, bus: Optional[queue.Queue] = None,
                 on_round_complete: Optional[Callable] = None):
        self.n_agents = n_agents
        self._bus = bus
        self._on_round_complete = on_round_complete
        self._current_round: dict = {}
        self._rounds_completed = 0

        if HAS_ROS2 and bus is None:
            rclpy.init()
            self._node = rclpy.create_node('wmcp_subscriber')
            self._sub = self._node.create_subscription(
                String, '/wmcp/messages', self._ros_callback, 10)
        else:
            self._node = None

    def _ros_callback(self, msg):
        wmcp_msg = WMCPMessage.from_json(msg.data)
        self._process_message(wmcp_msg)

    def _process_message(self, msg: WMCPMessage):
        if not msg.validate():
            return
        self._current_round[msg.agent_id] = msg
        if len(self._current_round) >= self.n_agents:
            self._rounds_completed += 1
            if self._on_round_complete:
                self._on_round_complete(dict(self._current_round))
            self._current_round = {}

    def poll(self, timeout: float = 1.0) -> Optional[WMCPMessage]:
        """Poll the bus for a message (non-ROS2 mode)."""
        if self._bus is None:
            return None
        try:
            data = self._bus.get(timeout=timeout)
            if data is None:
                return None
            msg = WMCPMessage.from_json(data)
            self._process_message(msg)
            return msg
        except queue.Empty:
            return None

    @property
    def stats(self) -> dict:
        return {"rounds_completed": self._rounds_completed}

    def shutdown(self):
        if self._node:
            self._node.destroy_node()
            rclpy.shutdown()
