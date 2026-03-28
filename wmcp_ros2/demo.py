"""
WMCP ROS 2 Demo — Two agents exchange messages through topic interface.

Usage:
    python -m wmcp_ros2.demo
"""

import time
import queue
import threading
import numpy as np
import torch
from wmcp_ros2.wmcp_msg import WMCPMessage
from wmcp_ros2.publisher_node import WMCPPublisher
from wmcp_ros2.subscriber_node import WMCPSubscriber


def run_demo():
    print("WMCP ROS 2 Integration Demo")
    print("=" * 50)

    # Message bus (simulates /wmcp/messages topic)
    bus = queue.Queue()

    # Create two agents
    pub_vjepa = WMCPPublisher(agent_id=0, encoder_type="vjepa2",
                               input_dim=1024, bus=bus)
    pub_dino = WMCPPublisher(agent_id=1, encoder_type="dinov2",
                              input_dim=384, bus=bus)

    rounds_data = []

    def on_round(messages):
        rounds_data.append(messages)

    sub = WMCPSubscriber(n_agents=2, bus=bus, on_round_complete=on_round)

    # Generate synthetic features
    rng = np.random.RandomState(42)
    n_rounds = 50
    latencies = []

    print(f"\nRunning {n_rounds} communication rounds...\n")
    print(f"  {'Round':>5s} │ {'Agent 0 (V-JEPA)':>18s} │ {'Agent 1 (DINOv2)':>18s} │ {'Latency':>8s}")
    print(f"  {'─'*5}─┼─{'─'*18}─┼─{'─'*18}─┼─{'─'*8}")

    for rd in range(n_rounds):
        feat_v = torch.randn(1, 4, 1024)
        feat_d = torch.randn(1, 4, 384)

        t_start = time.perf_counter()

        # Agent 0 publishes
        msg0 = pub_vjepa.publish(feat_v)

        # Agent 1 publishes
        msg1 = pub_dino.publish(feat_d)

        # Subscriber receives both
        sub.poll(timeout=0.1)
        sub.poll(timeout=0.1)

        t_end = time.perf_counter()
        lat = (t_end - t_start) * 1000
        latencies.append(lat)

        if rd < 10 or rd % 10 == 9:
            print(f"  {rd+1:5d} │ tokens={msg0.tokens!s:>12s} │ tokens={msg1.tokens!s:>12s} │ {lat:7.2f}ms")

    print(f"\n  {'─'*60}")
    lats = np.array(latencies)
    print(f"  Rounds completed: {sub.stats['rounds_completed']}")
    print(f"  Mean latency:     {np.mean(lats):.2f}ms")
    print(f"  P95 latency:      {np.percentile(lats, 95):.2f}ms")
    print(f"  Throughput:       {1000/np.mean(lats):.0f} rounds/s")

    # Verify message format
    if rounds_data:
        sample = list(rounds_data[0].values())[0]
        print(f"\n  Message format validation:")
        print(f"    Version:  {sample.version_string}")
        print(f"    K={sample.vocab_size}, L={sample.n_positions}")
        print(f"    Valid:    {sample.validate()}")
        print(f"    JSON size: {len(sample.to_json())} bytes")

    # Demonstrate JSON serialization (wire format)
    print(f"\n  Wire format (JSON):")
    print(f"    {msg0.to_json()}")

    print(f"\n  ROS 2 topic: /wmcp/messages")
    print(f"  ROS 2 available: {'YES' if hasattr(pub_vjepa, '_node') and pub_vjepa._node else 'NO (using Python queue simulation)'}")


if __name__ == "__main__":
    run_demo()
