"""
WMCP Protocol Simulator — Interactive Terminal Demo
=====================================================
War-game for the protocol: add/remove agents, inject noise,
corrupt agents, change domain. Live accuracy and message display.

Usage:
    python demo/simulator.py
"""

import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict


class MiniAgent(nn.Module):
    def __init__(self, dim, hd=64, K=3, nf=2):
        super().__init__()
        self.K = K
        self.proj = nn.Sequential(
            nn.Linear(dim * nf, hd), nn.ReLU(), nn.Linear(hd, hd), nn.ReLU())
        self.heads = nn.ModuleList([nn.Linear(hd, K), nn.Linear(hd, K)])

    def forward(self, x):
        h = self.proj(x.view(x.shape[0], -1))
        return [hd(h).argmax(-1).item() for hd in self.heads]


class Simulator:
    def __init__(self):
        self.agents = {}
        self.quarantined = set()
        self.history = []
        self.round = 0
        self.K = 3
        self.noise_sigma = 0.0
        self.corrupt_agent = None

    def add_agent(self, name, arch="vjepa", dim=1024):
        agent = MiniAgent(dim, K=self.K, nf=2)
        self.agents[name] = {"model": agent, "arch": arch, "dim": dim,
                              "messages": 0, "correct": 0}
        print(f"  + Agent '{name}' added ({arch}, {dim}-dim)")

    def remove_agent(self, name):
        if name in self.agents:
            del self.agents[name]
            self.quarantined.discard(name)
            print(f"  - Agent '{name}' removed")

    def quarantine_agent(self, name):
        if name in self.agents:
            self.quarantined.add(name)
            print(f"  ⚠ Agent '{name}' quarantined")

    def unquarantine_agent(self, name):
        self.quarantined.discard(name)
        print(f"  ✓ Agent '{name}' unquarantined")

    def set_noise(self, sigma):
        self.noise_sigma = sigma
        print(f"  Noise σ = {sigma}")

    def set_corrupt(self, name):
        self.corrupt_agent = name if name in self.agents else None
        if self.corrupt_agent:
            print(f"  ☠ Agent '{name}' set to adversarial mode")
        else:
            print(f"  Adversarial mode cleared")

    def run_round(self):
        if len(self.agents) < 1:
            print("  Need at least 1 agent")
            return

        self.round += 1
        active = {n: a for n, a in self.agents.items() if n not in self.quarantined}

        if not active:
            print("  All agents quarantined!")
            return

        # Each agent produces tokens
        messages = {}
        for name, agent in active.items():
            feat = torch.randn(1, 2, agent["dim"])
            if name == self.corrupt_agent:
                tokens = [np.random.randint(0, self.K), np.random.randint(0, self.K)]
            else:
                tokens = agent["model"](feat)
            messages[name] = tokens
            agent["messages"] += 1

        # Display
        parts = []
        for name, tokens in messages.items():
            status = "☠" if name == self.corrupt_agent else "●"
            parts.append(f"{status}{name}=[{tokens[0]},{tokens[1]}]")

        # Entropy-based anomaly check
        anomalies = []
        if len(self.history) > 10:
            for name in messages:
                recent = [h.get(name, [0, 0]) for h in self.history[-20:] if name in h]
                if len(recent) >= 5:
                    ent = 0
                    for pos in range(2):
                        vals = [r[pos] for r in recent]
                        counts = np.bincount(vals, minlength=self.K)
                        probs = counts / counts.sum()
                        probs = probs[probs > 0]
                        ent += -np.sum(probs * np.log(probs))
                    if ent < 0.3:
                        anomalies.append(name)

        self.history.append(messages)

        anomaly_str = f" ⚠ANOMALY:{','.join(anomalies)}" if anomalies else ""
        print(f"  R{self.round:4d} │ {' '.join(parts)}{anomaly_str}")

    def status(self):
        print(f"\n  ╔═══ Simulator Status ═══╗")
        print(f"  ║ Round: {self.round}")
        print(f"  ║ Agents: {len(self.agents)} ({len(self.quarantined)} quarantined)")
        print(f"  ║ Noise: σ={self.noise_sigma}")
        print(f"  ║ Corrupt: {self.corrupt_agent or 'none'}")
        for name, agent in self.agents.items():
            q = " [QUARANTINED]" if name in self.quarantined else ""
            c = " [ADVERSARIAL]" if name == self.corrupt_agent else ""
            print(f"  ║  {name}: {agent['arch']} {agent['dim']}d, "
                  f"{agent['messages']} msgs{q}{c}")
        print(f"  ╚═════════════════════════╝\n")


def run_interactive():
    print("WMCP Protocol Simulator")
    print("=" * 50)
    print("Commands: add <name> [arch] [dim], remove <name>, quarantine <name>,")
    print("          unquarantine <name>, noise <sigma>, corrupt <name>,")
    print("          run [N], status, auto [N], quit\n")

    sim = Simulator()
    # Default fleet
    sim.add_agent("cam-1", "vjepa", 1024)
    sim.add_agent("cam-2", "dinov2", 384)
    sim.add_agent("cam-3", "clip", 768)
    print()

    while True:
        try:
            cmd = input("wmcp> ").strip().split()
        except (EOFError, KeyboardInterrupt):
            break

        if not cmd:
            continue

        if cmd[0] == "quit":
            break
        elif cmd[0] == "add" and len(cmd) >= 2:
            arch = cmd[2] if len(cmd) > 2 else "vjepa"
            dim = int(cmd[3]) if len(cmd) > 3 else {"vjepa": 1024, "dinov2": 384, "clip": 768}.get(arch, 512)
            sim.add_agent(cmd[1], arch, dim)
        elif cmd[0] == "remove" and len(cmd) >= 2:
            sim.remove_agent(cmd[1])
        elif cmd[0] == "quarantine" and len(cmd) >= 2:
            sim.quarantine_agent(cmd[1])
        elif cmd[0] == "unquarantine" and len(cmd) >= 2:
            sim.unquarantine_agent(cmd[1])
        elif cmd[0] == "noise" and len(cmd) >= 2:
            sim.set_noise(float(cmd[1]))
        elif cmd[0] == "corrupt" and len(cmd) >= 2:
            sim.set_corrupt(cmd[1] if cmd[1] != "none" else None)
        elif cmd[0] == "run":
            n = int(cmd[1]) if len(cmd) > 1 else 1
            for _ in range(n):
                sim.run_round()
        elif cmd[0] == "auto":
            n = int(cmd[1]) if len(cmd) > 1 else 10
            for _ in range(n):
                sim.run_round()
                time.sleep(0.2)
        elif cmd[0] == "status":
            sim.status()
        else:
            print("  Unknown command. Try: add, remove, run, status, quit")


def run_demo():
    """Non-interactive demo for automated testing."""
    print("WMCP Simulator Demo (non-interactive)")
    print("=" * 50)
    sim = Simulator()
    sim.add_agent("vjepa-1", "vjepa", 1024)
    sim.add_agent("dino-1", "dinov2", 384)
    sim.add_agent("clip-1", "clip", 768)
    print()

    print("  Normal operation (10 rounds):")
    for _ in range(10):
        sim.run_round()

    print("\n  Corrupting clip-1:")
    sim.set_corrupt("clip-1")
    for _ in range(5):
        sim.run_round()

    print("\n  Quarantining clip-1:")
    sim.quarantine_agent("clip-1")
    for _ in range(5):
        sim.run_round()

    sim.status()


if __name__ == "__main__":
    import sys
    if "--demo" in sys.argv:
        run_demo()
    else:
        run_interactive()
