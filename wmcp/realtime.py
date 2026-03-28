"""WebSocket real-time protocol demo server."""

import json
import time
import asyncio
import threading
import numpy as np
import torch
from typing import Optional

try:
    import websockets
    HAS_WS = True
except ImportError:
    HAS_WS = False

from wmcp.protocol import Protocol


class RealtimeServer:
    """WebSocket server streaming live protocol messages.

    Usage:
        server = RealtimeServer(protocol, agent_views)
        server.start(port=8765)
    """

    def __init__(self, protocol: Optional[Protocol] = None,
                 agent_views=None, mass_values=None, interval_ms: int = 500):
        self.protocol = protocol or Protocol([(1024, 4), (384, 4)], vocab_size=3)
        self.protocol.eval()
        self.agent_views = agent_views
        self.mass_values = mass_values
        self.interval = interval_ms / 1000
        self.clients = set()
        self._round = 0

    async def handler(self, websocket):
        self.clients.add(websocket)
        try:
            async for _ in websocket:
                pass  # Client doesn't send, just receives
        finally:
            self.clients.discard(websocket)

    async def broadcast_loop(self):
        while True:
            self._round += 1
            # Generate a communication round
            if self.agent_views and len(self.agent_views[0]) > 0:
                n = len(self.agent_views[0])
                i, j = np.random.randint(0, n), np.random.randint(0, n)
                views_a = [v[i:i+1] for v in self.agent_views]
                views_b = [v[j:j+1] for v in self.agent_views]
            else:
                views_a = [torch.randn(1, 4, 1024), torch.randn(1, 4, 384)]
                views_b = [torch.randn(1, 4, 1024), torch.randn(1, 4, 384)]

            t0 = time.perf_counter()
            with torch.no_grad():
                msg_a, logits_a = self.protocol.encode(views_a)
                msg_b, logits_b = self.protocol.encode(views_b)
                pred = self.protocol.receivers[0](msg_a, msg_b).item()
            latency = (time.perf_counter() - t0) * 1000

            tokens_a = [l.argmax(-1).item() for l in logits_a]
            tokens_b = [l.argmax(-1).item() for l in logits_b]

            payload = json.dumps({
                "round": self._round,
                "tokens_a": tokens_a,
                "tokens_b": tokens_b,
                "prediction": float(pred),
                "a_greater": pred > 0,
                "latency_ms": round(latency, 2),
                "timestamp": time.time(),
                "agents": [
                    {"id": 0, "type": "vjepa2", "status": "online"},
                    {"id": 1, "type": "dinov2", "status": "online"},
                ],
            })

            if self.clients:
                await asyncio.gather(
                    *[c.send(payload) for c in self.clients],
                    return_exceptions=True)

            await asyncio.sleep(self.interval)

    def start(self, host: str = "localhost", port: int = 8765):
        """Start the WebSocket server."""
        if not HAS_WS:
            print("websockets not installed. Run: pip install websockets")
            return

        async def main():
            async with websockets.serve(self.handler, host, port):
                print(f"WMCP Realtime server on ws://{host}:{port}")
                await self.broadcast_loop()

        asyncio.run(main())


if __name__ == "__main__":
    RealtimeServer().start()
