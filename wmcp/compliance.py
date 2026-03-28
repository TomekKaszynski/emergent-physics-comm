"""Protocol compliance validation suite."""

import time
import numpy as np
import torch
from typing import Dict, List, Tuple
from wmcp.metrics import compute_posdis, compute_topsim, compute_bosdis, make_attributes


def validate_protocol(protocol, agent_views: List[torch.Tensor],
                      mass_values: np.ndarray, obj_names: List[str],
                      device: torch.device = torch.device("cpu")
                      ) -> Dict:
    """Run the 9-point WMCP compliance suite.

    Args:
        protocol: Trained Protocol instance.
        agent_views: List of (N, T, D) feature tensors per agent.
        mass_values: (N,) mass values.
        obj_names: List of N object names.
        device: Torch device for inference.

    Returns:
        Dict with test results and PASS/FAIL status.
    """
    results = []
    vocab_size = protocol.vocab_size
    n_agents = protocol.n_agents
    n_heads = protocol.n_heads

    protocol = protocol.to(device).eval()

    # Test 1: Message format dimensions
    with torch.no_grad():
        views = [v[:1].to(device) for v in agent_views]
        msg, logits = protocol.encode(views)
    expected_dim = n_agents * n_heads * vocab_size
    results.append(("Message format: correct dimensions",
                    msg.shape[1] == expected_dim,
                    f"Expected {expected_dim}, got {msg.shape[1]}"))

    # Test 2: One-hot per position
    msg_r = msg.view(1, n_agents * n_heads, vocab_size)
    is_onehot = all(
        abs(msg_r[0, p].sum().item() - 1.0) < 1e-5 and
        msg_r[0, p].max().item() > 0.99
        for p in range(n_agents * n_heads))
    results.append(("Message format: one-hot per position", is_onehot,
                    f"Checked {n_agents * n_heads} positions"))

    # Test 3: Extract tokens and compute PosDis
    tokens_list = []
    with torch.no_grad():
        for i in range(0, len(agent_views[0]), 32):
            views = [v[i:i+32].to(device) for v in agent_views]
            _, logits = protocol.encode(views)
            tokens_list.append(
                np.stack([l.argmax(-1).cpu().numpy() for l in logits], axis=1))
    all_tokens = np.concatenate(tokens_list, axis=0)

    attributes = make_attributes(mass_values, obj_names)
    posdis, mi_matrix, entropies = compute_posdis(all_tokens, attributes, vocab_size)
    results.append(("Compositionality: PosDis > 0.5",
                    posdis > 0.5, f"PosDis = {posdis:.3f}"))

    # Test 4: Noise tolerance at σ=0.5
    recv = protocol.receivers[0].to(device).eval()
    rng = np.random.RandomState(42)
    unique_objs = sorted(set(obj_names))
    holdout = set(rng.choice(unique_objs, max(4, len(unique_objs)//5), replace=False))
    holdout_ids = np.array([i for i, o in enumerate(obj_names) if o in holdout])

    if len(holdout_ids) >= 4:
        mass_t = torch.tensor(mass_values, dtype=torch.float32).to(device)
        baseline_c = noisy_c = total_n = 0
        er = np.random.RandomState(999)
        with torch.no_grad():
            for _ in range(100):
                ia = er.choice(holdout_ids, min(32, len(holdout_ids)))
                ib = er.choice(holdout_ids, min(32, len(holdout_ids)))
                s = ia == ib
                while s.any(): ib[s] = er.choice(holdout_ids, s.sum()); s = ia == ib
                md = np.abs(mass_values[ia] - mass_values[ib]); k = md > 0.5
                if k.sum() < 2: continue
                ia, ib = ia[k], ib[k]
                va = [v[ia].to(device) for v in agent_views]
                vb = [v[ib].to(device) for v in agent_views]
                la = mass_t[ia] > mass_t[ib]
                ma, _ = protocol.encode(va); mb, _ = protocol.encode(vb)
                baseline_c += ((recv(ma, mb) > 0) == la).sum().item()
                nma = ma + torch.randn_like(ma) * 0.5
                nmb = mb + torch.randn_like(mb) * 0.5
                noisy_c += ((recv(nma, nmb) > 0) == la).sum().item()
                total_n += len(la)
        ba = baseline_c / max(total_n, 1)
        na = noisy_c / max(total_n, 1)
        drop = ba - na
        results.append(("Noise tolerance: <10% drop at σ=0.5",
                        drop < 0.10, f"Baseline={ba:.1%}, Noisy={na:.1%}, Drop={drop:.1%}"))
    else:
        results.append(("Noise tolerance", False, "Insufficient holdout"))

    # Test 5: Latency < 10ms
    protocol_cpu = protocol.cpu()
    recv_cpu = recv.cpu()
    av_cpu = [v.cpu() for v in agent_views]
    for _ in range(10):
        with torch.no_grad():
            va = [v[:1] for v in av_cpu]; vb = [v[1:2] for v in av_cpu]
            protocol_cpu.communicate(va, vb)
    latencies = []
    for _ in range(200):
        i, j = np.random.randint(0, len(obj_names), 2)
        t_s = time.perf_counter()
        with torch.no_grad():
            protocol_cpu.communicate([v[i:i+1] for v in av_cpu],
                                     [v[j:j+1] for v in av_cpu])
        latencies.append((time.perf_counter() - t_s) * 1000)
    mean_lat = np.mean(latencies)
    results.append(("Latency: <10ms on CPU", mean_lat < 10.0,
                    f"Mean={mean_lat:.2f}ms"))
    protocol = protocol_cpu.to(device)

    # Test 6: Vocabulary size
    results.append(("Config: K=3 vocabulary",
                    vocab_size == 3, f"K={vocab_size}"))

    # Test 7: Heterogeneous check
    sender_dims = set()
    for s in protocol.senders:
        sender_dims.add(s.projection.temporal[0].in_channels)
    results.append(("Config: heterogeneous agents",
                    len(sender_dims) > 1,
                    f"Encoder dims: {sender_dims}"))

    # Test 8: MI mass > object
    mass_mi = np.mean(mi_matrix[:, 0])
    obj_mi = np.mean(mi_matrix[:, 1])
    results.append(("MI structure: mass > object identity",
                    mass_mi > obj_mi,
                    f"Mass MI={mass_mi:.3f}, Object MI={obj_mi:.3f}"))

    # Test 9: Holdout accuracy > 60%
    if len(holdout_ids) >= 4:
        results.append(("Task performance: holdout acc > 60%",
                        ba > 0.60, f"Accuracy={ba:.1%}"))
    else:
        results.append(("Task performance", False, "Insufficient holdout"))

    n_pass = sum(1 for _, p, _ in results if p)
    return {
        "tests": [{"name": n, "passed": p, "detail": d} for n, p, d in results],
        "n_pass": n_pass,
        "n_total": len(results),
        "all_pass": n_pass == len(results),
    }
