"""
Training loops for compositional communication with iterated learning.

Implements population-based training with periodic receiver reset (iterated
learning), Gumbel-Softmax temperature annealing, and entropy regularization.
"""

import time
import numpy as np
import torch
import torch.nn.functional as F

from .models import CompositionalReceiver
from .datasets import sample_pairs, split_views


def gumbel_schedule(epoch, total_epochs, tau_start=3.0, tau_end=1.0):
    """Compute Gumbel-Softmax temperature for current epoch."""
    return tau_start + (tau_end - tau_start) * (epoch - 1) / max(1, total_epochs - 1)


def evaluate_2agent(sender, receivers, data, prop1_bins, prop2_bins, ids, device,
                    batch_size=32, n_rounds=20):
    """Evaluate 2-agent communication, return best receiver's accuracy.

    Returns:
        (acc_prop1, acc_prop2, acc_both): Accuracy tuple from best receiver.
    """
    sender.eval()
    best_both = -1
    best_result = None

    for r in receivers:
        r.eval()
        correct_1, correct_2, correct_both, total = 0, 0, 0, 0
        rng = np.random.RandomState(999)
        with torch.no_grad():
            for _ in range(n_rounds):
                ia, ib = sample_pairs(ids, batch_size, rng)
                da = data[ia].to(device)
                db = data[ib].to(device)
                l1 = (prop1_bins[ia] > prop1_bins[ib]).astype(np.float32)
                l2 = (prop2_bins[ia] > prop2_bins[ib]).astype(np.float32)
                msg_a, _ = sender(da)
                msg_b, _ = sender(db)
                p1, p2 = r(msg_a, msg_b)
                p1 = (p1.cpu().numpy() > 0).astype(np.float32)
                p2 = (p2.cpu().numpy() > 0).astype(np.float32)
                correct_1 += (p1 == l1).sum()
                correct_2 += (p2 == l2).sum()
                correct_both += ((p1 == l1) & (p2 == l2)).sum()
                total += len(ia)
        acc_both = correct_both / total
        if acc_both > best_both:
            best_both = acc_both
            best_result = (correct_1 / total, correct_2 / total, acc_both)
    return best_result


def evaluate_4agent(multi_sender, receivers, data, prop1_bins, prop2_bins, ids,
                    device, n_agents, frames_per_agent, batch_size=32, n_rounds=20):
    """Evaluate 4-agent communication, return best receiver's accuracy."""
    multi_sender.eval()
    best_both = -1
    best_result = None

    for r in receivers:
        r.eval()
        correct_1, correct_2, correct_both, total = 0, 0, 0, 0
        rng = np.random.RandomState(999)
        with torch.no_grad():
            for _ in range(n_rounds):
                ia, ib = sample_pairs(ids, batch_size, rng)
                da = data[ia].to(device)
                db = data[ib].to(device)
                views_a = split_views(da, n_agents, frames_per_agent)
                views_b = split_views(db, n_agents, frames_per_agent)
                l1 = (prop1_bins[ia] > prop1_bins[ib]).astype(np.float32)
                l2 = (prop2_bins[ia] > prop2_bins[ib]).astype(np.float32)
                msg_a, _ = multi_sender(views_a)
                msg_b, _ = multi_sender(views_b)
                p1, p2 = r(msg_a, msg_b)
                p1 = (p1.cpu().numpy() > 0).astype(np.float32)
                p2 = (p2.cpu().numpy() > 0).astype(np.float32)
                correct_1 += (p1 == l1).sum()
                correct_2 += (p2 == l2).sum()
                correct_both += ((p1 == l1) & (p2 == l2)).sum()
                total += len(ia)
        acc_both = correct_both / total
        if acc_both > best_both:
            best_both = acc_both
            best_result = (correct_1 / total, correct_2 / total, acc_both)
    return best_result


def train_communication(sender, receivers, data, prop1_bins, prop2_bins,
                        train_ids, holdout_ids, device, msg_dim,
                        epochs=400, sender_lr=1e-3, receiver_lr=3e-3,
                        batch_size=32, tau_start=3.0, tau_end=1.0,
                        soft_warmup=30, entropy_threshold=0.1, entropy_coef=0.03,
                        receiver_reset_interval=40, hidden_dim=128,
                        seed=0, eval_fn=None, print_every=50):
    """Train sender-receiver communication with iterated learning.

    Population-based training: multiple receivers train simultaneously.
    Periodic receiver reset implements iterated learning pressure toward
    compositional protocols.

    Args:
        sender: CompositionalSender or MultiAgentSender.
        receivers: List of CompositionalReceiver instances.
        data: (N, T, D) feature tensor.
        prop1_bins, prop2_bins: (N,) property labels.
        train_ids, holdout_ids: Split indices.
        device: torch device.
        msg_dim: Message dimension for receiver reconstruction.
        epochs: Total training epochs.
        sender_lr, receiver_lr: Learning rates.
        batch_size: Batch size.
        tau_start, tau_end: Gumbel temperature schedule.
        soft_warmup: Epochs before switching to hard Gumbel.
        entropy_threshold: Minimum entropy before regularization kicks in.
        entropy_coef: Entropy regularization weight.
        receiver_reset_interval: Reset receivers every N epochs.
        hidden_dim: Hidden dim for new receivers on reset.
        seed: Random seed.
        eval_fn: Evaluation function (sender, receivers, data, bins, ids, device) -> (a1, a2, ab).
        print_every: Print interval.

    Returns:
        nan_count: Number of NaN batches encountered.
    """
    rng = np.random.RandomState(seed)
    sender_opt = torch.optim.Adam(sender.parameters(), lr=sender_lr)
    receiver_opts = [torch.optim.Adam(r.parameters(), lr=receiver_lr) for r in receivers]
    n_batches = max(1, len(train_ids) // batch_size)
    nan_count = 0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        # Iterated learning: reset receivers
        if epoch > 1 and (epoch - 1) % receiver_reset_interval == 0:
            for i in range(len(receivers)):
                receivers[i] = CompositionalReceiver(msg_dim, hidden_dim).to(device)
                receiver_opts[i] = torch.optim.Adam(receivers[i].parameters(), lr=receiver_lr)

        tau = gumbel_schedule(epoch, epochs, tau_start, tau_end)
        hard = epoch >= soft_warmup

        sender.train()
        for r in receivers:
            r.train()

        for _ in range(n_batches):
            ia, ib = sample_pairs(train_ids, batch_size, rng)
            da = data[ia].to(device)
            db = data[ib].to(device)
            label_1 = torch.tensor((prop1_bins[ia] > prop1_bins[ib]).astype(np.float32), device=device)
            label_2 = torch.tensor((prop2_bins[ia] > prop2_bins[ib]).astype(np.float32), device=device)

            # Handle multi-agent vs single-agent sender
            if hasattr(sender, 'senders'):
                n_agents = len(sender.senders)
                frames_per = da.shape[1] // n_agents
                views_a = split_views(da, n_agents, frames_per)
                views_b = split_views(db, n_agents, frames_per)
                msg_a, logits_a = sender(views_a, tau=tau, hard=hard)
                msg_b, logits_b = sender(views_b, tau=tau, hard=hard)
            else:
                msg_a, logits_a = sender(da, tau=tau, hard=hard)
                msg_b, logits_b = sender(db, tau=tau, hard=hard)

            if torch.isnan(msg_a).any() or torch.isnan(msg_b).any():
                nan_count += 1
                continue

            total_loss = torch.tensor(0.0, device=device)
            for r in receivers:
                p1, p2 = r(msg_a, msg_b)
                r_loss = F.binary_cross_entropy_with_logits(p1, label_1) + \
                         F.binary_cross_entropy_with_logits(p2, label_2)
                total_loss = total_loss + r_loss
            loss = total_loss / len(receivers)

            # Entropy regularization
            for logits_list in [logits_a, logits_b]:
                for lg in logits_list:
                    probs = F.softmax(lg, dim=-1)
                    ent = -(probs * (probs + 1e-10).log()).sum(dim=-1).mean()
                    if ent < entropy_threshold:
                        loss = loss - entropy_coef * ent

            sender_opt.zero_grad()
            for ro in receiver_opts:
                ro.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sender.parameters(), 1.0)
            for r in receivers:
                torch.nn.utils.clip_grad_norm_(r.parameters(), 1.0)
            sender_opt.step()
            for ro in receiver_opts:
                ro.step()

        if epoch % print_every == 0 or epoch == 1:
            if eval_fn is not None:
                t1, t2, tb = eval_fn(sender, receivers, data, prop1_bins, prop2_bins,
                                     train_ids, device)
                h1, h2, hb = eval_fn(sender, receivers, data, prop1_bins, prop2_bins,
                                     holdout_ids, device)
                eta = (time.time() - t0) / epoch * (epochs - epoch) / 60
                nan_str = f"  NaN={nan_count}" if nan_count > 0 else ""
                print(f"    Ep {epoch:3d}: tau={tau:.2f}  "
                      f"train[p1={t1*100:.1f}% p2={t2*100:.1f}% both={tb*100:.1f}%]  "
                      f"holdout[p1={h1*100:.1f}% p2={h2*100:.1f}% both={hb*100:.1f}%]"
                      f"{nan_str}  ETA {eta:.0f}min", flush=True)

        if epoch % 100 == 0 and hasattr(torch, 'mps') and device.type == 'mps':
            torch.mps.empty_cache()

    return nan_count
