"""
Phase 87b: Clean Restitution + Re-run Compositionality
=======================================================
1. Filter restitution to 0.05 < e < 0.95 (physically valid)
2. Re-run 2-property compositionality with cleaned labels, lower lr, grad clip
3. Re-run spring mass with stabilized training (lower lr, grad clip)

Run:
  PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=1 python3 _phase87b_clean_restitution.py
"""

import time, json, math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from scipy import stats
from collections import defaultdict

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
RESULTS_DIR = Path("results")

# Architecture constants
HIDDEN_DIM = 128
VJEPA_DIM = 1024
VOCAB_SIZE = 5
N_HEADS = 2
BATCH_SIZE = 32
COMM_EPOCHS = 400
# STABILIZED: lower lr, same grad clip
SENDER_LR = 3e-4
RECEIVER_LR = 1e-3
TAU_START = 3.0
TAU_END = 1.0
SOFT_WARMUP = 30
ENTROPY_THRESHOLD = 0.1
ENTROPY_COEF = 0.03
RECEIVER_RESET_INTERVAL = 40
N_RECEIVERS = 3
N_SEEDS = 10
GRAD_CLIP = 1.0


# ═══ Architecture ═══
class TemporalEncoder(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=1024, n_frames=4):
        super().__init__()
        ks = min(3, n_frames)
        self.temporal = nn.Sequential(
            nn.Conv1d(input_dim, 256, kernel_size=ks, padding=ks // 2), nn.ReLU(),
            nn.Conv1d(256, 128, kernel_size=ks, padding=ks // 2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1))
        self.fc = nn.Sequential(nn.Linear(128, hidden_dim), nn.ReLU())

    def forward(self, x):
        return self.fc(self.temporal(x.permute(0, 2, 1)).squeeze(-1))


class CompositionalSender(nn.Module):
    def __init__(self, encoder, hidden_dim, vocab_size, n_heads):
        super().__init__()
        self.encoder = encoder
        self.vocab_size = vocab_size
        self.n_heads = n_heads
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, vocab_size) for _ in range(n_heads)])

    def forward(self, x, tau=1.0, hard=True):
        h = self.encoder(x)
        messages, all_logits = [], []
        for head in self.heads:
            logits = head(h)
            if self.training:
                msg = F.gumbel_softmax(logits, tau=tau, hard=hard)
            else:
                msg = F.one_hot(logits.argmax(dim=-1), self.vocab_size).float()
            messages.append(msg)
            all_logits.append(logits)
        return torch.cat(messages, dim=-1), all_logits


class MultiAgentSender(nn.Module):
    def __init__(self, senders):
        super().__init__()
        self.senders = nn.ModuleList(senders)

    def forward(self, views, tau=1.0, hard=True):
        messages, all_logits = [], []
        for sender, view in zip(self.senders, views):
            msg, logits = sender(view, tau=tau, hard=hard)
            messages.append(msg)
            all_logits.extend(logits)
        return torch.cat(messages, dim=-1), all_logits


class SinglePropertyReceiver(nn.Module):
    def __init__(self, msg_dim, hidden_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(msg_dim * 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU())
        self.head = nn.Linear(hidden_dim // 2, 1)

    def forward(self, msg_a, msg_b):
        return self.head(self.shared(torch.cat([msg_a, msg_b], dim=-1))).squeeze(-1)


class TwoPropertyReceiver(nn.Module):
    def __init__(self, msg_dim, hidden_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(msg_dim * 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU())
        self.head_a = nn.Linear(hidden_dim // 2, 1)
        self.head_b = nn.Linear(hidden_dim // 2, 1)

    def forward(self, msg_a, msg_b):
        h = self.shared(torch.cat([msg_a, msg_b], dim=-1))
        return self.head_a(h).squeeze(-1), self.head_b(h).squeeze(-1)


# ═══ Compositionality metrics ═══
def compute_compositionality(multi, agent_views, features, mass_values, rest_values, n_pos):
    """Compute MI, PosDis, TopSim for two-property setup."""
    multi.eval()
    all_tokens = []
    with torch.no_grad():
        for i in range(0, len(features), BATCH_SIZE):
            views = [v[i:i+BATCH_SIZE].to(DEVICE) for v in agent_views]
            _, logits = multi(views)
            tokens = [l.argmax(dim=-1).cpu().numpy() for l in logits]
            all_tokens.append(np.stack(tokens, axis=1))
    all_tokens = np.concatenate(all_tokens, axis=0)

    # Bin mass and restitution into 5 levels
    mass_bins = np.digitize(mass_values, np.percentile(mass_values, [20, 40, 60, 80]))
    rest_bins = np.digitize(rest_values, np.percentile(rest_values, [20, 40, 60, 80]))
    attrs = np.stack([mass_bins, rest_bins], axis=1)

    # MI matrix
    mi_matrix = np.zeros((n_pos, 2))
    for p in range(n_pos):
        for a in range(2):
            x, y = all_tokens[:, p], attrs[:, a]
            xv, yv = np.unique(x), np.unique(y)
            n = len(x)
            mi = 0.0
            for xval in xv:
                for yval in yv:
                    pxy = np.sum((x == xval) & (y == yval)) / n
                    px = np.sum(x == xval) / n
                    py = np.sum(y == yval) / n
                    if pxy > 0 and px > 0 and py > 0:
                        mi += pxy * np.log(pxy / (px * py))
            mi_matrix[p, a] = mi

    # PosDis
    pos_dis = 0.0
    for p in range(n_pos):
        s = np.sort(mi_matrix[p])[::-1]
        if s[0] > 1e-10:
            pos_dis += (s[0] - s[1]) / s[0]
    pos_dis /= n_pos

    # TopSim
    rng_ts = np.random.RandomState(42)
    md_list, hd_list = [], []
    for _ in range(min(5000, len(features) * (len(features) - 1) // 2)):
        i, j = rng_ts.choice(len(features), 2, replace=False)
        md_list.append(abs(float(mass_bins[i]) - float(mass_bins[j])) +
                      abs(float(rest_bins[i]) - float(rest_bins[j])))
        hd_list.append(int((all_tokens[i] != all_tokens[j]).sum()))
    topsim, _ = stats.spearmanr(md_list, hd_list)
    if np.isnan(topsim):
        topsim = 0.0

    # Entropies
    entropies = []
    for p in range(n_pos):
        counts = np.bincount(all_tokens[:, p], minlength=VOCAB_SIZE)
        probs = counts / counts.sum()
        ent = -np.sum(probs * np.log(probs + 1e-10)) / np.log(VOCAB_SIZE)
        entropies.append(float(ent))

    return {
        "mi_matrix": mi_matrix.tolist(),
        "pos_dis": float(pos_dis),
        "topsim": float(topsim),
        "entropies": entropies,
    }


def compute_single_property_metrics(multi, agent_views, features, mass_values, n_pos):
    """Compute MI and TopSim for single-property setup."""
    multi.eval()
    all_tokens = []
    with torch.no_grad():
        for i in range(0, len(features), BATCH_SIZE):
            views = [v[i:i+BATCH_SIZE].to(DEVICE) for v in agent_views]
            _, logits = multi(views)
            tokens = [l.argmax(dim=-1).cpu().numpy() for l in logits]
            all_tokens.append(np.stack(tokens, axis=1))
    all_tokens = np.concatenate(all_tokens, axis=0)

    # MI per position
    mass_bins = np.digitize(mass_values, np.percentile(mass_values, [20, 40, 60, 80]))
    mi_values = []
    for p in range(n_pos):
        x, y = all_tokens[:, p], mass_bins
        xv, yv = np.unique(x), np.unique(y)
        n = len(x)
        mi = 0.0
        for xval in xv:
            for yval in yv:
                pxy = np.sum((x == xval) & (y == yval)) / n
                px = np.sum(x == xval) / n
                py = np.sum(y == yval) / n
                if pxy > 0 and px > 0 and py > 0:
                    mi += pxy * np.log(pxy / (px * py))
        mi_values.append(float(mi))

    # TopSim
    rng_ts = np.random.RandomState(42)
    md_list, hd_list = [], []
    for _ in range(min(5000, len(features) * (len(features) - 1) // 2)):
        i, j = rng_ts.choice(len(features), 2, replace=False)
        md_list.append(abs(mass_values[i] - mass_values[j]))
        hd_list.append(int((all_tokens[i] != all_tokens[j]).sum()))
    topsim, _ = stats.spearmanr(md_list, hd_list)
    if np.isnan(topsim):
        topsim = 0.0

    # Best rho
    best_rho = 0.0
    for p in range(n_pos):
        rho, _ = stats.spearmanr(all_tokens[:, p], mass_values)
        if abs(rho) > abs(best_rho):
            best_rho = rho

    entropies = []
    for p in range(n_pos):
        counts = np.bincount(all_tokens[:, p], minlength=VOCAB_SIZE)
        probs = counts / counts.sum()
        ent = -np.sum(probs * np.log(probs + 1e-10)) / np.log(VOCAB_SIZE)
        entropies.append(float(ent))

    return {
        "mi_values": mi_values,
        "topsim": float(topsim),
        "mass_symbol_rho": float(best_rho),
        "entropies": entropies,
    }


# ═══ Training loops ═══

def train_two_property(features, mass_values, rest_values, obj_names, n_agents, seed):
    """Train one seed of two-property communication with stabilized training."""
    n_frames = features.shape[1]
    fpa = n_frames // n_agents
    agent_views = [features[:, i*fpa:(i+1)*fpa, :].float() for i in range(n_agents)]
    msg_dim = n_agents * N_HEADS * VOCAB_SIZE

    unique_objs = sorted(set(obj_names))
    n_holdout = max(4, len(unique_objs) // 5)
    rng = np.random.RandomState(seed * 1000 + 42)
    holdout_objs = set(rng.choice(unique_objs, n_holdout, replace=False))
    train_ids = np.array([i for i, o in enumerate(obj_names) if o not in holdout_objs])
    holdout_ids = np.array([i for i, o in enumerate(obj_names) if o in holdout_objs])

    if len(holdout_ids) < 4:
        return None

    torch.manual_seed(seed)
    np.random.seed(seed)

    senders = [CompositionalSender(
        TemporalEncoder(HIDDEN_DIM, VJEPA_DIM, n_frames=fpa),
        HIDDEN_DIM, VOCAB_SIZE, N_HEADS
    ) for _ in range(n_agents)]
    multi = MultiAgentSender(senders).to(DEVICE)
    receivers = [TwoPropertyReceiver(msg_dim, HIDDEN_DIM).to(DEVICE) for _ in range(N_RECEIVERS)]
    s_opt = torch.optim.Adam(multi.parameters(), lr=SENDER_LR, weight_decay=1e-5)
    r_opts = [torch.optim.Adam(r.parameters(), lr=RECEIVER_LR, weight_decay=1e-5) for r in receivers]

    mass_dev = torch.tensor(mass_values, dtype=torch.float32).to(DEVICE)
    rest_dev = torch.tensor(rest_values, dtype=torch.float32).to(DEVICE)
    max_ent = math.log(VOCAB_SIZE)
    nb = max(1, len(train_ids) // BATCH_SIZE)
    best_both = 0.0
    best_state = None
    best_r_states = None
    nan_count = 0
    hm = hr = hb = 0.0
    t0 = time.time()

    for ep in range(COMM_EPOCHS):
        if ep > 0 and ep % RECEIVER_RESET_INTERVAL == 0:
            for i in range(len(receivers)):
                receivers[i] = TwoPropertyReceiver(msg_dim, HIDDEN_DIM).to(DEVICE)
                r_opts[i] = torch.optim.Adam(receivers[i].parameters(), lr=RECEIVER_LR, weight_decay=1e-5)

        multi.train()
        for r in receivers:
            r.train()
        tau = TAU_START + (TAU_END - TAU_START) * ep / max(1, COMM_EPOCHS - 1)
        hard = ep >= SOFT_WARMUP

        for _ in range(nb):
            ia = rng.choice(train_ids, BATCH_SIZE)
            ib = rng.choice(train_ids, BATCH_SIZE)
            same = ia == ib
            while same.any():
                ib[same] = rng.choice(train_ids, same.sum())
                same = ia == ib

            va = [v[ia].to(DEVICE) for v in agent_views]
            vb = [v[ib].to(DEVICE) for v in agent_views]
            l_mass = (mass_dev[ia] > mass_dev[ib]).float()
            l_rest = (rest_dev[ia] > rest_dev[ib]).float()
            ma, la = multi(va, tau=tau, hard=hard)
            mb, lb = multi(vb, tau=tau, hard=hard)

            total_loss = torch.tensor(0.0, device=DEVICE)
            for r in receivers:
                pm, pr = r(ma, mb)
                total_loss = total_loss + F.binary_cross_entropy_with_logits(pm, l_mass)
                total_loss = total_loss + F.binary_cross_entropy_with_logits(pr, l_rest)
            loss = total_loss / len(receivers)

            for logits in la + lb:
                lp = F.log_softmax(logits, dim=-1)
                p = lp.exp().clamp(min=1e-8)
                ent = -(p * lp).sum(dim=-1).mean()
                if ent / max_ent < ENTROPY_THRESHOLD:
                    loss = loss - ENTROPY_COEF * ent

            if torch.isnan(loss) or torch.isinf(loss):
                s_opt.zero_grad()
                for o in r_opts:
                    o.zero_grad()
                nan_count += 1
                continue

            s_opt.zero_grad()
            for o in r_opts:
                o.zero_grad()
            loss.backward()

            # Check for NaN grads
            has_nan = any(p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
                         for p in list(multi.parameters()) + [p for r in receivers for p in r.parameters()])
            if has_nan:
                s_opt.zero_grad()
                for o in r_opts:
                    o.zero_grad()
                nan_count += 1
                continue

            torch.nn.utils.clip_grad_norm_(multi.parameters(), GRAD_CLIP)
            for r in receivers:
                torch.nn.utils.clip_grad_norm_(r.parameters(), GRAD_CLIP)
            s_opt.step()
            for o in r_opts:
                o.step()

        if ep % 50 == 0:
            torch.mps.empty_cache()

        if (ep + 1) % 50 == 0 or ep == 0:
            multi.eval()
            for r in receivers:
                r.eval()
            with torch.no_grad():
                cm = cr = cb = tm = tr = tb = 0
                er = np.random.RandomState(999)
                for _ in range(30):
                    bs = min(BATCH_SIZE, len(holdout_ids))
                    ia_h = er.choice(holdout_ids, bs)
                    ib_h = er.choice(holdout_ids, bs)
                    same_h = ia_h == ib_h
                    while same_h.any():
                        ib_h[same_h] = er.choice(holdout_ids, same_h.sum())
                        same_h = ia_h == ib_h
                    va_h = [v[ia_h].to(DEVICE) for v in agent_views]
                    vb_h = [v[ib_h].to(DEVICE) for v in agent_views]
                    ma_h, _ = multi(va_h)
                    mb_h, _ = multi(vb_h)
                    lm_h = mass_dev[ia_h] > mass_dev[ib_h]
                    lr_h = rest_dev[ia_h] > rest_dev[ib_h]
                    for r in receivers:
                        pm_h, pr_h = r(ma_h, mb_h)
                        m_diff = np.abs(mass_values[ia_h] - mass_values[ib_h]) > 0.5
                        r_diff = np.abs(rest_values[ia_h] - rest_values[ib_h]) > 0.02
                        m_diff_t = torch.tensor(m_diff, device=DEVICE)
                        r_diff_t = torch.tensor(r_diff, device=DEVICE)
                        if m_diff_t.sum() > 0:
                            cm += ((pm_h[m_diff_t] > 0) == lm_h[m_diff_t]).sum().item()
                            tm += m_diff_t.sum().item()
                        if r_diff_t.sum() > 0:
                            cr += ((pr_h[r_diff_t] > 0) == lr_h[r_diff_t]).sum().item()
                            tr += r_diff_t.sum().item()
                        bd = m_diff_t & r_diff_t
                        if bd.sum() > 0:
                            ok = ((pm_h[bd] > 0) == lm_h[bd]) & ((pr_h[bd] > 0) == lr_h[bd])
                            cb += ok.sum().item()
                            tb += bd.sum().item()
                hm = cm / max(tm, 1)
                hr = cr / max(tr, 1)
                hb = cb / max(tb, 1)
                if hb > best_both:
                    best_both = hb
                    best_state = {k: v.cpu().clone() for k, v in multi.state_dict().items()}
                    best_r_states = [{k: v.cpu().clone() for k, v in r.state_dict().items()} for r in receivers]
            elapsed = time.time() - t0
            eta = elapsed / (ep + 1) * (COMM_EPOCHS - ep - 1)
            ns = f" NaN={nan_count}" if nan_count else ""
            print(f"      Ep {ep+1:3d}: mass={hm:.1%} rest={hr:.1%} both={hb:.1%}{ns}  ETA {eta/60:.0f}min", flush=True)

    if best_state:
        multi.load_state_dict(best_state)

    # Compositionality
    comp = compute_compositionality(multi, agent_views, features, mass_values, rest_values, n_agents * N_HEADS)

    dt = time.time() - t0
    return {
        "seed": seed,
        "holdout_mass": float(hm),
        "holdout_rest": float(hr),
        "holdout_both": float(best_both),
        "nan_count": nan_count,
        "time_sec": float(dt),
        **comp,
    }


def train_spring_mass(features, mass_values, obj_names, n_agents, seed):
    """Train one seed of single-property mass communication with stabilized training."""
    n_frames = features.shape[1]
    fpa = n_frames // n_agents
    agent_views = [features[:, i*fpa:(i+1)*fpa, :].float() for i in range(n_agents)]
    msg_dim = n_agents * N_HEADS * VOCAB_SIZE

    unique_objs = sorted(set(obj_names))
    n_holdout = max(4, len(unique_objs) // 5)
    rng = np.random.RandomState(seed * 1000 + 42)
    holdout_objs = set(rng.choice(unique_objs, n_holdout, replace=False))
    train_ids = np.array([i for i, o in enumerate(obj_names) if o not in holdout_objs])
    holdout_ids = np.array([i for i, o in enumerate(obj_names) if o in holdout_objs])

    if len(holdout_ids) < 4:
        return None

    torch.manual_seed(seed)
    np.random.seed(seed)

    senders = [CompositionalSender(
        TemporalEncoder(HIDDEN_DIM, VJEPA_DIM, n_frames=fpa),
        HIDDEN_DIM, VOCAB_SIZE, N_HEADS
    ) for _ in range(n_agents)]
    multi = MultiAgentSender(senders).to(DEVICE)
    receivers = [SinglePropertyReceiver(msg_dim, HIDDEN_DIM).to(DEVICE) for _ in range(N_RECEIVERS)]
    s_opt = torch.optim.Adam(multi.parameters(), lr=SENDER_LR, weight_decay=1e-5)
    r_opts = [torch.optim.Adam(r.parameters(), lr=RECEIVER_LR, weight_decay=1e-5) for r in receivers]

    mass_dev = torch.tensor(mass_values, dtype=torch.float32).to(DEVICE)
    max_ent = math.log(VOCAB_SIZE)
    nb = max(1, len(train_ids) // BATCH_SIZE)
    best_acc = 0.0
    best_state = None
    nan_count = 0
    t0 = time.time()

    for ep in range(COMM_EPOCHS):
        if ep > 0 and ep % RECEIVER_RESET_INTERVAL == 0:
            for i in range(len(receivers)):
                receivers[i] = SinglePropertyReceiver(msg_dim, HIDDEN_DIM).to(DEVICE)
                r_opts[i] = torch.optim.Adam(receivers[i].parameters(), lr=RECEIVER_LR, weight_decay=1e-5)

        multi.train()
        for r in receivers:
            r.train()
        tau = TAU_START + (TAU_END - TAU_START) * ep / max(1, COMM_EPOCHS - 1)
        hard = ep >= SOFT_WARMUP

        for _ in range(nb):
            ia = rng.choice(train_ids, BATCH_SIZE)
            ib = rng.choice(train_ids, BATCH_SIZE)
            same = ia == ib
            while same.any():
                ib[same] = rng.choice(train_ids, same.sum())
                same = ia == ib
            mass_diff = np.abs(mass_values[ia] - mass_values[ib])
            keep = mass_diff > 0.5
            if keep.sum() < 4:
                continue
            ia, ib = ia[keep], ib[keep]

            va = [v[ia].to(DEVICE) for v in agent_views]
            vb = [v[ib].to(DEVICE) for v in agent_views]
            label = (mass_dev[ia] > mass_dev[ib]).float()
            ma, la = multi(va, tau=tau, hard=hard)
            mb, lb = multi(vb, tau=tau, hard=hard)

            total_loss = torch.tensor(0.0, device=DEVICE)
            for r in receivers:
                pred = r(ma, mb)
                total_loss = total_loss + F.binary_cross_entropy_with_logits(pred, label)
            loss = total_loss / len(receivers)

            for logits in la + lb:
                lp = F.log_softmax(logits, dim=-1)
                p = lp.exp().clamp(min=1e-8)
                ent = -(p * lp).sum(dim=-1).mean()
                if ent / max_ent < ENTROPY_THRESHOLD:
                    loss = loss - ENTROPY_COEF * ent

            if torch.isnan(loss) or torch.isinf(loss):
                s_opt.zero_grad()
                for o in r_opts:
                    o.zero_grad()
                nan_count += 1
                continue

            s_opt.zero_grad()
            for o in r_opts:
                o.zero_grad()
            loss.backward()

            has_nan = any(p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
                         for p in list(multi.parameters()) + [p for r in receivers for p in r.parameters()])
            if has_nan:
                s_opt.zero_grad()
                for o in r_opts:
                    o.zero_grad()
                nan_count += 1
                continue

            torch.nn.utils.clip_grad_norm_(multi.parameters(), GRAD_CLIP)
            for r in receivers:
                torch.nn.utils.clip_grad_norm_(r.parameters(), GRAD_CLIP)
            s_opt.step()
            for o in r_opts:
                o.step()

        if ep % 50 == 0:
            torch.mps.empty_cache()

        if (ep + 1) % 50 == 0 or ep == 0:
            multi.eval()
            for r in receivers:
                r.eval()
            with torch.no_grad():
                correct = total = 0
                er = np.random.RandomState(999)
                for _ in range(30):
                    bs = min(BATCH_SIZE, len(holdout_ids))
                    ia_h = er.choice(holdout_ids, bs)
                    ib_h = er.choice(holdout_ids, bs)
                    same_h = ia_h == ib_h
                    while same_h.any():
                        ib_h[same_h] = er.choice(holdout_ids, same_h.sum())
                        same_h = ia_h == ib_h
                    md = np.abs(mass_values[ia_h] - mass_values[ib_h])
                    keep_h = md > 0.5
                    if keep_h.sum() < 2:
                        continue
                    ia_h, ib_h = ia_h[keep_h], ib_h[keep_h]
                    va_h = [v[ia_h].to(DEVICE) for v in agent_views]
                    vb_h = [v[ib_h].to(DEVICE) for v in agent_views]
                    la_h = mass_dev[ia_h] > mass_dev[ib_h]
                    ma_h, _ = multi(va_h)
                    mb_h, _ = multi(vb_h)
                    for r in receivers:
                        pred_h = r(ma_h, mb_h) > 0
                        correct += (pred_h == la_h).sum().item()
                        total += len(la_h)
                acc = correct / max(total, 1)
                if acc > best_acc:
                    best_acc = acc
                    best_state = {k: v.cpu().clone() for k, v in multi.state_dict().items()}
            elapsed = time.time() - t0
            eta = elapsed / (ep + 1) * (COMM_EPOCHS - ep - 1)
            ns = f" NaN={nan_count}" if nan_count else ""
            print(f"      Ep {ep+1:3d}: holdout={acc:.1%} best={best_acc:.1%}{ns}  ETA {eta/60:.0f}min", flush=True)

    if best_state:
        multi.load_state_dict(best_state)

    comp = compute_single_property_metrics(multi, agent_views, features, mass_values, n_agents * N_HEADS)
    dt = time.time() - t0
    return {
        "seed": seed,
        "holdout_acc": float(best_acc),
        "nan_count": nan_count,
        "time_sec": float(dt),
        **comp,
    }


# ═══ Main ═══
if __name__ == "__main__":
    t_start = time.time()
    print("=" * 70, flush=True)
    print("Phase 87b: Clean Restitution + Stabilized Training", flush=True)
    print("=" * 70, flush=True)
    print(f"Device: {DEVICE}", flush=True)
    print(f"Stabilized: lr_sender={SENDER_LR}, lr_receiver={RECEIVER_LR}, grad_clip={GRAD_CLIP}", flush=True)

    results = {}

    # ═══ PART 1: Clean restitution ═══
    print("\n" + "=" * 60, flush=True)
    print("PART 1: Clean Restitution Labels", flush=True)
    print("=" * 60, flush=True)

    with open(RESULTS_DIR / "phase87_phys101_restitution_labels.json") as f:
        rest_data = json.load(f)

    print(f"  Original valid trials: {len(rest_data)}", flush=True)

    # Filter: 0.05 < e < 0.95
    clean_data = [r for r in rest_data if 0.05 < r["restitution"] < 0.95]
    print(f"  After filtering 0.05 < e < 0.95: {len(clean_data)}", flush=True)

    # Also remove trials where bounce apex above drop point (h_bounce > h_drop)
    clean_data2 = [r for r in clean_data if r["h_bounce"] <= r["h_drop"]]
    print(f"  After removing bounce>drop: {len(clean_data2)}", flush=True)

    # Show distribution
    e_vals = [r["restitution"] for r in clean_data2]
    print(f"  Restitution range: [{min(e_vals):.3f}, {max(e_vals):.3f}]", flush=True)
    print(f"  Restitution mean: {np.mean(e_vals):.3f} ± {np.std(e_vals):.3f}", flush=True)

    by_mat = defaultdict(list)
    for r in clean_data2:
        mat = r["obj"].rsplit("_", 1)[0]
        by_mat[mat].append(r["restitution"])
    print("  By material:", flush=True)
    for mat in sorted(by_mat.keys()):
        vals = by_mat[mat]
        print(f"    {mat:15s}: n={len(vals):3d}, e={np.mean(vals):.3f} ± {np.std(vals):.3f}", flush=True)

    results["clean_restitution"] = {
        "original_count": len(rest_data),
        "filtered_count": len(clean_data2),
        "e_range": [float(min(e_vals)), float(max(e_vals))],
        "e_mean": float(np.mean(e_vals)),
        "e_std": float(np.std(e_vals)),
    }

    # ═══ PART 2: Two-property compositionality with cleaned data ═══
    if len(clean_data2) >= 100:
        print("\n" + "=" * 60, flush=True)
        print("PART 2: Two-Property Compositionality (Cleaned Restitution)", flush=True)
        print("=" * 60, flush=True)

        # Load fall features
        fall_data = torch.load(RESULTS_DIR / "phase87_phys101_fall_features.pt", weights_only=False)
        fall_feat = fall_data["features"]  # (666, 8, 1024)
        fall_objs = fall_data["obj_names"]
        fall_mass = fall_data["mass_values"]

        # Build per-object mean restitution from cleaned data
        obj_rest = defaultdict(list)
        for r in clean_data2:
            obj_rest[r["obj"]].append(r["restitution"])
        obj_rest_mean = {o: np.mean(v) for o, v in obj_rest.items()}

        # Match features to restitution
        matched_idx, matched_mass, matched_rest, matched_objs = [], [], [], []
        for i, obj in enumerate(fall_objs):
            if obj in obj_rest_mean:
                matched_idx.append(i)
                matched_mass.append(fall_mass[i])
                matched_rest.append(obj_rest_mean[obj])
                matched_objs.append(obj)

        n_matched = len(matched_idx)
        print(f"  Matched trials: {n_matched}", flush=True)

        if n_matched >= 50:
            matched_feat = fall_feat[matched_idx]
            matched_mass_arr = np.array(matched_mass)
            matched_rest_arr = np.array(matched_rest)

            rho, p = stats.spearmanr(matched_mass_arr, matched_rest_arr)
            print(f"  Mass-restitution Spearman rho = {rho:.3f} (p={p:.3f})", flush=True)

            n_agents = 2
            fpa = 8 // n_agents
            print(f"\n  --- 2 agents × {fpa} frames, stabilized training ---", flush=True)
            two_prop_results = []
            for seed in range(N_SEEDS):
                print(f"    Seed {seed}...", flush=True)
                r = train_two_property(matched_feat, matched_mass_arr, matched_rest_arr,
                                       matched_objs, n_agents, seed)
                if r:
                    two_prop_results.append(r)
                    print(f"      -> both={r['holdout_both']:.1%} PosDis={r['pos_dis']:.3f} "
                          f"TopSim={r['topsim']:.3f} NaN={r['nan_count']} ({r['time_sec']:.0f}s)", flush=True)

            if two_prop_results:
                both_accs = [r["holdout_both"] for r in two_prop_results]
                posdis = [r["pos_dis"] for r in two_prop_results]
                topsims = [r["topsim"] for r in two_prop_results]
                n_comp = sum(1 for r in two_prop_results if r["pos_dis"] > 0.4)
                print(f"\n  TWO-PROPERTY SUMMARY:", flush=True)
                print(f"    Both acc: {np.mean(both_accs):.1%} ± {np.std(both_accs):.1%}", flush=True)
                print(f"    Mass acc: {np.mean([r['holdout_mass'] for r in two_prop_results]):.1%}", flush=True)
                print(f"    Rest acc: {np.mean([r['holdout_rest'] for r in two_prop_results]):.1%}", flush=True)
                print(f"    PosDis: {np.mean(posdis):.3f} ± {np.std(posdis):.3f}", flush=True)
                print(f"    TopSim: {np.mean(topsims):.3f} ± {np.std(topsims):.3f}", flush=True)
                print(f"    Compositional (PosDis>0.4): {n_comp}/{len(two_prop_results)}", flush=True)
                total_nan = sum(r["nan_count"] for r in two_prop_results)
                print(f"    Total NaN events: {total_nan} (vs Phase 87 for comparison)", flush=True)

                results["two_property_clean"] = {
                    "per_seed": two_prop_results,
                    "mean_both": float(np.mean(both_accs)),
                    "std_both": float(np.std(both_accs)),
                    "mean_mass": float(np.mean([r["holdout_mass"] for r in two_prop_results])),
                    "mean_rest": float(np.mean([r["holdout_rest"] for r in two_prop_results])),
                    "mean_posdis": float(np.mean(posdis)),
                    "std_posdis": float(np.std(posdis)),
                    "mean_topsim": float(np.mean(topsims)),
                    "n_compositional": n_comp,
                    "n_seeds": len(two_prop_results),
                    "mass_rest_rho": float(rho),
                    "n_matched_trials": n_matched,
                }

    # ═══ PART 3: Stabilized spring mass ═══
    print("\n" + "=" * 60, flush=True)
    print("PART 3: Spring Mass Communication (Stabilized)", flush=True)
    print("=" * 60, flush=True)

    spring_data = torch.load(RESULTS_DIR / "phase87_phys101_spring_features.pt", weights_only=False)
    spring_feat = spring_data["features"]
    spring_objs = spring_data["obj_names"]
    spring_mass = spring_data["mass_values"]

    n_agents = 2
    fpa = 8 // n_agents
    print(f"  --- 2 agents × {fpa} frames ---", flush=True)
    spring_results = []
    for seed in range(N_SEEDS):
        print(f"    Seed {seed}...", flush=True)
        r = train_spring_mass(spring_feat, spring_mass, spring_objs, n_agents, seed)
        if r:
            spring_results.append(r)
            print(f"      -> acc={r['holdout_acc']:.1%} TopSim={r['topsim']:.3f} "
                  f"rho={r['mass_symbol_rho']:.3f} NaN={r['nan_count']} ({r['time_sec']:.0f}s)", flush=True)

    if spring_results:
        accs = [r["holdout_acc"] for r in spring_results]
        total_nan = sum(r["nan_count"] for r in spring_results)
        print(f"\n  SPRING MASS SUMMARY:", flush=True)
        print(f"    Holdout acc: {np.mean(accs):.1%} ± {np.std(accs):.1%}", flush=True)
        print(f"    TopSim: {np.mean([r['topsim'] for r in spring_results]):.3f}", flush=True)
        print(f"    Best rho: {np.mean([abs(r['mass_symbol_rho']) for r in spring_results]):.3f}", flush=True)
        print(f"    Total NaN events: {total_nan}", flush=True)
        print(f"    Phase 87 comparison: 84.1% ± 5.6%", flush=True)

        results["spring_mass_stabilized"] = {
            "per_seed": spring_results,
            "mean_acc": float(np.mean(accs)),
            "std_acc": float(np.std(accs)),
            "mean_topsim": float(np.mean([r["topsim"] for r in spring_results])),
            "total_nan": total_nan,
            "n_seeds": len(spring_results),
        }

    # ═══ Save ═══
    total_time = time.time() - t_start
    results["total_time_min"] = float(total_time / 60)
    results["config"] = {
        "sender_lr": SENDER_LR,
        "receiver_lr": RECEIVER_LR,
        "grad_clip": GRAD_CLIP,
        "weight_decay": 1e-5,
    }

    save_path = RESULTS_DIR / "phase87b_clean_restitution.json"
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {save_path}", flush=True)

    # ═══ Final comparison ═══
    print("\n" + "=" * 70, flush=True)
    print("PHASE 87b COMPARISON", flush=True)
    print("=" * 70, flush=True)

    if "two_property_clean" in results:
        r87b = results["two_property_clean"]
        print(f"\nTwo-Property (clean restitution, stabilized):", flush=True)
        print(f"  Both:   {r87b['mean_both']:.1%} ± {r87b['std_both']:.1%} (Phase 87: 55.0%)", flush=True)
        print(f"  PosDis: {r87b['mean_posdis']:.3f} ± {r87b['std_posdis']:.3f} (Phase 87: 0.318)", flush=True)
        print(f"  Comp:   {r87b['n_compositional']}/{r87b['n_seeds']} (Phase 87: 3/10)", flush=True)

    if "spring_mass_stabilized" in results:
        r87b_s = results["spring_mass_stabilized"]
        print(f"\nSpring Mass (stabilized):", flush=True)
        print(f"  Acc:    {r87b_s['mean_acc']:.1%} ± {r87b_s['std_acc']:.1%} (Phase 87: 84.1% ± 5.6%)", flush=True)
        print(f"  NaN:    {r87b_s['total_nan']} (Phase 87: many seeds had NaN)", flush=True)

    print(f"\nTotal time: {total_time/60:.1f} min", flush=True)
    print("Phase 87b complete.", flush=True)
