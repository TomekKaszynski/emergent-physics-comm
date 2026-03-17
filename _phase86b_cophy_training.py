"""
Phase 86b: Train Communication from Scratch on CoPhy
=====================================================
Train the full multi-agent pipeline on real CoPhy collision videos.
If compositionality emerges on real video, this transforms the paper.

Run:
  PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=1 python3 _phase86b_cophy_training.py
"""

import time, json, math, os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from scipy import stats

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
RESULTS_DIR = Path("results")

HIDDEN_DIM = 128; VJEPA_DIM = 1024; VOCAB_SIZE = 5; N_HEADS = 2
BATCH_SIZE = 32
COMM_EPOCHS = 400; SENDER_LR = 1e-3; RECEIVER_LR = 3e-3
TAU_START = 3.0; TAU_END = 1.0; SOFT_WARMUP = 30
ENTROPY_THRESHOLD = 0.1; ENTROPY_COEF = 0.03
RECEIVER_RESET_INTERVAL = 40; N_RECEIVERS = 3
N_SEEDS = 10

# CoPhy has 8 temporal positions (from 16 input frames)
# Use 4 agents × 2 frames each
N_AGENTS = 4
FRAMES_PER_AGENT = 2  # 8 temporal / 4 agents


# ═══ Architecture ═══
class TemporalEncoder(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=1024):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv1d(input_dim, 256, kernel_size=min(3, 2), padding=min(3, 2)//2), nn.ReLU(),
            nn.Conv1d(256, 128, kernel_size=min(3, 2), padding=min(3, 2)//2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1))
        self.fc = nn.Sequential(nn.Linear(128, hidden_dim), nn.ReLU())
    def forward(self, x):
        return self.fc(self.temporal(x.permute(0, 2, 1)).squeeze(-1))

class CompositionalSender(nn.Module):
    def __init__(self, encoder, hidden_dim, vocab_size, n_heads):
        super().__init__()
        self.encoder = encoder; self.vocab_size = vocab_size; self.n_heads = n_heads
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, vocab_size) for _ in range(n_heads)])
    def forward(self, x, tau=1.0, hard=True):
        h = self.encoder(x); messages = []; all_logits = []
        for head in self.heads:
            logits = head(h)
            if self.training: msg = F.gumbel_softmax(logits, tau=tau, hard=hard)
            else: msg = F.one_hot(logits.argmax(dim=-1), self.vocab_size).float()
            messages.append(msg); all_logits.append(logits)
        return torch.cat(messages, dim=-1), all_logits

class MultiAgentSender(nn.Module):
    def __init__(self, senders):
        super().__init__(); self.senders = nn.ModuleList(senders)
    def forward(self, views, tau=1.0, hard=True):
        messages = []; all_logits = []
        for sender, view in zip(self.senders, views):
            msg, logits = sender(view, tau=tau, hard=hard)
            messages.append(msg); all_logits.extend(logits)
        return torch.cat(messages, dim=-1), all_logits

class CompositionalReceiver(nn.Module):
    def __init__(self, msg_dim, hidden_dim):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(msg_dim * 2, hidden_dim), nn.ReLU(),
                                     nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU())
        self.head_a = nn.Linear(hidden_dim // 2, 1)
        self.head_b = nn.Linear(hidden_dim // 2, 1)
    def forward(self, msg_a, msg_b):
        h = self.shared(torch.cat([msg_a, msg_b], dim=-1))
        return self.head_a(h).squeeze(-1), self.head_b(h).squeeze(-1)


# ═══ Data ═══
def load_cophy_data():
    """Load CoPhy features and extract comparison labels."""
    print("  Loading CoPhy V-JEPA 2 features...", flush=True)
    feat_data = torch.load('results/phase85_cophy_vjepa2_features.pt', weights_only=True)
    features = feat_data['features'].float()  # (500, 8, 1024)
    n_scenes = features.shape[0]
    print(f"  Features: {features.shape}", flush=True)

    # Load confounders for the same 500 trials
    cophy_dir = Path("cophy/CoPhy_224/collisionCF")
    trial_dirs = sorted([d for d in cophy_dir.iterdir() if d.is_dir() and not d.name.startswith('.')])

    # Subsample same way as Phase 85
    rng_sub = np.random.RandomState(42)
    indices = rng_sub.choice(len(trial_dirs), min(500, len(trial_dirs)), replace=False)
    selected_dirs = [trial_dirs[i] for i in sorted(indices)]

    # Load confounders
    all_conf = []
    for td in selected_dirs[:n_scenes]:
        conf = np.load(td / 'confounders.npy')  # (4, 3)
        all_conf.append(conf)
    confounders = np.stack(all_conf)  # (500, 4, 3)

    # Find the two active objects per scene (mass > 0)
    # For the comparison task, focus on objects 0 and 2 (typically active)
    # CoPhy collision: objects 0 and 2 are the colliding objects
    # Use mass and restitution as the two comparison properties

    # Extract properties: use object 0's mass and restitution
    # Bin them into discrete levels for the comparison game
    mass_obj0 = confounders[:, 0, 0]  # mass of first active object
    rest_obj0 = confounders[:, 0, 2]  # restitution of first active object

    # Handle absent objects (mass=0): filter them out
    active = mass_obj0 > 0
    print(f"  Active scenes (obj0 mass > 0): {active.sum()}/{n_scenes}", flush=True)

    # Bin mass: {1, 2, 5} → bins {0, 1, 2}
    mass_bins = np.zeros(n_scenes, dtype=int)
    mass_bins[mass_obj0 == 1] = 0
    mass_bins[mass_obj0 == 2] = 1
    mass_bins[mass_obj0 == 5] = 2

    # Bin restitution: {0.1, 0.5, 1.0} → bins {0, 1, 2}
    rest_bins = np.zeros(n_scenes, dtype=int)
    rest_bins[rest_obj0 <= 0.15] = 0
    rest_bins[(rest_obj0 > 0.15) & (rest_obj0 <= 0.6)] = 1
    rest_bins[rest_obj0 > 0.6] = 2

    # Only keep active scenes
    active_idx = np.where(active)[0]
    features = features[active_idx]
    mass_bins = mass_bins[active_idx]
    rest_bins = rest_bins[active_idx]
    confounders = confounders[active_idx]

    print(f"  After filtering: {len(features)} scenes", flush=True)
    print(f"  Mass bins: {np.bincount(mass_bins, minlength=3)}", flush=True)
    print(f"  Rest bins: {np.bincount(rest_bins, minlength=3)}", flush=True)

    return features, mass_bins, rest_bins, confounders


def create_splits_3x3(mass_bins, rest_bins):
    """3×3 grid holdout: hold out diagonal cells."""
    holdout_cells = {(0, 0), (1, 1), (2, 2)}  # Diagonal
    train_ids, holdout_ids = [], []
    for i in range(len(mass_bins)):
        if (int(mass_bins[i]), int(rest_bins[i])) in holdout_cells:
            holdout_ids.append(i)
        else:
            train_ids.append(i)
    return np.array(train_ids), np.array(holdout_ids)


# ═══ Training ═══
def train_one_seed(features, mass_bins, rest_bins, seed):
    """Train one seed, return metrics."""
    n_frames = features.shape[1]
    fpa = n_frames // N_AGENTS
    agent_views = [features[:, i*fpa:(i+1)*fpa, :] for i in range(N_AGENTS)]
    train_ids, holdout_ids = create_splits_3x3(mass_bins, rest_bins)
    msg_dim = N_AGENTS * N_HEADS * VOCAB_SIZE

    torch.manual_seed(seed); np.random.seed(seed)
    senders = [CompositionalSender(TemporalEncoder(HIDDEN_DIM, VJEPA_DIM), HIDDEN_DIM, VOCAB_SIZE, N_HEADS) for _ in range(N_AGENTS)]
    multi = MultiAgentSender(senders).to(DEVICE)
    receivers = [CompositionalReceiver(msg_dim, HIDDEN_DIM).to(DEVICE) for _ in range(N_RECEIVERS)]
    s_opt = torch.optim.Adam(multi.parameters(), lr=SENDER_LR)
    r_opts = [torch.optim.Adam(r.parameters(), lr=RECEIVER_LR) for r in receivers]
    rng = np.random.RandomState(seed)
    e_dev = torch.tensor(mass_bins, dtype=torch.float32).to(DEVICE)
    f_dev = torch.tensor(rest_bins, dtype=torch.float32).to(DEVICE)
    max_ent = math.log(VOCAB_SIZE); nb = max(1, len(train_ids) // BATCH_SIZE)
    best_both = 0.0; best_state = None; best_r_states = None; nan_count = 0
    t0 = time.time()

    for ep in range(COMM_EPOCHS):
        if ep > 0 and ep % RECEIVER_RESET_INTERVAL == 0:
            for i in range(len(receivers)):
                receivers[i] = CompositionalReceiver(msg_dim, HIDDEN_DIM).to(DEVICE)
                r_opts[i] = torch.optim.Adam(receivers[i].parameters(), lr=RECEIVER_LR)
        multi.train()
        for r in receivers: r.train()
        tau = TAU_START + (TAU_END - TAU_START) * ep / max(1, COMM_EPOCHS - 1)
        hard = ep >= SOFT_WARMUP

        for _ in range(nb):
            ia, ib = rng.choice(train_ids, size=BATCH_SIZE), rng.choice(train_ids, size=BATCH_SIZE)
            same = ia == ib
            while same.any(): ib[same] = rng.choice(train_ids, size=same.sum()); same = ia == ib
            va = [v[ia].to(DEVICE) for v in agent_views]; vb = [v[ib].to(DEVICE) for v in agent_views]
            le = (e_dev[ia] > e_dev[ib]).float(); lf = (f_dev[ia] > f_dev[ib]).float()
            ma, la = multi(va, tau=tau, hard=hard); mb, lb = multi(vb, tau=tau, hard=hard)
            total = torch.tensor(0.0, device=DEVICE)
            for r in receivers:
                pe, pf = r(ma, mb)
                total = total + F.binary_cross_entropy_with_logits(pe, le) + F.binary_cross_entropy_with_logits(pf, lf)
            loss = total / len(receivers)
            for logits in la + lb:
                lp = F.log_softmax(logits, dim=-1); p = lp.exp().clamp(min=1e-8)
                ent = -(p * lp).sum(dim=-1).mean()
                if ent / max_ent < ENTROPY_THRESHOLD: loss = loss - ENTROPY_COEF * ent
            if torch.isnan(loss) or torch.isinf(loss):
                s_opt.zero_grad()
                for o in r_opts: o.zero_grad()
                nan_count += 1; continue
            s_opt.zero_grad()
            for o in r_opts: o.zero_grad()
            loss.backward()
            has_nan = any(p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
                         for p in list(multi.parameters()) + [p for r in receivers for p in r.parameters()])
            if has_nan:
                s_opt.zero_grad()
                for o in r_opts: o.zero_grad()
                nan_count += 1; continue
            torch.nn.utils.clip_grad_norm_(multi.parameters(), 1.0)
            for r in receivers: torch.nn.utils.clip_grad_norm_(r.parameters(), 1.0)
            s_opt.step()
            for o in r_opts: o.step()

        if ep % 50 == 0: torch.mps.empty_cache()

        if (ep + 1) % 50 == 0 or ep == 0:
            multi.eval()
            for r in receivers: r.eval()
            with torch.no_grad():
                # Evaluate on holdout
                ce = cf = cb = te = tf = tb = 0
                er = np.random.RandomState(999)
                for _ in range(20):
                    bs = min(BATCH_SIZE, len(holdout_ids))
                    ia, ib = er.choice(holdout_ids, bs), er.choice(holdout_ids, bs)
                    same = ia == ib
                    while same.any(): ib[same] = er.choice(holdout_ids, size=same.sum()); same = ia == ib
                    va = [v[ia].to(DEVICE) for v in agent_views]; vb = [v[ib].to(DEVICE) for v in agent_views]
                    ma, _ = multi(va); mb, _ = multi(vb)
                    for r in receivers:
                        pe, pf = r(ma, mb)
                        le = e_dev[ia] > e_dev[ib]; lf = f_dev[ia] > f_dev[ib]
                        ed = torch.tensor(mass_bins[ia] != mass_bins[ib], device=DEVICE)
                        fd = torch.tensor(rest_bins[ia] != rest_bins[ib], device=DEVICE)
                        if ed.sum() > 0: ce += ((pe[ed] > 0) == le[ed]).sum().item(); te += ed.sum().item()
                        if fd.sum() > 0: cf += ((pf[fd] > 0) == lf[fd]).sum().item(); tf += fd.sum().item()
                        bd = ed & fd
                        if bd.sum() > 0:
                            ok = ((pe[bd] > 0) == le[bd]) & ((pf[bd] > 0) == lf[bd])
                            cb += ok.sum().item(); tb += bd.sum().item()
                he = ce / max(te, 1); hf = cf / max(tf, 1); hb = cb / max(tb, 1)
                if hb > best_both:
                    best_both = hb
                    best_state = {k: v.cpu().clone() for k, v in multi.state_dict().items()}
                    best_r_states = [{k: v.cpu().clone() for k, v in r.state_dict().items()} for r in receivers]
            elapsed = time.time() - t0; eta = elapsed / (ep + 1) * (COMM_EPOCHS - ep - 1)
            ns = f"  NaN={nan_count}" if nan_count > 0 else ""
            print(f"    Ep {ep+1:3d}: tau={tau:.2f}  holdout[m={he:.1%} r={hf:.1%} both={hb:.1%}]{ns}  ETA {eta/60:.0f}min", flush=True)

    if best_state: multi.load_state_dict(best_state)
    if best_r_states:
        for r, s in zip(receivers, best_r_states): r.load_state_dict(s)

    # Compute compositionality
    multi.eval()
    all_tokens = []
    with torch.no_grad():
        for i in range(0, len(features), BATCH_SIZE):
            views = [v[i:i+BATCH_SIZE].to(DEVICE) for v in agent_views]
            _, logits = multi(views)
            tokens = [l.argmax(dim=-1).cpu().numpy() for l in logits]
            all_tokens.append(np.stack(tokens, axis=1))
    all_tokens = np.concatenate(all_tokens, axis=0)
    n_pos = all_tokens.shape[1]

    # MI and PosDis
    attrs = np.stack([mass_bins, rest_bins], axis=1)
    mi_matrix = np.zeros((n_pos, 2))
    for p in range(n_pos):
        for a in range(2):
            x, y = all_tokens[:, p], attrs[:, a]
            xv, yv = np.unique(x), np.unique(y)
            n = len(x); mi = 0.0
            for xval in xv:
                for yval in yv:
                    pxy = np.sum((x == xval) & (y == yval)) / n
                    px = np.sum(x == xval) / n; py = np.sum(y == yval) / n
                    if pxy > 0 and px > 0 and py > 0: mi += pxy * np.log(pxy / (px * py))
            mi_matrix[p, a] = mi

    pos_dis = 0.0
    for p in range(n_pos):
        s = np.sort(mi_matrix[p])[::-1]
        if s[0] > 1e-10: pos_dis += (s[0] - s[1]) / s[0]
    pos_dis /= n_pos

    # TopSim
    rng_ts = np.random.RandomState(42)
    md, hd = [], []
    for _ in range(min(5000, len(features) * (len(features) - 1) // 2)):
        i, j = rng_ts.choice(len(features), 2, replace=False)
        md.append(abs(int(mass_bins[i]) - int(mass_bins[j])) + abs(int(rest_bins[i]) - int(rest_bins[j])))
        hd.append(int((all_tokens[i] != all_tokens[j]).sum()))
    topsim, _ = stats.spearmanr(md, hd)
    if np.isnan(topsim): topsim = 0.0

    dt = time.time() - t0
    return {
        'seed': seed, 'holdout_both': float(best_both),
        'holdout_mass': float(he), 'holdout_rest': float(hf),
        'pos_dis': float(pos_dis), 'topsim': float(topsim),
        'mi_matrix': mi_matrix.tolist(), 'nan_count': nan_count,
        'time_sec': float(dt),
    }


if __name__ == "__main__":
    print("Phase 86b: Train Communication on CoPhy", flush=True)
    print(f"Device: {DEVICE}", flush=True)
    t_total = time.time()

    features, mass_bins, rest_bins, confounders = load_cophy_data()
    train_ids, holdout_ids = create_splits_3x3(mass_bins, rest_bins)
    print(f"  Train: {len(train_ids)}, Holdout: {len(holdout_ids)}", flush=True)

    all_results = []
    for seed in range(N_SEEDS):
        print(f"\n  --- Seed {seed} ---", flush=True)
        result = train_one_seed(features, mass_bins, rest_bins, seed)
        all_results.append(result)
        print(f"    -> holdout={result['holdout_both']:.1%}  PosDis={result['pos_dis']:.3f}  "
              f"TopSim={result['topsim']:.3f}  ({result['time_sec']:.0f}s)", flush=True)

    # Summary
    holdouts = [r['holdout_both'] for r in all_results]
    posdis = [r['pos_dis'] for r in all_results]
    topsims = [r['topsim'] for r in all_results]
    n_comp = sum(1 for r in all_results if r['pos_dis'] > 0.4)

    print(f"\n{'='*70}", flush=True)
    print(f"SUMMARY: CoPhy Training ({N_SEEDS} seeds)", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  Holdout: {np.mean(holdouts):.1%} ± {np.std(holdouts):.1%}", flush=True)
    print(f"  PosDis:  {np.mean(posdis):.3f} ± {np.std(posdis):.3f}", flush=True)
    print(f"  TopSim:  {np.mean(topsims):.3f} ± {np.std(topsims):.3f}", flush=True)
    print(f"  Compositional (PosDis>0.4): {n_comp}/{N_SEEDS}", flush=True)

    save_data = {
        'per_seed': all_results,
        'summary': {
            'holdout_mean': float(np.mean(holdouts)), 'holdout_std': float(np.std(holdouts)),
            'posdis_mean': float(np.mean(posdis)), 'posdis_std': float(np.std(posdis)),
            'topsim_mean': float(np.mean(topsims)), 'topsim_std': float(np.std(topsims)),
            'n_compositional': n_comp, 'n_seeds': N_SEEDS,
        },
        'config': {
            'n_agents': N_AGENTS, 'frames_per_agent': FRAMES_PER_AGENT,
            'n_scenes': len(features), 'dataset': 'CoPhy CollisionCF',
            'properties': 'mass (obj0) vs restitution (obj0)',
            'grid': '3x3 (mass∈{1,2,5} × rest∈{0.1,0.5,1})',
            'holdout': 'diagonal cells (0,0),(1,1),(2,2)',
        }
    }

    save_path = RESULTS_DIR / 'phase86b_cophy_training.json'
    with open(save_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved {save_path}", flush=True)

    dt = time.time() - t_total
    print(f"\nPhase 86b complete. Total time: {dt/60:.1f}min", flush=True)
