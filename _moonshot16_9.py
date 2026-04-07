"""
Moonshot #16: Audio Third Modality
Moonshot #9: End-to-End Perception

Run:
  PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=1 python3 _moonshot16_9.py
"""

import time, json, math, os, sys, traceback
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from itertools import combinations
from scipy import signal as scipy_signal
from scipy.optimize import linear_sum_assignment

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
RESULTS_BASE = Path("results")
HIDDEN_DIM = 128
VOCAB_SIZE = 3
N_HEADS = 2
N_AGENTS = 4
MSG_DIM = N_AGENTS * N_HEADS * VOCAB_SIZE
COMM_EPOCHS = 600
BATCH_SIZE = 32
EARLY_STOP = 200
START_TIME = time.time()
TIMING = []

sys.path.insert(0, os.path.dirname(__file__))
from _fix_exp3_exp4 import (
    TemporalEncoder, DiscreteSender, DiscreteMultiSender,
    ContinuousSender, ContinuousMultiSender, Receiver,
    compute_posdis, compute_topsim
)


def elapsed(): return f"{(time.time()-START_TIME)/60:.0f}min"


def load_spring():
    d = torch.load("results/phase87_phys101_spring_features.pt", weights_only=False)
    feat = d["features"].float(); obj = d["obj_names"]; mass = d["mass_values"]
    p1 = np.digitize(mass, np.quantile(mass, [0.2, 0.4, 0.6, 0.8]))
    uo = sorted(set(obj)); oi = {o: i for i, o in enumerate(uo)}
    p2 = np.digitize(np.array([oi[o] for o in obj]),
                      np.quantile(np.arange(len(uo)), [0.2, 0.4, 0.6, 0.8]))
    return feat, p1, p2, obj, mass


def train_modality(feat, mass, obj, seed):
    """Train discrete sender+receiver, return (sender, receiver, views, acc, tokens)."""
    t0 = time.time()
    n, nf, dim = feat.shape; fpa = max(1, nf // N_AGENTS)
    views = [feat[:, (i*fpa)%nf:(i*fpa)%nf+fpa, :] for i in range(N_AGENTS)]
    torch.manual_seed(seed); np.random.seed(seed)
    ss = [DiscreteSender(TemporalEncoder(HIDDEN_DIM, dim, fpa), HIDDEN_DIM, VOCAB_SIZE, N_HEADS)
          for _ in range(N_AGENTS)]
    sender = DiscreteMultiSender(ss).to(DEVICE)
    recvs = [Receiver(MSG_DIM, HIDDEN_DIM).to(DEVICE) for _ in range(3)]
    so = torch.optim.Adam(sender.parameters(), lr=1e-3)
    ros = [torch.optim.Adam(r.parameters(), lr=3e-3) for r in recvs]
    mass_dev = torch.tensor(mass, dtype=torch.float32).to(DEVICE)
    rng = np.random.RandomState(seed*1000+42)
    uo = sorted(set(obj)); ho = set(rng.choice(uo, max(4, len(uo)//5), replace=False))
    tr = np.array([i for i, o in enumerate(obj) if o not in ho])
    tei = np.array([i for i, o in enumerate(obj) if o in ho])
    nb = max(1, len(tr)//32); me = math.log(VOCAB_SIZE)
    ba, bst, bep = 0.0, None, 0
    for ep in range(COMM_EPOCHS):
        if ep-bep > EARLY_STOP and ba > 0.55: break
        if ep > 0 and ep % 40 == 0:
            for i in range(3): recvs[i]=Receiver(MSG_DIM,HIDDEN_DIM).to(DEVICE); ros[i]=torch.optim.Adam(recvs[i].parameters(),lr=3e-3)
        sender.train(); [r.train() for r in recvs]
        tau=3+(1-3)*ep/max(1,COMM_EPOCHS-1); hard=ep>=30
        for _ in range(nb):
            ia=rng.choice(tr,32);ib=rng.choice(tr,32);s=ia==ib
            while s.any():ib[s]=rng.choice(tr,s.sum());s=ia==ib
            md=np.abs(mass[ia]-mass[ib]);k=md>0.5
            if k.sum()<4:continue
            ia,ib=ia[k],ib[k]
            va=[v[ia].to(DEVICE) for v in views];vb=[v[ib].to(DEVICE) for v in views]
            lab=(mass_dev[ia]>mass_dev[ib]).float()
            ma,la=sender(va,tau,hard);mb,lb=sender(vb,tau,hard)
            loss=sum(F.binary_cross_entropy_with_logits(r(ma,mb),lab) for r in recvs)/3
            for lg in la+lb:
                lp=F.log_softmax(lg,-1);p=lp.exp().clamp(1e-8);ent=-(p*lp).sum(-1).mean()
                if ent/me<0.1:loss=loss-0.03*ent
            if torch.isnan(loss):so.zero_grad();[o.zero_grad() for o in ros];continue
            so.zero_grad();[o.zero_grad() for o in ros];loss.backward()
            torch.nn.utils.clip_grad_norm_(sender.parameters(),1.0);so.step();[o.step() for o in ros]
        if ep%50==0:torch.mps.empty_cache()
        if (ep+1)%50==0 or ep==0:
            sender.eval();[r.eval() for r in recvs]
            with torch.no_grad():
                c=t=0;er=np.random.RandomState(999)
                for _ in range(30):
                    ia_h=er.choice(tei,min(32,len(tei)));ib_h=er.choice(tei,min(32,len(tei)))
                    vh=[v[ia_h].to(DEVICE) for v in views];wh=[v[ib_h].to(DEVICE) for v in views]
                    mah,_=sender(vh);mbh,_=sender(wh)
                    for r in recvs:c+=((r(mah,mbh)>0)==(mass_dev[ia_h]>mass_dev[ib_h])).sum().item();t+=len(ia_h)
                acc=c/max(t,1)
                if acc>ba:ba=acc;bep=ep;bst={kk:vv.cpu().clone() for kk,vv in sender.state_dict().items()}
    if bst:sender.load_state_dict(bst)
    sender.eval();[r.eval() for r in recvs]
    # Tokens
    toks=[]
    with torch.no_grad():
        for i in range(0,n,BATCH_SIZE):
            v2=[vi[i:i+BATCH_SIZE].to(DEVICE) for vi in views]
            _,logits=sender(v2)
            toks.append(np.stack([l.argmax(-1).cpu().numpy() for l in logits],1))
    tokens=np.concatenate(toks,0)
    # Messages
    msgs=[]
    with torch.no_grad():
        for i in range(0,n,BATCH_SIZE):
            v2=[vi[i:i+BATCH_SIZE].to(DEVICE) for vi in views]
            m,_=sender(v2);msgs.append(m.cpu())
    msgs=torch.cat(msgs,0)
    elapsed_s=time.time()-t0; TIMING.append(elapsed_s)
    return sender, recvs[0], views, ba, tokens, msgs


def get_assignments(tokens, p1, p2):
    attrs = np.stack([p1, p2], axis=1)
    _, mi = compute_posdis(tokens, attrs, VOCAB_SIZE)
    return [int(np.argmax(mi[p])) for p in range(mi.shape[0])]


def cross_agreement(a_list, b_list):
    agree = []
    for a in a_list:
        for b in b_list:
            match = sum(1 for x, y in zip(a, b) if x == y)
            agree.append(match / len(a))
    return float(np.mean(agree))


def hungarian_transfer(sender_msgs, recv, mass, n):
    """Test receiver accuracy on sender's messages."""
    mass_dev = torch.tensor(mass, dtype=torch.float32).to(DEVICE)
    c = t = 0; er = np.random.RandomState(999)
    with torch.no_grad():
        for _ in range(100):
            ia = er.choice(n, min(32,n)); ib = er.choice(n, min(32,n))
            s = ia==ib
            while s.any(): ib[s]=er.choice(n,s.sum()); s=ia==ib
            md = np.abs(mass[ia]-mass[ib]); k = md>0.5
            if k.sum()<2: continue
            ia,ib = ia[k],ib[k]
            pred = recv(sender_msgs[ia].to(DEVICE), sender_msgs[ib].to(DEVICE)) > 0
            lab = mass_dev[ia] > mass_dev[ib]
            c += (pred==lab).sum().item(); t += len(lab)
    return c/max(t,1)


def align_and_transfer(src_tokens, tgt_tokens, src_msgs, recv, mass, n):
    """Hungarian-align src messages to tgt token space, then test with recv."""
    n_pos = src_tokens.shape[1]
    aligned = src_msgs.clone()
    for pos in range(n_pos):
        cost = np.zeros((VOCAB_SIZE, VOCAB_SIZE))
        for s in range(VOCAB_SIZE):
            for t in range(VOCAB_SIZE):
                cost[s, t] = -np.sum((src_tokens[:, pos]==s)&(tgt_tokens[:, pos]==t))
        ri, ci = linear_sum_assignment(cost)
        perm = np.zeros(VOCAB_SIZE, dtype=int)
        for r, c in zip(ri, ci): perm[r] = c
        start = pos * VOCAB_SIZE
        block = src_msgs[:, start:start+VOCAB_SIZE].clone()
        new_block = torch.zeros_like(block)
        for s in range(VOCAB_SIZE): new_block[:, perm[s]] = block[:, s]
        aligned[:, start:start+VOCAB_SIZE] = new_block
    return hungarian_transfer(aligned, recv, mass, n)


# ═══════════════════════════════════════════════════════════════
# MOONSHOT #16: Audio Third Modality
# ═══════════════════════════════════════════════════════════════

def moonshot16():
    print(f"\n{'#'*60}\n# MOONSHOT #16: Audio Third Modality\n# {elapsed()}\n{'#'*60}", flush=True)
    d = Path("results/crossmodal/audio"); d.mkdir(parents=True, exist_ok=True)

    _, p1, p2, obj, mass = load_spring()
    n = len(mass)

    # STEP 1: Synthesize audio
    print("  Step 1: Synthesizing physics audio...", flush=True)
    sr = 16000; duration = 1.0; n_samples = int(sr * duration)
    mass_norm = (mass - mass.min()) / (mass.max() - mass.min() + 1e-10)

    audio_clips = []
    for i in range(n):
        freq = 100 + (1 - mass_norm[i]) * 900  # Heavy=low, light=high
        t = np.linspace(0, duration, n_samples, dtype=np.float32)
        envelope = np.exp(-3 * t)  # Impact decay
        wave = np.sin(2 * np.pi * freq * t) * envelope
        # Add harmonics
        wave += 0.3 * np.sin(4 * np.pi * freq * t) * envelope
        wave += 0.1 * np.sin(6 * np.pi * freq * t) * envelope
        wave = wave / (np.abs(wave).max() + 1e-10)
        audio_clips.append(wave)

    print(f"    Synthesized {n} clips, {duration}s @ {sr}Hz", flush=True)

    # STEP 2: Extract audio features (mel spectrogram)
    cache = d / "audio_features.pt"
    if cache.exists():
        print("  Step 2: Loading cached audio features...", flush=True)
        audio_feat = torch.load(cache, weights_only=False)["features"]
    else:
        print("  Step 2: Extracting audio features (mel spectrogram)...", flush=True)
        # Compute mel spectrograms
        n_mels = 64; n_fft = 512; hop = 256
        features = []
        for i, clip in enumerate(audio_clips):
            # Simple STFT-based mel approximation
            f, t_bins, Sxx = scipy_signal.spectrogram(clip, fs=sr, nperseg=n_fft, noverlap=n_fft-hop)
            # Log power
            Sxx = np.log(Sxx + 1e-10)
            # Reduce frequency bins to n_mels
            if Sxx.shape[0] > n_mels:
                idx = np.linspace(0, Sxx.shape[0]-1, n_mels, dtype=int)
                Sxx = Sxx[idx]
            # Pool time to fixed length (8 frames to match other modalities)
            if Sxx.shape[1] > 8:
                t_idx = np.linspace(0, Sxx.shape[1]-1, 8, dtype=int)
                Sxx = Sxx[:, t_idx]
            elif Sxx.shape[1] < 8:
                Sxx = np.pad(Sxx, ((0,0),(0,8-Sxx.shape[1])))
            features.append(Sxx.T)  # [8, n_mels]

        audio_feat = torch.tensor(np.stack(features), dtype=torch.float32)  # [N, 8, n_mels]
        torch.save({"features": audio_feat, "sr": sr, "n_mels": n_mels}, cache)

    audio_dim = audio_feat.shape[-1]
    print(f"    Audio features: {audio_feat.shape} ({audio_dim}-dim)", flush=True)

    # STEP 3: Train WMCP on audio
    print(f"\n  Step 3: Training on audio...", flush=True)
    a_assigns = []; a_accs = []; a_senders = []; a_recvs = []; a_all_msgs = []

    for seed in range(10):
        sender, recv, views, acc, tokens, msgs = train_modality(audio_feat, mass, obj, seed)
        assign = get_assignments(tokens, p1, p2)
        a_assigns.append(assign); a_accs.append(acc)
        a_senders.append(sender); a_recvs.append(recv)
        a_all_msgs.append(msgs)

        if seed == 2:
            avg = np.mean(TIMING[-3:])
            print(f"    Measured: {avg:.1f}s/seed", flush=True)
        print(f"    Audio seed {seed}: acc={acc:.1%}", flush=True)
        torch.mps.empty_cache()

    # Train vision and text for comparison (5 seeds each for speed)
    print(f"\n  Training vision + text for comparison...", flush=True)
    vfeat = load_spring()[0]
    tfeat = torch.load("results/crossmodal/text_hidden_states.pt", weights_only=False)["features"].float()

    v_assigns = []; v_senders = []; v_recvs = []; v_all_msgs = []
    t_assigns = []; t_senders = []; t_recvs = []; t_all_msgs = []

    for seed in range(5):
        vs, vr, vv, va, vt, vm = train_modality(vfeat, mass, obj, seed)
        v_assigns.append(get_assignments(vt, p1, p2))
        v_senders.append(vs); v_recvs.append(vr); v_all_msgs.append(vm)

        ts, tr, tv, ta, tt, tm = train_modality(tfeat, mass, obj, seed + 100)
        t_assigns.append(get_assignments(tt, p1, p2))
        t_senders.append(ts); t_recvs.append(tr); t_all_msgs.append(tm)

        print(f"    V/T seed {seed}: v_acc={va:.1%} t_acc={ta:.1%}", flush=True)
        torch.mps.empty_cache()

    # STEP 4: Cross-modal comparison
    print(f"\n  Step 4: Cross-modal comparison...", flush=True)
    av_agree = cross_agreement(a_assigns[:5], v_assigns)
    at_agree = cross_agreement(a_assigns[:5], t_assigns)
    vt_agree = cross_agreement(v_assigns, t_assigns)
    aa_agree = cross_agreement(a_assigns, a_assigns)

    print(f"    Audio↔Vision: {av_agree:.1%}", flush=True)
    print(f"    Audio↔Text:   {at_agree:.1%}", flush=True)
    print(f"    Vision↔Text:  {vt_agree:.1%}", flush=True)
    print(f"    Audio within: {aa_agree:.1%}", flush=True)

    # 3×3 transfer matrix
    print(f"\n  Building 3×3 transfer matrix...", flush=True)
    modalities = {
        "vision": (v_senders, v_recvs, v_all_msgs, vfeat),
        "text": (t_senders, t_recvs, t_all_msgs, tfeat),
        "audio": (a_senders[:5], a_recvs[:5], a_all_msgs[:5], audio_feat),
    }
    mod_names = ["vision", "text", "audio"]

    # Get tokens for alignment
    mod_tokens = {}
    for name, (senders, _, _, feat) in modalities.items():
        toks = []
        fpa = max(1, feat.shape[1] // N_AGENTS)
        views = [feat[:, (i*fpa)%feat.shape[1]:(i*fpa)%feat.shape[1]+fpa, :] for i in range(N_AGENTS)]
        with torch.no_grad():
            for i in range(0, n, BATCH_SIZE):
                v2 = [vi[i:i+BATCH_SIZE].to(DEVICE) for vi in views]
                _, logits = senders[0](v2)
                toks.append(np.stack([l.argmax(-1).cpu().numpy() for l in logits], 1))
        mod_tokens[name] = np.concatenate(toks, 0)

    transfer_matrix = np.zeros((3, 3))
    for si, sname in enumerate(mod_names):
        for ri, rname in enumerate(mod_names):
            if si == ri:
                # Native
                accs = []
                senders_s, _, msgs_s, _ = modalities[sname]
                _, recvs_r, _, _ = modalities[rname]
                for k in range(min(5, len(senders_s), len(recvs_r))):
                    accs.append(hungarian_transfer(msgs_s[k], recvs_r[k], mass, n))
                transfer_matrix[si, ri] = np.mean(accs)
            else:
                # Cross-modal with alignment
                senders_s, _, msgs_s, _ = modalities[sname]
                _, recvs_r, _, _ = modalities[rname]
                accs = []
                for k in range(min(3, len(senders_s), len(recvs_r))):
                    acc = align_and_transfer(
                        mod_tokens[sname], mod_tokens[rname],
                        msgs_s[k], recvs_r[k], mass, n)
                    accs.append(acc)
                transfer_matrix[si, ri] = np.mean(accs)

    print(f"\n  3×3 Transfer Matrix:", flush=True)
    print(f"  {'':>10s} {'Vision':>8s} {'Text':>8s} {'Audio':>8s}", flush=True)
    for si, sname in enumerate(mod_names):
        row = f"  {sname:>10s}"
        for ri in range(3):
            row += f" {transfer_matrix[si, ri]:>7.1%}"
        print(row, flush=True)

    all_offdiag = [transfer_matrix[i, j] for i in range(3) for j in range(3) if i != j]

    if min(all_offdiag) > 0.70:
        print(f"\n  ╔══════════════════════════════════════════════════════╗", flush=True)
        print(f"  ║  THREE-MODALITY CONVERGENCE CONFIRMED                ║", flush=True)
        print(f"  ╚══════════════════════════════════════════════════════╝", flush=True)
        verdict = "THREE-MODALITY CONVERGENCE"
    elif np.mean(all_offdiag) > 0.60:
        verdict = "STRONG CROSS-MODAL TRANSFER ACROSS THREE MODALITIES"
    else:
        verdict = f"PARTIAL — mean off-diagonal {np.mean(all_offdiag):.1%}"

    results = {
        "audio_accuracy": f"{np.mean(a_accs):.1%}±{np.std(a_accs):.1%}",
        "audio_within_agreement": f"{aa_agree:.1%}",
        "audio_vision_agreement": f"{av_agree:.1%}",
        "audio_text_agreement": f"{at_agree:.1%}",
        "vision_text_agreement": f"{vt_agree:.1%}",
        "transfer_matrix": transfer_matrix.tolist(),
        "transfer_labels": mod_names,
        "verdict": verdict,
    }

    with open(d / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Plot transfer matrix
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(transfer_matrix, cmap='YlOrRd', vmin=0.3, vmax=1.0)
    ax.set_xticks(range(3)); ax.set_xticklabels(mod_names)
    ax.set_yticks(range(3)); ax.set_yticklabels(mod_names)
    ax.set_xlabel("Receiver"); ax.set_ylabel("Sender")
    ax.set_title("3-Modality Transfer Matrix (Accuracy)")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{transfer_matrix[i,j]:.0%}", ha='center', va='center',
                   fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax); plt.tight_layout()
    plt.savefig(d / "transfer_matrix.png", dpi=200, bbox_inches='tight'); plt.close()

    print(f"  Saved moonshot16 ({elapsed()})", flush=True)
    return results


# ═══════════════════════════════════════════════════════════════
# MOONSHOT #9: End-to-End Perception
# ═══════════════════════════════════════════════════════════════

class SmallEncoder(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(4),
        )
        self.fc = nn.Linear(128 * 4 * 4, out_dim)

    def forward(self, x):
        return self.fc(self.conv(x).flatten(1))


def moonshot9():
    print(f"\n{'#'*60}\n# MOONSHOT #9: End-to-End Perception\n# {elapsed()}\n{'#'*60}", flush=True)
    d = RESULTS_BASE / "neurips_battery" / "moonshot9_e2e"; d.mkdir(exist_ok=True)

    _, p1, p2, obj, mass = load_spring()
    n = len(mass)

    # Generate synthetic physics images
    print("  Generating synthetic physics images...", flush=True)
    imgs_path = d / "synth_images.pt"
    if imgs_path.exists():
        images = torch.load(imgs_path, weights_only=False)
    else:
        mass_norm = (mass - mass.min()) / (mass.max() - mass.min() + 1e-10)
        uo = sorted(set(obj)); oi = {o: i for i, o in enumerate(uo)}
        obj_idx = np.array([oi[o] for o in obj])
        obj_norm = obj_idx / max(len(uo) - 1, 1)

        images = torch.zeros(n, 3, 64, 64)
        colors = [[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1],
                  [0.5,0.5,0],[0.5,0,0.5],[0,0.5,0.5],[0.7,0.3,0]]

        rng = np.random.RandomState(42)
        for i in range(n):
            radius = int(5 + mass_norm[i] * 20)
            cx, cy = 16 + rng.randint(0, 32), 16 + rng.randint(0, 32)
            color = colors[obj_idx[i] % len(colors)]
            for c in range(3):
                yy, xx = np.ogrid[-cy:64-cy, -cx:64-cx]
                mask = xx*xx + yy*yy <= radius*radius
                images[i, c, mask] = color[c]

        torch.save(images, imgs_path)
    print(f"    Images: {images.shape}", flush=True)

    # Three conditions
    mass_dev = torch.tensor(mass, dtype=torch.float32).to(DEVICE)
    enc_dim = 256
    results = {"conditions": {}}

    for cond_name in ["frozen", "e2e", "supervised"]:
        print(f"\n  ── Condition: {cond_name} ──", flush=True)
        cond_accs = []; cond_pds = []

        for seed in range(10):
            t0 = time.time()
            torch.manual_seed(seed); np.random.seed(seed)

            encoder = SmallEncoder(enc_dim).to(DEVICE)
            if cond_name == "frozen":
                for p in encoder.parameters(): p.requires_grad = False

            # Extract features from images through encoder
            rng = np.random.RandomState(seed * 1000 + 42)
            uo = sorted(set(obj)); ho = set(rng.choice(uo, max(4, len(uo)//5), replace=False))
            tr = np.array([i for i, o in enumerate(obj) if o not in ho])
            tei = np.array([i for i, o in enumerate(obj) if o in ho])

            if cond_name == "supervised":
                # Simple classifier, no bottleneck
                classifier = nn.Sequential(
                    nn.Linear(enc_dim, HIDDEN_DIM), nn.ReLU(),
                    nn.Linear(HIDDEN_DIM, 1)).to(DEVICE)
                opt = torch.optim.Adam(list(encoder.parameters()) + list(classifier.parameters()), lr=1e-3)
                ba = 0
                for ep in range(400):
                    encoder.train(); classifier.train()
                    ia = rng.choice(tr, min(32, len(tr))); ib = rng.choice(tr, min(32, len(tr)))
                    s = ia==ib
                    while s.any(): ib[s]=rng.choice(tr,s.sum()); s=ia==ib
                    fa = encoder(images[ia].to(DEVICE)); fb = encoder(images[ib].to(DEVICE))
                    pred = classifier(fa - fb).squeeze(-1)
                    lab = (mass_dev[ia] > mass_dev[ib]).float()
                    loss = F.binary_cross_entropy_with_logits(pred, lab)
                    opt.zero_grad(); loss.backward(); opt.step()
                    if (ep+1) % 100 == 0:
                        encoder.eval(); classifier.eval()
                        with torch.no_grad():
                            c=t=0; er=np.random.RandomState(999)
                            for _ in range(20):
                                ia_h=er.choice(tei,min(32,len(tei)));ib_h=er.choice(tei,min(32,len(tei)))
                                fa=encoder(images[ia_h].to(DEVICE));fb=encoder(images[ib_h].to(DEVICE))
                                c+=((classifier(fa-fb).squeeze(-1)>0)==(mass_dev[ia_h]>mass_dev[ib_h])).sum().item()
                                t+=len(ia_h)
                            ba=max(ba, c/max(t,1))
                cond_accs.append(ba)
                # PosDis on encoder features
                encoder.eval()
                with torch.no_grad():
                    feats = encoder(images.to(DEVICE)).cpu().numpy()
                binned = np.zeros((n, enc_dim), dtype=int)
                for dim_i in range(min(enc_dim, 20)):  # Only check first 20 dims
                    try:
                        q = np.quantile(feats[:, dim_i], [0.33, 0.67])
                        binned[:, dim_i] = np.digitize(feats[:, dim_i], q)
                    except: pass
                attrs = np.stack([p1, p2], axis=1)
                pd, _ = compute_posdis(binned[:, :20], attrs, 3)
                cond_pds.append(pd)
            else:
                # Discrete bottleneck (frozen or e2e)
                # Build features: encoder output expanded to [N, 8, enc_dim]
                sender_heads = nn.ModuleList([nn.Linear(HIDDEN_DIM, VOCAB_SIZE) for _ in range(N_AGENTS * N_HEADS)]).to(DEVICE)
                enc_proj = nn.Sequential(nn.Linear(enc_dim, HIDDEN_DIM), nn.ReLU()).to(DEVICE)
                recv = Receiver(MSG_DIM, HIDDEN_DIM).to(DEVICE)

                params = list(sender_heads.parameters()) + list(enc_proj.parameters()) + list(recv.parameters())
                if cond_name == "e2e":
                    params += list(encoder.parameters())
                opt = torch.optim.Adam(params, lr=1e-3)

                ba = 0; me = math.log(VOCAB_SIZE)
                for ep in range(400):
                    encoder.train() if cond_name == "e2e" else encoder.eval()
                    enc_proj.train(); sender_heads.train(); recv.train()
                    ia = rng.choice(tr, min(32, len(tr))); ib = rng.choice(tr, min(32, len(tr)))
                    s = ia==ib
                    while s.any(): ib[s]=rng.choice(tr,s.sum()); s=ia==ib
                    fa = enc_proj(encoder(images[ia].to(DEVICE)))
                    fb = enc_proj(encoder(images[ib].to(DEVICE)))
                    lab = (mass_dev[ia] > mass_dev[ib]).float()
                    tau = 3+(1-3)*ep/399; hard = ep >= 30
                    # Generate messages
                    ma_parts = []; mb_parts = []
                    all_logits = []
                    for h in sender_heads:
                        la = h(fa); lb_h = h(fb)
                        if hard:
                            ma_parts.append(F.gumbel_softmax(la, tau=tau, hard=True))
                            mb_parts.append(F.gumbel_softmax(lb_h, tau=tau, hard=True))
                        else:
                            ma_parts.append(F.gumbel_softmax(la, tau=tau, hard=False))
                            mb_parts.append(F.gumbel_softmax(lb_h, tau=tau, hard=False))
                        all_logits.extend([la, lb_h])
                    ma = torch.cat(ma_parts, -1); mb = torch.cat(mb_parts, -1)
                    loss = F.binary_cross_entropy_with_logits(recv(ma, mb), lab)
                    for lg in all_logits:
                        lp=F.log_softmax(lg,-1);p=lp.exp().clamp(1e-8);ent=-(p*lp).sum(-1).mean()
                        if ent/me<0.1:loss=loss-0.03*ent
                    opt.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
                    if (ep+1) % 100 == 0:
                        encoder.eval(); enc_proj.eval(); sender_heads.eval(); recv.eval()
                        with torch.no_grad():
                            c=t=0; er=np.random.RandomState(999)
                            for _ in range(20):
                                ia_h=er.choice(tei,min(32,len(tei)));ib_h=er.choice(tei,min(32,len(tei)))
                                fah=enc_proj(encoder(images[ia_h].to(DEVICE)))
                                fbh=enc_proj(encoder(images[ib_h].to(DEVICE)))
                                mah=torch.cat([F.one_hot(h(fah).argmax(-1),VOCAB_SIZE).float() for h in sender_heads],-1)
                                mbh=torch.cat([F.one_hot(h(fbh).argmax(-1),VOCAB_SIZE).float() for h in sender_heads],-1)
                                c+=((recv(mah,mbh)>0)==(mass_dev[ia_h]>mass_dev[ib_h])).sum().item();t+=len(ia_h)
                            ba=max(ba, c/max(t,1))
                cond_accs.append(ba)

                # PosDis on bottleneck tokens
                encoder.eval(); enc_proj.eval()
                with torch.no_grad():
                    all_toks = []
                    for i in range(0, n, 32):
                        fe = enc_proj(encoder(images[i:i+32].to(DEVICE)))
                        toks = [h(fe).argmax(-1).cpu().numpy() for h in sender_heads]
                        all_toks.append(np.stack(toks, 1))
                    all_toks = np.concatenate(all_toks, 0)
                attrs = np.stack([p1, p2], axis=1)
                pd, _ = compute_posdis(all_toks, attrs, VOCAB_SIZE)
                cond_pds.append(pd)

            elapsed_s = time.time() - t0
            TIMING.append(elapsed_s)

            if seed == 2:
                avg = np.mean(TIMING[-3:])
                total_remain = (10 - seed - 1) + (2 - ["frozen","e2e","supervised"].index(cond_name)) * 10
                print(f"      Measured: {avg:.1f}s/seed, ~{total_remain * avg / 60:.0f}min remaining", flush=True)

            print(f"    {cond_name} seed {seed}: acc={ba:.1%} PD={cond_pds[-1]:.3f}", flush=True)
            torch.mps.empty_cache()

        results["conditions"][cond_name] = {
            "accuracy": f"{np.mean(cond_accs):.1%}±{np.std(cond_accs):.1%}",
            "posdis": f"{np.mean(cond_pds):.3f}±{np.std(cond_pds):.3f}",
            "acc_m": float(np.mean(cond_accs)),
            "pd_m": float(np.mean(cond_pds)),
        }
        print(f"  {cond_name}: acc={results['conditions'][cond_name]['accuracy']} "
              f"PD={results['conditions'][cond_name]['posdis']}", flush=True)

    # Verdict
    frozen = results["conditions"]["frozen"]
    e2e = results["conditions"]["e2e"]
    sup = results["conditions"]["supervised"]

    if e2e["pd_m"] > frozen["pd_m"] + 0.05:
        verdict = "E2E IMPROVES COMPOSITIONALITY — communication pressure creates structure"
    elif e2e["acc_m"] > frozen["acc_m"] + 0.05:
        verdict = "E2E IMPROVES ACCURACY but not compositionality"
    else:
        verdict = "NO E2E BENEFIT — WMCP works best as post-hoc analysis"
    results["verdict"] = verdict

    print(f"\n  ╔═══ END-TO-END PERCEPTION ═══╗", flush=True)
    print(f"  ║ Frozen:     acc={frozen['accuracy']} PD={frozen['posdis']}", flush=True)
    print(f"  ║ End-to-end: acc={e2e['accuracy']} PD={e2e['posdis']}", flush=True)
    print(f"  ║ Supervised: acc={sup['accuracy']} PD={sup['posdis']}", flush=True)
    print(f"  ║ {verdict}", flush=True)
    print(f"  ╚══════════════════════════════╝", flush=True)

    with open(d / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved moonshot9 ({elapsed()})", flush=True)
    return results


# ═══ Main ═══

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗", flush=True)
    print("║  MOONSHOTS #16 + #9                                      ║", flush=True)
    print(f"║  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}                          ║", flush=True)
    print("╚══════════════════════════════════════════════════════════╝", flush=True)

    for name, func in [("#16", moonshot16), ("#9", moonshot9)]:
        try:
            func()
            os.system(f'cd /Users/tomek/AI && git add results/ _moonshot16_9.py '
                      f'&& git commit -m "Moonshot {name}: results\n\n'
                      f'Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>" 2>/dev/null')
        except Exception as e:
            print(f"\n  Moonshot {name} FAILED: {e}", flush=True)
            traceback.print_exc()

    total_h = (time.time() - START_TIME) / 3600
    print(f"\n{'='*60}", flush=True)
    print(f"  COMPLETE. Total: {total_h:.1f} hours", flush=True)
    print(f"{'='*60}", flush=True)
