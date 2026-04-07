"""
Moonshot #18: DINOv2 Cross-Modal
Moonshot #17: LLM Scale Ladder

Run:
  PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=1 python3 _moonshot17_18.py
"""

import time, json, math, os, sys, traceback
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from itertools import combinations
from scipy.optimize import linear_sum_assignment

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
HIDDEN_DIM = 128; VOCAB_SIZE = 3; N_HEADS = 2; N_AGENTS = 4
MSG_DIM = N_AGENTS * N_HEADS * VOCAB_SIZE
COMM_EPOCHS = 600; BATCH_SIZE = 32; EARLY_STOP = 200
START_TIME = time.time(); TIMING = []

sys.path.insert(0, os.path.dirname(__file__))
from _fix_exp3_exp4 import (
    TemporalEncoder, DiscreteSender, DiscreteMultiSender, Receiver,
    compute_posdis, compute_topsim
)


def elapsed(): return f"{(time.time()-START_TIME)/60:.0f}min"


def load_metadata():
    d = torch.load("results/phase87_phys101_spring_features.pt", weights_only=False)
    obj = d["obj_names"]; mass = d["mass_values"]
    p1 = np.digitize(mass, np.quantile(mass, [0.2, 0.4, 0.6, 0.8]))
    uo = sorted(set(obj)); oi = {o: i for i, o in enumerate(uo)}
    p2 = np.digitize(np.array([oi[o] for o in obj]),
                      np.quantile(np.arange(len(uo)), [0.2, 0.4, 0.6, 0.8]))
    return p1, p2, obj, mass


def train_discrete(feat, mass, obj, seed):
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
    toks=[]
    with torch.no_grad():
        for i in range(0,n,BATCH_SIZE):
            v2=[vi[i:i+BATCH_SIZE].to(DEVICE) for vi in views]
            _,logits=sender(v2);toks.append(np.stack([l.argmax(-1).cpu().numpy() for l in logits],1))
    tokens=np.concatenate(toks,0)
    msgs=[]
    with torch.no_grad():
        for i in range(0,n,BATCH_SIZE):
            v2=[vi[i:i+BATCH_SIZE].to(DEVICE) for vi in views]
            m,_=sender(v2);msgs.append(m.cpu())
    msgs=torch.cat(msgs,0)
    TIMING.append(time.time()-t0)
    return sender, recvs[0], tokens, msgs, ba


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


def transfer_acc(src_tokens, tgt_tokens, src_msgs, recv, mass, n):
    n_pos = src_tokens.shape[1]
    aligned = src_msgs.clone()
    for pos in range(n_pos):
        cost = np.zeros((VOCAB_SIZE, VOCAB_SIZE))
        for s in range(VOCAB_SIZE):
            for t in range(VOCAB_SIZE):
                cost[s,t] = -np.sum((src_tokens[:,pos]==s)&(tgt_tokens[:,pos]==t))
        ri, ci = linear_sum_assignment(cost)
        perm = np.zeros(VOCAB_SIZE, dtype=int)
        for r, c in zip(ri, ci): perm[r] = c
        start = pos * VOCAB_SIZE
        block = src_msgs[:, start:start+VOCAB_SIZE].clone()
        new_block = torch.zeros_like(block)
        for s in range(VOCAB_SIZE): new_block[:, perm[s]] = block[:, s]
        aligned[:, start:start+VOCAB_SIZE] = new_block
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
            pred = recv(aligned[ia].to(DEVICE), aligned[ib].to(DEVICE)) > 0
            lab = mass_dev[ia] > mass_dev[ib]
            c += (pred==lab).sum().item(); t += len(lab)
    return c/max(t,1)


# ═══════════════════════════════════════════════════════════════
# MOONSHOT #18: DINOv2 Cross-Modal
# ═══════════════════════════════════════════════════════════════

def moonshot18():
    print(f"\n{'#'*60}\n# MOONSHOT #18: DINOv2 Cross-Modal\n# {elapsed()}\n{'#'*60}", flush=True)
    d = Path("results/crossmodal/dinov2_crossmodal"); d.mkdir(parents=True, exist_ok=True)

    p1, p2, obj, mass = load_metadata()
    n = len(mass)

    # Load ALL cached features
    vjepa_feat = torch.load("results/phase87_phys101_spring_features.pt", weights_only=False)["features"].float()
    dino_static = torch.load("results/phase87_phys101_spring_static.pt", weights_only=False)["features"].float()
    dino_feat = dino_static.unsqueeze(1).expand(-1, 8, -1).contiguous()
    text_feat = torch.load("results/crossmodal/text_hidden_states.pt", weights_only=False)["features"].float()

    # Audio features if available
    audio_path = Path("results/crossmodal/audio/audio_features.pt")
    audio_feat = torch.load(audio_path, weights_only=False)["features"].float() if audio_path.exists() else None

    # CLIP features if available
    clip_path = Path("results/phase96_phys101_spring_clip.pt")
    if clip_path.exists():
        clip_static = torch.load(clip_path, weights_only=False)["features"].float()
        clip_feat = clip_static.unsqueeze(1).expand(-1, 8, -1).contiguous()
    else:
        clip_feat = None

    # Train DINOv2 senders
    print("  Training DINOv2 senders (10 seeds)...", flush=True)
    dino_assigns = []; dino_tokens_all = []; dino_msgs_all = []; dino_recvs = []
    for seed in range(10):
        sender, recv, tokens, msgs, acc = train_discrete(dino_feat, mass, obj, seed)
        assign = get_assignments(tokens, p1, p2)
        dino_assigns.append(assign); dino_tokens_all.append(tokens)
        dino_msgs_all.append(msgs); dino_recvs.append(recv)
        if seed == 2:
            avg = np.mean(TIMING[-3:])
            print(f"    Measured: {avg:.1f}s/seed", flush=True)
        print(f"    DINOv2 seed {seed}: acc={acc:.1%}", flush=True)
        torch.mps.empty_cache()

    # Train comparison senders (5 seeds each for speed)
    modalities = {"dinov2": (dino_assigns[:5], dino_tokens_all[:5], dino_msgs_all[:5], dino_recvs[:5])}

    for mod_name, mod_feat in [("vjepa2", vjepa_feat), ("text", text_feat),
                                ("clip", clip_feat), ("audio", audio_feat)]:
        if mod_feat is None: continue
        print(f"  Training {mod_name} senders (5 seeds)...", flush=True)
        assigns = []; tokens_list = []; msgs_list = []; recvs_list = []
        for seed in range(5):
            sender, recv, tokens, msgs, acc = train_discrete(mod_feat, mass, obj, seed)
            assigns.append(get_assignments(tokens, p1, p2))
            tokens_list.append(tokens); msgs_list.append(msgs); recvs_list.append(recv)
            torch.mps.empty_cache()
        modalities[mod_name] = (assigns, tokens_list, msgs_list, recvs_list)
        print(f"    {mod_name} done", flush=True)

    # Build full agreement matrix
    mod_names = list(modalities.keys())
    n_mods = len(mod_names)
    agree_matrix = np.zeros((n_mods, n_mods))

    print(f"\n  Agreement matrix ({n_mods}×{n_mods}):", flush=True)
    for i, mi in enumerate(mod_names):
        for j, mj in enumerate(mod_names):
            agree_matrix[i, j] = cross_agreement(modalities[mi][0], modalities[mj][0])

    header = f"  {'':>10s}" + "".join(f"{m:>10s}" for m in mod_names)
    print(header, flush=True)
    for i, mi in enumerate(mod_names):
        row = f"  {mi:>10s}" + "".join(f"{agree_matrix[i,j]:>9.1%}" for j in range(n_mods))
        print(row, flush=True)

    # Transfer: DINOv2 ↔ Text
    print(f"\n  Cross-modal transfer (DINOv2 ↔ Text):", flush=True)
    d2t_accs = []; t2d_accs = []
    for k in range(min(5, len(dino_tokens_all), len(modalities["text"][1]))):
        d2t = transfer_acc(dino_tokens_all[k], modalities["text"][1][k],
                           dino_msgs_all[k], modalities["text"][3][k], mass, n)
        t2d = transfer_acc(modalities["text"][1][k], dino_tokens_all[k],
                           modalities["text"][2][k], dino_recvs[k], mass, n)
        d2t_accs.append(d2t); t2d_accs.append(t2d)
    print(f"    DINOv2→Text recv: {np.mean(d2t_accs):.1%}±{np.std(d2t_accs):.1%}", flush=True)
    print(f"    Text→DINOv2 recv: {np.mean(t2d_accs):.1%}±{np.std(t2d_accs):.1%}", flush=True)

    results = {
        "agreement_matrix": agree_matrix.tolist(),
        "mod_names": mod_names,
        "dinov2_text_transfer": f"{np.mean(d2t_accs):.1%}±{np.std(d2t_accs):.1%}",
        "text_dinov2_transfer": f"{np.mean(t2d_accs):.1%}±{np.std(t2d_accs):.1%}",
        "d2t_m": float(np.mean(d2t_accs)),
        "t2d_m": float(np.mean(t2d_accs)),
    }

    # Plot heatmap
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(agree_matrix, cmap='YlOrRd', vmin=0.5, vmax=1.0)
    ax.set_xticks(range(n_mods)); ax.set_xticklabels(mod_names, rotation=30)
    ax.set_yticks(range(n_mods)); ax.set_yticklabels(mod_names)
    ax.set_title("Full Cross-Modal/Architecture Agreement Matrix", fontweight='bold')
    for i in range(n_mods):
        for j in range(n_mods):
            ax.text(j, i, f"{agree_matrix[i,j]:.0%}", ha='center', va='center',
                   fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax); plt.tight_layout()
    plt.savefig(d / "full_agreement_matrix.png", dpi=200, bbox_inches='tight'); plt.close()

    with open(d / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved moonshot18 ({elapsed()})", flush=True)
    return results


# ═══════════════════════════════════════════════════════════════
# MOONSHOT #17: LLM Scale Ladder
# ═══════════════════════════════════════════════════════════════

def moonshot17():
    print(f"\n{'#'*60}\n# MOONSHOT #17: LLM Scale Ladder\n# {elapsed()}\n{'#'*60}", flush=True)
    d = Path("results/crossmodal/scale_ladder"); d.mkdir(parents=True, exist_ok=True)

    p1, p2, obj, mass = load_metadata()
    n = len(mass)
    vjepa_feat = torch.load("results/phase87_phys101_spring_features.pt", weights_only=False)["features"].float()
    tinyllama_feat = torch.load("results/crossmodal/text_hidden_states.pt", weights_only=False)["features"].float()

    # Generate text descriptions (same as cross-modal experiment)
    mass_q = np.quantile(mass, [0.33, 0.67])
    material_map = {'cardboard': ('cardboard','box'),'rubber': ('rubber','ball'),
                     'metal': ('metal','weight'),'wood': ('wooden','block'),
                     'plastic': ('plastic','container'),'foam': ('foam','cube')}
    descriptions = []
    for i, (o, m) in enumerate(zip(obj, mass)):
        base = o.split('_')[0].lower()
        mat, shape = material_map.get(base, (base, 'object'))
        weight = "light" if m < mass_q[0] else "heavy" if m > mass_q[1] else "medium-weight"
        descriptions.append(f"A {weight} {mat} {shape} oscillates on a spring. "
                           f"The object is weighing {m:.0f}g. Material: {mat}. Mass category: {weight}.")

    # Try downloading a larger LLM
    models_to_try = [
        ("Qwen/Qwen2.5-1.5B-Instruct", "qwen15b"),
        ("microsoft/phi-2", "phi2"),
        ("google/gemma-2-2b", "gemma2b"),
    ]

    new_feat = None; new_model_name = None

    for model_id, short_name in models_to_try:
        cache_path = d / f"{short_name}_features.pt"
        if cache_path.exists():
            print(f"  Loading cached {short_name} features...", flush=True)
            new_feat = torch.load(cache_path, weights_only=False)["features"].float()
            new_model_name = short_name
            break

        try:
            print(f"  Trying {model_id}...", flush=True)
            from transformers import AutoTokenizer, AutoModelForCausalLM
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch.float32, trust_remote_code=True,
                output_hidden_states=True)
            model.eval()
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            n_layers = model.config.num_hidden_layers
            target_layer = n_layers // 2
            hidden_dim = model.config.hidden_size
            print(f"    Loaded: {n_layers} layers, {hidden_dim}-dim, extracting layer {target_layer}", flush=True)

            features = []
            for i, desc in enumerate(descriptions):
                inputs = tokenizer(desc, return_tensors="pt", padding=True, truncation=True, max_length=128)
                with torch.no_grad():
                    outputs = model(**inputs)
                    hidden = outputs.hidden_states[target_layer]
                    pooled = hidden.mean(dim=1).squeeze(0)
                features.append(pooled)
                if (i+1) % 50 == 0: print(f"      {i+1}/{n}", flush=True)

            features = torch.stack(features).unsqueeze(1).expand(-1, 8, -1).contiguous().float()
            del model, tokenizer; torch.mps.empty_cache()
            torch.save({"features": features, "model_name": short_name, "model_id": model_id}, cache_path)
            new_feat = features; new_model_name = short_name
            print(f"    Saved: {features.shape}", flush=True)
            break
        except Exception as e:
            print(f"    Failed: {e}", flush=True)

    if new_feat is None:
        print("  No larger LLM available. Skipping scale comparison.", flush=True)
        results = {"error": "no model available"}
        with open(d / "results.json", "w") as f:
            json.dump(results, f, indent=2)
        return results

    print(f"\n  Using {new_model_name} ({new_feat.shape[-1]}-dim)", flush=True)

    # Train on new LLM features
    print(f"  Training WMCP on {new_model_name}...", flush=True)
    new_assigns = []; new_tokens_all = []; new_msgs_all = []; new_recvs = []; new_accs = []
    for seed in range(10):
        sender, recv, tokens, msgs, acc = train_discrete(new_feat, mass, obj, seed)
        new_assigns.append(get_assignments(tokens, p1, p2))
        new_tokens_all.append(tokens); new_msgs_all.append(msgs)
        new_recvs.append(recv); new_accs.append(acc)
        if seed == 2:
            avg = np.mean(TIMING[-3:])
            print(f"    Measured: {avg:.1f}s/seed", flush=True)
        print(f"    {new_model_name} seed {seed}: acc={acc:.1%}", flush=True)
        torch.mps.empty_cache()

    # Train V-JEPA and TinyLlama for comparison
    print(f"  Training V-JEPA 2 + TinyLlama for comparison...", flush=True)
    v_assigns = []; v_tokens = []; v_msgs = []; v_recvs = []
    t_assigns = []; t_tokens = []; t_msgs = []; t_recvs = []
    for seed in range(5):
        vs, vr, vt, vm, va = train_discrete(vjepa_feat, mass, obj, seed)
        v_assigns.append(get_assignments(vt, p1, p2))
        v_tokens.append(vt); v_msgs.append(vm); v_recvs.append(vr)

        ts, tr, tt, tm, ta = train_discrete(tinyllama_feat, mass, obj, seed + 100)
        t_assigns.append(get_assignments(tt, p1, p2))
        t_tokens.append(tt); t_msgs.append(tm); t_recvs.append(tr)
        torch.mps.empty_cache()

    # Agreements
    new_v_agree = cross_agreement(new_assigns[:5], v_assigns)
    new_t_agree = cross_agreement(new_assigns[:5], t_assigns)
    v_t_agree = cross_agreement(v_assigns, t_assigns)

    # Transfer
    new2v = []; v2new = []; new2t = []; t2new = []
    for k in range(min(5, len(new_tokens_all))):
        new2v.append(transfer_acc(new_tokens_all[k], v_tokens[k], new_msgs_all[k], v_recvs[k], mass, n))
        v2new.append(transfer_acc(v_tokens[k], new_tokens_all[k], v_msgs[k], new_recvs[k], mass, n))
        new2t.append(transfer_acc(new_tokens_all[k], t_tokens[k], new_msgs_all[k], t_recvs[k], mass, n))
        t2new.append(transfer_acc(t_tokens[k], new_tokens_all[k], t_msgs[k], new_recvs[k], mass, n))

    print(f"\n  ╔═══ SCALE LADDER ═══╗", flush=True)
    print(f"  ║ {'Model':<20s} │ {'↔V-JEPA':>8s} │ {'V→M':>6s} │ {'M→V':>6s}", flush=True)
    print(f"  ║ {'TinyLlama 1.1B':<20s} │ {v_t_agree:>7.1%} │ {'86.0%':>6s} │ {'81.5%':>6s}  (baseline)", flush=True)
    print(f"  ║ {new_model_name:<20s} │ {new_v_agree:>7.1%} │ {np.mean(v2new):>5.1%} │ {np.mean(new2v):>5.1%}", flush=True)
    print(f"  ║", flush=True)
    print(f"  ║ {new_model_name}↔TinyLlama: {new_t_agree:.1%}", flush=True)
    print(f"  ╚══════════════════════╝", flush=True)

    results = {
        "new_model": new_model_name,
        "new_accuracy": f"{np.mean(new_accs):.1%}±{np.std(new_accs):.1%}",
        "new_vjepa_agreement": f"{new_v_agree:.1%}",
        "new_tinyllama_agreement": f"{new_t_agree:.1%}",
        "vjepa_tinyllama_agreement": f"{v_t_agree:.1%}",
        "new_to_vjepa_transfer": f"{np.mean(new2v):.1%}±{np.std(new2v):.1%}",
        "vjepa_to_new_transfer": f"{np.mean(v2new):.1%}±{np.std(v2new):.1%}",
        "new_to_tinyllama_transfer": f"{np.mean(new2t):.1%}±{np.std(new2t):.1%}",
        "tinyllama_to_new_transfer": f"{np.mean(t2new):.1%}±{np.std(t2new):.1%}",
    }

    with open(d / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved moonshot17 ({elapsed()})", flush=True)
    return results


# ═══ Main ═══

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗", flush=True)
    print("║  MOONSHOTS #18 + #17                                     ║", flush=True)
    print(f"║  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}                          ║", flush=True)
    print("╚══════════════════════════════════════════════════════════╝", flush=True)

    for name, func in [("#18", moonshot18), ("#17", moonshot17)]:
        try:
            func()
            os.system(f'cd /Users/tomek/AI && git add results/crossmodal/ _moonshot17_18.py '
                      f'&& git commit -m "Moonshot {name}: results\n\n'
                      f'Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>" 2>/dev/null')
        except Exception as e:
            print(f"\n  Moonshot {name} FAILED: {e}", flush=True)
            traceback.print_exc()

    total_h = (time.time() - START_TIME) / 3600
    print(f"\n{'='*60}", flush=True)
    print(f"  COMPLETE. Total: {total_h:.1f} hours", flush=True)
    print(f"{'='*60}", flush=True)
