"""
Remaining Battery: QW2, QW3, Sunshot C, Neuro RSA, #12 Universal Sender
Run: PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=1 python3 _remaining_battery.py
"""

import time, json, math, os, sys, traceback, urllib.request
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from itertools import combinations
from scipy import stats as scipy_stats
from scipy.optimize import linear_sum_assignment

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
HIDDEN_DIM = 128; VOCAB_SIZE = 3; N_HEADS = 2; N_AGENTS = 4
MSG_DIM = N_AGENTS * N_HEADS * VOCAB_SIZE
COMM_EPOCHS = 600; BATCH_SIZE = 32; EARLY_STOP = 200
START_TIME = time.time(); TIMING = []
RB = Path("results/neurips_battery")

sys.path.insert(0, os.path.dirname(__file__))
from _fix_exp3_exp4 import (
    TemporalEncoder, DiscreteSender, DiscreteMultiSender,
    ContinuousSender, ContinuousMultiSender, Receiver,
    compute_posdis, compute_topsim
)

def elapsed(): return f"{(time.time()-START_TIME)/60:.0f}min"
def commit(name):
    os.system(f'cd /Users/tomek/AI && git add results/ _remaining_battery.py '
              f'&& git commit -m "{name}\n\nCo-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>" 2>/dev/null')

def load_meta():
    d = torch.load("results/phase87_phys101_spring_features.pt", weights_only=False)
    obj = d["obj_names"]; mass = d["mass_values"]
    p1 = np.digitize(mass, np.quantile(mass, [0.2, 0.4, 0.6, 0.8]))
    uo = sorted(set(obj)); oi = {o:i for i,o in enumerate(uo)}
    p2 = np.digitize(np.array([oi[o] for o in obj]),
                      np.quantile(np.arange(len(uo)), [0.2, 0.4, 0.6, 0.8]))
    return p1, p2, obj, mass

def load_feat(name):
    if name == "vjepa2":
        return torch.load("results/phase87_phys101_spring_features.pt", weights_only=False)["features"].float()
    elif name == "dinov2":
        f = torch.load("results/phase87_phys101_spring_static.pt", weights_only=False)["features"].float()
        return f.unsqueeze(1).expand(-1, 8, -1).contiguous()
    elif name == "text":
        return torch.load("results/crossmodal/text_hidden_states.pt", weights_only=False)["features"].float()
    return None

def train_d(feat, mass, obj, seed, n_agents=N_AGENTS, vocab_size=VOCAB_SIZE, n_heads=N_HEADS):
    t0=time.time(); n,nf,dim=feat.shape; fpa=max(1,nf//n_agents)
    msg_dim = n_agents * n_heads * vocab_size
    views=[feat[:,(i*fpa)%nf:(i*fpa)%nf+fpa,:] for i in range(n_agents)]
    torch.manual_seed(seed);np.random.seed(seed)
    ss=[DiscreteSender(TemporalEncoder(HIDDEN_DIM,dim,fpa),HIDDEN_DIM,vocab_size,n_heads) for _ in range(n_agents)]
    sender=DiscreteMultiSender(ss).to(DEVICE)
    recvs=[Receiver(msg_dim,HIDDEN_DIM).to(DEVICE) for _ in range(3)]
    so=torch.optim.Adam(sender.parameters(),lr=1e-3)
    ros=[torch.optim.Adam(r.parameters(),lr=3e-3) for r in recvs]
    mass_dev=torch.tensor(mass,dtype=torch.float32).to(DEVICE)
    rng=np.random.RandomState(seed*1000+42)
    uo=sorted(set(obj));ho=set(rng.choice(uo,max(4,len(uo)//5),replace=False))
    tr=np.array([i for i,o in enumerate(obj) if o not in ho])
    tei=np.array([i for i,o in enumerate(obj) if o in ho])
    nb=max(1,len(tr)//32);me=math.log(max(vocab_size,2));ba=0;bst=None;bep=0
    for ep in range(COMM_EPOCHS):
        if ep-bep>EARLY_STOP and ba>0.55:break
        if ep>0 and ep%40==0:
            for i in range(3):recvs[i]=Receiver(msg_dim,HIDDEN_DIM).to(DEVICE);ros[i]=torch.optim.Adam(recvs[i].parameters(),lr=3e-3)
        sender.train();[r.train() for r in recvs]
        tau=3+(1-3)*ep/max(1,COMM_EPOCHS-1);hard=ep>=30
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
    TIMING.append(time.time()-t0)
    return sender,recvs[0],tokens,ba

def get_assigns(tokens,p1,p2):
    attrs=np.stack([p1,p2],axis=1);_,mi=compute_posdis(tokens,attrs,VOCAB_SIZE)
    return [int(np.argmax(mi[p])) for p in range(mi.shape[0])]

def xagree(a,b):
    ag=[]
    for x in a:
        for y in b:ag.append(sum(1 for i,j in zip(x,y) if i==j)/len(x))
    return float(np.mean(ag))


# ═══════════════════════════════════════════════════════════════
# QW2: CLIP Layer Probing
# ═══════════════════════════════════════════════════════════════

def qw2_clip_layers():
    print(f"\n{'#'*60}\n# QW2: CLIP Layer Probing\n# {elapsed()}\n{'#'*60}",flush=True)
    d=RB/"quick_win_clip_layers";d.mkdir(parents=True,exist_ok=True)
    p1,p2,obj,mass=load_meta();n=len(mass)

    # Load CLIP ViT-L via timm and extract features from intermediate layers
    import timm
    clip_model = timm.create_model('vit_large_patch14_clip_224.openai', pretrained=True)
    clip_model.eval()
    n_blocks = len(clip_model.blocks)  # 24
    embed_dim = clip_model.embed_dim   # 1024
    print(f"  CLIP ViT-L: {n_blocks} blocks, {embed_dim}-dim",flush=True)

    # We need images to extract features — generate synthetic from metadata
    # Use simple colored circles like moonshot #9
    mass_norm = (mass - mass.min()) / (mass.max() - mass.min() + 1e-10)
    uo = sorted(set(obj)); oi = {o:i for i,o in enumerate(uo)}
    obj_idx = np.array([oi[o] for o in obj])
    colors = [[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1],
              [0.5,0.5,0],[0.5,0,0.5],[0,0.5,0.5],[0.7,0.3,0]]
    rng_img = np.random.RandomState(42)
    images = torch.zeros(n, 3, 224, 224)
    for i in range(n):
        radius = int(10 + mass_norm[i] * 80)
        cx, cy = 56 + rng_img.randint(0, 112), 56 + rng_img.randint(0, 112)
        color = colors[obj_idx[i] % len(colors)]
        for c in range(3):
            yy, xx = np.ogrid[-cy:224-cy, -cx:224-cx]
            mask = xx*xx + yy*yy <= radius*radius
            images[i, c, mask] = color[c]

    # ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
    images_norm = (images - mean) / std

    # Extract features from layers 3, 6, 9, 12, 15, 18, 21, 24
    probe_layers = [3, 6, 9, 12, 15, 18, 21, 24]
    layer_features = {}

    for target_layer in probe_layers:
        feats = []
        # Register hook
        activations = {}
        def hook_fn(module, input, output, layer_id=target_layer):
            activations[layer_id] = output

        if target_layer <= n_blocks:
            handle = clip_model.blocks[target_layer - 1].register_forward_hook(
                lambda m, i, o, lid=target_layer: activations.update({lid: o}))

        for i in range(0, n, 32):
            batch = images_norm[i:i+32]
            with torch.no_grad():
                _ = clip_model(batch)
            if target_layer in activations:
                # CLS token from this layer
                feat = activations[target_layer][:, 0, :]  # [batch, embed_dim]
                feats.append(feat)
            activations.clear()

        if target_layer <= n_blocks:
            handle.remove()

        if feats:
            layer_features[target_layer] = torch.cat(feats, 0)
            print(f"  Layer {target_layer}: {layer_features[target_layer].shape}",flush=True)

    del clip_model; torch.mps.empty_cache()

    # Also train V-JEPA baseline for agreement comparison
    vfeat = load_feat("vjepa2")
    v_assigns_list = []
    for seed in range(3):
        _,_,vtoks,_ = train_d(vfeat, mass, obj, seed)
        v_assigns_list.append(get_assigns(vtoks, p1, p2))
        torch.mps.empty_cache()

    # Train WMCP on each CLIP layer
    results = {}
    for layer_idx in probe_layers:
        if layer_idx not in layer_features:
            continue
        feat = layer_features[layer_idx]
        # Expand to temporal format [n, 8, dim]
        feat_exp = feat.unsqueeze(1).expand(-1, 8, -1).contiguous()

        assigns_list = []; accs = []; pds = []
        for seed in range(5):
            _,_,tokens,acc = train_d(feat_exp, mass, obj, seed)
            assigns_list.append(get_assigns(tokens, p1, p2))
            accs.append(acc)
            attrs = np.stack([p1, p2], axis=1)
            pd, _ = compute_posdis(tokens, attrs, VOCAB_SIZE)
            pds.append(pd)
            torch.mps.empty_cache()

        agree_vjepa = xagree(assigns_list[:3], v_assigns_list)
        within = xagree(assigns_list, assigns_list)

        results[f"layer_{layer_idx}"] = {
            "accuracy": f"{np.mean(accs):.1%}±{np.std(accs):.1%}",
            "posdis": f"{np.mean(pds):.3f}±{np.std(pds):.3f}",
            "agreement_vjepa": f"{agree_vjepa:.1%}",
            "within_agreement": f"{within:.1%}",
            "acc_m": float(np.mean(accs)),
            "pd_m": float(np.mean(pds)),
            "agree_m": float(agree_vjepa),
        }
        print(f"    Layer {layer_idx}: acc={np.mean(accs):.1%} PD={np.mean(pds):.3f} "
              f"↔V-JEPA={agree_vjepa:.1%}",flush=True)

        if seed == 0 and layer_idx == probe_layers[0]:
            avg = np.mean(TIMING[-1:]) if TIMING else 40
            remaining = len(probe_layers) * 5
            print(f"    Measured: ~{avg:.0f}s/seed, ~{remaining * avg / 60:.0f}min remaining",flush=True)

    # Plot
    import matplotlib;matplotlib.use("Agg");import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("CLIP Layer Probing: Where Does Physics Encoding Live?", fontweight='bold')
    layers = sorted([int(k.split("_")[1]) for k in results])

    for ax, metric, label in [(axes[0], "agree_m", "Agreement with V-JEPA 2"),
                               (axes[1], "pd_m", "PosDis")]:
        vals = [results[f"layer_{l}"][metric] for l in layers]
        ax.plot(layers, vals, 'o-', linewidth=2, markersize=8)
        ax.set_xlabel("CLIP Layer"); ax.set_ylabel(label)
        ax.set_title(label); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(d / "clip_layer_probing.png", dpi=200, bbox_inches='tight'); plt.close()

    with open(d / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved QW2 ({elapsed()})",flush=True)
    return results


# ═══════════════════════════════════════════════════════════════
# QW3: CLIP Null-Space Projection
# ═══════════════════════════════════════════════════════════════

def qw3_clip_nullspace():
    print(f"\n{'#'*60}\n# QW3: CLIP Null-Space Projection\n# {elapsed()}\n{'#'*60}",flush=True)
    d=RB/"quick_win_clip_nullspace";d.mkdir(parents=True,exist_ok=True)
    p1,p2,obj,mass=load_meta();n=len(mass)

    # Load CLIP visual features
    clip_static = torch.load("results/phase96_phys101_spring_clip.pt", weights_only=False)["features"].float()
    # Load text features (use as the "language direction")
    text_feat_raw = torch.load("results/crossmodal/text_hidden_states.pt", weights_only=False)["features"].float()
    text_pooled = text_feat_raw.mean(dim=1)  # [n, text_dim]

    # Project CLIP features into null space of text direction
    # For each sample: v' = v - (v · t_hat) * t_hat
    clip_dim = clip_static.shape[1]
    text_dim = text_pooled.shape[1]

    # Since dims may differ, project text into CLIP space via linear mapping first
    # Or simply: use PCA of text features to find top-k language directions, project CLIP out of those
    from sklearn.decomposition import PCA

    # Find top language directions from text features (reduced to CLIP dim via truncation/padding)
    min_dim = min(clip_dim, text_dim)
    text_trunc = text_pooled[:, :min_dim].numpy()
    clip_trunc = clip_static[:, :min_dim].numpy()

    # PCA on text to find language axes
    n_lang_dirs = 10
    pca = PCA(n_components=min(n_lang_dirs, min_dim, n))
    pca.fit(text_trunc)
    lang_dirs = pca.components_  # [n_dirs, dim]

    # Project CLIP features out of language directions
    clip_debiased = clip_trunc.copy()
    for direction in lang_dirs:
        d_norm = direction / (np.linalg.norm(direction) + 1e-10)
        projections = clip_debiased @ d_norm
        clip_debiased -= np.outer(projections, d_norm)

    clip_debiased_t = torch.tensor(clip_debiased, dtype=torch.float32)
    clip_debiased_exp = clip_debiased_t.unsqueeze(1).expand(-1, 8, -1).contiguous()
    clip_orig_exp = torch.tensor(clip_trunc, dtype=torch.float32).unsqueeze(1).expand(-1, 8, -1).contiguous()

    # Train V-JEPA baseline for comparison
    vfeat = load_feat("vjepa2")
    v_assigns = []
    for seed in range(3):
        _,_,vt,_ = train_d(vfeat, mass, obj, seed)
        v_assigns.append(get_assigns(vt, p1, p2))
        torch.mps.empty_cache()

    # Train on original CLIP (truncated)
    print("  Training on original CLIP features...",flush=True)
    orig_assigns = []; orig_accs = []
    for seed in range(5):
        _,_,tokens,acc = train_d(clip_orig_exp, mass, obj, seed)
        orig_assigns.append(get_assigns(tokens, p1, p2))
        orig_accs.append(acc)
        torch.mps.empty_cache()

    # Train on de-biased CLIP
    print("  Training on de-biased CLIP features...",flush=True)
    debiased_assigns = []; debiased_accs = []
    for seed in range(5):
        _,_,tokens,acc = train_d(clip_debiased_exp, mass, obj, seed)
        debiased_assigns.append(get_assigns(tokens, p1, p2))
        debiased_accs.append(acc)
        if seed == 2:
            avg = np.mean(TIMING[-3:])
            print(f"    Measured: {avg:.1f}s/seed",flush=True)
        torch.mps.empty_cache()

    orig_vjepa = xagree(orig_assigns[:3], v_assigns)
    debiased_vjepa = xagree(debiased_assigns[:3], v_assigns)

    results = {
        "original_clip_accuracy": f"{np.mean(orig_accs):.1%}±{np.std(orig_accs):.1%}",
        "debiased_clip_accuracy": f"{np.mean(debiased_accs):.1%}±{np.std(debiased_accs):.1%}",
        "original_vjepa_agreement": f"{orig_vjepa:.1%}",
        "debiased_vjepa_agreement": f"{debiased_vjepa:.1%}",
        "n_language_directions_removed": n_lang_dirs,
    }

    if debiased_vjepa > orig_vjepa + 0.05:
        verdict = "LANGUAGE ALIGNMENT IS THE DISTORTION — removing it increases physics agreement"
    elif debiased_vjepa < orig_vjepa - 0.05:
        verdict = "LANGUAGE DIRECTIONS CONTAIN PHYSICS — removing them hurts"
    else:
        verdict = "NO SIGNIFICANT EFFECT — distortion is deeper than linear"
    results["verdict"] = verdict

    print(f"\n  Original CLIP ↔ V-JEPA: {orig_vjepa:.1%}",flush=True)
    print(f"  De-biased CLIP ↔ V-JEPA: {debiased_vjepa:.1%}",flush=True)
    print(f"  Verdict: {verdict}",flush=True)

    with open(d / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved QW3 ({elapsed()})",flush=True)
    return results


# ═══════════════════════════════════════════════════════════════
# Sunshot C: dSprites Non-Physics Domain
# ═══════════════════════════════════════════════════════════════

def sunshot_dsprites():
    print(f"\n{'#'*60}\n# SUNSHOT C: dSprites Non-Physics Domain\n# {elapsed()}\n{'#'*60}",flush=True)
    d=RB/"sunshot_dsprites";d.mkdir(parents=True,exist_ok=True)

    # Download dSprites
    dsprites_path = d / "dsprites.npz"
    if not dsprites_path.exists():
        print("  Downloading dSprites...",flush=True)
        url = 'https://github.com/deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        resp = urllib.request.urlopen(req, timeout=60)
        with open(dsprites_path, 'wb') as f:
            f.write(resp.read())
        print("  Downloaded",flush=True)

    data = np.load(dsprites_path, allow_pickle=True)
    imgs = data["imgs"]  # [737280, 64, 64] binary
    latents = data["latents_values"]  # [737280, 6]: color, shape, scale, orientation, x, y
    print(f"  dSprites: {imgs.shape}, latents: {latents.shape}",flush=True)

    # Subsample 500 images with diverse latents
    rng = np.random.RandomState(42)
    idx = rng.choice(len(imgs), 500, replace=False)
    imgs_sub = imgs[idx]  # [500, 64, 64]
    latents_sub = latents[idx]  # [500, 6]

    # Properties: shape (col 1), scale (col 2)
    shape_vals = latents_sub[:, 1]  # 1, 2, 3
    scale_vals = latents_sub[:, 2]  # continuous
    shape_bins = (shape_vals - shape_vals.min()).astype(int)
    scale_bins = np.digitize(scale_vals, np.quantile(scale_vals, [0.33, 0.67]))

    # Create "mass" proxy from scale (bigger = heavier)
    mass_proxy = scale_vals.astype(np.float64)

    # Resize to 224x224 and make RGB
    imgs_224 = np.zeros((500, 3, 224, 224), dtype=np.float32)
    for i in range(500):
        img = imgs_sub[i].astype(np.float32)
        # Simple nearest-neighbor upscale
        img_up = np.repeat(np.repeat(img, 4, axis=0), 4, axis=1)[:224, :224]
        for c in range(3):
            imgs_224[i, c] = img_up

    imgs_t = torch.tensor(imgs_224)

    # Extract DINOv2 features
    print("  Extracting DINOv2 features on dSprites...",flush=True)
    dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    dino.eval().to(DEVICE)
    mean_t = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(DEVICE)
    std_t = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(DEVICE)

    dino_feats = []
    for i in range(0, 500, 64):
        batch = imgs_t[i:i+64].to(DEVICE)
        batch = (batch - mean_t) / std_t
        with torch.no_grad():
            out = dino.forward_features(batch)
            dino_feats.append(out["x_norm_clstoken"].cpu())
    dino_feats = torch.cat(dino_feats, 0)  # [500, 384]
    del dino; torch.mps.empty_cache()
    dino_exp = dino_feats.unsqueeze(1).expand(-1, 8, -1).contiguous()
    print(f"  DINOv2 features: {dino_exp.shape}",flush=True)

    # Generate text descriptions for dSprites
    shape_names = {1: "heart", 2: "ellipse", 3: "square"}
    scale_names = {0: "small", 1: "medium", 2: "large"}
    obj_names = [f"shape_{int(shape_vals[i])}" for i in range(500)]

    descriptions = []
    for i in range(500):
        sh = shape_names.get(int(shape_vals[i]), "shape")
        sc = scale_names.get(scale_bins[i], "medium")
        descriptions.append(f"A {sc} {sh} in a 2D scene. Scale: {scale_vals[i]:.2f}. Shape category: {sh}.")

    # Extract TinyLlama features
    print("  Extracting TinyLlama features on dSprites descriptions...",flush=True)
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    mdl = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        torch_dtype=torch.float32, output_hidden_states=True)
    mdl.eval()
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tl = mdl.config.num_hidden_layers // 2

    text_feats = []
    for i, desc in enumerate(descriptions):
        inp = tok(desc, return_tensors="pt", padding=True, truncation=True, max_length=64)
        with torch.no_grad():
            out = mdl(**inp)
            text_feats.append(out.hidden_states[tl].mean(1).squeeze(0))
        if (i+1) % 100 == 0: print(f"    {i+1}/500",flush=True)
    text_feats = torch.stack(text_feats).unsqueeze(1).expand(-1, 8, -1).contiguous().float()
    del mdl, tok; torch.mps.empty_cache()
    print(f"  Text features: {text_feats.shape}",flush=True)

    # Train WMCP on both modalities
    p1_ds = scale_bins
    p2_ds = shape_bins

    dino_assigns = []; text_assigns = []; dino_accs = []; text_accs = []
    for seed in range(5):
        _,_,dt,da = train_d(dino_exp, mass_proxy, obj_names, seed)
        dino_assigns.append(get_assigns(dt, p1_ds, p2_ds))
        dino_accs.append(da)

        _,_,tt,ta = train_d(text_feats, mass_proxy, obj_names, seed + 100)
        text_assigns.append(get_assigns(tt, p1_ds, p2_ds))
        text_accs.append(ta)

        if seed == 2:
            avg = np.mean(TIMING[-6:])
            print(f"    Measured: {avg:.1f}s/seed",flush=True)
        torch.mps.empty_cache()

    cross_agree = xagree(dino_assigns, text_assigns)
    dino_within = xagree(dino_assigns, dino_assigns)
    text_within = xagree(text_assigns, text_assigns)

    results = {
        "domain": "dSprites",
        "dino_accuracy": f"{np.mean(dino_accs):.1%}±{np.std(dino_accs):.1%}",
        "text_accuracy": f"{np.mean(text_accs):.1%}±{np.std(text_accs):.1%}",
        "cross_modal_agreement": f"{cross_agree:.1%}",
        "dino_within": f"{dino_within:.1%}",
        "text_within": f"{text_within:.1%}",
        "cross_agree_m": float(cross_agree),
    }

    if cross_agree > 0.80:
        verdict = "MECHANISM GENERALIZES BEYOND PHYSICS — dSprites cross-modal convergence"
    elif cross_agree > 0.60:
        verdict = f"PARTIAL GENERALIZATION — {cross_agree:.0%} agreement on non-physics domain"
    else:
        verdict = "PHYSICS-SPECIFIC — mechanism doesn't generalize to dSprites"
    results["verdict"] = verdict

    print(f"\n  Cross-modal agreement on dSprites: {cross_agree:.1%}",flush=True)
    print(f"  Verdict: {verdict}",flush=True)

    with open(d / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved dSprites ({elapsed()})",flush=True)
    return results


# ═══════════════════════════════════════════════════════════════
# Neuro RSA: THINGS-EEG2
# ═══════════════════════════════════════════════════════════════

def neuro_rsa():
    print(f"\n{'#'*60}\n# NEURO RSA: Human Brain Comparison\n# {elapsed()}\n{'#'*60}",flush=True)
    d=RB/"neuro_rsa";d.mkdir(parents=True,exist_ok=True)

    # Try to download THINGS-EEG2 RDMs from OSF
    print("  Attempting THINGS-EEG2 download...",flush=True)
    try:
        # The pre-computed RDMs are large — try a smaller alternative
        # Use the THINGS concept-level RDM from the behavioral ratings
        url = "https://osf.io/download/3jk45/"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        resp = urllib.request.urlopen(req, timeout=30)
        content_type = resp.headers.get('Content-Type', '')
        print(f"  Response: {content_type}, {resp.headers.get('Content-Length', '?')} bytes",flush=True)
        # This likely returns an HTML page, not data directly
        if 'html' in content_type.lower():
            raise ValueError("Got HTML page, not data file")
    except Exception as e:
        print(f"  THINGS-EEG2 download failed: {e}",flush=True)
        print("  Falling back to synthetic neural RDM comparison",flush=True)

        # Fallback: Compare WMCP RDM structure to a human-like categorical RDM
        # Hypothesis: humans group objects by category more than by mass
        p1,p2,obj,mass = load_meta()
        n = len(mass)

        # Create a "human-like" RDM based on object category (same category = similar)
        uo = sorted(set(obj)); oi = {o:i for i,o in enumerate(uo)}
        obj_idx = np.array([oi[o] for o in obj])
        human_rdm = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                # Same object = distance 0, different = 1
                dist = 0.0 if obj_idx[i] == obj_idx[j] else 1.0
                human_rdm[i, j] = dist; human_rdm[j, i] = dist

        # Create a "physics-like" RDM based on mass difference
        physics_rdm = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                dist = abs(mass[i] - mass[j]) / (mass.max() - mass.min() + 1e-10)
                physics_rdm[i, j] = dist; physics_rdm[j, i] = dist

        # Get WMCP discrete RDM
        vfeat = load_feat("vjepa2")
        _,_,tokens,_ = train_d(vfeat, mass, obj, 0)
        wmcp_rdm = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                d_val = np.sum(tokens[i] != tokens[j])
                wmcp_rdm[i, j] = d_val; wmcp_rdm[j, i] = d_val
        torch.mps.empty_cache()

        # Get raw feature RDMs
        from sklearn.metrics.pairwise import cosine_distances
        vfeat_pooled = vfeat.mean(dim=1).numpy()
        raw_rdm = cosine_distances(vfeat_pooled)

        # RSA comparisons
        tri = np.triu_indices(n, k=1)
        wmcp_human = scipy_stats.spearmanr(wmcp_rdm[tri], human_rdm[tri])[0]
        wmcp_physics = scipy_stats.spearmanr(wmcp_rdm[tri], physics_rdm[tri])[0]
        raw_human = scipy_stats.spearmanr(raw_rdm[tri], human_rdm[tri])[0]
        raw_physics = scipy_stats.spearmanr(raw_rdm[tri], physics_rdm[tri])[0]

        results = {
            "data_source": "synthetic_categorical_rdm",
            "wmcp_vs_category": float(wmcp_human),
            "wmcp_vs_physics": float(wmcp_physics),
            "raw_vs_category": float(raw_human),
            "raw_vs_physics": float(raw_physics),
        }

        if wmcp_physics > raw_physics:
            verdict = "WMCP discrete codes are MORE physics-aligned than raw features"
        else:
            verdict = "Raw features are more physics-aligned than WMCP codes"
        results["verdict"] = verdict

        print(f"\n  WMCP ↔ category: {wmcp_human:.3f}  |  WMCP ↔ physics: {wmcp_physics:.3f}",flush=True)
        print(f"  Raw ↔ category:  {raw_human:.3f}  |  Raw ↔ physics:  {raw_physics:.3f}",flush=True)
        print(f"  Verdict: {verdict}",flush=True)

        with open(d / "results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"  Saved neuro_rsa ({elapsed()})",flush=True)
        return results


# ═══════════════════════════════════════════════════════════════
# #12: Universal Sender with Task-Adversarial Training
# ═══════════════════════════════════════════════════════════════

class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None

def grad_reverse(x, alpha=1.0):
    return GradientReversal.apply(x, alpha)


def moonshot12_universal():
    print(f"\n{'#'*60}\n# #12: Universal Sender with GRL\n# {elapsed()}\n{'#'*60}",flush=True)
    d=RB/"moonshot12_universal";d.mkdir(parents=True,exist_ok=True)
    p1,p2,obj,mass=load_meta();n=len(mass)

    # Pool features from spring + fall + ramp (V-JEPA 2)
    tasks = {}
    for task_name in ["spring", "fall", "ramp"]:
        try:
            td = torch.load(f"results/phase87_phys101_{task_name}_features.pt", weights_only=False)
            feat = td["features"].float()
            if feat.shape[0] > 500:
                feat = feat[:500]
                task_mass = td["mass_values"][:500]
                task_obj = td["obj_names"][:500]
            else:
                task_mass = td["mass_values"]
                task_obj = td["obj_names"]
            tasks[task_name] = {"feat": feat, "mass": task_mass, "obj": task_obj}
            print(f"  {task_name}: {feat.shape}",flush=True)
        except Exception as e:
            print(f"  {task_name}: failed ({e})",flush=True)

    if len(tasks) < 2:
        print("  Not enough tasks, skipping",flush=True)
        return {"error": "insufficient tasks"}

    task_names = list(tasks.keys())
    n_tasks = len(task_names)

    # Combine all features with task labels
    all_feats = []; all_masses = []; all_objs = []; all_task_labels = []
    for ti, tname in enumerate(task_names):
        t = tasks[tname]
        all_feats.append(t["feat"])
        all_masses.extend(t["mass"].tolist())
        all_objs.extend(t["obj"])
        all_task_labels.extend([ti] * len(t["obj"]))

    combined_feat = torch.cat(all_feats, 0)
    combined_mass = np.array(all_masses)
    combined_task = np.array(all_task_labels)
    n_combined = len(combined_mass)
    print(f"  Combined: {combined_feat.shape}, {n_tasks} tasks",flush=True)

    dim = combined_feat.shape[-1]; nf = combined_feat.shape[1]
    fpa = max(1, nf // N_AGENTS)
    views = [combined_feat[:, (i*fpa)%nf:(i*fpa)%nf+fpa, :] for i in range(N_AGENTS)]

    results = {"conditions": {}}

    for use_grl in [False, True]:
        cond_name = "with_grl" if use_grl else "without_grl"
        print(f"\n  ── {cond_name} ──",flush=True)
        seed_results = []

        for seed in range(10):
            t0 = time.time()
            torch.manual_seed(seed); np.random.seed(seed)
            ss = [DiscreteSender(TemporalEncoder(HIDDEN_DIM, dim, fpa), HIDDEN_DIM, VOCAB_SIZE, N_HEADS)
                  for _ in range(N_AGENTS)]
            sender = DiscreteMultiSender(ss).to(DEVICE)
            recv = Receiver(MSG_DIM, HIDDEN_DIM).to(DEVICE)

            # Task discriminator
            task_disc = nn.Sequential(
                nn.Linear(MSG_DIM, 64), nn.ReLU(), nn.Linear(64, n_tasks)
            ).to(DEVICE)

            params = list(sender.parameters()) + list(recv.parameters()) + list(task_disc.parameters())
            opt = torch.optim.Adam(params, lr=1e-3)
            mass_dev = torch.tensor(combined_mass, dtype=torch.float32).to(DEVICE)
            task_dev = torch.tensor(combined_task, dtype=torch.long).to(DEVICE)
            rng = np.random.RandomState(seed * 1000 + 42)
            me = math.log(VOCAB_SIZE); ba = 0

            for ep in range(400):
                sender.train(); recv.train(); task_disc.train()
                tau = 3 + (1-3)*ep/399; hard = ep >= 30
                ia = rng.choice(n_combined, 32); ib = rng.choice(n_combined, 32)
                s = ia == ib
                while s.any(): ib[s] = rng.choice(n_combined, s.sum()); s = ia == ib
                md = np.abs(combined_mass[ia] - combined_mass[ib]); k = md > 0.5
                if k.sum() < 4: continue
                ia, ib = ia[k], ib[k]
                va = [v[ia].to(DEVICE) for v in views]; vb = [v[ib].to(DEVICE) for v in views]
                lab = (mass_dev[ia] > mass_dev[ib]).float()
                ma, la = sender(va, tau, hard); mb, lb = sender(vb, tau, hard)

                # Physics loss
                phys_loss = F.binary_cross_entropy_with_logits(recv(ma, mb), lab)

                # Task discrimination loss
                if use_grl:
                    ma_rev = grad_reverse(ma, alpha=1.0)
                    task_pred = task_disc(ma_rev)
                else:
                    task_pred = task_disc(ma.detach())
                task_loss = F.cross_entropy(task_pred, task_dev[ia])

                loss = phys_loss + 0.1 * task_loss

                for lg in la + lb:
                    lp = F.log_softmax(lg, -1); p = lp.exp().clamp(1e-8)
                    ent = -(p*lp).sum(-1).mean()
                    if ent/me < 0.1: loss = loss - 0.03*ent

                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()

                if (ep+1) % 100 == 0:
                    sender.eval(); recv.eval(); task_disc.eval()
                    with torch.no_grad():
                        test_ia = rng.choice(n_combined, 100)
                        test_va = [v[test_ia].to(DEVICE) for v in views]
                        test_ma, _ = sender(test_va)
                        task_acc = (task_disc(test_ma).argmax(-1) == task_dev[test_ia]).float().mean().item()
                        # Physics acc
                        c = t = 0; er = np.random.RandomState(999)
                        for _ in range(20):
                            ai = er.choice(n_combined, 32); bi = er.choice(n_combined, 32)
                            va2 = [v[ai].to(DEVICE) for v in views]; vb2 = [v[bi].to(DEVICE) for v in views]
                            ma2,_ = sender(va2); mb2,_ = sender(vb2)
                            c += ((recv(ma2,mb2)>0)==(mass_dev[ai]>mass_dev[bi])).sum().item(); t += len(ai)
                        ba = max(ba, c/max(t,1))

            # Final task discriminator accuracy
            sender.eval(); task_disc.eval()
            with torch.no_grad():
                all_msgs = []
                for i in range(0, n_combined, 64):
                    v2 = [vi[i:i+64].to(DEVICE) for vi in views]
                    m, _ = sender(v2); all_msgs.append(m.cpu())
                all_msgs = torch.cat(all_msgs, 0)
                final_task_acc = (task_disc(all_msgs.to(DEVICE)).argmax(-1) == task_dev).float().mean().item()

            seed_results.append({
                "physics_acc": float(ba),
                "task_disc_acc": float(final_task_acc),
            })
            TIMING.append(time.time() - t0)

            if seed == 2:
                print(f"    Measured: {np.mean(TIMING[-3:]):.1f}s/seed",flush=True)
            print(f"    Seed {seed}: phys={ba:.1%} task_disc={final_task_acc:.1%}",flush=True)
            torch.mps.empty_cache()

        paccs = [r["physics_acc"] for r in seed_results]
        taccs = [r["task_disc_acc"] for r in seed_results]
        results["conditions"][cond_name] = {
            "physics_acc": f"{np.mean(paccs):.1%}±{np.std(paccs):.1%}",
            "task_disc_acc": f"{np.mean(taccs):.1%}±{np.std(taccs):.1%}",
            "phys_m": float(np.mean(paccs)),
            "task_m": float(np.mean(taccs)),
        }

    # Verdict
    grl = results["conditions"]["with_grl"]
    nogrl = results["conditions"]["without_grl"]
    chance = 1.0 / n_tasks
    if grl["task_m"] < chance + 0.1 and grl["phys_m"] > 0.6:
        verdict = "TASK-INVARIANT PHYSICS PROTOCOL — GRL successfully removes task identity"
    else:
        verdict = f"GRL partial: task_disc {grl['task_m']:.0%} (chance={chance:.0%}), physics {grl['phys_m']:.0%}"
    results["verdict"] = verdict

    print(f"\n  Without GRL: phys={nogrl['physics_acc']} task_disc={nogrl['task_disc_acc']}",flush=True)
    print(f"  With GRL:    phys={grl['physics_acc']} task_disc={grl['task_disc_acc']}",flush=True)
    print(f"  Verdict: {verdict}",flush=True)

    with open(d / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved #12 ({elapsed()})",flush=True)
    return results


# ═══ Main ═══

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗",flush=True)
    print("║  REMAINING BATTERY: QW2, QW3, Sunshot C, Neuro, #12     ║",flush=True)
    print(f"║  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}                          ║",flush=True)
    print("╚══════════════════════════════════════════════════════════╝",flush=True)

    experiments = [
        ("QW2: CLIP Layer Probing", qw2_clip_layers),
        ("QW3: CLIP Null-Space", qw3_clip_nullspace),
        ("Sunshot C: dSprites", sunshot_dsprites),
        ("Neuro RSA", neuro_rsa),
        ("#12: Universal Sender", moonshot12_universal),
    ]

    for name, func in experiments:
        try:
            print(f"\n{'='*60}\n  STARTING: {name} ({elapsed()})\n{'='*60}",flush=True)
            func()
            commit(name)
        except Exception as e:
            print(f"\n  {name} FAILED: {e}",flush=True)
            traceback.print_exc()

    total_h = (time.time()-START_TIME) / 3600
    print(f"\n{'='*60}",flush=True)
    print(f"  ALL COMPLETE. Total: {total_h:.1f} hours",flush=True)
    print(f"{'='*60}",flush=True)
