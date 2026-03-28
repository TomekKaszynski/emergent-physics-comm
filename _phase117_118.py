"""
Phases 117-118: Manipulation domain validation + HuggingFace model cards
=========================================================================
117: Use ramp scenario as manipulation-adjacent domain validation
118: Generate HuggingFace model cards for three protocol instances

Run:
  PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=1 python3 -c "from _phase117_118 import run_all; run_all()"
"""

import time, json, math, os, sys, traceback
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "emergent-physics-comm", "src"))
from metrics import positional_disentanglement, topographic_similarity

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
RESULTS_DIR = Path("results")
HIDDEN_DIM = 128; N_HEADS = 2; VOCAB_SIZE = 3


class TemporalEncoder(nn.Module):
    def __init__(self, hd=128, ind=1024, nf=4):
        super().__init__()
        ks = min(3, max(1, nf))
        self.temporal = nn.Sequential(nn.Conv1d(ind, 256, ks, padding=ks//2), nn.ReLU(),
            nn.Conv1d(256, 128, ks, padding=ks//2), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
        self.fc = nn.Sequential(nn.Linear(128, hd), nn.ReLU())
    def forward(self, x): return self.fc(self.temporal(x.permute(0,2,1)).squeeze(-1))

class CompositionalSender(nn.Module):
    def __init__(self, enc, hd, vs, nh):
        super().__init__()
        self.enc=enc; self.vs=vs; self.nh=nh
        self.heads = nn.ModuleList([nn.Linear(hd, vs) for _ in range(nh)])
    def forward(self, x, tau=1.0, hard=True):
        h=self.enc(x); msgs,lgs=[],[]
        for hd in self.heads:
            l=hd(h)
            m=F.gumbel_softmax(l,tau=tau,hard=hard) if self.training else F.one_hot(l.argmax(-1),self.vs).float()
            msgs.append(m);lgs.append(l)
        return torch.cat(msgs,-1),lgs

class MultiAgentSender(nn.Module):
    def __init__(self,ss): super().__init__(); self.senders=nn.ModuleList(ss)
    def forward(self,views,tau=1.0,hard=True):
        msgs,lgs=[],[]
        for s,v in zip(self.senders,views):
            m,l=s(v,tau,hard);msgs.append(m);lgs.extend(l)
        return torch.cat(msgs,-1),lgs

class CompositionalReceiver(nn.Module):
    def __init__(self,md,hd):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(md*2,hd),nn.ReLU(),nn.Linear(hd,hd//2),nn.ReLU(),nn.Linear(hd//2,1))
    def forward(self,a,b): return self.net(torch.cat([a,b],-1)).squeeze(-1)


def train_quick(configs, mass, obj, vs, seed, epochs=400, patience=150):
    na=len(configs); md=na*N_HEADS*vs
    av=[f.float() for f,_ in configs]
    uo=sorted(set(obj)); rng=np.random.RandomState(seed*1000+42)
    ho=set(rng.choice(uo,max(4,len(uo)//5),replace=False))
    tri=np.array([i for i,o in enumerate(obj) if o not in ho])
    tei=np.array([i for i,o in enumerate(obj) if o in ho])
    if len(tei)<4: return None
    torch.manual_seed(seed); np.random.seed(seed)
    ss=[CompositionalSender(TemporalEncoder(HIDDEN_DIM,d,f.shape[1]),HIDDEN_DIM,vs,N_HEADS) for f,d in configs]
    multi=MultiAgentSender(ss).to(DEVICE)
    recv=CompositionalReceiver(md,HIDDEN_DIM).to(DEVICE)
    so=torch.optim.Adam(multi.parameters(),lr=1e-3)
    ro=torch.optim.Adam(recv.parameters(),lr=3e-3)
    mdev=torch.tensor(mass,dtype=torch.float32).to(DEVICE)
    me=math.log(vs);nb=max(1,len(tri)//32);ba=0.0;bs=None;be=0
    for ep in range(epochs):
        if ep-be>patience and ba>0.55:break
        if ep>0 and ep%40==0: recv=CompositionalReceiver(md,HIDDEN_DIM).to(DEVICE);ro=torch.optim.Adam(recv.parameters(),lr=3e-3)
        multi.train();recv.train();tau=3+(1-3)*ep/max(1,epochs-1);hard=ep>=30
        for _ in range(nb):
            ia=rng.choice(tri,32);ib=rng.choice(tri,32);s=ia==ib
            while s.any():ib[s]=rng.choice(tri,s.sum());s=ia==ib
            md2=np.abs(mass[ia]-mass[ib]);k=md2>0.5
            if k.sum()<4:continue
            ia,ib=ia[k],ib[k]
            va=[v[ia].to(DEVICE) for v in av];vb=[v[ib].to(DEVICE) for v in av]
            lab=(mdev[ia]>mdev[ib]).float()
            ma,la=multi(va,tau,hard);mb,lb=multi(vb,tau,hard)
            loss=F.binary_cross_entropy_with_logits(recv(ma,mb),lab)
            for lg in la+lb:
                lp=F.log_softmax(lg,-1);p=lp.exp().clamp(min=1e-8);ent=-(p*lp).sum(-1).mean()
                if ent/me<0.1:loss=loss-0.03*ent
            if torch.isnan(loss):so.zero_grad();ro.zero_grad();continue
            so.zero_grad();ro.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(multi.parameters(),1.0);so.step();ro.step()
        if ep%50==0:torch.mps.empty_cache()
        if (ep+1)%50==0 or ep==0:
            multi.eval();recv.eval()
            with torch.no_grad():
                c=t=0;er=np.random.RandomState(999)
                for _ in range(30):
                    ia_h=er.choice(tei,min(32,len(tei)));ib_h=er.choice(tei,min(32,len(tei)))
                    mdh=np.abs(mass[ia_h]-mass[ib_h]);kh=mdh>0.5
                    if kh.sum()<2:continue
                    ia_h,ib_h=ia_h[kh],ib_h[kh]
                    vh=[v[ia_h].to(DEVICE) for v in av];wh=[v[ib_h].to(DEVICE) for v in av]
                    c+=((recv(multi(vh)[0],multi(wh)[0])>0)==(mdev[ia_h]>mdev[ib_h])).sum().item();t+=len(ia_h)
                acc=c/max(t,1)
                if acc>ba:ba=acc;be=ep;bs={k:v.cpu().clone() for k,v in multi.state_dict().items()}
    if bs:multi.load_state_dict(bs)
    multi.eval()
    all_tokens=[]
    with torch.no_grad():
        for i in range(0,len(av[0]),32):
            views=[v[i:i+32].to(DEVICE) for v in av]
            _,logits=multi(views)
            all_tokens.append(np.stack([l.argmax(-1).cpu().numpy() for l in logits],axis=1))
    all_tokens=np.concatenate(all_tokens,axis=0)
    mass_bins=np.digitize(mass,np.quantile(mass,[0.2,0.4,0.6,0.8]))
    uo2=sorted(set(obj));oi={o:i for i,o in enumerate(uo2)}
    obj_bins=np.digitize(np.array([oi[o] for o in obj]),np.quantile(np.arange(len(uo2)),[0.2,0.4,0.6,0.8]))
    pd,_,_=positional_disentanglement(all_tokens,np.stack([mass_bins,obj_bins],1),vs)
    ts=topographic_similarity(all_tokens,mass_bins,obj_bins)
    return {"accuracy":float(ba),"posdis":float(pd),"topsim":float(ts)}


# ═══ PHASE 117: Manipulation Domain Validation ═══
def run_phase117():
    print("\n╔══════════════════════════════════════════════════════════╗",flush=True)
    print("║  Phase 117: Manipulation Domain Validation (Ramp)        ║",flush=True)
    print("╚══════════════════════════════════════════════════════════╝",flush=True)
    t0=time.time()

    # Ramp: objects slide down inclined surfaces — properties relevant to manipulation:
    # mass (gripping force), surface friction (slip risk), material (rigidity)
    vd=torch.load(RESULTS_DIR/"phase87_phys101_ramp_features.pt",weights_only=False)
    dd=torch.load(RESULTS_DIR/"phase87_phys101_ramp_static.pt",weights_only=False)
    rng=np.random.RandomState(42)
    idx=rng.choice(len(vd["obj_names"]),500,replace=False);idx.sort()
    vf=vd["features"].float()[idx]
    df=dd["features"].float()[idx]
    obj=[vd["obj_names"][i] for i in idx]
    mass=vd["mass_values"][idx]
    nf=vf.shape[1];fpa=nf//2
    dt=df.unsqueeze(1).expand(-1,nf,-1).contiguous()

    print(f"  Ramp scenario: {len(obj)} clips, {len(set(obj))} objects",flush=True)
    print(f"  Mass range: {mass.min():.1f}g – {mass.max():.1f}g",flush=True)
    print(f"  Manipulation relevance: mass → grip force, surface → slip risk\n",flush=True)

    n_seeds=10
    all_results={}
    for cond_name in ["heterogeneous","homo_vjepa","homo_dino"]:
        print(f"  ── {cond_name} ──",flush=True)
        seed_results=[]
        for seed in range(n_seeds):
            if cond_name=="heterogeneous":
                configs=[(vf[:,:fpa,:],1024),(dt[:,fpa:,:],384)]
            elif cond_name=="homo_vjepa":
                configs=[(vf[:,:fpa,:],1024),(vf[:,fpa:,:],1024)]
            else:
                configs=[(dt[:,:fpa,:],384),(dt[:,fpa:,:],384)]
            r=train_quick(configs,mass,obj,3,seed)
            if r:seed_results.append(r)
            torch.mps.empty_cache()
        if seed_results:
            accs=[r["accuracy"] for r in seed_results]
            pds=[r["posdis"] for r in seed_results]
            tss=[r["topsim"] for r in seed_results]
            all_results[cond_name]={
                "acc":f"{np.mean(accs):.1%}±{np.std(accs):.1%}",
                "posdis":f"{np.mean(pds):.3f}±{np.std(pds):.3f}",
                "topsim":f"{np.mean(tss):.3f}±{np.std(tss):.3f}",
                "acc_mean":float(np.mean(accs)),"pd_mean":float(np.mean(pds)),
            }
            print(f"    acc={np.mean(accs):.1%} PD={np.mean(pds):.3f} TS={np.mean(tss):.3f}",flush=True)

    save_path=RESULTS_DIR/"phase117_manipulation.json"
    with open(save_path,"w") as f:json.dump(all_results,f,indent=2,default=str)
    print(f"\n  Saved {save_path} ({(time.time()-t0)/60:.1f}min)",flush=True)
    return all_results


# ═══ PHASE 118: HuggingFace Model Cards ═══
def run_phase118():
    print("\n╔══════════════════════════════════════════════════════════╗",flush=True)
    print("║  Phase 118: HuggingFace Model Cards                      ║",flush=True)
    print("╚══════════════════════════════════════════════════════════╝",flush=True)

    cards_dir=Path("protocol-spec/model-cards")
    cards_dir.mkdir(exist_ok=True)

    scenarios={
        "spring":{
            "name":"wmcp-physics-spring",
            "properties":"mass, elasticity",
            "clips":206,"objects":26,
            "het_acc":"81.8%","het_pd":"0.764",
            "homo_vv_acc":"82.5%","homo_dd_acc":"74.5%",
            "description":"Protocol trained on MIT Physics 101 spring scenario. Objects are released between two springs; agents communicate about mass properties inferred from oscillation dynamics.",
            "use_case":"Robotics manipulation where grip force must be estimated from visual observation of object response to applied forces.",
        },
        "fall":{
            "name":"wmcp-physics-fall",
            "properties":"mass, restitution",
            "clips":666,"objects":33,
            "het_acc":"86.7%","het_pd":"0.494",
            "homo_vv_acc":"85.8%","homo_dd_acc":"81.7%",
            "description":"Protocol trained on MIT Physics 101 fall scenario. Objects are dropped from fixed height; agents communicate about mass and bounciness inferred from impact dynamics.",
            "use_case":"Quality inspection where material properties must be assessed from observed behavior (e.g., drop tests on production lines).",
        },
        "ramp":{
            "name":"wmcp-physics-ramp",
            "properties":"mass, surface friction",
            "clips":"500 (subsampled from 1801)","objects":101,
            "het_acc":"82.1%","het_pd":"0.520",
            "homo_vv_acc":"75.7%","homo_dd_acc":"81.3%",
            "description":"Protocol trained on MIT Physics 101 ramp scenario. Objects slide down inclined surfaces; agents communicate about mass and friction inferred from sliding dynamics.",
            "use_case":"Autonomous vehicle terrain assessment, warehouse logistics (predicting object sliding behavior on conveyor surfaces).",
        },
    }

    for scenario, info in scenarios.items():
        card = f"""---
language: en
tags:
- wmcp
- emergent-communication
- world-model
- compositional
- physics
- {scenario}
license: apache-2.0
library_name: wmcp
pipeline_tag: feature-extraction
---

# {info['name']}

{info['description']}

## Model Details

- **Protocol version:** WMCP v0.1.0
- **Vocabulary size (K):** 3 symbols per position
- **Message positions (L):** 2 per agent
- **Message capacity:** 6.3 bits per 2-agent pair
- **Physical properties:** {info['properties']}
- **Training data:** MIT Physics 101 — {scenario} scenario ({info['clips']} clips, {info['objects']} objects)
- **Validated encoders:** V-JEPA 2 ViT-L (1024-dim), DINOv2 ViT-S/14 (384-dim), CLIP ViT-L/14 (768-dim)

## Performance

| Condition | Accuracy | PosDis |
|-----------|----------|--------|
| Heterogeneous (V-JEPA+DINOv2) | {info['het_acc']} | {info['het_pd']} |
| Homogeneous V-JEPA | {info['homo_vv_acc']} | — |
| Homogeneous DINOv2 | {info['homo_dd_acc']} | — |

## Intended Use

{info['use_case']}

The protocol enables heterogeneous vision models to communicate about physical scene properties through discrete compositional messages, without requiring shared architecture or explicit representation alignment.

## How to Use

```python
from wmcp import Protocol

# Load protocol (2-agent, K=3)
protocol = Protocol(
    agent_configs=[(1024, 4), (384, 4)],  # V-JEPA + DINOv2
    vocab_size=3, n_heads=2)

# Load weights
protocol.load_state_dict(torch.load("{info['name']}.pt"))

# Communicate about two scenes
prediction = protocol.communicate(views_a, views_b)
# prediction > 0 means scene A has higher property value
```

## Training Procedure

- **Objective:** Pairwise property comparison via binary cross-entropy
- **Optimizer:** Adam (sender lr=1e-3, receiver lr=3e-3)
- **Discretization:** Gumbel-Softmax (τ: 3.0→1.0, hard after 30 epochs)
- **Population pressure:** 3 receivers, reset every 40 epochs
- **Entropy regularization:** Penalty when position entropy < 0.1
- **Validation:** Object-level holdout (20% of unique objects)

## Limitations

- Validated on {scenario} physics only; does not transfer to other scenarios
- Requires frozen encoder features as input (not raw images)
- Pairwise comparison task only (not absolute property estimation)
- Co-training required for new encoder architectures

## Citation

```bibtex
@article{{kaszynski2026emergent,
  title={{Emergent Compositional Communication for Latent World Properties}},
  author={{Kaszy{{\\'n}}ski, Tomek}},
  year={{2026}},
  doi={{10.5281/zenodo.19197757}}
}}
```
"""
        card_path = cards_dir / f"{info['name']}.md"
        with open(card_path, "w") as f:
            f.write(card)
        print(f"  Saved {card_path}", flush=True)

    print(f"  Generated 3 model cards", flush=True)


def run_all():
    t=time.time()
    try:
        run_phase117()
        torch.mps.empty_cache()
    except Exception as e:
        print(f"  Phase 117 FAILED: {e}",flush=True);traceback.print_exc()
    try:
        run_phase118()
    except Exception as e:
        print(f"  Phase 118 FAILED: {e}",flush=True);traceback.print_exc()
    print(f"\n  Phases 117-118 complete. Total: {(time.time()-t)/60:.1f}min",flush=True)


if __name__=="__main__": run_all()
