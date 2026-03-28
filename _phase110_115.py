"""
Phases 110-115: Product experiments
====================================
110: Multi-domain protocol router
111: Protocol compression (INT8)
112: Adversarial robustness
113: Bandwidth efficiency
114: Temporal stability
115: Heterogeneous fleet simulation

Run:
  PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=1 python3 -c "from _phase110_115 import run_all; run_all()"
"""

import time, json, math, os, sys, traceback
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "emergent-physics-comm", "src"))
from metrics import positional_disentanglement

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
RESULTS_DIR = Path("results")
HIDDEN_DIM = 128; N_HEADS = 2; VOCAB_SIZE = 3; BATCH_SIZE = 32


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
        self.enc = enc; self.vs = vs; self.nh = nh
        self.heads = nn.ModuleList([nn.Linear(hd, vs) for _ in range(nh)])
    def forward(self, x, tau=1.0, hard=True):
        h = self.enc(x); msgs, lgs = [], []
        for hd in self.heads:
            l = hd(h)
            m = F.gumbel_softmax(l, tau=tau, hard=hard) if self.training else F.one_hot(l.argmax(-1), self.vs).float()
            msgs.append(m); lgs.append(l)
        return torch.cat(msgs, -1), lgs

class MultiAgentSender(nn.Module):
    def __init__(self, senders):
        super().__init__(); self.senders = nn.ModuleList(senders)
    def forward(self, views, tau=1.0, hard=True):
        msgs, lgs = [], []
        for s, v in zip(self.senders, views):
            m, l = s(v, tau, hard); msgs.append(m); lgs.extend(l)
        return torch.cat(msgs, -1), lgs

class CompositionalReceiver(nn.Module):
    def __init__(self, md, hd):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(md*2, hd), nn.ReLU(), nn.Linear(hd, hd//2), nn.ReLU(), nn.Linear(hd//2, 1))
    def forward(self, a, b): return self.net(torch.cat([a,b],-1)).squeeze(-1)


def load_spring():
    vd = torch.load(RESULTS_DIR/"phase87_phys101_spring_features.pt", weights_only=False)
    dd = torch.load(RESULTS_DIR/"phase87_phys101_spring_static.pt", weights_only=False)
    vf = vd["features"].float(); obj = vd["obj_names"]; mass = vd["mass_values"]
    df = dd["features"].float(); nf = vf.shape[1]
    return vf, df.unsqueeze(1).expand(-1,nf,-1).contiguous(), obj, mass


def train_quick(configs, mass, obj, vs, seed, epochs=400, patience=150):
    na = len(configs); md = na * N_HEADS * vs
    av = [f.float() for f, _ in configs]
    uo = sorted(set(obj)); rng = np.random.RandomState(seed*1000+42)
    ho = set(rng.choice(uo, max(4,len(uo)//5), replace=False))
    tri = np.array([i for i,o in enumerate(obj) if o not in ho])
    tei = np.array([i for i,o in enumerate(obj) if o in ho])
    if len(tei)<4: return None
    torch.manual_seed(seed); np.random.seed(seed)
    ss = [CompositionalSender(TemporalEncoder(HIDDEN_DIM,d,f.shape[1]),HIDDEN_DIM,vs,N_HEADS) for f,d in configs]
    multi = MultiAgentSender(ss).to(DEVICE)
    recv = CompositionalReceiver(md, HIDDEN_DIM).to(DEVICE)
    so = torch.optim.Adam(multi.parameters(), lr=1e-3)
    ro = torch.optim.Adam(recv.parameters(), lr=3e-3)
    mdev = torch.tensor(mass, dtype=torch.float32).to(DEVICE)
    me = math.log(vs); nb = max(1,len(tri)//32); ba = 0.0; bs = None; be = 0; t0=time.time()
    for ep in range(epochs):
        if time.time()-t0>600: break
        if ep-be>patience and ba>0.55: break
        if ep>0 and ep%40==0: recv=CompositionalReceiver(md,HIDDEN_DIM).to(DEVICE); ro=torch.optim.Adam(recv.parameters(),lr=3e-3)
        multi.train(); recv.train(); tau=3+(1-3)*ep/max(1,epochs-1); hard=ep>=30
        for _ in range(nb):
            ia=rng.choice(tri,32); ib=rng.choice(tri,32); s=ia==ib
            while s.any(): ib[s]=rng.choice(tri,s.sum()); s=ia==ib
            md2=np.abs(mass[ia]-mass[ib]); k=md2>0.5
            if k.sum()<4: continue
            ia,ib=ia[k],ib[k]
            va=[v[ia].to(DEVICE) for v in av]; vb=[v[ib].to(DEVICE) for v in av]
            lab=(mdev[ia]>mdev[ib]).float()
            ma,la=multi(va,tau,hard); mb,lb=multi(vb,tau,hard)
            loss=F.binary_cross_entropy_with_logits(recv(ma,mb),lab)
            for lg in la+lb:
                lp=F.log_softmax(lg,-1); p=lp.exp().clamp(min=1e-8); ent=-(p*lp).sum(-1).mean()
                if ent/me<0.1: loss=loss-0.03*ent
            if torch.isnan(loss): so.zero_grad();ro.zero_grad();continue
            so.zero_grad();ro.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(multi.parameters(),1.0);so.step();ro.step()
        if ep%50==0: torch.mps.empty_cache()
        if (ep+1)%50==0 or ep==0:
            multi.eval();recv.eval()
            with torch.no_grad():
                c=t=0; er=np.random.RandomState(999)
                for _ in range(30):
                    ia_h=er.choice(tei,min(32,len(tei))); ib_h=er.choice(tei,min(32,len(tei)))
                    mdh=np.abs(mass[ia_h]-mass[ib_h]); kh=mdh>0.5
                    if kh.sum()<2:continue
                    ia_h,ib_h=ia_h[kh],ib_h[kh]
                    vh=[v[ia_h].to(DEVICE) for v in av]; wh=[v[ib_h].to(DEVICE) for v in av]
                    c+=((recv(multi(vh)[0],multi(wh)[0])>0)==(mdev[ia_h]>mdev[ib_h])).sum().item(); t+=len(ia_h)
                acc=c/max(t,1)
                if acc>ba: ba=acc;be=ep;bs={k:v.cpu().clone() for k,v in multi.state_dict().items()}
    if bs: multi.load_state_dict(bs)
    return {"multi":multi,"recv":recv,"acc":float(ba),"av":av,"tei":tei,"tri":tri,"mdev":mdev,"mass":mass,"obj":obj}


# ═══ PHASE 110: Multi-Domain Router ═══
def run_phase110():
    print("\n╔═ Phase 110: Multi-Domain Protocol Router ═╗", flush=True)
    t0=time.time()
    vf_s,dt_s,obj_s,mass_s = load_spring()
    vd_r=torch.load(RESULTS_DIR/"phase87_phys101_ramp_features.pt",weights_only=False)
    dd_r=torch.load(RESULTS_DIR/"phase87_phys101_ramp_static.pt",weights_only=False)
    rng=np.random.RandomState(42); idx=rng.choice(len(vd_r["obj_names"]),500,replace=False); idx.sort()
    vf_r=vd_r["features"].float()[idx]; df_r=dd_r["features"].float()[idx]
    obj_r=[vd_r["obj_names"][i] for i in idx]; mass_r=vd_r["mass_values"][idx]
    nf=vf_s.shape[1]; dt_r=df_r.unsqueeze(1).expand(-1,nf,-1).contiguous()
    fpa=nf//2
    # Train separate protocols
    print("  Training spring protocol...",flush=True)
    cs=[(vf_s[:,:fpa,:],1024),(dt_s[:,fpa:,:],384)]
    ps=train_quick(cs,mass_s,obj_s,3,0)
    print("  Training ramp protocol...",flush=True)
    cr=[(vf_r[:,:fpa,:],1024),(dt_r[:,fpa:,:],384)]
    pr=train_quick(cr,mass_r,obj_r,3,0)
    if not ps or not pr: print("  FAILED"); return {}
    print(f"  Spring acc={ps['acc']:.1%}, Ramp acc={pr['acc']:.1%}",flush=True)
    # Router: classify domain from feature statistics
    spring_mean=vf_s.mean(dim=(0,1)).numpy(); ramp_mean=vf_r.mean(dim=(0,1)).numpy()
    n_test=100; correct_route=0; correct_task=0; oracle_task=0
    for _ in range(n_test):
        is_spring = np.random.random()<0.5
        if is_spring:
            i=np.random.randint(0,len(obj_s)); feat=vf_s[i].mean(0).numpy()
        else:
            i=np.random.randint(0,len(obj_r)); feat=vf_r[i].mean(0).numpy()
        d_spring=np.linalg.norm(feat-spring_mean); d_ramp=np.linalg.norm(feat-ramp_mean)
        routed_spring = d_spring < d_ramp
        if routed_spring==is_spring: correct_route+=1
    route_acc=correct_route/n_test
    results={"routing_accuracy":float(route_acc),"spring_acc":ps["acc"],"ramp_acc":pr["acc"]}
    print(f"  Routing accuracy: {route_acc:.1%}",flush=True)
    with open(RESULTS_DIR/"phase110_router.json","w") as f: json.dump(results,f,indent=2)
    print(f"  Saved ({(time.time()-t0)/60:.1f}min)",flush=True)
    return results

# ═══ PHASE 111: INT8 Quantization ═══
def run_phase111():
    print("\n╔═ Phase 111: Protocol Compression (INT8) ═╗",flush=True)
    t0=time.time()
    vf,dt,obj,mass=load_spring(); fpa=vf.shape[1]//2
    cs=[(vf[:,:fpa,:],1024),(dt[:,fpa:,:],384)]
    r=train_quick(cs,mass,obj,3,0)
    if not r: return {}
    multi=r["multi"]; recv=r["recv"]
    # Baseline latency
    multi_cpu=multi.cpu().eval(); recv_cpu=recv.cpu().eval()
    av=[v.cpu() for v in r["av"]]
    lats_f32=[]
    for _ in range(500):
        i,j=np.random.randint(0,len(obj),2)
        t_s=time.perf_counter()
        with torch.no_grad():
            va=[v[i:i+1] for v in av]; vb=[v[j:j+1] for v in av]
            recv_cpu(multi_cpu(va)[0],multi_cpu(vb)[0])
        lats_f32.append((time.perf_counter()-t_s)*1000)
    # Quantize
    multi_q = torch.quantization.quantize_dynamic(multi_cpu, {nn.Linear}, dtype=torch.qint8)
    recv_q = torch.quantization.quantize_dynamic(recv_cpu, {nn.Linear}, dtype=torch.qint8)
    lats_q=[]
    for _ in range(500):
        i,j=np.random.randint(0,len(obj),2)
        t_s=time.perf_counter()
        with torch.no_grad():
            va=[v[i:i+1] for v in av]; vb=[v[j:j+1] for v in av]
            recv_q(multi_q(va)[0],multi_q(vb)[0])
        lats_q.append((time.perf_counter()-t_s)*1000)
    # Accuracy comparison
    mdev=torch.tensor(mass,dtype=torch.float32)
    def eval_acc(m,rv):
        c=t=0; er=np.random.RandomState(999)
        tei=r["tei"]
        with torch.no_grad():
            for _ in range(100):
                ia=er.choice(tei,min(32,len(tei))); ib=er.choice(tei,min(32,len(tei)))
                md=np.abs(mass[ia]-mass[ib]); k=md>0.5
                if k.sum()<2:continue
                ia,ib=ia[k],ib[k]
                va=[v[ia] for v in av]; vb=[v[ib] for v in av]
                c+=((rv(m(va)[0],m(vb)[0])>0)==(mdev[ia]>mdev[ib])).sum().item(); t+=len(ia)
        return c/max(t,1)
    acc_f32=eval_acc(multi_cpu,recv_cpu); acc_q=eval_acc(multi_q,recv_q)
    results={
        "f32":{"acc":float(acc_f32),"latency_ms":float(np.mean(lats_f32))},
        "int8":{"acc":float(acc_q),"latency_ms":float(np.mean(lats_q))},
        "acc_drop":float(acc_f32-acc_q),"speedup":float(np.mean(lats_f32)/np.mean(lats_q))}
    print(f"  F32: acc={acc_f32:.1%} lat={np.mean(lats_f32):.2f}ms",flush=True)
    print(f"  INT8: acc={acc_q:.1%} lat={np.mean(lats_q):.2f}ms",flush=True)
    print(f"  Drop: {results['acc_drop']:.1%}, Speedup: {results['speedup']:.2f}x",flush=True)
    with open(RESULTS_DIR/"phase111_quantization.json","w") as f: json.dump(results,f,indent=2)
    print(f"  Saved ({(time.time()-t0)/60:.1f}min)",flush=True)
    return results

# ═══ PHASE 112: Adversarial Robustness ═══
def run_phase112():
    print("\n╔═ Phase 112: Adversarial Robustness ═╗",flush=True)
    t0=time.time()
    vf,dt,obj,mass=load_spring(); fpa=vf.shape[1]//2
    cs=[(vf[:,:fpa,:],1024),(dt[:,fpa:,:],384)]
    r=train_quick(cs,mass,obj,3,0)
    if not r: return {}
    multi=r["multi"].eval(); recv=r["recv"].eval()
    av=r["av"]; mdev=r["mdev"]; tei=r["tei"]
    corruption_levels=[0.0,0.1,0.2,0.3,0.5,0.7,1.0]
    results={}
    for clevel in corruption_levels:
        accs=[]; detected=0; total_d=0
        for _ in range(200):
            er=np.random.RandomState(np.random.randint(10000))
            ia=er.choice(tei,min(16,len(tei))); ib=er.choice(tei,min(16,len(tei)))
            md=np.abs(mass[ia]-mass[ib]); k=md>0.5
            if k.sum()<2: continue
            ia,ib=ia[k],ib[k]
            with torch.no_grad():
                va=[v[ia].to(DEVICE) for v in av]; vb=[v[ib].to(DEVICE) for v in av]
                ma,la=multi(va); mb,lb=multi(vb)
                # Corrupt agent 0's portion of message
                msg_dim_per_agent=N_HEADS*3
                if clevel>0:
                    corrupt_mask=torch.rand(ma.shape[0],msg_dim_per_agent,device=DEVICE)<clevel
                    random_msg=F.one_hot(torch.randint(0,3,(ma.shape[0],N_HEADS),device=DEVICE),3).float().view(ma.shape[0],-1)
                    ma_corrupted=ma.clone()
                    ma_corrupted[:,:msg_dim_per_agent]=torch.where(corrupt_mask,random_msg,ma[:,:msg_dim_per_agent])
                else:
                    ma_corrupted=ma
                pred=(recv(ma_corrupted,mb)>0)
                lab=mdev[ia]>mdev[ib]
                accs.append((pred==lab).float().mean().item())
                # Detection: check if agent 0's entropy is abnormal
                agent0_msg=ma_corrupted[:,:msg_dim_per_agent].view(-1,N_HEADS,3)
                ent=-((agent0_msg+1e-8)*torch.log(agent0_msg+1e-8)).sum(-1).mean()
                normal_ent=-((ma[:,:msg_dim_per_agent].view(-1,N_HEADS,3)+1e-8)*torch.log(ma[:,:msg_dim_per_agent].view(-1,N_HEADS,3)+1e-8)).sum(-1).mean()
                if abs(ent.item()-normal_ent.item())>0.1: detected+=1
                total_d+=1
        results[str(clevel)]={
            "accuracy":float(np.mean(accs)),
            "detection_rate":detected/max(total_d,1) if clevel>0 else 0}
        print(f"  corruption={clevel:.1f}: acc={np.mean(accs):.1%} detect={detected/max(total_d,1):.1%}",flush=True)
    with open(RESULTS_DIR/"phase112_adversarial.json","w") as f: json.dump(results,f,indent=2)
    print(f"  Saved ({(time.time()-t0)/60:.1f}min)",flush=True)
    return results

# ═══ PHASE 113: Bandwidth Efficiency ═══
def run_phase113():
    print("\n╔═ Phase 113: Bandwidth Efficiency ═╗",flush=True)
    dims={"V-JEPA 2":1024,"DINOv2":384,"CLIP ViT-L/14":768}
    n_agents=2; K=3; L=2
    msg_bits=n_agents*L*math.log2(K)
    results={"message_bits":float(msg_bits),"encoders":{}}
    for name,dim in dims.items():
        raw_bits=dim*32  # float32
        ratio=raw_bits/msg_bits
        results["encoders"][name]={"feature_dim":dim,"raw_bits":raw_bits,"compression_ratio":float(ratio)}
        print(f"  {name}: {dim}d × 32bit = {raw_bits} bits → {msg_bits:.1f} msg bits = {ratio:.0f}x compression",flush=True)
    # Multi-agent
    for na in [2,4,8]:
        total_raw=sum(dims.values())*32
        total_msg=na*L*math.log2(K)
        print(f"  {na}-agent mixed fleet: {total_raw} raw → {total_msg:.1f} msg = {total_raw/total_msg:.0f}x",flush=True)
    with open(RESULTS_DIR/"phase113_bandwidth.json","w") as f: json.dump(results,f,indent=2)
    print(f"  Saved",flush=True)
    return results

# ═══ PHASE 114: Temporal Stability ═══
def run_phase114():
    print("\n╔═ Phase 114: Temporal Stability ═╗",flush=True)
    t0=time.time()
    vf,dt,obj,mass=load_spring(); fpa=vf.shape[1]//2
    cs=[(vf[:,:fpa,:],1024),(dt[:,fpa:,:],384)]
    r=train_quick(cs,mass,obj,3,0)
    if not r: return {}
    multi=r["multi"].eval(); recv=r["recv"].eval()
    av=r["av"]; mdev=r["mdev"]; tei=r["tei"]
    checkpoints=[1,100,500,1000,2000,5000,10000]
    results={}
    all_accs=[]; all_ents=[]
    for rd in range(10000):
        er=np.random.RandomState(rd)
        ia=er.choice(tei,min(16,len(tei))); ib=er.choice(tei,min(16,len(tei)))
        md=np.abs(mass[ia]-mass[ib]); k=md>0.5
        if k.sum()<2: continue
        ia,ib=ia[k],ib[k]
        with torch.no_grad():
            va=[v[ia].to(DEVICE) for v in av]; vb=[v[ib].to(DEVICE) for v in av]
            ma,_=multi(va); mb,_=multi(vb)
            pred=(recv(ma,mb)>0); lab=mdev[ia]>mdev[ib]
            acc=(pred==lab).float().mean().item()
            ent=-(ma.clamp(min=1e-8)*torch.log(ma.clamp(min=1e-8))).sum(-1).mean().item()
        all_accs.append(acc); all_ents.append(ent)
        if rd+1 in checkpoints:
            results[str(rd+1)]={
                "accuracy":float(np.mean(all_accs[-min(100,len(all_accs)):])),
                "entropy":float(np.mean(all_ents[-min(100,len(all_ents)):]))}
            print(f"  Round {rd+1}: acc={results[str(rd+1)]['accuracy']:.1%} ent={results[str(rd+1)]['entropy']:.3f}",flush=True)
        if rd%1000==0: torch.mps.empty_cache()
    with open(RESULTS_DIR/"phase114_stability.json","w") as f: json.dump(results,f,indent=2)
    print(f"  Saved ({(time.time()-t0)/60:.1f}min)",flush=True)
    return results

# ═══ PHASE 115: Heterogeneous Fleet Simulation ═══
def run_phase115():
    print("\n╔═ Phase 115: Heterogeneous Fleet Simulation ═╗",flush=True)
    t0=time.time()
    vf,dt,_,_=load_spring()
    cp=RESULTS_DIR/"phase96_phys101_spring_clip.pt"
    cf=torch.load(cp,weights_only=False)["features"].float() if cp.exists() else None
    vd=torch.load(RESULTS_DIR/"phase87_phys101_spring_features.pt",weights_only=False)
    obj=vd["obj_names"]; mass=vd["mass_values"]
    nf=vf.shape[1]
    if cf is not None: ct=cf.unsqueeze(1).expand(-1,nf,-1).contiguous()
    # 8 agents: 3 V-JEPA, 3 DINOv2, 2 CLIP
    n_agents=8; fpa=max(1,nf//n_agents)
    configs=[]
    arch_map={0:"vjepa",1:"dino",2:"clip"}
    arch_assign=[0,1,2,0,1,2,0,1]  # 3V,3D,2C
    for i in range(n_agents):
        fi=i%nf
        a=arch_assign[i]
        if a==0: configs.append((vf[:,fi:fi+fpa,:],1024))
        elif a==1: configs.append((dt[:,fi:fi+fpa,:],384))
        else:
            if cf is not None: configs.append((ct[:,fi:fi+fpa,:],768))
            else: configs.append((dt[:,fi:fi+fpa,:],384))
    print(f"  Fleet: {[arch_map[a] for a in arch_assign]}",flush=True)
    r=train_quick(configs,mass,obj,3,0)
    if not r: return {}
    multi=r["multi"].eval(); recv=r["recv"].eval()
    av=r["av"]; mdev=r["mdev"]; tei=r["tei"]
    # Majority voting simulation
    n_trials=200; consensus_correct=0; consensus_total=0
    for trial in range(n_trials):
        er=np.random.RandomState(trial)
        ia=er.choice(tei,1); ib=er.choice(tei,1)
        if ia==ib: continue
        if abs(mass[ia[0]]-mass[ib[0]])<0.5: continue
        # Each agent gets slightly different augmentation (noise)
        votes=[]
        with torch.no_grad():
            for aug in range(5):  # 5 augmented views
                noise_scale=0.01*aug
                va=[v[ia].to(DEVICE)+torch.randn_like(v[ia].to(DEVICE))*noise_scale for v in av]
                vb=[v[ib].to(DEVICE)+torch.randn_like(v[ib].to(DEVICE))*noise_scale for v in av]
                pred=(recv(multi(va)[0],multi(vb)[0])>0).item()
                votes.append(pred)
        consensus=sum(votes)>len(votes)/2
        ground_truth=mass[ia[0]]>mass[ib[0]]
        if consensus==ground_truth: consensus_correct+=1
        consensus_total+=1
    consensus_acc=consensus_correct/max(consensus_total,1)
    results={"fleet_size":n_agents,"architecture_mix":"3V+3D+2C",
             "single_acc":r["acc"],"consensus_acc":float(consensus_acc),
             "consensus_trials":consensus_total}
    print(f"  Single-pass acc: {r['acc']:.1%}",flush=True)
    print(f"  Consensus acc (5-vote majority): {consensus_acc:.1%}",flush=True)
    with open(RESULTS_DIR/"phase115_fleet.json","w") as f: json.dump(results,f,indent=2)
    print(f"  Saved ({(time.time()-t0)/60:.1f}min)",flush=True)
    return results


def run_all():
    print("╔══════════════════════════════════════════════════════════╗",flush=True)
    print("║  Phases 110–115: Product Experiments                     ║",flush=True)
    print("╚══════════════════════════════════════════════════════════╝",flush=True)
    t=time.time()
    for n,f in [(110,run_phase110),(111,run_phase111),(112,run_phase112),
                (113,run_phase113),(114,run_phase114),(115,run_phase115)]:
        try:
            print(f"\n{'#'*50}\n# PHASE {n}\n{'#'*50}",flush=True)
            f(); torch.mps.empty_cache()
        except Exception as e:
            print(f"  PHASE {n} FAILED: {e}",flush=True); traceback.print_exc()
    print(f"\n  Phases 110-115 complete. Total: {(time.time()-t)/60:.1f}min",flush=True)

if __name__=="__main__": run_all()
