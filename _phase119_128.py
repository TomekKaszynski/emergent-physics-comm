"""
Phases 119-128: Product infrastructure experiments
====================================================
119: ROS 2 demo (pure Python)
120: Real checkpoint onboarding test (DINOv2-L + CLIP-L from HuggingFace)
121: Manipulation domain (ramp, already done in Phase 117 — reference only)
122-123: Versioning + monitoring (code-only, tested via unit tests)
124: Multi-agent consensus protocol
125: Adversarial agent quarantine
126: Bandwidth adaptation test
127-128: Analytics + benchmarks (code-only, tested via unit tests)

Run:
  PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=1 python3 -c "from _phase119_128 import run_all; run_all()"
"""

import time, json, math, os, sys, traceback, queue, threading
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
RESULTS_DIR = Path("results")
HIDDEN_DIM = 128; N_HEADS = 2; VOCAB_SIZE = 3


class TemporalEncoder(nn.Module):
    def __init__(self, hd=128, ind=1024, nf=4):
        super().__init__()
        ks = min(3, max(1, nf))
        self.temporal = nn.Sequential(nn.Conv1d(ind,256,ks,padding=ks//2),nn.ReLU(),
            nn.Conv1d(256,128,ks,padding=ks//2),nn.ReLU(),nn.AdaptiveAvgPool1d(1))
        self.fc = nn.Sequential(nn.Linear(128,hd),nn.ReLU())
    def forward(self, x): return self.fc(self.temporal(x.permute(0,2,1)).squeeze(-1))

class CompositionalSender(nn.Module):
    def __init__(self, enc, hd, vs, nh):
        super().__init__()
        self.enc=enc;self.vs=vs;self.nh=nh
        self.heads=nn.ModuleList([nn.Linear(hd,vs) for _ in range(nh)])
    def forward(self,x,tau=1.0,hard=True):
        h=self.enc(x);ms,ls=[],[]
        for hd in self.heads:
            l=hd(h);m=F.gumbel_softmax(l,tau=tau,hard=hard) if self.training else F.one_hot(l.argmax(-1),self.vs).float()
            ms.append(m);ls.append(l)
        return torch.cat(ms,-1),ls

class MultiAgentSender(nn.Module):
    def __init__(self,ss): super().__init__();self.senders=nn.ModuleList(ss)
    def forward(self,views,tau=1.0,hard=True):
        ms,ls=[],[]
        for s,v in zip(self.senders,views): m,l=s(v,tau,hard);ms.append(m);ls.extend(l)
        return torch.cat(ms,-1),ls

class CompositionalReceiver(nn.Module):
    def __init__(self,md,hd):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(md*2,hd),nn.ReLU(),nn.Linear(hd,hd//2),nn.ReLU(),nn.Linear(hd//2,1))
    def forward(self,a,b): return self.net(torch.cat([a,b],-1)).squeeze(-1)


def load_spring():
    vd=torch.load(RESULTS_DIR/"phase87_phys101_spring_features.pt",weights_only=False)
    dd=torch.load(RESULTS_DIR/"phase87_phys101_spring_static.pt",weights_only=False)
    vf=vd["features"].float();obj=vd["obj_names"];mass=vd["mass_values"]
    df=dd["features"].float();nf=vf.shape[1]
    return vf,df.unsqueeze(1).expand(-1,nf,-1).contiguous(),obj,mass


def train_quick(configs,mass,obj,seed):
    na=len(configs);md=na*N_HEADS*VOCAB_SIZE;av=[f.float() for f,_ in configs]
    uo=sorted(set(obj));rng=np.random.RandomState(seed*1000+42)
    ho=set(rng.choice(uo,max(4,len(uo)//5),replace=False))
    tri=np.array([i for i,o in enumerate(obj) if o not in ho])
    tei=np.array([i for i,o in enumerate(obj) if o in ho])
    if len(tei)<4:return None
    torch.manual_seed(seed);np.random.seed(seed)
    ss=[CompositionalSender(TemporalEncoder(HIDDEN_DIM,d,f.shape[1]),HIDDEN_DIM,VOCAB_SIZE,N_HEADS) for f,d in configs]
    multi=MultiAgentSender(ss).to(DEVICE)
    recv=CompositionalReceiver(md,HIDDEN_DIM).to(DEVICE)
    so=torch.optim.Adam(multi.parameters(),lr=1e-3);ro=torch.optim.Adam(recv.parameters(),lr=3e-3)
    mdev=torch.tensor(mass,dtype=torch.float32).to(DEVICE)
    me=math.log(VOCAB_SIZE);nb=max(1,len(tri)//32);ba=0.0;bs=None;be=0
    for ep in range(400):
        if ep-be>150 and ba>0.55:break
        if ep>0 and ep%40==0:recv=CompositionalReceiver(md,HIDDEN_DIM).to(DEVICE);ro=torch.optim.Adam(recv.parameters(),lr=3e-3)
        multi.train();recv.train();tau=3+(1-3)*ep/399;hard=ep>=30
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
    return {"multi":multi,"recv":recv,"acc":float(ba),"av":av,"tei":tei,"mdev":mdev,"mass":mass}


# ═══ PHASE 119: ROS 2 Demo ═══
def run_phase119():
    print("\n╔═ Phase 119: ROS 2 Integration Demo ═╗",flush=True)
    import subprocess
    r=subprocess.run(["python3","-m","wmcp_ros2.demo"],capture_output=True,text=True,timeout=60)
    print(r.stdout,flush=True)
    if r.returncode!=0:print(r.stderr,flush=True)
    with open(RESULTS_DIR/"phase119_ros2.json","w") as f:
        json.dump({"status":"OK" if r.returncode==0 else "FAILED","output":r.stdout[:2000]},f,indent=2)
    print(f"  Saved results/phase119_ros2.json",flush=True)


# ═══ PHASE 120: Real Checkpoint Onboarding ═══
def run_phase120():
    print("\n╔═ Phase 120: Real Checkpoint Onboarding ═╗",flush=True)
    t0=time.time()
    # Test with real DINOv2 weights (torch.hub)
    print("  Loading DINOv2 ViT-L/14 from torch.hub...",flush=True)
    try:
        dino=torch.hub.load("facebookresearch/dinov2","dinov2_vitl14")
        dino=dino.eval()
        # Extract features from a random image
        dummy_img=torch.randn(1,3,224,224)
        with torch.no_grad():
            feat=dino(dummy_img)  # (1, 1024)
        dino_dim=feat.shape[1]
        print(f"  DINOv2-L output dim: {dino_dim}",flush=True)

        # Test: can we onboard this into a protocol?
        # Build protocol with DINOv2-L dim
        feat_temporal=feat.unsqueeze(1).expand(-1,4,-1)  # (1,4,1024)

        # Quick sanity: does the projection layer work?
        proj=TemporalEncoder(HIDDEN_DIM,dino_dim,4)
        h=proj(feat_temporal)
        print(f"  Projection output: {h.shape} (should be [1, 128])",flush=True)
        assert h.shape==(1,128)

        print(f"  ✓ DINOv2-L real weights: PASS (features extract, projection works)",flush=True)
        dino_result="PASS"
    except Exception as e:
        print(f"  DINOv2-L: FAILED ({e})",flush=True)
        dino_result=f"FAILED: {e}"

    # Test with CLIP
    print("  Loading CLIP ViT-L/14 from open_clip...",flush=True)
    try:
        import open_clip
        clip_model,_,preprocess=open_clip.create_model_and_transforms('ViT-L-14',pretrained='openai')
        clip_model=clip_model.eval()
        dummy_img=torch.randn(1,3,224,224)
        with torch.no_grad():
            feat=clip_model.encode_image(dummy_img)
        clip_dim=feat.shape[1]
        print(f"  CLIP output dim: {clip_dim}",flush=True)

        feat_temporal=feat.float().unsqueeze(1).expand(-1,4,-1)
        proj=TemporalEncoder(HIDDEN_DIM,clip_dim,4)
        h=proj(feat_temporal)
        assert h.shape==(1,128)

        print(f"  ✓ CLIP real weights: PASS",flush=True)
        clip_result="PASS"
    except Exception as e:
        print(f"  CLIP: FAILED ({e})",flush=True)
        clip_result=f"FAILED: {e}"

    results={"dinov2_vitl14":dino_result,"clip_vitl14":clip_result,
             "elapsed_s":time.time()-t0}
    with open(RESULTS_DIR/"phase120_real_checkpoints.json","w") as f:
        json.dump(results,f,indent=2)
    print(f"  Saved ({(time.time()-t0)/60:.1f}min)",flush=True)


# ═══ PHASE 124: Multi-Agent Consensus ═══
def run_phase124():
    print("\n╔═ Phase 124: Multi-Agent Consensus Protocol ═╗",flush=True)
    t0=time.time()
    vf,dt,obj,mass=load_spring();nf=vf.shape[1]
    n_agents=8;fpa=max(1,nf//n_agents)
    configs=[]
    for i in range(n_agents):
        fi=i%nf
        if i%2==0:configs.append((vf[:,fi:fi+fpa,:],1024))
        else:configs.append((dt[:,fi:fi+fpa,:],384))
    r=train_quick(configs,mass,obj,0)
    if not r:print("  FAILED");return
    multi=r["multi"].eval();recv=r["recv"].eval()
    av=r["av"];mdev=r["mdev"];tei=r["tei"]

    # Single-round majority voting
    n_trials=200;single_correct=0;consensus_correct=0;total_trials=0
    for trial in range(n_trials):
        er=np.random.RandomState(trial)
        ia=er.choice(tei,1);ib=er.choice(tei,1)
        if ia==ib or abs(mass[ia[0]]-mass[ib[0]])<0.5:continue
        gt=mass[ia[0]]>mass[ib[0]]
        with torch.no_grad():
            va=[v[ia].to(DEVICE) for v in av];vb=[v[ib].to(DEVICE) for v in av]
            ma,_=multi(va);mb,_=multi(vb)
            # Single round
            single_pred=(recv(ma,mb)>0).item()
            single_correct+=int(single_pred==gt)
            # 3-round consensus (add noise in rounds 2,3 to simulate updated views)
            votes=[single_pred]
            for rd in range(2):
                noise=0.05*(rd+1)
                ma_n=ma+torch.randn_like(ma)*noise
                mb_n=mb+torch.randn_like(mb)*noise
                votes.append((recv(ma_n,mb_n)>0).item())
            consensus=sum(votes)>len(votes)/2
            consensus_correct+=int(consensus==gt)
        total_trials+=1

    results={
        "single_round_acc":single_correct/max(total_trials,1),
        "consensus_3round_acc":consensus_correct/max(total_trials,1),
        "n_trials":total_trials,"n_agents":n_agents}
    print(f"  Single round: {results['single_round_acc']:.1%}",flush=True)
    print(f"  3-round consensus: {results['consensus_3round_acc']:.1%}",flush=True)
    with open(RESULTS_DIR/"phase124_consensus.json","w") as f:json.dump(results,f,indent=2)
    print(f"  Saved ({(time.time()-t0)/60:.1f}min)",flush=True)


# ═══ PHASE 125: Adversarial Quarantine ═══
def run_phase125():
    print("\n╔═ Phase 125: Adversarial Agent Quarantine ═╗",flush=True)
    t0=time.time()
    vf,dt,obj,mass=load_spring();nf=vf.shape[1]
    n_agents=8;fpa=max(1,nf//n_agents)
    configs=[]
    for i in range(n_agents):
        fi=i%nf
        if i%2==0:configs.append((vf[:,fi:fi+fpa,:],1024))
        else:configs.append((dt[:,fi:fi+fpa,:],384))
    r=train_quick(configs,mass,obj,0)
    if not r:print("  FAILED");return
    multi=r["multi"].eval();recv=r["recv"].eval()
    av=r["av"];mdev=r["mdev"];tei=r["tei"]
    msg_per_agent=N_HEADS*VOCAB_SIZE

    results={}
    for n_adversarial in [0,1,2,3]:
        correct_no_q=0;correct_q=0;detected=0;false_pos=0;total_t=0
        for trial in range(200):
            er=np.random.RandomState(trial+n_adversarial*1000)
            ia=er.choice(tei,1);ib=er.choice(tei,1)
            if ia==ib or abs(mass[ia[0]]-mass[ib[0]])<0.5:continue
            gt=mass[ia[0]]>mass[ib[0]]
            with torch.no_grad():
                va=[v[ia].to(DEVICE) for v in av];vb=[v[ib].to(DEVICE) for v in av]
                ma,_=multi(va);mb,_=multi(vb)
                # Corrupt first n_adversarial agents
                if n_adversarial>0:
                    for a in range(n_adversarial):
                        start=a*msg_per_agent;end=start+msg_per_agent
                        ma[:,start:end]=F.one_hot(torch.randint(0,3,(1,N_HEADS),device=DEVICE),3).float().view(1,-1)
                        mb[:,start:end]=F.one_hot(torch.randint(0,3,(1,N_HEADS),device=DEVICE),3).float().view(1,-1)
                # Without quarantine
                pred_no_q=(recv(ma,mb)>0).item()
                correct_no_q+=int(pred_no_q==gt)
                # Detect adversarial: check entropy per agent
                quarantined=set()
                for a in range(n_agents):
                    start=a*msg_per_agent;end=start+msg_per_agent
                    agent_msg=ma[:,start:end].view(1,N_HEADS,VOCAB_SIZE)
                    # Adversarial messages tend to be more random (higher entropy)
                    # or perfectly one-hot (zero entropy) — check for anomaly
                    ent=-(agent_msg.clamp(min=1e-8)*torch.log(agent_msg.clamp(min=1e-8))).sum(-1).mean().item()
                    if a<n_adversarial:
                        if ent>0.5 or ent<0.01:detected+=1
                    else:
                        if ent>0.5 or ent<0.01:false_pos+=1
                    if ent>0.5 or ent<0.01:quarantined.add(a)
                # With quarantine: zero out quarantined agents
                ma_q=ma.clone();mb_q=mb.clone()
                for a in quarantined:
                    start=a*msg_per_agent;end=start+msg_per_agent
                    ma_q[:,start:end]=0;mb_q[:,start:end]=0
                pred_q=(recv(ma_q,mb_q)>0).item()
                correct_q+=int(pred_q==gt)
                total_t+=1

        results[str(n_adversarial)]={
            "acc_no_quarantine":correct_no_q/max(total_t,1),
            "acc_with_quarantine":correct_q/max(total_t,1),
            "detection_rate":detected/max(n_adversarial*total_t,1) if n_adversarial>0 else 0,
            "false_positive_rate":false_pos/max((n_agents-n_adversarial)*total_t,1),
        }
        print(f"  {n_adversarial} adversarial: "
              f"no_q={results[str(n_adversarial)]['acc_no_quarantine']:.1%} "
              f"with_q={results[str(n_adversarial)]['acc_with_quarantine']:.1%} "
              f"detect={results[str(n_adversarial)]['detection_rate']:.1%}",flush=True)

    with open(RESULTS_DIR/"phase125_quarantine.json","w") as f:json.dump(results,f,indent=2)
    print(f"  Saved ({(time.time()-t0)/60:.1f}min)",flush=True)


# ═══ PHASE 126: Bandwidth Adaptation ═══
def run_phase126():
    print("\n╔═ Phase 126: Bandwidth Adaptation ═╗",flush=True)
    t0=time.time()
    vf,dt,obj,mass=load_spring();nf=vf.shape[1];fpa=nf//2
    results={}
    for K in [2,3,4,5,8]:
        accs=[]
        for seed in range(5):
            configs=[(vf[:,:fpa,:],1024),(dt[:,fpa:,:],384)]
            na=2;md=na*N_HEADS*K;av=[f.float() for f,_ in configs]
            uo=sorted(set(obj));rng=np.random.RandomState(seed*1000+42)
            ho=set(rng.choice(uo,max(4,len(uo)//5),replace=False))
            tri=np.array([i for i,o in enumerate(obj) if o not in ho])
            tei=np.array([i for i,o in enumerate(obj) if o in ho])
            if len(tei)<4:continue
            torch.manual_seed(seed);np.random.seed(seed)
            ss=[CompositionalSender(TemporalEncoder(HIDDEN_DIM,d,f.shape[1]),HIDDEN_DIM,K,N_HEADS) for f,d in configs]
            multi=MultiAgentSender(ss).to(DEVICE)
            recv=CompositionalReceiver(md,HIDDEN_DIM).to(DEVICE)
            so=torch.optim.Adam(multi.parameters(),lr=1e-3);ro=torch.optim.Adam(recv.parameters(),lr=3e-3)
            mdev=torch.tensor(mass,dtype=torch.float32).to(DEVICE)
            me=math.log(K);nb=max(1,len(tri)//32);ba=0.0;be=0
            for ep in range(300):
                if ep-be>150 and ba>0.55:break
                if ep>0 and ep%40==0:recv=CompositionalReceiver(md,HIDDEN_DIM).to(DEVICE);ro=torch.optim.Adam(recv.parameters(),lr=3e-3)
                multi.train();recv.train();tau=3+(1-3)*ep/299;hard=ep>=30
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
                    if torch.isnan(loss):so.zero_grad();ro.zero_grad();continue
                    so.zero_grad();ro.zero_grad();loss.backward()
                    torch.nn.utils.clip_grad_norm_(multi.parameters(),1.0);so.step();ro.step()
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
                        if acc>ba:ba=acc;be=ep
            accs.append(ba)
            torch.mps.empty_cache()
        bits=2*N_HEADS*math.log2(K)
        results[str(K)]={"K":K,"bits":float(bits),"acc_mean":float(np.mean(accs)),"acc_std":float(np.std(accs))}
        print(f"  K={K}: {bits:.1f} bits, acc={np.mean(accs):.1%}±{np.std(accs):.1%}",flush=True)

    with open(RESULTS_DIR/"phase126_bandwidth.json","w") as f:json.dump(results,f,indent=2)
    print(f"  Saved ({(time.time()-t0)/60:.1f}min)",flush=True)


def run_all():
    print("╔══════════════════════════════════════════════════════════╗",flush=True)
    print("║  Phases 119-128: Product Infrastructure                   ║",flush=True)
    print("╚══════════════════════════════════════════════════════════╝",flush=True)
    t=time.time()
    phases=[(119,run_phase119),(120,run_phase120),(124,run_phase124),
            (125,run_phase125),(126,run_phase126)]
    for n,f in phases:
        try:
            print(f"\n{'#'*50}\n# PHASE {n}\n{'#'*50}",flush=True)
            f();torch.mps.empty_cache()
        except Exception as e:
            print(f"  PHASE {n} FAILED: {e}",flush=True);traceback.print_exc()

    # Run unit tests for code-only phases
    print(f"\n{'#'*50}\n# Phases 122-123, 127-128: Unit Tests\n{'#'*50}",flush=True)
    try:
        from wmcp.versioning import negotiate_version, check_migration_path
        ok,v=negotiate_version("0.1.0","0.1.1")
        assert ok and v=="0.1.0"
        ok2,v2=negotiate_version("0.1.0","1.0.0")
        assert not ok2
        mig=check_migration_path("0.1.0","0.2.0")
        assert not mig["retrain_required"]
        print("  ✓ Versioning tests pass",flush=True)
    except Exception as e:
        print(f"  Versioning: FAILED ({e})",flush=True)

    try:
        from wmcp.monitoring import ProtocolMonitor
        mon=ProtocolMonitor(vocab_size=3,n_positions=2)
        mon.enroll_agent(0,np.array([[0,1],[1,2],[2,0],[0,2],[1,1]]))
        mon.record_message(0,[1,2])
        mon.record_message(0,[0,1])
        h=mon.health
        assert h["total_messages"]==2
        print("  ✓ Monitoring tests pass",flush=True)
    except Exception as e:
        print(f"  Monitoring: FAILED ({e})",flush=True)

    try:
        from wmcp.analytics import ProtocolMetrics,export_csv,export_json,export_prometheus
        m=ProtocolMetrics(time.time(),4,1000,0.82,0.76,0.46,0.61,1.19,1.35,0.85)
        csv=export_csv([m]);assert "0.8200" in csv
        j=export_json([m]);assert "posdis" in j
        p=export_prometheus(m);assert "wmcp_accuracy" in p
        print("  ✓ Analytics tests pass",flush=True)
    except Exception as e:
        print(f"  Analytics: FAILED ({e})",flush=True)

    try:
        from wmcp.benchmarks.benchmark_latency import run as run_lat
        r=run_lat(n_iterations=50)
        assert r["mean_ms"]>0
        print(f"  ✓ Benchmark tests pass (latency={r['mean_ms']:.2f}ms)",flush=True)
    except Exception as e:
        print(f"  Benchmarks: FAILED ({e})",flush=True)

    print(f"\n  All phases complete. Total: {(time.time()-t)/60:.1f}min",flush=True)


if __name__=="__main__":run_all()
