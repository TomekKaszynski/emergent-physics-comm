"""
Final Overnight Battery — 10 Experiments
Run: PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=1 python3 _final_overnight.py
"""

import time, json, math, os, sys, traceback
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from itertools import combinations, permutations
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
    os.system(f'cd /Users/tomek/AI && git add results/ _final_overnight.py '
              f'&& git commit -m "{name}\n\nCo-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>" 2>/dev/null')

def load_meta():
    d = torch.load("results/phase87_phys101_spring_features.pt", weights_only=False)
    obj = d["obj_names"]; mass = d["mass_values"]
    p1 = np.digitize(mass, np.quantile(mass, [0.2, 0.4, 0.6, 0.8]))
    uo = sorted(set(obj)); oi = {o:i for i,o in enumerate(uo)}
    p2 = np.digitize(np.array([oi[o] for o in obj]), np.quantile(np.arange(len(uo)), [0.2, 0.4, 0.6, 0.8]))
    return p1, p2, obj, mass

def load_feat(name):
    if name == "vjepa2":
        return torch.load("results/phase87_phys101_spring_features.pt", weights_only=False)["features"].float()
    elif name == "dinov2":
        f = torch.load("results/phase87_phys101_spring_static.pt", weights_only=False)["features"].float()
        return f.unsqueeze(1).expand(-1, 8, -1).contiguous()
    elif name == "clip":
        f = torch.load("results/phase96_phys101_spring_clip.pt", weights_only=False)["features"].float()
        return f.unsqueeze(1).expand(-1, 8, -1).contiguous()
    elif name == "text":
        return torch.load("results/crossmodal/text_hidden_states.pt", weights_only=False)["features"].float()
    elif name == "audio":
        return torch.load("results/crossmodal/audio/audio_features.pt", weights_only=False)["features"].float()
    return None

def train_d(feat, mass, obj, seed):
    t0=time.time(); n,nf,dim=feat.shape; fpa=max(1,nf//N_AGENTS)
    views=[feat[:,(i*fpa)%nf:(i*fpa)%nf+fpa,:] for i in range(N_AGENTS)]
    torch.manual_seed(seed);np.random.seed(seed)
    ss=[DiscreteSender(TemporalEncoder(HIDDEN_DIM,dim,fpa),HIDDEN_DIM,VOCAB_SIZE,N_HEADS) for _ in range(N_AGENTS)]
    sender=DiscreteMultiSender(ss).to(DEVICE)
    recvs=[Receiver(MSG_DIM,HIDDEN_DIM).to(DEVICE) for _ in range(3)]
    so=torch.optim.Adam(sender.parameters(),lr=1e-3)
    ros=[torch.optim.Adam(r.parameters(),lr=3e-3) for r in recvs]
    mass_dev=torch.tensor(mass,dtype=torch.float32).to(DEVICE)
    rng=np.random.RandomState(seed*1000+42)
    uo=sorted(set(obj));ho=set(rng.choice(uo,max(4,len(uo)//5),replace=False))
    tr=np.array([i for i,o in enumerate(obj) if o not in ho])
    tei=np.array([i for i,o in enumerate(obj) if o in ho])
    nb=max(1,len(tr)//32);me=math.log(VOCAB_SIZE);ba=0;bst=None;bep=0
    for ep in range(COMM_EPOCHS):
        if ep-bep>EARLY_STOP and ba>0.55:break
        if ep>0 and ep%40==0:
            for i in range(3):recvs[i]=Receiver(MSG_DIM,HIDDEN_DIM).to(DEVICE);ros[i]=torch.optim.Adam(recvs[i].parameters(),lr=3e-3)
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
    toks=[];msgs=[]
    with torch.no_grad():
        for i in range(0,n,BATCH_SIZE):
            v2=[vi[i:i+BATCH_SIZE].to(DEVICE) for vi in views]
            m,logits=sender(v2);msgs.append(m.cpu())
            toks.append(np.stack([l.argmax(-1).cpu().numpy() for l in logits],1))
    TIMING.append(time.time()-t0)
    return sender,recvs[0],np.concatenate(toks,0),torch.cat(msgs,0),ba,views

def get_assigns(tokens,p1,p2):
    attrs=np.stack([p1,p2],axis=1);_,mi=compute_posdis(tokens,attrs,VOCAB_SIZE)
    return [int(np.argmax(mi[p])) for p in range(mi.shape[0])]

def xagree(a,b):
    ag=[];
    for x in a:
        for y in b: ag.append(sum(1 for i,j in zip(x,y) if i==j)/len(x))
    return float(np.mean(ag))


# ═══════════════════════════════════════════════════════════════
# QW1: Permutation-Aligned Transfer
# ═══════════════════════════════════════════════════════════════

def qw1_permutation():
    print(f"\n{'#'*60}\n# QW1: Permutation-Aligned Transfer\n# {elapsed()}\n{'#'*60}",flush=True)
    d=RB/"quick_win_permutation";d.mkdir(parents=True,exist_ok=True)
    p1,p2,obj,mass=load_meta();n=len(mass)
    mass_dev=torch.tensor(mass,dtype=torch.float32).to(DEVICE)
    mods=["vjepa2","dinov2","clip","text","audio"]

    # Train one sender+receiver per modality (seed=0)
    trained={}
    for mod in mods:
        feat=load_feat(mod)
        if feat is None:continue
        sender,recv,tokens,msgs,acc,views=train_d(feat,mass,obj,0)
        trained[mod]={"sender":sender,"recv":recv,"tokens":tokens,"msgs":msgs,"acc":acc}
        print(f"  {mod}: acc={acc:.1%}",flush=True)
        torch.mps.empty_cache()

    avail=list(trained.keys())

    # Build transfer matrices: unaligned and aligned
    n_mods=len(avail)
    unaligned=np.zeros((n_mods,n_mods))
    aligned=np.zeros((n_mods,n_mods))

    def eval_recv(msgs_tensor,recv):
        c=t=0;er=np.random.RandomState(999)
        with torch.no_grad():
            for _ in range(100):
                ia=er.choice(n,min(32,n));ib=er.choice(n,min(32,n))
                s=ia==ib
                while s.any():ib[s]=er.choice(n,s.sum());s=ia==ib
                md=np.abs(mass[ia]-mass[ib]);k=md>0.5
                if k.sum()<2:continue
                ia,ib=ia[k],ib[k]
                pred=recv(msgs_tensor[ia].to(DEVICE),msgs_tensor[ib].to(DEVICE))>0
                lab=mass_dev[ia]>mass_dev[ib]
                c+=(pred==lab).sum().item();t+=len(lab)
        return c/max(t,1)

    def permute_msgs(msgs,perm_map):
        """Apply per-position token permutation to one-hot messages."""
        out=msgs.clone()
        n_pos=N_AGENTS*N_HEADS
        for pos in range(n_pos):
            start=pos*VOCAB_SIZE
            block=msgs[:,start:start+VOCAB_SIZE].clone()
            new_block=torch.zeros_like(block)
            for s in range(VOCAB_SIZE):new_block[:,perm_map[pos][s]]=block[:,s]
            out[:,start:start+VOCAB_SIZE]=new_block
        return out

    for si,sname in enumerate(avail):
        for ri,rname in enumerate(avail):
            src_msgs=trained[sname]["msgs"]
            recv=trained[rname]["recv"]
            # Unaligned
            unaligned[si,ri]=eval_recv(src_msgs,recv)

            if si==ri:
                aligned[si,ri]=unaligned[si,ri]
                continue
            # Try all K! permutations per position — but K=3 means 6 perms
            # For speed: use Hungarian matching on tokens
            src_tokens=trained[sname]["tokens"]
            tgt_tokens=trained[rname]["tokens"]
            n_pos=src_tokens.shape[1]
            best_perm_map={}
            for pos in range(n_pos):
                cost=np.zeros((VOCAB_SIZE,VOCAB_SIZE))
                for s in range(VOCAB_SIZE):
                    for t in range(VOCAB_SIZE):
                        cost[s,t]=-np.sum((src_tokens[:,pos]==s)&(tgt_tokens[:,pos]==t))
                ri2,ci2=linear_sum_assignment(cost)
                pm=np.zeros(VOCAB_SIZE,dtype=int)
                for r,c in zip(ri2,ci2):pm[r]=c
                best_perm_map[pos]=pm
            aligned_msgs=permute_msgs(src_msgs,best_perm_map)
            aligned[si,ri]=eval_recv(aligned_msgs,recv)

    print(f"\n  Unaligned transfer:",flush=True)
    hdr=f"  {'':>8s}"+"".join(f"{m:>8s}" for m in avail);print(hdr,flush=True)
    for si,sn in enumerate(avail):
        row=f"  {sn:>8s}"+"".join(f"{unaligned[si,ri]:>7.1%}" for ri in range(n_mods));print(row,flush=True)
    print(f"\n  Aligned transfer:",flush=True)
    print(hdr,flush=True)
    for si,sn in enumerate(avail):
        row=f"  {sn:>8s}"+"".join(f"{aligned[si,ri]:>7.1%}" for ri in range(n_mods));print(row,flush=True)

    improvement=aligned-unaligned
    off_diag_imp=[improvement[i,j] for i in range(n_mods) for j in range(n_mods) if i!=j]
    print(f"\n  Mean improvement from alignment: {np.mean(off_diag_imp):+.1%}",flush=True)
    off_diag_aligned=[aligned[i,j] for i in range(n_mods) for j in range(n_mods) if i!=j]
    print(f"  Mean aligned off-diagonal: {np.mean(off_diag_aligned):.1%}",flush=True)

    results={"unaligned":unaligned.tolist(),"aligned":aligned.tolist(),
             "modalities":avail,"mean_improvement":float(np.mean(off_diag_imp)),
             "mean_aligned_offdiag":float(np.mean(off_diag_aligned))}

    # Plot
    import matplotlib;matplotlib.use("Agg");import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    fig.suptitle("Transfer: Before vs After Permutation Alignment",fontweight='bold')
    for ax,mat,title in [(axes[0],unaligned,"Unaligned"),(axes[1],aligned,"Permutation-Aligned")]:
        im=ax.imshow(mat,cmap='YlOrRd',vmin=0.4,vmax=1.0)
        ax.set_xticks(range(n_mods));ax.set_xticklabels(avail,rotation=30)
        ax.set_yticks(range(n_mods));ax.set_yticklabels(avail)
        ax.set_title(title)
        for i in range(n_mods):
            for j in range(n_mods):
                ax.text(j,i,f"{mat[i,j]:.0%}",ha='center',va='center',fontsize=10,fontweight='bold')
        plt.colorbar(im,ax=ax)
    plt.tight_layout();plt.savefig(d/"permutation_transfer.png",dpi=200,bbox_inches='tight');plt.close()

    with open(d/"results.json","w") as f:json.dump(results,f,indent=2,default=str)
    print(f"  Saved QW1 ({elapsed()})",flush=True)
    return results


# ═══════════════════════════════════════════════════════════════
# PRE-BOTTLENECK RSA
# ═══════════════════════════════════════════════════════════════

def pre_bottleneck_rsa():
    print(f"\n{'#'*60}\n# Pre-Bottleneck RSA\n# {elapsed()}\n{'#'*60}",flush=True)
    d=RB/"pre_bottleneck_rsa";d.mkdir(parents=True,exist_ok=True)
    p1,p2,obj,mass=load_meta();n=len(mass)
    mods=["vjepa2","dinov2","clip","text","audio"]

    # Compute RDMs BEFORE bottleneck
    raw_rdms={}
    for mod in mods:
        feat=load_feat(mod)
        if feat is None:continue
        # Pool to single vector per scene
        pooled=feat.mean(dim=1).numpy()  # [n, dim]
        # Cosine distance RDM
        from sklearn.metrics.pairwise import cosine_distances
        rdm=cosine_distances(pooled)
        raw_rdms[mod]=rdm
        print(f"  {mod}: raw RDM {rdm.shape}",flush=True)

    # Compute RDMs AFTER bottleneck
    disc_rdms={}
    for mod in mods:
        feat=load_feat(mod)
        if feat is None:continue
        _,_,tokens,_,_,_=train_d(feat,mass,obj,0)
        # Hamming distance RDM
        rdm=np.zeros((n,n))
        for i in range(n):
            for j in range(i+1,n):
                d_val=np.sum(tokens[i]!=tokens[j])
                rdm[i,j]=d_val;rdm[j,i]=d_val
        disc_rdms[mod]=rdm
        torch.mps.empty_cache()

    avail_mods=[m for m in mods if m in raw_rdms and m in disc_rdms]
    nm=len(avail_mods)

    # RSA matrices
    def rsa_matrix(rdms):
        mat=np.zeros((nm,nm))
        for i,mi in enumerate(avail_mods):
            for j,mj in enumerate(avail_mods):
                # Upper triangle of RDMs
                ui=rdms[mi][np.triu_indices(n,k=1)]
                uj=rdms[mj][np.triu_indices(n,k=1)]
                mat[i,j],_=scipy_stats.spearmanr(ui,uj)
        return mat

    raw_rsa=rsa_matrix(raw_rdms)
    disc_rsa=rsa_matrix(disc_rdms)

    print(f"\n  Raw RSA (before bottleneck):",flush=True)
    hdr=f"  {'':>8s}"+"".join(f"{m:>8s}" for m in avail_mods);print(hdr,flush=True)
    for i,m in enumerate(avail_mods):
        print(f"  {m:>8s}"+"".join(f"{raw_rsa[i,j]:>7.3f}" for j in range(nm)),flush=True)

    print(f"\n  Discrete RSA (after bottleneck):",flush=True)
    print(hdr,flush=True)
    for i,m in enumerate(avail_mods):
        print(f"  {m:>8s}"+"".join(f"{disc_rsa[i,j]:>7.3f}" for j in range(nm)),flush=True)

    # Compare
    raw_off=[raw_rsa[i,j] for i in range(nm) for j in range(nm) if i!=j]
    disc_off=[disc_rsa[i,j] for i in range(nm) for j in range(nm) if i!=j]
    print(f"\n  Mean off-diag RSA: raw={np.mean(raw_off):.3f} → discrete={np.mean(disc_off):.3f}",flush=True)

    if np.mean(disc_off) > np.mean(raw_off) + 0.05:
        verdict="WMCP AMPLIFIES pre-existing alignment"
    elif np.mean(raw_off) > 0.5:
        verdict="Encoders were ALREADY aligned, WMCP reveals it"
    else:
        verdict="WMCP CREATES alignment that wasn't there"
    print(f"  Verdict: {verdict}",flush=True)

    results={"raw_rsa":raw_rsa.tolist(),"disc_rsa":disc_rsa.tolist(),
             "modalities":avail_mods,"verdict":verdict,
             "raw_mean_offdiag":float(np.mean(raw_off)),
             "disc_mean_offdiag":float(np.mean(disc_off))}

    # Plot
    import matplotlib;matplotlib.use("Agg");import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    fig.suptitle("RSA: Before vs After Discrete Bottleneck",fontweight='bold')
    for ax,mat,title in [(axes[0],raw_rsa,"Raw Features (before WMCP)"),(axes[1],disc_rsa,"Discrete Codes (after WMCP)")]:
        im=ax.imshow(mat,cmap='RdYlBu_r',vmin=-0.2,vmax=1.0)
        ax.set_xticks(range(nm));ax.set_xticklabels(avail_mods,rotation=30)
        ax.set_yticks(range(nm));ax.set_yticklabels(avail_mods)
        ax.set_title(title)
        for i in range(nm):
            for j in range(nm):
                ax.text(j,i,f"{mat[i,j]:.2f}",ha='center',va='center',fontsize=9)
        plt.colorbar(im,ax=ax)
    plt.tight_layout();plt.savefig(d/"rsa_comparison.png",dpi=200,bbox_inches='tight');plt.close()

    with open(d/"results.json","w") as f:json.dump(results,f,indent=2,default=str)
    print(f"  Saved RSA ({elapsed()})",flush=True)
    return results


# ═══════════════════════════════════════════════════════════════
# TEMPORAL / PARAPHRASE CONSISTENCY
# ═══════════════════════════════════════════════════════════════

def temporal_paraphrase():
    print(f"\n{'#'*60}\n# Temporal/Paraphrase Consistency\n# {elapsed()}\n{'#'*60}",flush=True)
    d=RB/"temporal_paraphrase";d.mkdir(parents=True,exist_ok=True)
    p1,p2,obj,mass=load_meta();n=len(mass)

    # TEST A: Temporal consistency
    vfeat=load_feat("vjepa2")  # [206, 8, 1024]
    sender,recv,full_tokens,_,acc,_=train_d(vfeat,mass,obj,0)

    # Run sender on first-half and second-half frames
    fpa=4//N_AGENTS  # 1 frame per agent for half-clip
    if fpa<1:fpa=1

    half1_feat=vfeat[:,:4,:]  # frames 0-3
    half2_feat=vfeat[:,4:,:]  # frames 4-7
    views1=[half1_feat[:,(i*fpa)%4:(i*fpa)%4+fpa,:] for i in range(N_AGENTS)]
    views2=[half2_feat[:,(i*fpa)%4:(i*fpa)%4+fpa,:] for i in range(N_AGENTS)]

    t1=[];t2=[]
    with torch.no_grad():
        for i in range(0,n,BATCH_SIZE):
            v1=[vi[i:i+BATCH_SIZE].to(DEVICE) for vi in views1]
            v2=[vi[i:i+BATCH_SIZE].to(DEVICE) for vi in views2]
            _,l1=sender(v1);_,l2=sender(v2)
            t1.append(np.stack([l.argmax(-1).cpu().numpy() for l in l1],1))
            t2.append(np.stack([l.argmax(-1).cpu().numpy() for l in l2],1))
    t1=np.concatenate(t1,0);t2=np.concatenate(t2,0)

    temporal_agreement=np.mean([np.all(t1[i]==t2[i]) for i in range(n)])
    per_pos_agree=np.mean(t1==t2,axis=0)
    print(f"  Temporal: full-code agreement={temporal_agreement:.1%}",flush=True)
    print(f"  Per-position: {per_pos_agree}",flush=True)

    # TEST B: Paraphrase consistency
    tfeat=load_feat("text")
    t_sender,_,orig_tokens,_,_,_=train_d(tfeat,mass,obj,0)

    # Generate paraphrases
    mass_q=np.quantile(mass,[0.33,0.67])
    mat_map={'cardboard':('cardboard','box'),'rubber':('rubber','ball'),
             'metal':('metal','weight'),'wood':('wooden','block'),
             'plastic':('plastic','container'),'foam':('foam','cube')}

    templates=[
        "A {wt} {mat} {shape} on a spring, {m:.0f}g, {mat} material",
        "{mat} {shape} ({wt}), spring oscillation, mass {m:.0f}g",
        "Spring system: {wt} {shape} made of {mat}, weighing {m:.0f}g",
        "The {m:.0f}g {mat} {shape} bounces on a spring ({wt})",
        "{wt} {mat} {shape}. Mass: {m:.0f}g. Attached to spring.",
    ]

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok=AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    mdl=AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        torch_dtype=torch.float32,output_hidden_states=True)
    mdl.eval()
    if tok.pad_token is None:tok.pad_token=tok.eos_token
    tl=mdl.config.num_hidden_layers//2

    paraphrase_agrees=[]
    for si in range(min(50,n)):
        base=obj[si].split('_')[0].lower()
        mat,shape=mat_map.get(base,(base,'object'))
        wt="light" if mass[si]<mass_q[0] else "heavy" if mass[si]>mass_q[1] else "medium"
        orig_code=tuple(orig_tokens[si].tolist())

        matches=0
        for tmpl in templates:
            desc=tmpl.format(wt=wt,mat=mat,shape=shape,m=mass[si])
            inp=tok(desc,return_tensors="pt",padding=True,truncation=True,max_length=128)
            with torch.no_grad():
                out=mdl(**inp)
                feat_para=out.hidden_states[tl].mean(1).squeeze(0)
            feat_para=feat_para.unsqueeze(0).unsqueeze(0).expand(1,8,-1).contiguous()
            fpa_t=max(1,8//N_AGENTS)
            views_p=[feat_para[:,(i*fpa_t)%8:(i*fpa_t)%8+fpa_t,:] for i in range(N_AGENTS)]
            with torch.no_grad():
                _,logits=t_sender([v.to(DEVICE) for v in views_p])
                para_code=tuple(l.argmax(-1).cpu().item() for l in logits)
            if para_code==orig_code:matches+=1
        paraphrase_agrees.append(matches/len(templates))

    del mdl,tok;torch.mps.empty_cache()
    para_rate=np.mean(paraphrase_agrees)
    print(f"  Paraphrase agreement: {para_rate:.1%}",flush=True)

    results={
        "temporal_full_agreement":float(temporal_agreement),
        "temporal_per_position":per_pos_agree.tolist(),
        "paraphrase_agreement":float(para_rate),
        "n_paraphrase_scenes":len(paraphrase_agrees),
    }
    with open(d/"results.json","w") as f:json.dump(results,f,indent=2,default=str)
    print(f"  Saved temporal/paraphrase ({elapsed()})",flush=True)
    return results


# ═══════════════════════════════════════════════════════════════
# SUNSHOT A: Adversarial Impossibility
# ═══════════════════════════════════════════════════════════════

def sunshot_adversarial():
    print(f"\n{'#'*60}\n# SUNSHOT A: Adversarial Impossibility\n# {elapsed()}\n{'#'*60}",flush=True)
    d=RB/"sunshot_adversarial";d.mkdir(parents=True,exist_ok=True)
    p1,p2,obj,mass=load_meta();n=len(mass)

    # Train frozen canonical protocol (V-JEPA 2)
    vfeat=load_feat("vjepa2")
    canon_sender,_,canon_tokens,_,canon_acc,_=train_d(vfeat,mass,obj,0)
    canon_assigns=get_assigns(canon_tokens,p1,p2)

    # Get canonical soft logits for all scenes
    fpa=max(1,8//N_AGENTS)
    canon_views=[vfeat[:,(i*fpa)%8:(i*fpa)%8+fpa,:] for i in range(N_AGENTS)]
    with torch.no_grad():
        canon_logits_all=[]
        for i in range(0,n,BATCH_SIZE):
            v=[vi[i:i+BATCH_SIZE].to(DEVICE) for vi in canon_views]
            _,logits=canon_sender(v)
            canon_logits_all.append([l.cpu() for l in logits])
    # Flatten: list of [n_pos] lists of [batch, vocab] tensors
    canon_soft=[torch.cat([batch[p] for batch in canon_logits_all],0) for p in range(N_AGENTS*N_HEADS)]

    # Train adversarial DINOv2 senders
    dfeat=load_feat("dinov2")
    results={"lambdas":{}}

    for lam in [0.0,0.1,0.3,0.5,1.0]:
        print(f"\n  ── lambda={lam} ──",flush=True)
        seed_results=[]
        for seed in range(5):
            t0=time.time()
            dim=dfeat.shape[-1];nf=8;fpa2=max(1,nf//N_AGENTS)
            views=[dfeat[:,(i*fpa2)%nf:(i*fpa2)%nf+fpa2,:] for i in range(N_AGENTS)]
            torch.manual_seed(seed+500);np.random.seed(seed+500)
            ss=[DiscreteSender(TemporalEncoder(HIDDEN_DIM,dim,fpa2),HIDDEN_DIM,VOCAB_SIZE,N_HEADS) for _ in range(N_AGENTS)]
            sender=DiscreteMultiSender(ss).to(DEVICE)
            recvs=[Receiver(MSG_DIM,HIDDEN_DIM).to(DEVICE) for _ in range(3)]
            so=torch.optim.Adam(sender.parameters(),lr=1e-3)
            ros=[torch.optim.Adam(r.parameters(),lr=3e-3) for r in recvs]
            mass_dev=torch.tensor(mass,dtype=torch.float32).to(DEVICE)
            rng=np.random.RandomState(seed*1000+42)
            uo=sorted(set(obj));ho=set(rng.choice(uo,max(4,len(uo)//5),replace=False))
            tr=np.array([i for i,o in enumerate(obj) if o not in ho])
            tei=np.array([i for i,o in enumerate(obj) if o in ho])
            nb=max(1,len(tr)//32);me=math.log(VOCAB_SIZE);ba=0;bst=None;bep=0

            for ep in range(400):
                if ep-bep>150 and ba>0.55:break
                if ep>0 and ep%40==0:
                    for i in range(3):recvs[i]=Receiver(MSG_DIM,HIDDEN_DIM).to(DEVICE);ros[i]=torch.optim.Adam(recvs[i].parameters(),lr=3e-3)
                sender.train();[r.train() for r in recvs]
                tau=3+(1-3)*ep/399;hard=ep>=30
                for _ in range(nb):
                    ia=rng.choice(tr,32);ib=rng.choice(tr,32);s=ia==ib
                    while s.any():ib[s]=rng.choice(tr,s.sum());s=ia==ib
                    md=np.abs(mass[ia]-mass[ib]);k=md>0.5
                    if k.sum()<4:continue
                    ia,ib=ia[k],ib[k]
                    va=[v[ia].to(DEVICE) for v in views];vb=[v[ib].to(DEVICE) for v in views]
                    lab=(mass_dev[ia]>mass_dev[ib]).float()
                    ma,la=sender(va,tau,hard);mb,lb=sender(vb,tau,hard)
                    task_loss=sum(F.binary_cross_entropy_with_logits(r(ma,mb),lab) for r in recvs)/3

                    # Adversarial: reward disagreement with canon
                    if lam>0:
                        disagree=0
                        for pi,lg in enumerate(la):
                            canon_target=canon_soft[pi][ia].to(DEVICE)
                            disagree+=F.kl_div(F.log_softmax(lg,-1),F.softmax(canon_target,-1),reduction='batchmean')
                        loss=task_loss-lam*disagree
                    else:
                        loss=task_loss

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
                        for _ in range(20):
                            ia_h=er.choice(tei,min(32,len(tei)));ib_h=er.choice(tei,min(32,len(tei)))
                            vh=[v[ia_h].to(DEVICE) for v in views];wh=[v[ib_h].to(DEVICE) for v in views]
                            mah,_=sender(vh);mbh,_=sender(wh)
                            for r in recvs:c+=((r(mah,mbh)>0)==(mass_dev[ia_h]>mass_dev[ib_h])).sum().item();t+=len(ia_h)
                        acc=c/max(t,1)
                        if acc>ba:ba=acc;bep=ep;bst={kk:vv.cpu().clone() for kk,vv in sender.state_dict().items()}
            if bst:sender.load_state_dict(bst)
            sender.eval()
            toks=[]
            with torch.no_grad():
                for i in range(0,n,BATCH_SIZE):
                    v2=[vi[i:i+BATCH_SIZE].to(DEVICE) for vi in views]
                    _,logits=sender(v2);toks.append(np.stack([l.argmax(-1).cpu().numpy() for l in logits],1))
            adv_tokens=np.concatenate(toks,0)
            adv_assigns=get_assigns(adv_tokens,p1,p2)
            agree=sum(1 for a,b in zip(canon_assigns,adv_assigns) if a==b)/len(canon_assigns)
            attrs=np.stack([p1,p2],axis=1);pd,_=compute_posdis(adv_tokens,attrs,VOCAB_SIZE)
            seed_results.append({"acc":float(ba),"agreement":float(agree),"posdis":float(pd)})
            TIMING.append(time.time()-t0)
            print(f"    seed {seed}: acc={ba:.1%} agree={agree:.1%} PD={pd:.3f}",flush=True)
            torch.mps.empty_cache()

        accs=[r["acc"] for r in seed_results]
        agrees=[r["agreement"] for r in seed_results]
        results["lambdas"][str(lam)]={
            "acc":f"{np.mean(accs):.1%}±{np.std(accs):.1%}",
            "agreement":f"{np.mean(agrees):.1%}±{np.std(agrees):.1%}",
            "agree_m":float(np.mean(agrees)),
        }

    # Verdict
    lam0_agree=results["lambdas"]["0.0"]["agree_m"]
    lam1_agree=results["lambdas"]["1.0"]["agree_m"]
    if lam1_agree>0.8:
        verdict="IMPOSSIBILITY OF DISAGREEMENT — convergence is an attractor"
    elif lam1_agree<0.4:
        verdict="ALTERNATIVE PROTOCOL FOUND — convergence is not unique"
    else:
        verdict=f"PARTIAL — agreement drops from {lam0_agree:.0%} to {lam1_agree:.0%}"
    results["verdict"]=verdict
    print(f"\n  {verdict}",flush=True)

    with open(d/"results.json","w") as f:json.dump(results,f,indent=2,default=str)
    print(f"  Saved sunshot_adversarial ({elapsed()})",flush=True)
    return results


# ═══════════════════════════════════════════════════════════════
# SUNSHOT B: Iterated Protocol Chain
# ═══════════════════════════════════════════════════════════════

def sunshot_chain():
    print(f"\n{'#'*60}\n# SUNSHOT B: Iterated Protocol Chain\n# {elapsed()}\n{'#'*60}",flush=True)
    d=RB/"sunshot_chain";d.mkdir(parents=True,exist_ok=True)
    p1,p2,obj,mass=load_meta();n=len(mass)

    chain_order=["vjepa2","text","audio","dinov2","vjepa2"]
    results={"hops":[]}

    # Hop 0: Original V-JEPA protocol
    feat0=load_feat("vjepa2")
    s0,r0,tokens0,_,acc0,_=train_d(feat0,mass,obj,0)
    orig_assigns=get_assigns(tokens0,p1,p2)
    attrs=np.stack([p1,p2],axis=1);pd0,_=compute_posdis(tokens0,attrs,VOCAB_SIZE)
    results["hops"].append({"hop":0,"modality":"vjepa2","agreement":"100%",
                             "accuracy":f"{acc0:.1%}","posdis":f"{pd0:.3f}"})
    print(f"  Hop 0 (vjepa2): acc={acc0:.1%} PD={pd0:.3f}",flush=True)

    prev_assigns=orig_assigns
    for hop_i in range(1,len(chain_order)):
        mod=chain_order[hop_i]
        feat=load_feat(mod)
        if feat is None:
            print(f"  Hop {hop_i} ({mod}): skipped (no features)",flush=True)
            continue
        s,r,tokens,_,acc,_=train_d(feat,mass,obj,hop_i)
        assigns=get_assigns(tokens,p1,p2)
        agree_orig=sum(1 for a,b in zip(orig_assigns,assigns) if a==b)/len(orig_assigns)
        agree_prev=sum(1 for a,b in zip(prev_assigns,assigns) if a==b)/len(prev_assigns)
        pd,_=compute_posdis(tokens,attrs,VOCAB_SIZE)
        results["hops"].append({"hop":hop_i,"modality":mod,
                                 "agreement_with_original":f"{agree_orig:.1%}",
                                 "agreement_with_previous":f"{agree_prev:.1%}",
                                 "accuracy":f"{acc:.1%}","posdis":f"{pd:.3f}",
                                 "agree_orig_m":float(agree_orig)})
        prev_assigns=assigns
        print(f"  Hop {hop_i} ({mod}): acc={acc:.1%} PD={pd:.3f} agree_orig={agree_orig:.1%}",flush=True)
        torch.mps.empty_cache()

    # Check round-trip
    final_agree=results["hops"][-1].get("agree_orig_m",0) if len(results["hops"])>1 else 0
    if final_agree>0.85:
        verdict="FIXED-POINT ATTRACTOR — protocol survives full chain"
    else:
        verdict=f"DEGRADATION — agreement drops to {final_agree:.0%} after chain"
    results["verdict"]=verdict
    print(f"\n  {verdict}",flush=True)

    with open(d/"results.json","w") as f:json.dump(results,f,indent=2,default=str)
    print(f"  Saved sunshot_chain ({elapsed()})",flush=True)
    return results


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

if __name__=="__main__":
    print("╔══════════════════════════════════════════════════════════╗",flush=True)
    print("║  FINAL OVERNIGHT BATTERY                                 ║",flush=True)
    print(f"║  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}                          ║",flush=True)
    print("╚══════════════════════════════════════════════════════════╝",flush=True)

    experiments=[
        ("QW1: Permutation Transfer",qw1_permutation),
        ("Pre-Bottleneck RSA",pre_bottleneck_rsa),
        ("Temporal/Paraphrase",temporal_paraphrase),
        ("Sunshot A: Adversarial",sunshot_adversarial),
        ("Sunshot B: Chain",sunshot_chain),
    ]

    for name,func in experiments:
        try:
            print(f"\n{'='*60}\n  STARTING: {name} ({elapsed()})\n{'='*60}",flush=True)
            func()
            commit(name)
        except Exception as e:
            print(f"\n  {name} FAILED: {e}",flush=True)
            traceback.print_exc()

    total_h=(time.time()-START_TIME)/3600
    print(f"\n{'='*60}",flush=True)
    print(f"  OVERNIGHT BATTERY COMPLETE. Total: {total_h:.1f} hours",flush=True)
    print(f"{'='*60}",flush=True)
