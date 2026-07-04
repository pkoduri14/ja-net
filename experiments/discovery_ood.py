"""Does the discovery ranking hold on strictly-novel solutes?
Split held-out systems into solute-seen vs novel and recompute the synergy-ranking enrichment on each."""
import sys, collections, numpy as np, pandas as pd
sys.stdout.reconfigure(line_buffering=True)
from scipy.stats import spearmanr
import torch, janet as J
D = J.load_features()
fda = set(pd.read_csv("data/mixturesoldb.csv", usecols=["SMILES_Solute","FDA_Approved"])
          .pipe(lambda d: d.SMILES_Solute[d.FDA_Approved==True]))
by = collections.defaultdict(list)
for i in range(len(D.sysid)): by[D.sysid[i]].append(i)
targets={}
for s,idx in by.items():
    idx=np.array(idx)
    if D.soluteid[idx[0]] not in fda: continue
    modal=collections.Counter(np.round(D.T[idx],1)).most_common(1)[0][0]
    m=idx[np.round(D.T[idx],1)==modal]; xx=D.x1[m]
    interior=m[(xx>0.05)&(xx<0.95)]
    if xx.min()<=0.05 and xx.max()>=0.95 and len(interior)>=6: targets[s]=m[np.argsort(D.x1[m])]
target_sys=set(targets)
train_solutes=set(D.soluteid[[i for i in range(len(D.y)) if D.sysid[i] not in target_sys]])
nets=[]
for sd in (0,1,2):
    n=J.JANet(D.fdim,D.tdim); n.load_state_dict(torch.load(f"models/ja_ens_{sd}.pt")["state"]); n.eval(); nets.append(n)
t=torch.tensor
def C0(r):
    with torch.no_grad():
        return np.mean([nets[k](t(D.SOLn[[r]]),t(D.V1n[[r]]),t(D.V2n[[r]]),t(D.Tfn[[r]]),t(D.x1[[r]]),t(D.x2[[r]]))[3].numpy()[0,0] for k in range(3)])
rows=[]
for s,m in targets.items():
    e2,e1=D.y[m[0]],D.y[m[-1]]; interior=m[1:-1]
    ll=D.x1[interior]*e1+(1-D.x1[interior])*e2
    rows.append(dict(pred=float(C0(m[0])), meas=float((D.y[interior]-ll).max()),
                     novel=D.soluteid[m[0]] not in train_solutes))
df=pd.DataFrame(rows)
def report(name, d):
    if len(d)<20: print(f"  {name}: n={len(d)} (too few)"); return
    hi=d.meas>=df.meas.quantile(0.75); base=hi.mean()
    order=d.sort_values("pred",ascending=False)
    p10=order.iloc[:10].meas.ge(df.meas.quantile(0.75)).mean()
    sr=spearmanr(d.pred,d.meas).statistic
    print(f"  {name:16s} n={len(d):3d}  Spearman {sr:+.2f}  precision@10 {p10:.0%}  (base {base:.0%}, enrich {p10/base:.1f}x)")
print(f"total {len(df)} systems | novel-solute {int(df.novel.sum())}\n== discovery ranking: SEEN vs NOVEL solutes ==")
report("solute-SEEN", df[~df.novel]); report("solute-NOVEL", df[df.novel]); report("ALL", df)
