"""Mechanism: what molecular features drive cosolvency synergy, and did the model learn them?
Regress measured synergy and predicted C0 on physical descriptors (solvent polarity mismatch, water, H-bonding)."""
import sys, collections, numpy as np, pandas as pd
sys.stdout.reconfigure(line_buffering=True)
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
from scipy.stats import spearmanr
import torch, janet as J

D = J.load_features()
fda = set(pd.read_csv("data/mixturesoldb.csv", usecols=["SMILES_Solute", "FDA_Approved"])
          .pipe(lambda d: d.SMILES_Solute[d.FDA_Approved == True]))
by = collections.defaultdict(list)
for i in range(len(D.sysid)): by[D.sysid[i]].append(i)
targets = {}
for s, idx in by.items():
    idx = np.array(idx)
    if D.soluteid[idx[0]] not in fda: continue
    modal = collections.Counter(np.round(D.T[idx], 1)).most_common(1)[0][0]
    m = idx[np.round(D.T[idx], 1) == modal]; xx = D.x1[m]
    interior = m[(xx > 0.05) & (xx < 0.95)]
    if xx.min() <= 0.05 and xx.max() >= 0.95 and len(interior) >= 6:
        targets[s] = m[np.argsort(D.x1[m])]

nets = []
for sd in (0, 1, 2):
    n = J.JANet(D.fdim, D.tdim); n.load_state_dict(torch.load(f"models/ja_ens_{sd}.pt")["state"]); n.eval(); nets.append(n)
t = torch.tensor
def C0(row):
    with torch.no_grad():
        return np.mean([nets[k](t(D.SOLn[[row]]), t(D.V1n[[row]]), t(D.V2n[[row]]), t(D.Tfn[[row]]),
                                t(D.x1[[row]]), t(D.x2[[row]]))[3].numpy()[0, 0] for k in range(3)])

_dc = {}
def desc(smi):
    if smi in _dc: return _dc[smi]
    m = Chem.MolFromSmiles(smi); _dc[smi] = m; return m
def logp(smi):
    m = desc(smi); return Crippen.MolLogP(m) if m else np.nan

rows = []
for s, m in targets.items():
    e2, e1 = D.y[m[0]], D.y[m[-1]]; interior = m[1:-1]
    ll = D.x1[interior] * e1 + (1 - D.x1[interior]) * e2
    syn = float((D.y[interior] - ll).max())                 # measured synergy height
    su, v1, v2 = s.split("|"); mm = desc(su)
    if mm is None: continue
    rows.append(dict(
        syn=syn, predC0=float(C0(m[0])),
        solLogP=Crippen.MolLogP(mm), solHBD=rdMolDescriptors.CalcNumHBD(mm),
        solHBA=rdMolDescriptors.CalcNumHBA(mm), solTPSA=rdMolDescriptors.CalcTPSA(mm),
        solMW=Descriptors.MolWt(mm), solArom=rdMolDescriptors.CalcNumAromaticRings(mm),
        water=float(v1 == "O" or v2 == "O"), polMismatch=abs(logp(v1) - logp(v2))))
df = pd.DataFrame(rows).dropna().reset_index(drop=True)
print(f"{len(df)} held-out FDA systems with measured synergy\n")

FEATS = ["solLogP", "solHBD", "solHBA", "solTPSA", "solMW", "solArom", "water", "polMismatch"]
print("== univariate drivers of MEASURED synergy (Spearman) ==")
uni = []
for f in FEATS:
    r, p = spearmanr(df[f], df.syn); uni.append((f, r, p))
for f, r, p in sorted(uni, key=lambda z: -abs(z[1])):
    print(f"   {f:12s} rho={r:+.3f}  (p={p:.1e})")

def stdreg(target):                                          # standardized linear model -> which drivers dominate
    X = (df[FEATS] - df[FEATS].mean()) / (df[FEATS].std() + 1e-9)
    yv = (df[target] - df[target].mean()) / df[target].std()
    A = np.hstack([X.values, np.ones((len(X), 1))])
    coef, *_ = np.linalg.lstsq(A, yv.values, rcond=None)
    r2 = 1 - ((A @ coef - yv.values) ** 2).sum() / ((yv.values - yv.values.mean()) ** 2).sum()
    return dict(zip(FEATS, coef[:-1])), r2

bm, r2m = stdreg("syn"); bp, r2p = stdreg("predC0")
print(f"\n== standardized regression: measured synergy (R2={r2m:.2f}) vs model C0 (R2={r2p:.2f}) ==")
print(f"   {'feature':12s} {'->measured':>11s} {'->model':>9s}   agree?")
for f in sorted(FEATS, key=lambda z: -abs(bm[z])):
    ag = "YES" if np.sign(bm[f]) == np.sign(bp[f]) and abs(bm[f]) > 0.05 else ""
    print(f"   {f:12s} {bm[f]:>+11.2f} {bp[f]:>+9.2f}   {ag}")
agree = np.mean([np.sign(bm[f]) == np.sign(bp[f]) for f in FEATS if abs(bm[f]) > 0.05])
print(f"\n   model recovers the sign of {agree:.0%} of the meaningful measured-synergy drivers.")

