"""Chase the residual: what explains the synergy variance simple descriptors miss?
Test an intermediate-polarity bracketing feature (cross-validated R2) and whether the model learned it."""
import sys, collections, numpy as np, pandas as pd
sys.stdout.reconfigure(line_buffering=True)
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
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

_c = {}
def mol(s):
    if s not in _c: _c[s] = Chem.MolFromSmiles(s)
    return _c[s]
def lp(s): m = mol(s); return Crippen.MolLogP(m) if m else np.nan
def hbd(s): m = mol(s); return rdMolDescriptors.CalcNumHBD(m) if m else np.nan
def hba(s): m = mol(s); return rdMolDescriptors.CalcNumHBA(m) if m else np.nan

rows = []
for s, m in targets.items():
    e2, e1 = D.y[m[0]], D.y[m[-1]]; interior = m[1:-1]
    ll = D.x1[interior] * e1 + (1 - D.x1[interior]) * e2
    syn = float((D.y[interior] - ll).max())
    su, v1, v2 = s.split("|"); mm = mol(su)
    if mm is None or mol(v1) is None or mol(v2) is None: continue
    sL, s1L, s2L = Crippen.MolLogP(mm), lp(v1), lp(v2)
    lo, hi = min(s1L, s2L), max(s1L, s2L); gap = hi - lo
    pos = (sL - lo) / gap if gap > 1e-6 else 0.5          # solute position within the solvent-polarity bracket
    bracket = 1.0 if 0 <= pos <= 1 else 0.0               # is the solute polarity BETWEEN the two solvents?
    centrality = bracket * (1 - abs(2 * pos - 1))         # peaks (=1) when solute at the midpoint -> max synergy
    rows.append(dict(
        syn=syn, predC0=float(C0(m[0])),
        # simple set (the original ~0.27 story)
        solLogP=sL, solHBD=rdMolDescriptors.CalcNumHBD(mm), solHBA=rdMolDescriptors.CalcNumHBA(mm),
        solTPSA=rdMolDescriptors.CalcTPSA(mm), solMW=Descriptors.MolWt(mm),
        water=float(v1 == "O" or v2 == "O"), polMismatch=gap,
        # rich set (the residual hypothesis)
        bracket=bracket, centrality=centrality, centrality_x_gap=centrality * gap,
        hb_complement=abs(hbd(v1) - hbd(v2)) + abs(hba(v1) - hba(v2)),   # one donor-rich, one acceptor-rich
        pos_dist_mid=abs(2 * pos - 1)))
df = pd.DataFrame(rows).dropna().reset_index(drop=True)
print(f"{len(df)} held-out FDA systems\n")

SIMPLE = ["solLogP", "solHBD", "solHBA", "solTPSA", "solMW", "water", "polMismatch"]
RICH = SIMPLE + ["bracket", "centrality", "centrality_x_gap", "hb_complement", "pos_dist_mid"]

def cvr2(feats, target="syn", nonlin=True):
    X, y = df[feats].values, df[target].values
    est = HistGradientBoostingRegressor(max_iter=300, max_depth=3, learning_rate=0.05,
                                         l2_regularization=1.0, random_state=0) if nonlin else LinearRegression()
    return cross_val_score(est, X, y, cv=KFold(5, shuffle=True, random_state=0), scoring="r2").mean()

print("== cross-validated R2 explaining MEASURED synergy ==")
print(f"  simple  features, linear : {cvr2(SIMPLE, nonlin=False):+.3f}")
print(f"  simple  features, GBDT   : {cvr2(SIMPLE):+.3f}")
print(f"  RICH    features, linear : {cvr2(RICH, nonlin=False):+.3f}")
print(f"  RICH    features, GBDT   : {cvr2(RICH):+.3f}   <- did bracketing/centrality eat the residual?")

print("\n== the residual hypothesis, univariate (Spearman vs measured synergy) ==")
for f in ["polMismatch", "centrality", "centrality_x_gap", "bracket", "hb_complement", "pos_dist_mid"]:
    r, p = spearmanr(df[f], df.syn); print(f"   {f:16s} rho={r:+.3f} (p={p:.1e})")

# does the model see the sharper (centrality) rule, beyond raw polMismatch?
print("\n== did JA-NET learn the sharper rule? (predicted C0 vs the residual features) ==")
for f in ["polMismatch", "centrality", "centrality_x_gap"]:
    r, _ = spearmanr(df[f], df.predC0); print(f"   C0 vs {f:16s} rho={r:+.3f}")
# partial signal: does centrality predict synergy AFTER removing polMismatch?
from numpy.polynomial import polynomial as _p
res_syn = df.syn - np.poly1d(np.polyfit(df.polMismatch, df.syn, 1))(df.polMismatch)
res_cen = df.centrality - np.poly1d(np.polyfit(df.polMismatch, df.centrality, 1))(df.polMismatch)
rp, pp = spearmanr(res_cen, res_syn)
print(f"\n   centrality vs synergy, AFTER partialling out polMismatch: rho={rp:+.3f} (p={pp:.1e})")

