"""Model diagnostics (reuses the leave-all-FDA-out ensemble -> all predictions here are out-of-fold).
  error analysis:      residuals bucketed by solvent class / solute size / temperature / composition
  applicability domain: does error rise with distance (max-Tanimoto) from the training set?
  interpretability CI:  bootstrap 95% CI on the structure->JA coefficient recovery, at two R^2 thresholds
  physical validity:    fraction of predictions violating logS(mole-fraction) <= 0 (x_s > 1 is impossible)
No training. Run datacost_vs_classical.py first (needs models/ja_ens_{0,1,2}.pt).
"""
import sys, collections, numpy as np, pandas as pd
sys.stdout.reconfigure(line_buffering=True)
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
from scipy.stats import pearsonr
import torch
import janet as J

D = J.load_features()
fda = set(pd.read_csv("data/mixturesoldb.csv", usecols=["SMILES_Solute", "FDA_Approved"])
          .pipe(lambda d: d.SMILES_Solute[d.FDA_Approved == True]))

# reconstruct the exact held-out target systems (all their rows are OOF)
by = collections.defaultdict(list)
for i in range(len(D.sysid)): by[D.sysid[i]].append(i)
target_sys = set()
for s, idx in by.items():
    idx = np.array(idx)
    if D.soluteid[idx[0]] not in fda: continue
    modal = collections.Counter(np.round(D.T[idx], 1)).most_common(1)[0][0]
    m = idx[np.round(D.T[idx], 1) == modal]; xx = D.x1[m]
    if len(m) >= 10 and xx.max() - xx.min() >= 0.6 and len(np.unique(np.round(xx, 2))) >= 6:
        target_sys.add(s)
oof = np.array([i for i in range(len(D.y)) if D.sysid[i] in target_sys])
train_solutes = set(D.soluteid[[i for i in range(len(D.y)) if D.sysid[i] not in target_sys]])
print(f"OOF rows {len(oof)} across {len(target_sys)} held-out FDA systems | train solutes {len(train_solutes)}")

nets = []
for sd in (0, 1, 2):
    net = J.JANet(D.fdim, D.tdim); net.load_state_dict(torch.load(f"models/ja_ens_{sd}.pt")["state"]); net.eval()
    nets.append(net)
t = torch.tensor
def predict(rows):
    with torch.no_grad():
        return np.mean([nets[k](t(D.SOLn[rows]), t(D.V1n[rows]), t(D.V2n[rows]), t(D.Tfn[rows]),
                                t(D.x1[rows]), t(D.x2[rows]))[0].numpy() for k in range(3)], 0)

pred = predict(oof); y = D.y[oof]; err = np.abs(pred - y)
print(f"\noverall OOF MAE {err.mean():.3f}")

# ---------- error analysis by class ----------
print("\n== error by class (MAE) ==")
sol_smi = np.array([s.split("|")[0] for s in D.sysid[oof]])
v1 = np.array([s.split("|")[1] for s in D.sysid[oof]]); v2 = np.array([s.split("|")[2] for s in D.sysid[oof]])
aqueous = (v1 == "O") | (v2 == "O")
def bucket(name, mask):
    if mask.sum() == 0: return
    print(f"   {name:28s} n={mask.sum():6d}  MAE {err[mask].mean():.3f}")
bucket("aqueous (water present)", aqueous); bucket("non-aqueous", ~aqueous)
# solute size
_mw = {}
def mw(s):
    if s not in _mw:
        m = Chem.MolFromSmiles(s); _mw[s] = Descriptors.MolWt(m) if m else np.nan
    return _mw[s]
MW = np.array([mw(s) for s in sol_smi])
for lo, hi in [(0, 250), (250, 400), (400, 1e9)]:
    bucket(f"solute MW [{lo},{hi})", (MW >= lo) & (MW < hi))
for lo, hi in [(200, 288), (288, 310), (310, 450)]:
    bucket(f"T [{lo},{hi}) K", (D.T[oof] >= lo) & (D.T[oof] < hi))
bucket("endpoint (x<=.05 or >=.95)", (D.x1[oof] <= 0.05) | (D.x1[oof] >= 0.95))
bucket("interior", (D.x1[oof] > 0.05) & (D.x1[oof] < 0.95))

# ---------- applicability domain: error vs max-Tanimoto to training solutes ----------
print("\n== applicability domain (error vs solute similarity to train) ==")
def ecfp(s, _c={}):
    if s not in _c:
        m = Chem.MolFromSmiles(s); _c[s] = rdMolDescriptors.GetMorganFingerprintAsBitVect(m, 2, 2048) if m else None
    return _c[s]
tr_fps = [ecfp(s) for s in train_solutes if ecfp(s)]
sim = {}
for s in set(sol_smi):
    f = ecfp(s); sim[s] = max(DataStructs.BulkTanimotoSimilarity(f, tr_fps)) if f else 0.0
S = np.array([sim[s] for s in sol_smi])
for lo, hi in [(0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.01)]:
    m = (S >= lo) & (S < hi)
    if m.sum(): print(f"   max-Tanimoto to train [{lo:.1f},{hi:.1f})  n={m.sum():6d}  MAE {err[m].mean():.3f}")


# ---------- interpretability CI (bootstrap) ----------
print("\n== coefficient recovery with bootstrap 95% CI ==")
def structC(row):
    with torch.no_grad():
        return np.mean([nets[k](t(D.SOLn[[row]]), t(D.V1n[[row]]), t(D.V2n[[row]]), t(D.Tfn[[row]]),
                                t(D.x1[[row]]), t(D.x2[[row]]))[3].numpy()[0] for k in range(3)], 0)
def recover(r2min):
    Jf, Cp = [], []
    for s in target_sys:
        m = np.array(sorted([i for i in by[s]], key=lambda i: D.x1[i]))
        x1, x2 = D.x1[m], D.x2[m]
        if len(m) < 10 or x1.max() - x1.min() < 0.5 or len(np.unique(np.round(x1, 2))) < 5: continue
        d = x1 - x2; A = np.stack([x1, x2, x1 * x2, x1 * x2 * d, x1 * x2 * d * d], 1)
        coef, *_ = np.linalg.lstsq(A, D.y[m], rcond=None)
        res = D.y[m] - A @ coef; r2 = 1 - (res ** 2).sum() / (((D.y[m] - D.y[m].mean()) ** 2).sum() + 1e-9)
        if r2 < r2min or abs(coef[2]) > 8: continue
        Jf.append(coef[2:5]); Cp.append(structC(m[0]))
    return np.array(Jf), np.array(Cp)
rng = np.random.RandomState(0)
for r2min in (0.9, 0.7):
    Jf, Cp = recover(r2min)
    if len(Jf) < 5: continue
    r0 = pearsonr(Jf[:, 0], Cp[:, 0])[0]
    boot = [pearsonr(Jf[ix, 0], Cp[ix, 0])[0] for ix in (rng.randint(0, len(Jf), len(Jf)) for _ in range(2000))]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    sign = np.mean(np.sign(Jf[:, 0]) == np.sign(Cp[:, 0]))
    print(f"   R2>{r2min}: n={len(Jf):4d}  C0 r={r0:+.2f} [95%CI {lo:+.2f},{hi:+.2f}]  sign-agreement {sign:.0%}")

# ---------- physical validity ----------
print("\n== physical validity: logS(mole fraction) must be <= 0 (x_s <= 1) ==")
bad_pred = np.mean(pred > 1e-6); bad_data = np.mean(y > 1e-6)
print(f"   predictions with logS>0 (x_s>1, impossible): {bad_pred:.2%}   (measured labels >0: {bad_data:.2%})")
print(f"   worst positive prediction: {pred.max():+.2f} logS")
