"""Featurization ablation: does 128-bit vs 2048-bit Morgan matter?
Rebuild features at each bit size, same leave-system-out protocol, 3 repeated splits."""
import sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)
from types import SimpleNamespace
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import janet as J

DESC = [f for _, f in Descriptors._descList]

def feat(smi, nbits, _c={}):
    k = (smi, nbits)
    if k in _c: return _c[k]
    m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    if m is None: _c[k] = None; return None
    d = np.array([f(m) for f in DESC], float)
    fp = np.zeros(nbits, np.int8)
    DataStructs.ConvertToNumpyArray(rdMolDescriptors.GetMorganFingerprintAsBitVect(m, 2, nBits=nbits), fp)
    _c[k] = np.concatenate([d, fp]).astype(np.float32); return _c[k]


def build_D(nbits, csv="data/mixturesoldb.csv"):
    cols = ["SMILES_Solute", "SMILES_Solvent1", "SMILES_Solvent2", "Fraction_Solvent1",
            "Fraction_Solvent2", "Temperature_K", "LogS(mole_fraction)"]
    d = pd.read_csv(csv)[cols].dropna()
    d = d.rename(columns={"LogS(mole_fraction)": "y", "Temperature_K": "TK",
                          "Fraction_Solvent1": "x1", "Fraction_Solvent2": "x2"})
    d = d[np.isfinite(d.y) & (d.TK > 200) & (d.TK < 450)].reset_index(drop=True)
    SOL, V1, V2, keep = [], [], [], []
    for r in d.itertuples():
        a, b, c = feat(r.SMILES_Solute, nbits), feat(r.SMILES_Solvent1, nbits), feat(r.SMILES_Solvent2, nbits)
        if a is None or b is None or c is None: keep.append(False); continue
        keep.append(True); SOL.append(a); V1.append(b); V2.append(c)
    d = d[keep].reset_index(drop=True)
    SOL, V1, V2 = np.array(SOL), np.array(V1), np.array(V2)
    for A in (SOL, V1, V2): A[~np.isfinite(A)] = 0
    D = SimpleNamespace(y=d.y.to_numpy(np.float32), x1=d.x1.to_numpy(np.float32), x2=d.x2.to_numpy(np.float32),
        sysid=(d.SMILES_Solute + "|" + d.SMILES_Solvent1 + "|" + d.SMILES_Solvent2).values.astype(str))
    sm, ss = SOL.mean(0), SOL.std(0) + 1e-6
    vm = np.vstack([V1, V2]).mean(0); vs = np.vstack([V1, V2]).std(0) + 1e-6
    Tf = np.stack([d.TK, 1000.0 / d.TK, np.log(d.TK)], 1); Tm, Ts = Tf.mean(0), Tf.std(0) + 1e-6
    D.SOLn = ((SOL - sm) / ss).astype(np.float32)
    D.V1n = ((V1 - vm) / vs).astype(np.float32); D.V2n = ((V2 - vm) / vs).astype(np.float32)
    D.Tfn = ((Tf - Tm) / Ts).astype(np.float32)
    D.fdim, D.tdim = SOL.shape[1], 3
    return D


rmse = lambda p, y: np.sqrt(mean_squared_error(y, p))
print("=== featurization ablation: JA-NET 128-bit vs 2048-bit Morgan (leave-system-out, 3 splits) ===")
res = {128: [], 2048: []}
for nb in (128, 2048):
    print(f"\nbuilding {nb}-bit features ...")
    D = build_D(nb)
    print(f"  {len(D.y)} rows, {len(set(D.sysid))} systems, fdim={D.fdim}")
    for rs in range(3):
        tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=rs).split(np.arange(len(D.y)), groups=D.sysid))
        pj = np.mean([J.train_ja(D, tr, te, s, vgroups=D.sysid) for s in (0, 1, 2)], 0)
        m, r = mean_absolute_error(D.y[te], pj), rmse(pj, D.y[te])
        res[nb].append((m, r)); print(f"  {nb}-bit rs={rs}: MAE {m:.3f} RMSE {r:.3f}")

print("\n=== RESULT (mean +/- std across 3 splits) ===")
for nb in (128, 2048):
    v = np.array(res[nb])
    print(f"  {nb:>4}-bit Morgan: MAE {v[:,0].mean():.3f} +/- {v[:,0].std():.3f}   RMSE {v[:,1].mean():.3f} +/- {v[:,1].std():.3f}")
d128, d2048 = np.array(res[128])[:, 0].mean(), np.array(res[2048])[:, 0].mean()
print(f"  2048-bit vs 128-bit Morgan: MAE change {d2048 - d128:+.3f} ({(d2048-d128)/d128*100:+.1f}%)")
