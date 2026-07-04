"""Apples-to-apples vs the published mixed-solvent SOTA (Bao 2024 / Amiri 2025).
Their exact 27k dataset, units, and splits; JA-NET vs concat/GBDT on random / solvent-disjoint / system-disjoint."""
import sys, numpy as np, pandas as pd
sys.stdout.reconfigure(line_buffering=True)
from types import SimpleNamespace
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
import janet as J

# load + join (drug/solvent -> SMILES)
xl = "data/bao27k.xlsx"
sol = pd.read_excel(xl, sheet_name="Solubility").rename(columns={
    "Solubility (mol/mol)": "xs", "Temperature (K)": "TK", "Solvent_1_weight_fraction": "w1",
    "Solvent_1_mol_fraction": "f1in", "Solvent_1": "S1", "Solvent_2": "S2"})
drugs = pd.read_excel(xl, sheet_name="Drugs"); solv = pd.read_excel(xl, sheet_name="Solvents")
d2s = dict(zip(drugs.Drug, drugs.SMILES)); s2s = dict(zip(solv.Solvent, solv.SMILES))

def mw(smi, _c={}):
    if smi in _c: return _c[smi]
    m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    _c[smi] = Descriptors.MolWt(m) if m else None; return _c[smi]

# convert solute mole fraction -> logS (g solute / 100 g solvent); build features
SOL, V1, V2, x1, T, y, drug, pair, keep = [], [], [], [], [], [], [], [], 0
for r in sol.itertuples():
    su, v1, v2 = d2s.get(r.Drug), s2s.get(r.S1), s2s.get(r.S2)
    xs, w1, f1in, TK = r.xs, r.w1, r.f1in, r.TK
    if not (isinstance(su, str) and isinstance(v1, str) and isinstance(v2, str)): continue
    if not (np.isfinite(xs) and 0 < xs < 1 and np.isfinite(TK)): continue
    Ms, M1, M2 = mw(su), mw(v1), mw(v2)
    if not (Ms and M1 and M2): continue
    # solvent-1 mole fraction from whichever composition column is present
    if np.isfinite(f1in) and 0 <= f1in <= 1:
        f1 = f1in
    elif np.isfinite(w1) and 0 <= w1 <= 1:
        fa, fb = w1 / M1, (1 - w1) / M2; f1 = fa / (fa + fb)
    else:
        continue
    Msolv = f1 * M1 + (1 - f1) * M2                   # solvent avg molar mass
    S = 100 * xs * Ms / ((1 - xs) * Msolv)            # g solute / 100 g solvent
    if not (S > 0 and np.isfinite(S)): continue
    a, b, c = J._feat(su), J._feat(v1), J._feat(v2)
    if a is None or b is None or c is None: continue
    SOL.append(a); V1.append(b); V2.append(c); x1.append(f1); T.append(TK)
    y.append(np.log10(S)); drug.append(r.Drug); pair.append("|".join(sorted([str(r.S1), str(r.S2)]))); keep += 1

SOL = np.array(SOL); V1 = np.array(V1); V2 = np.array(V2)
for A in (SOL, V1, V2): A[~np.isfinite(A)] = 0
x1 = np.array(x1, np.float32); x2 = (1 - x1).astype(np.float32); T = np.array(T, np.float32); y = np.array(y, np.float32)
drug = np.array(drug, object); pair = np.array(pair, object)
print(f"{keep} rows joined | {len(set(drug))} drugs | {len(set(pair))} binary-solvent combos | logS range [{y.min():.2f},{y.max():.2f}]")

# standardize -> D bundle reused by janet.evaluate / train_*
sm, ss = SOL.mean(0), SOL.std(0) + 1e-6
vm = np.vstack([V1, V2]).mean(0); vs = np.vstack([V1, V2]).std(0) + 1e-6
Tf = np.stack([T, 1000.0 / T, np.log(T)], 1); Tm, Ts = Tf.mean(0), Tf.std(0) + 1e-6
D = SimpleNamespace(y=y, x1=x1, x2=x2,
    SOLn=((SOL - sm) / ss).astype(np.float32), V1n=((V1 - vm) / vs).astype(np.float32),
    V2n=((V2 - vm) / vs).astype(np.float32), Tfn=((Tf - Tm) / Ts).astype(np.float32))
D.Xcat = np.hstack([D.SOLn, D.V1n, D.V2n, x1[:, None], x2[:, None], D.Tfn]).astype(np.float32)
D.fdim, D.tdim = SOL.shape[1], D.Tfn.shape[1]

# random (sanity vs 0.28-0.33), solvent-disjoint (vs GCN 0.41), system-disjoint (vs Bao 0.33)
J.evaluate(D, "RANDOM (sanity vs published 0.28-0.33)", np.arange(len(y)))
J.evaluate(D, "SOLVENT-DISJOINT (vs published GCN 0.41)", pair)
J.evaluate(D, "SYSTEM-DISJOINT (vs Bao LightGBM 0.33)", np.array([f"{d}|{p}" for d, p in zip(drug, pair)], object))
