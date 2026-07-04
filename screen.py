"""Screen your own solute-solvent systems for predicted cosolvency synergy.

Usage:
    python screen.py candidates.csv

Input CSV columns:
    solute_smiles, solvent1_smiles, solvent2_smiles   (required)
    name, temperature_K                               (optional; temperature defaults to 298.15 K)

Writes <input>_predictions.csv, ranked by predicted synergy (the solubility maximum above additive mixing),
with the recovered Jouyban-Acree coefficients, an ensemble-uncertainty estimate, and the solvent polarity gap.
Runs from the saved ensemble (models/ja_ens_*.pt) and normalization stats (models/norm.npz), so no training data
is needed. Prediction is for the cosolvency curve SHAPE (synergy); absolute solubility is not predicted here.
"""
import sys, os, numpy as np, pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
import torch
from janet import _feat, JANet

z = np.load("models/norm.npz")
sol_mean, sol_std = z["sol_mean"], z["sol_std"]
slv_mean, slv_std = z["slv_mean"], z["slv_std"]
t_mean, t_std = z["t_mean"], z["t_std"]
fdim, tdim = int(z["fdim"]), int(z["tdim"])

nets = []
for s in (0, 1, 2):
    net = JANet(fdim, tdim)
    net.load_state_dict(torch.load(f"models/ja_ens_{s}.pt")["state"])
    net.eval()
    nets.append(net)

xs = np.linspace(0, 1, 101).astype(np.float32)      # composition grid
d = xs - (1 - xs)                                    # d = x1 - x2


def standardize(smi, mean, std):
    f = _feat(smi)
    return None if f is None else ((np.nan_to_num(f) - mean) / std).astype(np.float32)


def predict(solute, solvent1, solvent2, T):
    """Structure-predicted Jouyban-Acree coefficients + the synergy (peak bulge above additive mixing)."""
    zs = standardize(solute, sol_mean, sol_std)
    v1 = standardize(solvent1, slv_mean, slv_std)
    v2 = standardize(solvent2, slv_mean, slv_std)
    if zs is None or v1 is None or v2 is None:
        return None
    tf = (np.array([T, 1000 / T, np.log(T)], np.float32) - t_mean) / t_std
    t = torch.tensor
    coeffs = []
    with torch.no_grad():
        for net in nets:
            _, _, _, C = net(t(zs[None]), t(v1[None]), t(v2[None]), t(tf[None]), t([0.5]), t([0.5]))
            coeffs.append(C.numpy()[0])              # (C0, C1, C2) — independent of composition
    coeffs = np.array(coeffs)
    peak_per_model = [np.max(xs * (1 - xs) * (c[0] + c[1] * d + c[2] * d * d)) for c in coeffs]
    curve = xs * (1 - xs) * (coeffs[:, 0].mean() + coeffs[:, 1].mean() * d + coeffs[:, 2].mean() * d * d)
    return dict(C0=coeffs[:, 0].mean(), C1=coeffs[:, 1].mean(), C2=coeffs[:, 2].mean(),
                peak_synergy=float(curve.max()), peak_x1=float(xs[curve.argmax()]),
                uncertainty=float(np.std(peak_per_model)))


def logp(smi):
    m = Chem.MolFromSmiles(smi)
    return Crippen.MolLogP(m) if m else np.nan


def is_water(smi):
    m = Chem.MolFromSmiles(smi)
    return m is not None and Chem.MolToSmiles(m) == "O"


inp = sys.argv[1] if len(sys.argv) > 1 else "example_candidates.csv"
df = pd.read_csv(inp)
if "temperature_K" not in df.columns:
    df["temperature_K"] = 298.15
if "name" not in df.columns:
    df["name"] = ""

rows = []
for r in df.itertuples():
    p = predict(r.solute_smiles, r.solvent1_smiles, r.solvent2_smiles, float(r.temperature_K))
    if p is None:
        print(f"skipped (unparseable SMILES): {r.name or r.solute_smiles}")
        continue
    rows.append(dict(
        name=r.name, solute_smiles=r.solute_smiles, solvent1=r.solvent1_smiles, solvent2=r.solvent2_smiles,
        temperature_K=r.temperature_K, pred_peak_synergy=round(p["peak_synergy"], 3), peak_x1=round(p["peak_x1"], 2),
        C0=round(p["C0"], 3), C1=round(p["C1"], 3), C2=round(p["C2"], 3), uncertainty=round(p["uncertainty"], 3),
        polarity_gap=round(abs(logp(r.solvent1_smiles) - logp(r.solvent2_smiles)), 2),
        water=("yes" if is_water(r.solvent1_smiles) or is_water(r.solvent2_smiles) else "")))

out = pd.DataFrame(rows).sort_values("pred_peak_synergy", ascending=False).reset_index(drop=True)
outpath = os.path.splitext(inp)[0] + "_predictions.csv"
out.to_csv(outpath, index=False)
print(f"\n{len(out)} systems scored -> {outpath}\n")
print(out[["name", "pred_peak_synergy", "peak_x1", "C0", "uncertainty", "polarity_gap", "water"]].to_string(index=False))
