"""Prospective target list: candidate systems to measure (structure-predicted; none appear in the training data).
Score unmeasured FDA-drug x solvent-pair systems, rank by predicted synergy gated on confidence + physical rationale."""
import sys, collections, numpy as np, pandas as pd
sys.stdout.reconfigure(line_buffering=True)
from rdkit import Chem
from rdkit.Chem import Crippen
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
import torch, janet as J

D = J.load_features()
sm, ss, vm, vs, Tm, Ts = D.norm
meta = pd.read_csv("data/mixturesoldb.csv", usecols=["SMILES_Solute", "Compound_Name", "FDA_Approved"])
name = dict(zip(meta.SMILES_Solute, meta.Compound_Name))
fda_solutes = sorted(set(meta.SMILES_Solute[meta.FDA_Approved == True]))

def canon(s):
    m = Chem.MolFromSmiles(s); return Chem.MolToSmiles(m) if m else None
# existing (solute, unordered solvent-pair) systems, canonicalized -> exclude these
existing = set()
for s in set(D.sysid):
    a, b, c = s.split("|"); ca, cb, cc = canon(a), canon(b), canon(c)
    if ca and cb and cc: existing.add((ca, frozenset([cb, cc])))

# pharma cosolvents (name -> SMILES)
SOLV = {"water": "O", "ethanol": "CCO", "methanol": "CO", "isopropanol": "CC(C)O", "n-propanol": "CCCO",
        "propylene glycol": "CC(O)CO", "glycerol": "OCC(O)CO", "PEG(tetra)": "OCCOCCOCCOCCO",
        "acetone": "CC(C)=O", "acetonitrile": "CC#N", "DMSO": "CS(C)=O", "DMF": "CN(C)C=O",
        "1,4-dioxane": "C1COCCO1", "ethyl acetate": "CCOC(C)=O", "THF": "C1CCOC1", "NMP": "CN1CCCC1=O"}
slogp = {k: Crippen.MolLogP(Chem.MolFromSmiles(v)) for k, v in SOLV.items()}
# candidate solvent pairs: water + each organic, plus a few organic-organic blends
pairs = [("water", o) for o in SOLV if o != "water"]
pairs += [("ethanol", "ethyl acetate"), ("ethanol", "acetone"), ("methanol", "acetonitrile"),
          ("ethanol", "1,4-dioxane"), ("acetone", "ethyl acetate"), ("DMSO", "acetonitrile")]

nets = []
for sd in (0, 1, 2):
    n = J.JANet(D.fdim, D.tdim); n.load_state_dict(torch.load(f"models/ja_ens_{sd}.pt")["state"]); n.eval(); nets.append(n)
t = torch.tensor
Tf = ((np.array([298.15, 1000 / 298.15, np.log(298.15)], np.float32) - Tm) / Ts)[None]

def fz(smi, mean, std):
    f = J._feat(smi)
    if f is None: return None
    return ((np.nan_to_num(f) - mean) / std).astype(np.float32)[None]

rows = []
for su in fda_solutes:
    zs = fz(su, sm, ss)
    if zs is None: continue
    for a, b in pairs:
        va, vb = fz(SOLV[a], vm, vs), fz(SOLV[b], vm, vs)
        key = (canon(su), frozenset([canon(SOLV[a]), canon(SOLV[b])]))
        if key in existing or va is None or vb is None: continue
        with torch.no_grad():
            Cs = np.array([nets[k](t(zs), t(va), t(vb), t(Tf), t([0.5]), t([0.5]))[3].numpy()[0] for k in range(3)])
        c0 = Cs[:, 0]
        rows.append(dict(drug=name.get(su, "?"), solA=a, solB=b, synergy=float(c0.mean()),
                         unc=float(c0.std()), polMismatch=abs(slogp[a] - slogp[b]),
                         water=("yes" if a == "water" or b == "water" else "")))
df = pd.DataFrame(rows)
print(f"{len(df)} unmeasured FDA drug x solvent-pair candidates scored")

# rank: confident (low ensemble disagreement) + strongly synergistic; require a real polarity gap for sanity
conf = df[(df.unc <= df.unc.quantile(0.5)) & (df.polMismatch >= 1.0)].copy()
conf = conf.sort_values("synergy", ascending=False).drop_duplicates(["drug", "solA", "solB"]).head(25).reset_index(drop=True)

conf.to_csv("results/prospective_targets.csv", index=False)
md = ["# Prospective cosolvency-synergy targets (structure-predicted, unmeasured)\n",
      "Top structure-predicted strong-synergy drug x solvent systems with **no data in MixtureSolDB** — ranked by",
      "predicted Jouyban-Acree synergy (C0) among high-confidence predictions; each has a large solvent polarity gap",
      "(the physical driver), so it is sanity-checkable. Predicted at 298 K. These systems have no measured data; measuring a few would test the predictions.\n",
      "| # | Drug | Solvent A | Solvent B | Predicted synergy | Confidence | \\|ΔlogP\\| | Water |",
      "|---|------|-----------|-----------|------------------:|-----------:|--------:|:-----:|"]
for i, r in conf.iterrows():
    md.append(f"| {i+1} | {r.drug} | {r.solA} | {r.solB} | {r.synergy:+.2f} | {1-r.unc/df.unc.max():.2f} | {r.polMismatch:.2f} | {r.water} |")
open("results/prospective_targets.md", "w", encoding="utf-8").write("\n".join(md) + "\n")

asc = lambda s: s.encode("ascii", "replace").decode()
print("\n===== TOP PROSPECTIVE TARGETS TO MEASURE (predicted strong synergy, high confidence) =====")
print(f"{'#':>2}  {'drug':26s} {'solvent A + B':24s} {'predSyn':>7} {'conf':>5} {'|dLogP|':>7} water")
for i, r in conf.iterrows():
    print(f"{i+1:>2}  {asc(r.drug)[:26]:26s} {(r.solA+' + '+r.solB)[:24]:24s} {r.synergy:>+7.2f} "
          f"{1-r.unc/df.unc.max():>5.2f} {r.polMismatch:>7.2f} {r.water}")
print("\nsaved prospective_targets.csv + prospective_targets.md")
