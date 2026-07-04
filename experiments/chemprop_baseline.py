"""Strong modern baseline: a ChemProp multicomponent D-MPNN, re-run on the same data/splits.
Three molecule graphs + composition/T features -> logS; JA-NET vs ChemProp vs GBDT on a matched subsample."""
import os, sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)
import torch
from lightning import pytorch as pl
from chemprop import data, featurizers, nn, models
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor

SMOKE = os.environ.get("SMOKE") == "1"
N = 300 if SMOKE else 40000
EPOCHS = 2 if SMOKE else 25
torch.manual_seed(0)

d = pd.read_csv("data/mixturesoldb.csv", usecols=["SMILES_Solute", "SMILES_Solvent1", "SMILES_Solvent2",
    "Fraction_Solvent1", "Fraction_Solvent2", "Temperature_K", "LogS(mole_fraction)"]).rename(
    columns={"LogS(mole_fraction)": "y", "Temperature_K": "TK", "Fraction_Solvent1": "x1", "Fraction_Solvent2": "x2"})
for c in ("y", "x1", "x2", "TK"): d[c] = pd.to_numeric(d[c], errors="coerce")
d = d.dropna(subset=["SMILES_Solute", "SMILES_Solvent1", "SMILES_Solvent2", "y", "x1", "x2", "TK"])
d = d[np.isfinite(d.y) & (d.TK > 200) & (d.TK < 450)].reset_index(drop=True)
from rdkit import Chem as _Chem
from rdkit import RDLogger as _RDL; _RDL.DisableLog("rdApp.*")
ok = np.array([(_Chem.MolFromSmiles(a) is not None and _Chem.MolFromSmiles(b) is not None
                and _Chem.MolFromSmiles(c) is not None)
               for a, b, c in zip(d.SMILES_Solute, d.SMILES_Solvent1, d.SMILES_Solvent2)])
d = d[ok].reset_index(drop=True)
print(f"{len(d)} rows with valid SMILES")
rng = np.random.RandomState(0)
d = d.iloc[rng.choice(len(d), min(N, len(d)), replace=False)].reset_index(drop=True)
sysid = (d.SMILES_Solute + "|" + d.SMILES_Solvent1 + "|" + d.SMILES_Solvent2).values
tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=0).split(np.arange(len(d)), groups=sysid))
print(f"{len(d)} rows | {len(set(sysid))} systems | train {len(tr)} test {len(te)}")

smis = [d.SMILES_Solute.values, d.SMILES_Solvent1.values, d.SMILES_Solvent2.values]
xd = np.stack([d.x1, d.x2, 1000.0 / d.TK], 1).astype(np.float32)
y = d.y.values.astype(np.float32)[:, None]
feat = featurizers.SimpleMoleculeMolGraphFeaturizer()

def build(ix):
    comps = []
    for k in range(3):
        dps = []
        for j in ix:
            kw = dict(x_d=xd[j]) if k == 0 else {}
            yy = y[j] if k == 0 else None
            dps.append(data.MoleculeDatapoint.from_smi(smis[k][j], y=yy, **kw))
        comps.append(data.MoleculeDataset(dps, feat))
    return comps

tr_c, te_c = build(tr), build(te)
yscale = tr_c[0].normalize_targets()
for c in te_c: pass
te_c[0].Y = te_c[0].Y  # test targets left raw; we unscale preds
xdscale = tr_c[0].normalize_inputs("X_d")
te_c[0].normalize_inputs("X_d", xdscale)
tr_ds, te_ds = data.MulticomponentDataset(tr_c), data.MulticomponentDataset(te_c)
tr_dl = data.build_dataloader(tr_ds, batch_size=64, num_workers=0, shuffle=True)
te_dl = data.build_dataloader(te_ds, batch_size=256, num_workers=0, shuffle=False)

mp = nn.MulticomponentMessagePassing([nn.BondMessagePassing() for _ in range(3)], n_components=3)
agg = nn.MeanAggregation()
ffn = nn.RegressionFFN(input_dim=mp.output_dim + xd.shape[1], output_transform=yscale.inverse_transform_module()
                       if hasattr(yscale, "inverse_transform_module") else None)
from chemprop.nn.transforms import UnscaleTransform
ffn = nn.RegressionFFN(input_dim=mp.output_dim + xd.shape[1],
                       output_transform=UnscaleTransform.from_standard_scaler(yscale))
model = models.MulticomponentMPNN(mp, agg, ffn, batch_norm=True,
                                  X_d_transform=None)
trainer = pl.Trainer(max_epochs=EPOCHS, accelerator="cpu", devices=1, enable_checkpointing=False,
                     enable_progress_bar=False, logger=False, enable_model_summary=False)
print("training ChemProp D-MPNN ...")
trainer.fit(model, tr_dl)
pred = np.concatenate([b.numpy() for b in trainer.predict(model, te_dl)]).ravel()
cp_mae = mean_absolute_error(d.y.values[te], pred); cp_rmse = np.sqrt(mean_squared_error(d.y.values[te], pred))

print(f"\nChemProp D-MPNN:  MAE {cp_mae:.3f}  RMSE {cp_rmse:.3f}")
if not SMOKE:
    import janet as J
    Dj = J.load_features()
    # align the subsample rows to janet's D by sysid+x1+T is complex; instead retrain JA-NET on this exact subset
    # quick: GBDT on simple concat features for reference on THIS split
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    from rdkit import DataStructs
    DESC = [f for _, f in Descriptors._descList]
    def fe(s, _c={}):
        if s in _c: return _c[s]
        m = Chem.MolFromSmiles(s);
        if m is None: _c[s] = None; return None
        dd = np.array([f(m) for f in DESC], float); fp = np.zeros(128, np.int8)
        DataStructs.ConvertToNumpyArray(rdMolDescriptors.GetMorganFingerprintAsBitVect(m, 2, nBits=128), fp)
        _c[s] = np.nan_to_num(np.concatenate([dd, fp])).astype(np.float32); return _c[s]
    SOL = np.nan_to_num(np.array([fe(s) for s in smis[0]])); V1 = np.nan_to_num(np.array([fe(s) for s in smis[1]]))
    V2 = np.nan_to_num(np.array([fe(s) for s in smis[2]]))
    X = np.hstack([SOL, V1, V2, xd])
    g = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, l2_regularization=1.0, random_state=0)
    g.fit(X[tr], d.y.values[tr]); pg = g.predict(X[te])
    # JA-NET on the SAME subsample + split (fair three-way)
    import janet as J
    from types import SimpleNamespace
    sm, ss = SOL.mean(0), SOL.std(0) + 1e-6; vm = np.vstack([V1, V2]).mean(0); vs = np.vstack([V1, V2]).std(0) + 1e-6
    Tf = np.stack([d.TK, 1000.0 / d.TK, np.log(d.TK)], 1); Tm, Ts = Tf.mean(0), Tf.std(0) + 1e-6
    Dj = SimpleNamespace(y=d.y.values.astype(np.float32), x1=d.x1.values.astype(np.float32),
        x2=d.x2.values.astype(np.float32), sysid=sysid,
        SOLn=((SOL - sm) / ss).astype(np.float32), V1n=((V1 - vm) / vs).astype(np.float32),
        V2n=((V2 - vm) / vs).astype(np.float32), Tfn=((Tf - Tm) / Ts).astype(np.float32),
        fdim=SOL.shape[1], tdim=3)
    pj = np.mean([J.train_ja(Dj, tr, te, s, vgroups=sysid) for s in (0, 1, 2)], 0)
    mae = lambda p: mean_absolute_error(d.y.values[te], p); rms = lambda p: np.sqrt(mean_squared_error(d.y.values[te], p))
    print(f"\n===== THREE-WAY on the same {N}-row subsample, leave-system-out =====")
    print(f"  GBDT (concat)     MAE {mae(pg):.3f}  RMSE {rms(pg):.3f}")
    print(f"  ChemProp D-MPNN   MAE {cp_mae:.3f}  RMSE {cp_rmse:.3f}")
    print(f"  JA-NET (3-seed)   MAE {mae(pj):.3f}  RMSE {rms(pj):.3f}")
print("DONE")
