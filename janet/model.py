"""JA-NET: a Jouyban-Acree-structured net for mixed-solvent solubility from structure alone.
logS = x1*e1 + x2*e2 + x1*x2*(C0+C1*d+C2*d^2); importable (load_features/JANet/train_ja/train_concat/evaluate) or run for the leave-system/solute-out comparison."""
import os
import warnings
import numpy as np
import pandas as pd
from types import SimpleNamespace

warnings.filterwarnings("ignore")
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error

DESC = [f for _, f in Descriptors._descList]   # the RDKit 2D descriptor functions
C_SCALE = 3.0                                   # tanh bound on the cosolvency cross-term
BATCH = 4096
PATIENCE = 7                                     # early-stopping patience (epochs without val improvement)


def _val_split(train_idx, seed, vgroups=None, frac=1 / 12):
    """Carve an early-stopping validation set out of the training rows.

    vgroups given  -> val is DISJOINT from train by group (matches a leave-group-out target -> honest selection).
    vgroups None   -> legacy random-row val (systems may overlap train).
    """
    rng = np.random.RandomState(seed)
    if vgroups is None:
        perm = rng.permutation(train_idx)
        val = perm[:max(1, int(len(perm) * frac))]
        return np.setdiff1d(train_idx, val), val
    train_pos, val_pos = next(GroupShuffleSplit(1, test_size=frac, random_state=seed)
                              .split(train_idx, groups=vgroups[train_idx]))
    return train_idx[train_pos], train_idx[val_pos]


def _feat(smi, _cache={}):
    """Featurize one molecule: RDKit 2D descriptors + a 128-bit Morgan fingerprint (cached per SMILES)."""
    if smi in _cache:
        return _cache[smi]
    mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    if mol is None:
        _cache[smi] = None
        return None
    descriptors = np.array([f(mol) for f in DESC], float)
    fp = np.zeros(128, np.int8)
    DataStructs.ConvertToNumpyArray(rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=128), fp)
    _cache[smi] = np.concatenate([descriptors, fp]).astype(np.float32)
    return _cache[smi]


def load_features(csv="data/mixturesoldb.csv", npz="data/feat.npz"):
    """Load MixtureSolDB into a data bundle: raw features (cached to npz), standardized versions, and the concat matrix."""
    if os.path.exists(npz):
        z = np.load(npz, allow_pickle=True)
        D = SimpleNamespace(**{k: z[k] for k in z.files})
    else:
        # build features from the CSV once, then cache the raw arrays
        cols = ["SMILES_Solute", "SMILES_Solvent1", "SMILES_Solvent2", "Fraction_Solvent1",
                "Fraction_Solvent2", "Temperature_K", "LogS(mole_fraction)", "IsPureSolventEndpoint"]
        d = pd.read_csv(csv)[cols].dropna(subset=cols[:7])
        d = d.rename(columns={"LogS(mole_fraction)": "y", "Temperature_K": "TK",
                              "Fraction_Solvent1": "x1", "Fraction_Solvent2": "x2"})
        d = d[np.isfinite(d.y) & (d.TK > 200) & (d.TK < 450)].reset_index(drop=True)

        sol_feats, v1_feats, v2_feats, keep = [], [], [], []
        for r in d.itertuples():
            a = _feat(r.SMILES_Solute)
            b = _feat(r.SMILES_Solvent1)
            c = _feat(r.SMILES_Solvent2)
            if a is None or b is None or c is None:   # drop rows with an unparseable molecule
                keep.append(False)
                continue
            keep.append(True)
            sol_feats.append(a); v1_feats.append(b); v2_feats.append(c)
        d = d[keep].reset_index(drop=True)

        SOL, V1, V2 = np.array(sol_feats), np.array(v1_feats), np.array(v2_feats)
        for A in (SOL, V1, V2):
            A[~np.isfinite(A)] = 0                     # a few descriptors can be NaN/inf -> zero them
        D = SimpleNamespace(
            SOL=SOL, V1=V1, V2=V2,
            x1=d.x1.to_numpy(np.float32), x2=d.x2.to_numpy(np.float32),
            T=d.TK.to_numpy(np.float32), y=d.y.to_numpy(np.float32),
            sysid=(d.SMILES_Solute + "|" + d.SMILES_Solvent1 + "|" + d.SMILES_Solvent2).values.astype(str),
            soluteid=d.SMILES_Solute.values.astype(str),
            endp=(d.IsPureSolventEndpoint.astype(str).str.lower() == "true").to_numpy())
        np.savez(npz, **vars(D))

    # standardize features + temperature, and build the concat-baseline design matrix
    sol_mean, sol_std = D.SOL.mean(0), D.SOL.std(0) + 1e-6
    slv_mean = np.vstack([D.V1, D.V2]).mean(0)
    slv_std = np.vstack([D.V1, D.V2]).std(0) + 1e-6
    t_feats = np.stack([D.T, 1000.0 / D.T, np.log(D.T)], 1)    # T, 1000/T, log T
    t_mean, t_std = t_feats.mean(0), t_feats.std(0) + 1e-6

    D.SOLn = ((D.SOL - sol_mean) / sol_std).astype(np.float32)
    D.V1n = ((D.V1 - slv_mean) / slv_std).astype(np.float32)
    D.V2n = ((D.V2 - slv_mean) / slv_std).astype(np.float32)
    D.Tfn = ((t_feats - t_mean) / t_std).astype(np.float32)
    D.norm = (sol_mean, sol_std, slv_mean, slv_std, t_mean, t_std)   # kept in this order (unpacked elsewhere)
    D.Xcat = np.hstack([D.SOLn, D.V1n, D.V2n, D.x1[:, None], D.x2[:, None], D.Tfn]).astype(np.float32)
    D.fdim, D.tdim = D.SOL.shape[1], D.Tfn.shape[1]
    return D


class JANet(nn.Module):
    """Shared solute/solvent encoders -> a single-solvent endpoint head + a bounded cosolvency cross-term head,
    assembled through the Jouyban-Acree equation so composition enters only via the physical form."""

    def __init__(self, fdim, tdim, h=256, z=128, c_scale=C_SCALE):
        super().__init__()
        self.c_scale = c_scale
        self.sol_enc = nn.Sequential(nn.Linear(fdim, h), nn.ReLU(), nn.Linear(h, z), nn.ReLU())
        self.slv_enc = nn.Sequential(nn.Linear(fdim, h), nn.ReLU(), nn.Linear(h, z), nn.ReLU())
        self.endp = nn.Sequential(nn.Linear(2 * z + tdim, h), nn.ReLU(), nn.Linear(h, 64), nn.ReLU(), nn.Linear(64, 1))
        self.cross = nn.Sequential(nn.Linear(3 * z + tdim, h), nn.ReLU(), nn.Linear(h, 64), nn.ReLU(), nn.Linear(64, 3))

    def forward(self, sol, v1, v2, tf, x1, x2):
        z_sol, z_v1, z_v2 = self.sol_enc(sol), self.slv_enc(v1), self.slv_enc(v2)
        e1 = self.endp(torch.cat([z_sol, z_v1, tf], 1)).squeeze(1)   # pure-solvent-1 endpoint
        e2 = self.endp(torch.cat([z_sol, z_v2, tf], 1)).squeeze(1)   # pure-solvent-2 endpoint
        C = self.c_scale * torch.tanh(self.cross(torch.cat([z_sol, z_v1, z_v2, tf], 1)))   # bounded C0,C1,C2
        d = x1 - x2
        logS = x1 * e1 + x2 * e2 + x1 * x2 * (C[:, 0] + C[:, 1] * d + C[:, 2] * d * d)      # Jouyban-Acree readout
        return logS, e1, e2, C


class ConcatMLP(nn.Module):
    """Baseline: concatenate everything (solute, both solvents, mole fractions, T) and regress logS directly."""

    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 512), nn.ReLU(), nn.Dropout(0.1),
                                 nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 1))

    def forward(self, x):
        return self.net(x).squeeze(1)


def train_ja(D, train_idx, test_idx, seed=0, epochs=45, save=None, z=128, lr=1e-3,
             c_scale=C_SCALE, wd=2e-5, vgroups=None):
    """Train JA-NET (solvent-swap augmented for symmetry, early-stopped on val MAE); return test predictions."""
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    tensor = torch.tensor
    net = JANet(D.fdim, D.tdim, z=z, c_scale=c_scale)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    fit_idx, val_idx = _val_split(train_idx, seed, vgroups)
    y = tensor(D.y)

    def forward_batch(idx, swap=False):
        # swap=True feeds solvents in the other order (with mole fractions swapped) to enforce solvent symmetry
        va, vb = (D.V2n, D.V1n) if swap else (D.V1n, D.V2n)
        xa, xb = (D.x2, D.x1) if swap else (D.x1, D.x2)
        return net(tensor(D.SOLn[idx]), tensor(va[idx]), tensor(vb[idx]),
                   tensor(D.Tfn[idx]), tensor(xa[idx]), tensor(xb[idx]))

    best_val, best_state, stale = 1e9, None, 0
    for _ in range(epochs):
        net.train()
        perm = fit_idx[rng.permutation(len(fit_idx))]
        for start in range(0, len(perm), BATCH):
            batch = perm[start:start + BATCH]
            opt.zero_grad()
            pred = forward_batch(batch, swap=rng.rand() < 0.5)[0]
            ((pred - y[batch]) ** 2).mean().backward()          # MSE
            opt.step()
        net.eval()
        with torch.no_grad():
            val_mae = mean_absolute_error(D.y[val_idx], forward_batch(val_idx)[0].numpy())
        if val_mae < best_val - 1e-4:
            best_val, best_state, stale = val_mae, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break

    net.load_state_dict(best_state)
    net.eval()
    if save:
        torch.save({"state": best_state}, save)
    with torch.no_grad():
        return forward_batch(test_idx)[0].numpy()


def train_concat(D, train_idx, test_idx, seed=0, epochs=45, vgroups=None):
    """Train the concat-MLP baseline (standardized target, early-stopped on val MAE); return test predictions."""
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    tensor = torch.tensor
    y_mean, y_std = D.y[train_idx].mean(), D.y[train_idx].std()
    net = ConcatMLP(D.Xcat.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    mse = nn.MSELoss()
    X = tensor(D.Xcat)
    fit_idx, val_idx = _val_split(train_idx, seed, vgroups)
    y_std_target = tensor((D.y - y_mean) / y_std)

    best_val, best_state, stale = 1e9, None, 0
    for _ in range(epochs):
        net.train()
        perm = fit_idx[rng.permutation(len(fit_idx))]
        for start in range(0, len(perm), BATCH):
            batch = perm[start:start + BATCH]
            opt.zero_grad()
            mse(net(X[batch]), y_std_target[batch]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            val_pred = net(X[tensor(val_idx)]).numpy() * y_std + y_mean
            val_mae = mean_absolute_error(D.y[val_idx], val_pred)
        if val_mae < best_val - 1e-4:
            best_val, best_state, stale = val_mae, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break

    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        return net(X[tensor(test_idx)]).numpy() * y_std + y_mean


def evaluate(D, name, groups, seeds=(0, 1, 2), rs=0, quiet=False):
    """80/20 group split (rs varies it) + the three-model comparison; val is disjoint by the same groups."""
    train_idx, test_idx = next(GroupShuffleSplit(1, test_size=0.2, random_state=rs)
                               .split(np.arange(len(D.y)), groups=groups))
    if not quiet:
        print(f"### {name} [split rs={rs}]: train {len(train_idx)} / test {len(test_idx)}")

    gbdt = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, l2_regularization=1.0, random_state=0)
    gbdt.fit(D.Xcat[train_idx], D.y[train_idx])
    pred_gbdt = gbdt.predict(D.Xcat[test_idx])
    pred_concat = train_concat(D, train_idx, test_idx, vgroups=groups)
    pred_ja = np.mean([train_ja(D, train_idx, test_idx, s, vgroups=groups) for s in seeds], 0)

    preds = {"CONCAT-GBDT": pred_gbdt, "CONCAT-MLP": pred_concat, f"JA-NET ({len(seeds)}-seed)": pred_ja}
    if not quiet:
        for nm, p in preds.items():
            print(f"  {nm:20s} MAE {mean_absolute_error(D.y[test_idx], p):.3f}   "
                  f"RMSE {np.sqrt(mean_squared_error(D.y[test_idx], p)):.3f}")
    return train_idx, test_idx, preds


def demo_cosolvency_curve(D, min_pts=12, seed=0):
    """Predict one held-out system's full composition curve at its median temperature (for the interpretability figure)."""
    import collections
    candidates = [s for s, c in collections.Counter(D.sysid).items() if c >= min_pts]
    system = max(candidates, key=lambda s: D.x1[D.sysid == s].max() - D.x1[D.sysid == s].min())
    mask = D.sysid == system
    train_ja(D, np.where(~mask)[0], np.where(mask)[0][:1], seed=seed, save="models/ja_demo.pt")
    net = JANet(D.fdim, D.tdim)
    net.load_state_dict(torch.load("models/ja_demo.pt")["state"])
    net.eval()

    rows = np.where(mask)[0]
    anchor = rows[0]
    T_med = float(np.median(D.T[rows]))
    _, _, _, _, t_mean, t_std = D.norm
    tensor = torch.tensor
    xs = np.linspace(0, 1, 101).astype(np.float32)
    t_feat = (np.array([T_med, 1000 / T_med, np.log(T_med)], np.float32) - t_mean) / t_std
    tf = np.repeat(t_feat[None], len(xs), 0)
    tile = lambda a: np.repeat(a[None], len(xs), 0)
    with torch.no_grad():
        pred, e1, e2, C = net(tensor(tile(D.SOLn[anchor])), tensor(tile(D.V1n[anchor])), tensor(tile(D.V2n[anchor])),
                              tensor(tf), tensor(xs), tensor(1 - xs))
    pred, e1, e2 = pred.numpy(), e1.numpy(), e2.numpy()
    return dict(name=system, xs=xs, pred=pred, loglin=xs * e1 + (1 - xs) * e2,
                mx=D.x1[rows], my=D.y[rows], Tmed=T_med, C=C.numpy().mean(0))


if __name__ == "__main__":
    D = load_features()
    print(f"{len(D.y)} rows | {len(set(D.sysid))} systems | {len(set(D.soluteid))} solutes")
    evaluate(D, "LEAVE-BINARY-SYSTEM-OUT", D.sysid)
    evaluate(D, "LEAVE-SOLUTE-OUT", D.soluteid)
