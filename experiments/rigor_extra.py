"""Additional robustness checks: single-seed fairness, matched JA-equation ablation, OOD conformal coverage.
System-disjoint train/calib/test; per-seed JA-NET vs single baselines, FreeNet vs JA readout, conformal coverage on novel solutes."""
import sys, numpy as np
sys.stdout.reconfigure(line_buffering=True)
import torch, torch.nn as nn
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import janet as J

D = J.load_features()
idx = np.arange(len(D.y))
tr0, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=0).split(idx, groups=D.sysid))
sub_tr, sub_cal = next(GroupShuffleSplit(1, test_size=0.2, random_state=1).split(tr0, groups=D.sysid[tr0]))
tr, cal = tr0[sub_tr], tr0[sub_cal]
tr_solutes = set(D.soluteid[tr])
te_novel = te[np.array([D.soluteid[i] not in tr_solutes for i in te])]
print(f"train {len(tr)} / calib {len(cal)} / test {len(te)} (novel-solute subset {len(te_novel)})")
rmse = lambda p, y: np.sqrt(mean_squared_error(y, p))


class FreeNet(nn.Module):   # matched encoders, composition through a free MLP (NO JA equation)
    def __init__(self, fdim, tdim, h=256, z=128):
        super().__init__()
        self.sol_enc = nn.Sequential(nn.Linear(fdim, h), nn.ReLU(), nn.Linear(h, z), nn.ReLU())
        self.slv_enc = nn.Sequential(nn.Linear(fdim, h), nn.ReLU(), nn.Linear(h, z), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(3 * z + tdim + 2, h), nn.ReLU(), nn.Linear(h, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, sol, v1, v2, tf, x1, x2):
        zs, z1, z2 = self.sol_enc(sol), self.slv_enc(v1), self.slv_enc(v2)
        h = torch.cat([zs, z1, z2, tf, x1[:, None], x2[:, None]], 1)
        return self.head(h).squeeze(1)


def train_free(tr, te, seed=0, epochs=45, vgroups=None):
    torch.manual_seed(seed); rng = np.random.RandomState(seed); t = torch.tensor
    net = FreeNet(D.fdim, D.tdim); opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=2e-5)
    ti, vi = J._val_split(tr, seed, vgroups); yT = t(D.y)
    def fwd(ix, swap=False):
        a, b = (D.V2n, D.V1n) if swap else (D.V1n, D.V2n); xa, xb = (D.x2, D.x1) if swap else (D.x1, D.x2)
        return net(t(D.SOLn[ix]), t(a[ix]), t(b[ix]), t(D.Tfn[ix]), t(xa[ix]), t(xb[ix]))
    best, bs, bad = 1e9, None, 0
    for _ in range(epochs):
        net.train(); p = ti[rng.permutation(len(ti))]
        for i in range(0, len(p), 4096):
            b = p[i:i + 4096]; opt.zero_grad(); fwd(b, rng.rand() < 0.5).sub(yT[b]).pow(2).mean().backward(); opt.step()
        net.eval()
        with torch.no_grad(): vmae = mean_absolute_error(D.y[vi], fwd(vi).numpy())
        if vmae < best - 1e-4: best, bs, bad = vmae, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 7: break
    net.load_state_dict(bs); net.eval()
    with torch.no_grad(): return fwd(te).numpy()


# JA-NET per-seed (train once per seed; predict calib+test from the same model)
ja_te = [J.train_ja(D, tr, te, s, vgroups=D.sysid) for s in (0, 1, 2)]
ja_cal = [J.train_ja(D, tr, cal, s, vgroups=D.sysid) for s in (0, 1, 2)]
ja_ens = np.mean(ja_te, 0)
# baselines (single models)
g = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, l2_regularization=1.0, random_state=0)
g.fit(D.Xcat[tr], D.y[tr]); pg = g.predict(D.Xcat[te])
pc = J.train_concat(D, tr, te, vgroups=D.sysid)
# FreeNet (matched ablation)
free_ens = np.mean([train_free(tr, te, s, vgroups=D.sysid) for s in (0, 1, 2)], 0)

seed_maes = [mean_absolute_error(D.y[te], p) for p in ja_te]
print("\n== single-seed fairness (leave-system-out test) ==")
print(f"  JA-NET single-seed  MAE {min(seed_maes):.3f}-{max(seed_maes):.3f} (mean {np.mean(seed_maes):.3f})")
print(f"  JA-NET 3-seed ens   MAE {mean_absolute_error(D.y[te], ja_ens):.3f}")
print(f"  GBDT (single)       MAE {mean_absolute_error(D.y[te], pg):.3f}")
print(f"  concat-MLP (single) MAE {mean_absolute_error(D.y[te], pc):.3f}")


print("\n== matched JA-equation ablation (same encoders, free MLP composition vs JA readout) ==")
print(f"  FreeNet (free comp) MAE {mean_absolute_error(D.y[te], free_ens):.3f}  RMSE {rmse(free_ens, D.y[te]):.3f}")
print(f"  JA-NET (JA readout) MAE {mean_absolute_error(D.y[te], ja_ens):.3f}  RMSE {rmse(ja_ens, D.y[te]):.3f}")
dd = mean_absolute_error(D.y[te], free_ens) - mean_absolute_error(D.y[te], ja_ens)
print(f"  JA readout vs free-composition head: {dd:+.3f} MAE")

print("\n== OOD conformal coverage (90% target; calibrated on system-disjoint calib) ==")
mu_cal = np.mean(ja_cal, 0); mu_te = np.mean(ja_te, 0); sd_te = np.std(ja_te, 0) + 1e-6
score = np.abs(mu_cal - D.y[cal]); n = len(score); q = np.quantile(score, np.ceil((n + 1) * 0.9) / n)
cov_all = np.mean(np.abs(mu_te - D.y[te]) <= q)
nv = np.array([D.soluteid[i] not in tr_solutes for i in te])
cov_novel = np.mean(np.abs(mu_te[nv] - D.y[te[nv]]) <= q) if nv.sum() else float("nan")
print(f"  coverage on full system-disjoint test: {cov_all:.2f}")
print(f"  coverage on NOVEL-SOLUTE subset:       {cov_novel:.2f}  (n={int(nv.sum())})")

