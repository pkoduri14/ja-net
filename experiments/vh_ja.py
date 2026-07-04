"""Van't Hoff-Jouyban-Acree readout: temperature enters only through the physical 1/T form.
Structure-only coefficients (A,B,J0-J2); base vs VH-JA on leave-system-out and temperature-extrapolation."""
import os, sys, numpy as np
sys.stdout.reconfigure(line_buffering=True)
import torch, torch.nn as nn
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import janet as J

SMOKE = os.environ.get("SMOKE") == "1"
EP = 6 if SMOKE else 45
D = J.load_features()
uphys = (300.0 / D.T).astype(np.float32)   # positive, O(1), proportional to 1/T (Van't Hoff variable)


class JANetVH(nn.Module):
    def __init__(self, fdim, h=256, z=128, c_scale=3.0):
        super().__init__(); self.c_scale = c_scale
        self.sol_enc = nn.Sequential(nn.Linear(fdim, h), nn.ReLU(), nn.Linear(h, z), nn.ReLU())
        self.slv_enc = nn.Sequential(nn.Linear(fdim, h), nn.ReLU(), nn.Linear(h, z), nn.ReLU())
        self.endp = nn.Sequential(nn.Linear(2 * z, h), nn.ReLU(), nn.Linear(h, 64), nn.ReLU(), nn.Linear(64, 2))
        self.cross = nn.Sequential(nn.Linear(3 * z, h), nn.ReLU(), nn.Linear(h, 64), nn.ReLU(), nn.Linear(64, 3))

    def forward(self, sol, v1, v2, u, x1, x2):
        zs, z1, z2 = self.sol_enc(sol), self.slv_enc(v1), self.slv_enc(v2)
        AB1 = self.endp(torch.cat([zs, z1], 1)); AB2 = self.endp(torch.cat([zs, z2], 1))
        e1 = AB1[:, 0] + AB1[:, 1] * u                       # Van't Hoff endpoints (linear in 1/T)
        e2 = AB2[:, 0] + AB2[:, 1] * u
        Jc = self.c_scale * torch.tanh(self.cross(torch.cat([zs, z1, z2], 1)))
        d = x1 - x2
        cross = x1 * x2 * u * (Jc[:, 0] + Jc[:, 1] * d + Jc[:, 2] * d * d)   # cosolvency term carries 1/T
        return x1 * e1 + x2 * e2 + cross, Jc


def train_vh(tr, te, seed=0, epochs=EP, vgroups=None):
    torch.manual_seed(seed); rng = np.random.RandomState(seed); t = torch.tensor
    net = JANetVH(D.fdim); opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=2e-5)
    ti, vi = J._val_split(tr, seed, vgroups); yT = t(D.y)
    def fwd(idx, swap=False):
        a, b = (D.V2n, D.V1n) if swap else (D.V1n, D.V2n); xa, xb = (D.x2, D.x1) if swap else (D.x1, D.x2)
        return net(t(D.SOLn[idx]), t(a[idx]), t(b[idx]), t(uphys[idx]), t(xa[idx]), t(xb[idx]))
    best, bs, bad = 1e9, None, 0
    for _ in range(epochs):
        net.train(); p = ti[rng.permutation(len(ti))]
        for i in range(0, len(p), 4096):
            b = p[i:i + 4096]; opt.zero_grad()
            fwd(b, rng.rand() < 0.5)[0].sub(yT[b]).pow(2).mean().backward(); opt.step()
        net.eval()
        with torch.no_grad(): vmae = mean_absolute_error(D.y[vi], fwd(vi)[0].numpy())
        if vmae < best - 1e-4: best, bs, bad = vmae, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 7: break
    net.load_state_dict(bs); net.eval()
    with torch.no_grad(): return fwd(te)[0].numpy()


rmse = lambda p, y: np.sqrt(mean_squared_error(y, p))
seeds = (0,) if SMOKE else (0, 1, 2)

# 1) leave-system-out (does the physical T form hold up on the main split?)
tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=0).split(np.arange(len(D.y)), groups=D.sysid))
vh = np.mean([train_vh(tr, te, s, vgroups=D.sysid) for s in seeds], 0)
base = np.mean([J.train_ja(D, tr, te, s, vgroups=D.sysid) for s in seeds], 0)
print("\n=== leave-system-out ===")
print(f"  base JA-NET  MAE {mean_absolute_error(D.y[te], base):.3f}  RMSE {rmse(base, D.y[te]):.3f}")
print(f"  VH-JA        MAE {mean_absolute_error(D.y[te], vh):.3f}  RMSE {rmse(vh, D.y[te]):.3f}")

# 2) temperature-extrapolation: train cool, test hot (the axis VH physics should help)
thr = np.quantile(D.T, 0.8)
trH = np.where(D.T < thr)[0]; teH = np.where(D.T >= thr)[0]
print(f"\n=== temperature extrapolation (train T<{thr:.0f}K n={len(trH)}, test T>={thr:.0f}K n={len(teH)}) ===")
vhH = np.mean([train_vh(trH, teH, s, vgroups=D.sysid) for s in seeds], 0)
baseH = np.mean([J.train_ja(D, trH, teH, s, vgroups=D.sysid) for s in seeds], 0)
print(f"  base JA-NET  MAE {mean_absolute_error(D.y[teH], baseH):.3f}  RMSE {rmse(baseH, D.y[teH]):.3f}")
print(f"  VH-JA        MAE {mean_absolute_error(D.y[teH], vhH):.3f}  RMSE {rmse(vhH, D.y[teH]):.3f}")

