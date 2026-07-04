"""Ablation: does the end-to-end JA readout beat the two-stage 'fit coefficients then plug in'?
Same architecture and leave-system-out test; decomposes the gain into objective vs data-efficiency."""
import sys, numpy as np
sys.stdout.reconfigure(line_buffering=True)
import torch
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import janet as J

D = J.load_features()
idx = np.arange(len(D.y))
tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=0).split(idx, groups=D.sysid))
CS = J.C_SCALE
mae = lambda p: mean_absolute_error(D.y[te], p); rmse = lambda p: np.sqrt(mean_squared_error(D.y[te], p))

# stage 1: per-training-system least-squares fit of (a1,a2,J0,J1,J2); keep stable, in-range fits
tr_sys = D.sysid[tr]; reps, tgts = [], []
for s in set(tr_sys):
    g = tr[tr_sys == s]
    x1 = D.x1[g]
    if len(g) < 8 or x1.max() - x1.min() < 0.5 or len(np.unique(np.round(x1, 2))) < 5:
        continue
    d = x1 - D.x2[g]; A = np.stack([x1, D.x2[g], x1 * D.x2[g], x1 * D.x2[g] * d, x1 * D.x2[g] * d * d], 1)
    coef, *_ = np.linalg.lstsq(A, D.y[g], rcond=None)
    res = D.y[g] - A @ coef; r2 = 1 - (res ** 2).sum() / (((D.y[g] - D.y[g].mean()) ** 2).sum() + 1e-9)
    if r2 < 0.9 or abs(coef[2]) > CS:
        continue
    coef[2:5] = np.clip(coef[2:5], -CS, CS)          # clip C-targets into the net's representable range
    reps.append(g[0]); tgts.append(coef)
reps = np.array(reps); tgts = np.array(tgts)
fitable = set(D.sysid[reps])
tr_sub = tr[np.isin(D.sysid[tr], list(fitable))]      # all points of fit-able systems (for condition 2)
tm, ts = tgts.mean(0), tgts.std(0) + 1e-6
print(f"{len(reps)} fit-able systems ({len(tr_sub)} of {len(tr)} train points); test {len(te)}")


# two-stage: same JANet, but loss matches per-system fitted coefficients (then JA-assemble at test)
def train_twostage(seed, epochs=120):
    torch.manual_seed(seed); rng = np.random.RandomState(seed); t = torch.tensor
    net = J.JANet(D.fdim, D.tdim); opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=2e-5)
    Tg = t(((tgts - tm) / ts).astype(np.float32))
    N = len(reps); vi = rng.permutation(N)[:N // 8]; ti = np.setdiff1d(np.arange(N), vi)
    def coefs(ix):
        r = reps[ix]
        _, e1, e2, C = net(t(D.SOLn[r]), t(D.V1n[r]), t(D.V2n[r]), t(D.Tfn[r]), t(D.x1[r]), t(D.x2[r]))
        pred = torch.stack([e1, e2, C[:, 0], C[:, 1], C[:, 2]], 1)
        return (pred - t(tm.astype(np.float32))) / t(ts.astype(np.float32))
    best, bs, bad = 1e9, None, 0
    for _ in range(epochs):
        net.train(); p = ti[rng.permutation(len(ti))]
        for i in range(0, len(p), 512):
            b = p[i:i + 512]; opt.zero_grad(); ((coefs(b) - Tg[b]) ** 2).mean().backward(); opt.step()
        net.eval()
        with torch.no_grad(): vl = ((coefs(vi) - Tg[vi]) ** 2).mean().item()
        if vl < best - 1e-5: best, bs, bad = vl, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 12: break
    net.load_state_dict(bs); net.eval()
    with torch.no_grad():
        pr, *_ = net(t(D.SOLn[te]), t(D.V1n[te]), t(D.V2n[te]), t(D.Tfn[te]), t(D.x1[te]), t(D.x2[te]))
    return pr.numpy()


print("training (3 seeds each) ...")
p1 = np.mean([J.train_ja(D, tr, te, seed=s) for s in (0, 1, 2)], 0)        # end-to-end, all data
p2 = np.mean([J.train_ja(D, tr_sub, te, seed=s) for s in (0, 1, 2)], 0)    # end-to-end, fit-able only
p3 = np.mean([train_twostage(s) for s in (0, 1, 2)], 0)                    # two-stage

print("\n===== END-TO-END vs TWO-STAGE (leave-system-out test) =====")
for nm, p in [("1. END-TO-END (all data)", p1), ("2. END-TO-END (fit-able only)", p2), ("3. TWO-STAGE (fit-able)", p3)]:
    print(f"  {nm:30s} MAE {mae(p):.3f}   RMSE {rmse(p):.3f}")
print(f"\n  objective effect (2 vs 3):  end-to-end {mae(p2):.3f} vs two-stage {mae(p3):.3f}")
print(f"  total end-to-end gain (1 vs 3): {mae(p1):.3f} vs {mae(p3):.3f}")

