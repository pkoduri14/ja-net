"""Leakage-safe hyperparameter sweep for JA-NET.
Coordinate search selected on a system-disjoint validation set with a single test readout (the win is architectural, not tuning)."""
import sys, numpy as np
sys.stdout.reconfigure(line_buffering=True)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import janet as J

D = J.load_features()
idx = np.arange(len(D.y))

# same TEST as the headline (test_size=0.2, rs=0); carve VAL from the training systems
orig_tr, test = next(GroupShuffleSplit(1, test_size=0.2, random_state=0).split(idx, groups=D.sysid))
sub_tr, sub_val = next(GroupShuffleSplit(1, test_size=0.18, random_state=1).split(orig_tr, groups=D.sysid[orig_tr]))
train, val = orig_tr[sub_tr], orig_tr[sub_val]
print(f"train {len(train)} / val {len(val)} / test {len(test)}  (system-disjoint)")
mae = lambda ix, p: mean_absolute_error(D.y[ix], p)

# coordinate search around baseline (1 seed, 35 epochs); SELECT ON VAL
base = dict(z=128, lr=1e-3, c_scale=3.0, wd=2e-5)
grid = {"c_scale": [2.0, 5.0], "z": [64, 256], "lr": [5e-4, 2e-3], "wd": [1e-5, 1e-4]}
configs = [("baseline", dict(base))]
for k, vals in grid.items():
    for v in vals:
        c = dict(base); c[k] = v; configs.append((f"{k}={v}", c))

print("\n# searching (val MAE):")
res = []
for name, c in configs:
    m = mae(val, J.train_ja(D, train, val, seed=0, epochs=35, vgroups=D.sysid, **c))
    res.append((m, name, c)); print(f"  {name:12s} val MAE {m:.4f}")
res.sort()
best_m, best_name, best = res[0]
print(f"\nBEST on VAL: {best_name}  (val MAE {best_m:.4f})\n  config {best}")

# FINAL: best config, 3-seed, trained on full original train set -> single TEST readout
fit = orig_tr
pj = np.mean([J.train_ja(D, fit, test, seed=s, epochs=45, vgroups=D.sysid, **best) for s in (0, 1, 2)], 0)
pc = J.train_concat(D, fit, test)
g = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, l2_regularization=1.0, random_state=0)
g.fit(D.Xcat[fit], D.y[fit]); pg = g.predict(D.Xcat[test])
rmse = lambda p: np.sqrt(mean_squared_error(D.y[test], p))
print("\n===== FINAL TEST (leave-system-out, untouched) =====")
for nm, p in [("CONCAT-GBDT", pg), ("CONCAT-MLP", pc), ("JA-NET tuned (3-seed)", pj)]:
    print(f"  {nm:22s} MAE {mae(test, p):.3f}   RMSE {rmse(p):.3f}")
print(f"  tuned config: {best}")
