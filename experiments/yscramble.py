"""Chuang-Keiser y-scramble negative control.
Train on permuted labels; a genuine model must collapse toward the mean-predictor (reported vs real + mean baselines)."""
import sys, numpy as np
sys.stdout.reconfigure(line_buffering=True)
from types import SimpleNamespace
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import janet as J

D = J.load_features()
idx = np.arange(len(D.y))
tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=0).split(idx, groups=D.sysid))
rmse = lambda p, y: np.sqrt(mean_squared_error(y, p))

# real
pj = np.mean([J.train_ja(D, tr, te, s, vgroups=D.sysid) for s in (0, 1, 2)], 0)
real_mae = mean_absolute_error(D.y[te], pj)

# scrambled: permute labels globally, keep everything else identical
rng = np.random.RandomState(0)
Dp = SimpleNamespace(**vars(D)); Dp.y = D.y[rng.permutation(len(D.y))].copy()
ps = np.mean([J.train_ja(Dp, tr, te, s, vgroups=Dp.sysid) for s in (0, 1, 2)], 0)
scr_mae = mean_absolute_error(Dp.y[te], ps)

mean_mae = mean_absolute_error(D.y[te], np.full(len(te), D.y[tr].mean()))   # trivial mean-predictor
print("\n== Y-SCRAMBLE negative control (leave-system-out) ==")
print(f"  real JA-NET         test MAE {real_mae:.3f}")
print(f"  y-scrambled JA-NET  test MAE {scr_mae:.3f}")
print(f"  mean-predictor      test MAE {mean_mae:.3f}")
verdict = "PASS" \
    if scr_mae >= 0.9 * mean_mae else "FAIL"
print(f"  y-scramble control: {verdict}")
