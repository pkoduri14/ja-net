"""Uncertainty quantification: calibration + conformal prediction intervals for JA-NET.
Does ensemble spread rank error, a recalibration scale, and split-conformal 90% coverage (system-disjoint)."""
import sys, numpy as np
sys.stdout.reconfigure(line_buffering=True)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr
import janet as J

D = J.load_features()
idx = np.arange(len(D.y))
# system-disjoint train / calib / test
tr0, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=0).split(idx, groups=D.sysid))
sub_tr, sub_cal = next(GroupShuffleSplit(1, test_size=0.2, random_state=1).split(tr0, groups=D.sysid[tr0]))
tr, cal = tr0[sub_tr], tr0[sub_cal]
print(f"train {len(tr)} / calib {len(cal)} / test {len(te)}  (system-disjoint)")

SEEDS = (0, 1, 2, 3, 4)
print(f"training {len(SEEDS)}-seed ensemble ...")
P_cal = np.stack([J.train_ja(D, tr, cal, seed=s, vgroups=D.sysid) for s in SEEDS], 1)   # (n_cal, seeds)
P_te = np.stack([J.train_ja(D, tr, te, seed=s, vgroups=D.sysid) for s in SEEDS], 1)
mu_cal, sd_cal = P_cal.mean(1), P_cal.std(1) + 1e-6
mu_te, sd_te = P_te.mean(1), P_te.std(1) + 1e-6
err_cal, err_te = np.abs(mu_cal - D.y[cal]), np.abs(mu_te - D.y[te])
print(f"ensemble test MAE {err_te.mean():.3f}\n")

# 1. does sigma rank error?
sr = spearmanr(sd_te, err_te)
print(f"1. does ensemble sigma rank error?  Spearman(sigma, |err|) = {sr.statistic:+.3f} (p={sr.pvalue:.1e})")
qs = np.quantile(sd_te, [0, .2, .4, .6, .8, 1.0])
print("   reliability by sigma quintile (test):")
for i in range(5):
    m = (sd_te >= qs[i]) & (sd_te <= qs[i + 1])
    print(f"     sigma in [{qs[i]:.2f},{qs[i+1]:.2f}]  n={m.sum():5d}  actual RMSE {np.sqrt((err_te[m]**2).mean()):.3f}")

# 2. calibration scale (are raw stds the right size?): observed vs predicted error
cal_scale = err_cal.mean() / (sd_cal.mean() * np.sqrt(2 / np.pi))     # ratio of actual to ensemble-implied MAE
print(f"\n2. calibration: raw ensemble sigma is {'UNDER' if cal_scale>1 else 'OVER'}-confident "
      f"(actual/implied error ~ {cal_scale:.2f}x). Recalibration scale s={cal_scale:.2f}.")

# 3. split-conformal intervals @ 90% (guarantee from the calibration set)
alpha = 0.10
def conformal(score_cal, score_te_num, width_unit):
    n = len(score_cal); q = np.quantile(score_cal, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n))
    cover = np.mean(score_te_num <= q); width = 2 * q * np.mean(width_unit)
    return q, cover, width
# standard (fixed width)
q_s, cov_s, w_s = conformal(err_cal, err_te, np.ones_like(sd_te))
# normalized / adaptive (width scales with sigma)
q_a, cov_a, w_a = conformal(err_cal / sd_cal, err_te / sd_te, sd_te)
print(f"\n3. split-conformal 90% prediction intervals (empirical coverage should be ~0.90):")
print(f"   standard  : coverage {cov_s:.2f}   mean width {w_s:.2f} logS  (+/-{q_s:.2f})")
print(f"   adaptive  : coverage {cov_a:.2f}   mean width {w_a:.2f} logS  (scales with sigma)")


# figure: reliability diagram
plt.figure(figsize=(7, 4.6))
cent = [(qs[i] + qs[i + 1]) / 2 for i in range(5)]
rmse_bin = [np.sqrt((err_te[(sd_te >= qs[i]) & (sd_te <= qs[i + 1])] ** 2).mean()) for i in range(5)]
plt.plot(cent, rmse_bin, "o-", color="#1f6f4a", lw=2, label="observed RMSE per sigma-bin")
plt.plot([min(cent), max(cent)], [min(cent) * cal_scale * np.sqrt(2), max(cent) * cal_scale * np.sqrt(2)],
         "k--", alpha=0.5, label="calibrated ideal")
plt.xlabel("ensemble sigma (predicted uncertainty)"); plt.ylabel("actual RMSE")
plt.title(f"Ensemble uncertainty vs. prediction error (Spearman {sr.statistic:+.2f})")
plt.legend(); plt.tight_layout(); plt.savefig("figures/uncertainty_reliability.png", dpi=150)
print("\nsaved figures/uncertainty_reliability.png")
