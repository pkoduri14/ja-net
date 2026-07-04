"""Interpretability: does JA-NET recover the Jouyban-Acree coefficients from structure?
Compares structure-predicted C0,C1,C2 to the classical JA fit on each held-out system's own data."""
import sys, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.stdout.reconfigure(line_buffering=True)
from sklearn.model_selection import GroupShuffleSplit
from scipy.stats import pearsonr, spearmanr
import torch, janet as J

D = J.load_features()
idx = np.arange(len(D.y))

# leave-system-out: train on train systems, analyze HELD-OUT systems
train, test = next(GroupShuffleSplit(1, test_size=0.25, random_state=0).split(idx, groups=D.sysid))
print("training JA-NET (held-out systems will be analyzed) ...")
J.train_ja(D, train, train[:1], seed=0, epochs=45, save="models/ja_interp.pt")
net = J.JANet(D.fdim, D.tdim); net.load_state_dict(torch.load("models/ja_interp.pt")["state"]); net.eval()
t = torch.tensor

# per held-out system: data-fit J0,J1,J2 (keep only well-conditioned fits) ; model-predict C0,C1,C2
Jf, Cp, names, dropped = [], [], [], 0
for s in set(D.sysid[test]):
    m = np.where(D.sysid == s)[0]
    x1, x2 = D.x1[m], D.x2[m]
    if len(m) < 10 or x1.max() - x1.min() < 0.5 or len(np.unique(np.round(x1, 2))) < 5:
        continue
    d = x1 - x2
    A = np.stack([x1, x2, x1 * x2, x1 * x2 * d, x1 * x2 * d * d], 1)   # [a1,a2,J0,J1,J2]
    coef, *_ = np.linalg.lstsq(A, D.y[m], rcond=None)
    # keep only systems where classical JA actually fits (so J is meaningful) and is stable
    res = D.y[m] - A @ coef; r2 = 1 - (res ** 2).sum() / ((D.y[m] - D.y[m].mean()) ** 2 + 1e-9).sum()
    if r2 < 0.9 or abs(coef[2]) > 8:
        dropped += 1; continue
    i0 = m[0]
    with torch.no_grad():
        _, _, _, C = net(t(D.SOLn[i0][None]), t(D.V1n[i0][None]), t(D.V2n[i0][None]),
                         t(D.Tfn[i0][None]), t(D.x1[i0][None]), t(D.x2[i0][None]))
    Jf.append(coef[2:5]); Cp.append(C.numpy()[0]); names.append(s)
Jf, Cp = np.array(Jf), np.array(Cp)
print(f"\n{len(Jf)} held-out systems with stable JA fits (R2>0.9); dropped {dropped} ill-conditioned")

print("\nstructure-predicted C  vs  data-fitted J (held-out systems):")
for i, nm in enumerate(["C0 (synergy/height)", "C1 (skew)", "C2 (curvature)"]):
    r, p = pearsonr(Jf[:, i], Cp[:, i]); rs, _ = spearmanr(Jf[:, i], Cp[:, i])
    print(f"  {nm:22s} pearson r={r:+.3f} (p={p:.1e})  spearman={rs:+.3f}")
agree = np.mean(np.sign(Jf[:, 0]) == np.sign(Cp[:, 0]))
print(f"  C0 sign agreement (synergistic vs antagonistic): {agree:.1%}")

# figure: predicted C0 vs fitted J0
plt.figure(figsize=(6, 6))
plt.scatter(Jf[:, 0], Cp[:, 0], alpha=0.45, edgecolor="none")
lo, hi = np.percentile(Jf[:, 0], [1, 99]); lo = min(lo, -1); hi = max(hi, 4)
plt.plot([lo, hi], [lo, hi], "k--", alpha=0.4, label="y=x"); plt.xlim(lo, hi); plt.ylim(lo, hi)
plt.axhline(0, c="gray", lw=0.6); plt.axvline(0, c="gray", lw=0.6)
r0 = pearsonr(Jf[:, 0], Cp[:, 0])[0]
plt.xlabel("classical JA  J0  (fit to each system's data)")
plt.ylabel("JA-NET  C0  (predicted from structure)")
plt.title(f"Structure-predicted vs. fitted Jouyban-Acree coefficient C0 (r={r0:.2f})")
plt.legend(); plt.tight_layout(); plt.savefig("figures/interpretability_C0.png", dpi=150)
print("\nsaved figures/interpretability_C0.png")
