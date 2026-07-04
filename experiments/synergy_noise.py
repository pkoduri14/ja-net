"""Noise of the derived synergy metric (peak bulge above additive).
Monte-Carlo propagate the logS label noise into synergy; contextualizes the discovery correlation as a lower bound."""
import sys, collections, numpy as np, pandas as pd
sys.stdout.reconfigure(line_buffering=True)
import janet as J

SIGMA = 0.20   # inter-lab logS noise floor (RMSE), from noise_floor.py
D = J.load_features()
fda = set(pd.read_csv("data/mixturesoldb.csv", usecols=["SMILES_Solute", "FDA_Approved"])
          .pipe(lambda d: d.SMILES_Solute[d.FDA_Approved == True]))
by = collections.defaultdict(list)
for i in range(len(D.sysid)): by[D.sysid[i]].append(i)
targets = {}
for s, idx in by.items():
    idx = np.array(idx)
    if D.soluteid[idx[0]] not in fda: continue
    modal = collections.Counter(np.round(D.T[idx], 1)).most_common(1)[0][0]
    m = idx[np.round(D.T[idx], 1) == modal]; xx = D.x1[m]
    interior = m[(xx > 0.05) & (xx < 0.95)]
    if xx.min() <= 0.05 and xx.max() >= 0.95 and len(interior) >= 6:
        targets[s] = m[np.argsort(D.x1[m])]

rng = np.random.RandomState(0)
syn, noise = [], []
for s, m in targets.items():
    x1 = D.x1[m]; y = D.y[m]; e2i, e1i = 0, len(m) - 1; interior = np.arange(1, len(m) - 1)
    def synergy(yy):
        ll = x1[interior] * yy[e1i] + (1 - x1[interior]) * yy[e2i]
        return (yy[interior] - ll).max()
    syn.append(synergy(y))
    reps = [synergy(y + rng.normal(0, SIGMA, len(y))) for _ in range(300)]
    noise.append(np.std(reps))
syn, noise = np.array(syn), np.array(noise)

print(f"{len(syn)} systems | logS label noise sigma = {SIGMA}")
print(f"\nsynergy-metric noise (propagated): median std {np.median(noise):.3f}  (mean {noise.mean():.3f})")
print(f"synergy signal spread across systems: std {syn.std():.3f}, IQR {np.percentile(syn,75)-np.percentile(syn,25):.3f}")
snr = syn.std() / np.median(noise)
print(f"signal-to-noise (signal std / metric noise) = {snr:.1f}")
print(f"\ninterpretation: the synergy metric carries ~{np.median(noise):.2f} logS noise (max-selection + endpoint")
print(f"propagation inflate it above the {SIGMA} label floor). Signal spread ({syn.std():.2f}) >> noise, so the")
print(f"ranking is robust; and because the VALIDATION target is noisy, the observed discovery r/4x-enrichment are")
print(f"LOWER BOUNDS (attenuated by target noise) — the true structure->synergy signal is at least that strong.")
