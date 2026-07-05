"""Discovery engine: structure as a cosolvency-synergy hypothesis engine (which systems to measure).
Rank held-out systems by structure-predicted synergy + uncertainty; validate the enrichment against measured synergy."""
import sys, collections, json, numpy as np, pandas as pd
sys.stdout.reconfigure(line_buffering=True)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
import torch
import janet as J

D = J.load_features()
fda = set(pd.read_csv("data/mixturesoldb.csv", usecols=["SMILES_Solute", "FDA_Approved"])
          .pipe(lambda d: d.SMILES_Solute[d.FDA_Approved == True]))
nm = pd.read_csv("data/mixturesoldb.csv", usecols=["SMILES_Solute", "Compound_Name", "Solvent1", "Solvent2"])
name_map = dict(zip(nm.SMILES_Solute, nm.Compound_Name))

# targets: isothermal FDA systems with both endpoints + >=6 interior (so a synergy height is measurable)
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
print(f"target systems (OOF): {len(targets)}")

nets = []
for sd in (0, 1, 2):
    net = J.JANet(D.fdim, D.tdim); net.load_state_dict(torch.load(f"models/ja_ens_{sd}.pt")["state"]); net.eval()
    nets.append(net)
t = torch.tensor
def struct_C(row):                                       # per-model C0 (synergy coeff) from structure
    with torch.no_grad():
        return np.array([net(t(D.SOLn[[row]]), t(D.V1n[[row]]), t(D.V2n[[row]]), t(D.Tfn[[row]]),
                             t(D.x1[[row]]), t(D.x2[[row]]))[3].numpy()[0, 0] for net in nets])

rows = []
for s, m in targets.items():
    e2, e1 = D.y[m[0]], D.y[m[-1]]; interior = m[1:-1]
    loglin = D.x1[interior] * e1 + (1 - D.x1[interior]) * e2
    syn_meas = float((D.y[interior] - loglin).max())     # measured synergy height (peak bulge above additive)
    c0 = struct_C(m[0])
    rows.append(dict(sys=s, drug=name_map.get(D.soluteid[m[0]], "?"),
                     pred=float(c0.mean()), unc=float(c0.std()), meas=syn_meas))
df = pd.DataFrame(rows)

# --- does structure rank true synergy? ---
sr = spearmanr(df.pred, df.meas); pr = pearsonr(df.pred, df.meas)
hi = df.meas >= df.meas.quantile(0.75)                    # "strongly synergistic" = top-quartile measured
base = hi.mean()
print(f"\nstructure-predicted synergy vs MEASURED synergy: Spearman {sr.statistic:+.3f} (p={sr.pvalue:.1e}), "
      f"Pearson {pr.statistic:+.3f}")
print(f"base rate of strongly-synergistic (top-quartile) systems: {base:.0%}\n")

print("=== hypothesis-engine test: rank by structure-predicted synergy, check the measured hit-rate ===")
order = df.sort_values("pred", ascending=False).reset_index(drop=True)
for k in (10, 20, 50):
    p_at_k = order.iloc[:k].meas.ge(df.meas.quantile(0.75)).mean()
    print(f"  precision@{k:<3d} {p_at_k:.0%}   enrichment {p_at_k/base:.2f}x over random")

# confidence gating: restrict to the MOST-CONFIDENT half (low ensemble disagreement), re-rank
conf = df[df.unc <= df.unc.median()].sort_values("pred", ascending=False)
for k in (10, 20):
    p = conf.iloc[:k].meas.ge(df.meas.quantile(0.75)).mean()
    print(f"  precision@{k:<3d} (confident half only) {p:.0%}   enrichment {p/base:.2f}x")

print("\n=== top structure-flagged MEASUREMENT TARGETS (predicted strong synergy, high confidence) ===")
flag = df[df.unc <= df.unc.median()].sort_values("pred", ascending=False).head(12)
for _, r in flag.iterrows():
    a = r.sys.split("|")
    truth = "HIT" if r.meas >= df.meas.quantile(0.75) else ("mid" if r.meas >= df.meas.median() else "low")
    print(f"  predC0={r.pred:+.2f} unc={r.unc:.2f} | measured-synergy={r.meas:.2f} [{truth}] | {r.drug}")

# --- figure ---
plt.figure(figsize=(7, 5))
sc = plt.scatter(df.pred, df.meas, c=df.unc, cmap="viridis_r", s=28, alpha=0.8)
plt.colorbar(sc, label="ensemble uncertainty (low = confident)")
plt.axhline(df.meas.quantile(0.75), color="crimson", lw=0.9, ls="--", label="strong-synergy threshold (top quartile)")
top = df[df.unc <= df.unc.median()].sort_values("pred", ascending=False).head(12)
plt.scatter(top.pred, top.meas, facecolors="none", edgecolors="crimson", s=110, linewidths=1.6,
            label="top-12 confident flags")
plt.xlabel("structure-predicted synergy  (JA-NET C0, out-of-fold)")
plt.ylabel("MEASURED synergy height (bulge above additive)")
plt.title("Structure-predicted vs. measured cosolvency synergy")
plt.legend(fontsize=8); plt.tight_layout(); plt.savefig("figures/discovery_synergy.png", dpi=150)
print("\nsaved figures/discovery_synergy.png")
json.dump(dict(n=len(df), spearman=sr.statistic, base_rate=float(base)), open("discovery_summary.json", "w"), indent=2)
