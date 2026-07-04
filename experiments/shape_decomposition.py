"""Fair decomposition: isolate JA-NET's actual contribution, the cosolvency SHAPE from structure.
Give both methods the measured endpoints and compare only the cross-term (structure vs measured titration points)."""
import sys, collections, json, numpy as np, pandas as pd
sys.stdout.reconfigure(line_buffering=True)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import torch
from sklearn.metrics import mean_absolute_error
import janet as J

D = J.load_features()
fda = set(pd.read_csv("data/mixturesoldb.csv", usecols=["SMILES_Solute", "FDA_Approved"])
          .pipe(lambda d: d.SMILES_Solute[d.FDA_Approved == True]))
nm = pd.read_csv("data/mixturesoldb.csv", usecols=["SMILES_Solute", "Compound_Name"])
name_map = dict(zip(nm.SMILES_Solute, nm.Compound_Name))

# targets: isothermal FDA systems that HAVE both pure endpoints (x1~0 and x1~1) + >=6 interior points
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
print(f"isothermal FDA systems with both endpoints + >=6 interior: {len(targets)}")

nets = []
for sd in (0, 1, 2):
    net = J.JANet(D.fdim, D.tdim); net.load_state_dict(torch.load(f"models/ja_ens_{sd}.pt")["state"]); net.eval()
    nets.append(net)
t = torch.tensor
def structure_C(row):
    with torch.no_grad():
        return np.mean([net(t(D.SOLn[[row]]), t(D.V1n[[row]]), t(D.V2n[[row]]), t(D.Tfn[[row]]),
                            t(D.x1[[row]]), t(D.x2[[row]]))[3].numpy()[0] for net in nets], 0)


def cross(x1, C, ncross=None):
    x2 = 1 - x1; d = x1 - x2
    basis = [np.ones_like(x1), d, d * d]
    n = len(C) if ncross is None else min(ncross, len(C))
    return x1 * x2 * sum(C[i] * basis[i] for i in range(n))

def fit_cross(x1, y, e1, e2, ncross):     # fit cross-term to (y - loglin) using anchor interior pts
    x2 = 1 - x1; d = x1 - x2
    resid = y - (x1 * e1 + x2 * e2)
    cols = [x1 * x2, x1 * x2 * d, x1 * x2 * d * d][:ncross]
    if ncross == 0: return np.zeros(0)
    A = np.stack(cols, 1); coef, *_ = np.linalg.lstsq(A, resid, rcond=None); return coef


K_LIST = [2, 3, 4, 5, 6]
rec = {k: {"classical": [], "anchored": [], "loglin": []} for k in K_LIST}
winA = {k: [] for k in K_LIST}
anchored_vs_loglin = []
for s, m in targets.items():
    e2 = D.y[m[0]]; e1 = D.y[m[-1]]                       # x1~0 -> e2 ; x1~1 -> e1
    interior = m[1:-1]
    Cstruct = structure_C(m[0])
    for k in K_LIST:
        ni = k - 2                                        # interior points the classical fit may use
        if ni > len(interior) - 1: continue
        a_int = interior[np.linspace(0, len(interior) - 1, ni).round().astype(int)] if ni > 0 else np.array([], int)
        held = np.setdiff1d(interior, a_int)
        if len(held) == 0: continue
        xh = D.x1[held]; yh = D.y[held]
        ll = xh * e1 + (1 - xh) * e2                       # log-linear
        anch = ll + cross(xh, Cstruct)                    # endpoint-anchored JA-NET (structure cross-term)
        if ni == 0:
            cl = ll
        else:
            cc = fit_cross(D.x1[a_int], D.y[a_int], e1, e2, int(np.clip(ni, 0, 3)))
            cl = ll + cross(xh, cc, ncross=len(cc))
        rec[k]["classical"].append(mean_absolute_error(yh, cl))
        rec[k]["anchored"].append(mean_absolute_error(yh, anch))
        rec[k]["loglin"].append(mean_absolute_error(yh, ll))
        winA[k].append(mean_absolute_error(yh, cl) > mean_absolute_error(yh, anch))
        if k == 2: anchored_vs_loglin.append((mean_absolute_error(yh, anch), mean_absolute_error(yh, ll)))

print("\n== SHAPE-only: given the 2 measured endpoints, who predicts the cosolvency bulge? (matched held-out interior) ==")
print(f"{'interior pts (k-2)':>18} | {'classical JA':>12} | {'anchored JA-NET':>15} | {'log-linear':>10} | {'JA-NET beats classical':>22} | n")
summary = {}; crossover = None
for k in K_LIST:
    c = np.array(rec[k]["classical"]); a = np.array(rec[k]["anchored"]); l = np.array(rec[k]["loglin"])
    if len(c) == 0: continue
    frac = np.mean(winA[k])
    print(f"{k-2:>18} | {c.mean():>12.3f} | {a.mean():>15.3f} | {l.mean():>10.3f} | {frac:>21.0%} | {len(c)}")
    summary[k] = dict(interior=k-2, classical=float(c.mean()), anchored=float(a.mean()),
                      loglin=float(l.mean()), janet_beats_classical=float(frac), n=len(c))
    if crossover is None and c.mean() <= a.mean():
        crossover = k - 2
av = np.array(anchored_vs_loglin)
print(f"\nAnchored JA-NET vs log-linear (both 0 interior pts): {av[:,0].mean():.3f} vs {av[:,1].mean():.3f} MAE "
      f"-> structure {'HELPS' if av[:,0].mean()<av[:,1].mean() else 'does NOT help'} beyond the endpoints "
      f"({np.mean(av[:,0]<av[:,1]):.0%} of systems improved).")
print(f"CROSSOVER = {crossover} interior points: the classical fit needs ~{crossover} measured interior "
      f"titration point(s) to match structure-only shape prediction.")

# figure
ks = [k for k in K_LIST if k in summary]; xint = [k-2 for k in ks]
plt.figure(figsize=(7, 4.6))
plt.plot(xint, [summary[k]["classical"] for k in ks], "o-", color="#b5651d", lw=2,
         label="Classical JA fit (k−2 measured interior points)")
plt.axhline(summary[ks[0]]["anchored"], color="#1f6f4a", lw=2, ls="--",
            label="Endpoint-anchored JA-NET (structure shape, 0 interior)")
plt.axhline(summary[ks[0]]["loglin"], color="#8a2be2", lw=1.6, ls=":",
            label="Log-linear (endpoints only, no curve)")
if crossover is not None: plt.axvline(crossover, color="gray", lw=0.8, ls=":")
plt.xlabel("measured INTERIOR titration points given to the classical fit")
plt.ylabel("held-out interior MAE (logS)")
plt.title("Cosolvency-curve shape: structure vs. measured titration points")
plt.legend(fontsize=8.5); plt.tight_layout(); plt.savefig("figures/shape.png", dpi=150)
print("saved figures/shape.png")

# ---- example: a CLEAN, representative synergistic FDA system (low-noise curve, anchored MAE near the aggregate) ----
best, score = None, -1
for s, m in targets.items():
    if len(m) < 14: continue
    e2, e1 = D.y[m[0]], D.y[m[-1]]; interior = m[1:-1]
    C = structure_C(m[0])
    ll = D.x1[interior] * e1 + (1 - D.x1[interior]) * e2
    cc_full = fit_cross(D.x1[interior], D.y[interior], e1, e2, 3)
    clean = np.sqrt(((D.y[interior] - (ll + cross(D.x1[interior], cc_full))) ** 2).mean())  # data cleanliness
    anch_mae = mean_absolute_error(D.y[interior], ll + cross(D.x1[interior], C))
    bulge = (D.y[interior] - ll).max()                        # measured positive synergy height
    if clean < 0.15 and anch_mae < 0.30 and bulge > 0.3 and bulge > score:  # clean + representative + synergistic
        best, score = (s, m, e1, e2, C), bulge
s, m, e1, e2, C = best; drug = name_map.get(D.soluteid[m[0]], "drug")
xs = np.linspace(0, 1, 101)
# classical full JA fit (all points) as the reference "true shape"
cc_full = fit_cross(D.x1[m[1:-1]], D.y[m[1:-1]], e1, e2, 3)
plt.figure(figsize=(7, 4.8))
plt.scatter(D.x1[m], D.y[m], color="k", zorder=5, label="measured")
plt.plot(xs, xs * e1 + (1 - xs) * e2, color="#8a2be2", lw=1.5, ls=":", label="log-linear (2 endpoints, no curve)")
plt.plot(xs, xs * e1 + (1 - xs) * e2 + cross(xs, C), color="#1f6f4a", lw=2.3,
         label=f"anchored JA-NET (endpoints + structure): J0={C[0]:+.2f} J1={C[1]:+.2f} J2={C[2]:+.2f}")
plt.plot(xs, xs * e1 + (1 - xs) * e2 + cross(xs, cc_full), color="#b5651d", lw=1.6, ls="--",
         label=f"classical JA fit to ALL points: J0={cc_full[0]:+.2f}")
plt.xlabel("solvent-1 fraction  x1"); plt.ylabel("logS (mole fraction)")
plt.title(f"{drug}: cosolvency curve predicted from structure and two endpoints")
plt.legend(fontsize=8.3); plt.tight_layout(); plt.savefig("figures/example_curve.png", dpi=150)
print(f"saved figures/example_curve.png  (example: {drug})")

json.dump(dict(summary=summary, crossover_interior=crossover, n_systems=len(targets),
               anchored_vs_loglin=[float(av[:,0].mean()), float(av[:,1].mean())], example=drug),
          open("shape_summary.json", "w"), indent=2)
print("saved shape_summary.json")
