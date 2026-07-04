"""Head-to-head: the data-cost of the classical Jouyban-Acree route vs a SMILES-only model.
On FDA-drug cosolvency curves: how many measurements the classical fit needs to match zero-measurement prediction."""
import sys, collections, json, numpy as np, pandas as pd
sys.stdout.reconfigure(line_buffering=True)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import torch
from sklearn.metrics import mean_absolute_error
import janet as J

D = J.load_features()
fda = set(pd.read_csv("data/mixturesoldb.csv", usecols=["SMILES_Solute", "FDA_Approved"])
          .pipe(lambda d: d.SMILES_Solute[d.FDA_Approved == True]))
print(f"{len(D.y)} rows | {len(fda)} FDA solutes")

# ---- isothermal FDA target systems (modal T, curve-bearing, >=10 pts so held-out survives to k=8) ----
by = collections.defaultdict(list)
for i in range(len(D.sysid)): by[D.sysid[i]].append(i)
targets = {}
for s, idx in by.items():
    idx = np.array(idx)
    if D.soluteid[idx[0]] not in fda: continue
    modal = collections.Counter(np.round(D.T[idx], 1)).most_common(1)[0][0]
    m = idx[np.round(D.T[idx], 1) == modal]
    xx = D.x1[m]
    if len(m) >= 10 and xx.max() - xx.min() >= 0.6 and len(np.unique(np.round(xx, 2))) >= 6:
        targets[s] = m[np.argsort(xx)]                       # sorted by composition
target_sys = set(targets)
print(f"isothermal FDA target systems (n>=10): {len(target_sys)}")

# ---- train JA-NET leave-ALL-targets-out (system-disjoint), 3-seed ensemble ----
tr = np.array([i for i in range(len(D.y)) if D.sysid[i] not in target_sys])
print(f"training JA-NET on {len(tr)} rows (all target systems withheld) ...")
seeds = (0, 1, 2)
nets = []
for sd in seeds:
    J.train_ja(D, tr, tr[:1], seed=sd, epochs=45, save=f"models/ja_ens_{sd}.pt", vgroups=D.sysid)
    net = J.JANet(D.fdim, D.tdim); net.load_state_dict(torch.load(f"models/ja_ens_{sd}.pt")["state"]); net.eval()
    nets.append(net)
t = torch.tensor
def janet_pred(rows):
    with torch.no_grad():
        ps = [net(t(D.SOLn[rows]), t(D.V1n[rows]), t(D.V2n[rows]), t(D.Tfn[rows]),
                  t(D.x1[rows]), t(D.x2[rows]))[0].numpy() for net in nets]
    return np.mean(ps, 0)
def janet_coeffs(row):                                       # the recovered J0,J1,J2 for one system
    with torch.no_grad():
        Cs = [net(t(D.SOLn[[row]]), t(D.V1n[[row]]), t(D.V2n[[row]]), t(D.Tfn[[row]]),
                  t(D.x1[[row]]), t(D.x2[[row]]))[3].numpy()[0] for net in nets]
    return np.mean(Cs, 0)


def fit_ja(x1, y, ncross):                                   # classical least-squares JA (reduced by ncross)
    x2 = 1 - x1; d = x1 - x2
    cols = [x1, x2] + [x1 * x2, x1 * x2 * d, x1 * x2 * d * d][:ncross]
    A = np.stack(cols, 1); coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coef, cols_fn(ncross)
def cols_fn(ncross):
    def pred(x1, coef):
        x2 = 1 - x1; d = x1 - x2
        cols = [x1, x2] + [x1 * x2, x1 * x2 * d, x1 * x2 * d * d][:ncross]
        return np.stack(cols, 1) @ coef
    return pred
def anchors(m, k):                                           # k points: both endpoints + evenly-spread interior
    if k >= len(m): return m.copy()
    int_idx = np.linspace(1, len(m) - 2, k - 2).round().astype(int)
    return m[np.concatenate([[0], np.unique(int_idx), [len(m) - 1]])][:k]


# ---- data-cost sweep ----
K_LIST = [2, 3, 4, 5, 6, 8]
rec = {k: {"classical": [], "janet": []} for k in K_LIST}
win0 = {k: [] for k in K_LIST}                                # does JA-NET(0 pts) beat classical(k pts)?
for s, m in targets.items():
    x1m, ym = D.x1[m], D.y[m]
    for k in K_LIST:
        if k > len(m) - 1: continue
        a = anchors(m, k); held = np.setdiff1d(m, a)
        if len(held) == 0: continue
        ncross = int(np.clip(k - 2, 0, 3))
        coef, pred = fit_ja(D.x1[a], D.y[a], ncross)
        c_mae = mean_absolute_error(D.y[held], pred(D.x1[held], coef))
        j_mae = mean_absolute_error(D.y[held], janet_pred(held))
        rec[k]["classical"].append(c_mae); rec[k]["janet"].append(j_mae)
        win0[k].append(c_mae > j_mae)

print("\n===== DATA-COST: classical Jouyban-Acree fit(k) vs SMILES-only JA-NET(0 pts), matched held-out =====")
print(f"{'k (measured pts)':>16} | {'classical JA MAE':>16} | {'JA-NET(0) MAE':>13} | {'JA-NET better on':>16} | n_sys")
summary = {}
crossover = None
for k in K_LIST:
    c = np.array(rec[k]["classical"]); j = np.array(rec[k]["janet"])
    if len(c) == 0: continue
    frac = np.mean(win0[k])
    tag = "log-linear (Yalkowsky)" if k == 2 else ("full JA" if k >= 5 else f"JA+{k-2} cross")
    print(f"{k:>16} | {c.mean():>16.3f} | {j.mean():>13.3f} | {frac:>15.0%} | {len(c)}   [{tag}]")
    summary[k] = dict(classical=float(c.mean()), janet=float(j.mean()), janet_better_frac=float(frac), n=len(c))
    if crossover is None and c.mean() <= j.mean():
        crossover = k
print(f"\nCROSSOVER k* = {crossover}  (fewest measured points at which the classical JA fit matches the "
      f"zero-measurement SMILES model). Below k*, structure-only wins.")

# ---- aggregate figure ----
ks = [k for k in K_LIST if k in summary]
cvals = [summary[k]["classical"] for k in ks]; jval = summary[ks[0]]["janet"]
jline = np.mean([summary[k]["janet"] for k in ks])
plt.figure(figsize=(7, 4.6))
plt.plot(ks, cvals, "o-", color="#b5651d", lw=2, label="Classical Jouyban–Acree fit (k measured points)")
plt.axhline(jline, color="#1f6f4a", lw=2, ls="--", label="JA-NET — SMILES only (0 measured points)")
plt.scatter([2], [summary[2]["classical"]], color="#8a2be2", zorder=5,
            label="Yalkowsky log-linear (2 endpoints)")
if crossover: plt.axvline(crossover, color="gray", lw=0.8, ls=":")
plt.xlabel("measured points per system available to the classical fit  (k)")
plt.ylabel("prediction MAE on held-out points  (logS)")
plt.title("Measurements required for the classical fit to match a structure-only prediction")
plt.legend(fontsize=8.5); plt.tight_layout(); plt.savefig("figures/datacost.png", dpi=150)
print("saved figures/datacost.png")

# ---- example curve+coefficients figure: a well-measured synergistic FDA system ----
name_map = dict(zip(pd.read_csv("data/mixturesoldb.csv", usecols=["SMILES_Solute", "Compound_Name"])
                    .SMILES_Solute, pd.read_csv("data/mixturesoldb.csv",
                    usecols=["SMILES_Solute", "Compound_Name"]).Compound_Name))
best, bestscore = None, -1
for s, m in targets.items():
    if len(m) < 14: continue
    coef, _ = fit_ja(D.x1[m], D.y[m], 3)
    span = D.y[m].max() - D.y[m].min()
    if coef[2] > 0.5 and span * abs(coef[2]) > bestscore:     # positive synergy, well-resolved
        best, bestscore = (s, m), span * abs(coef[2])
s, m = best; drug = name_map.get(D.soluteid[m[0]], "drug")
xs = np.linspace(0, 1, 101)
i0 = m[0]
with torch.no_grad():
    rep = lambda a: np.repeat(a[None], len(xs), 0)
    jc = np.mean([net(t(rep(D.SOLn[i0])), t(rep(D.V1n[i0])), t(rep(D.V2n[i0])),
                      t(rep(D.Tfn[i0])), t(xs.astype("float32")), t((1 - xs).astype("float32")))[0].numpy()
                  for net in nets], 0)
Jc = janet_coeffs(i0)
cc, predc = fit_ja(D.x1[m], D.y[m], 3); ll_e = D.y[m[[0, -1]]]
plt.figure(figsize=(7, 4.8))
plt.scatter(D.x1[m], D.y[m], color="k", zorder=5, label="measured (this system)")
plt.plot(xs, jc, color="#1f6f4a", lw=2.2, label=f"JA-NET (SMILES only): J0={Jc[0]:+.2f}, J1={Jc[1]:+.2f}, J2={Jc[2]:+.2f}")
plt.plot(xs, predc(xs, cc), color="#b5651d", lw=1.8, ls="-", label=f"classical JA fit: J0={cc[2]:+.2f}")
plt.plot(xs, xs * ll_e[-1] + (1 - xs) * ll_e[0], color="#8a2be2", lw=1.4, ls=":", label="log-linear (2 endpoints)")
plt.xlabel("solvent-1 fraction  x1"); plt.ylabel("logS (mole fraction)")
plt.title(f"{drug}: cosolvency curve predicted from structure and two endpoints")
plt.legend(fontsize=8.5); plt.tight_layout(); plt.savefig("figures/datacost_curve.png", dpi=150)
print(f"saved figures/datacost_curve.png  (example system: {drug})")

json.dump(dict(summary=summary, crossover=crossover, n_target_systems=len(targets),
               example=drug), open("datacost_summary.json", "w"), indent=2)
print("saved datacost_summary.json")
