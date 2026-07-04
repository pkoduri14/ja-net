"""Label-noise floor of MixtureSolDB (the irreducible error no model can beat).
Inter-lab disagreement across replicate measurements of the same solute / solvents / composition / temperature."""
import sys, numpy as np, pandas as pd
sys.stdout.reconfigure(line_buffering=True)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

d = pd.read_csv("data/mixturesoldb.csv", usecols=["SMILES_Solute", "SMILES_Solvent1", "SMILES_Solvent2",
    "Fraction_Solvent1", "Temperature_K", "LogS(mole_fraction)", "Source"]).rename(
    columns={"LogS(mole_fraction)": "y", "Fraction_Solvent1": "x1", "Temperature_K": "TK"})
for c in ("y", "x1", "TK"):
    d[c] = pd.to_numeric(d[c], errors="coerce")
d = d[np.isfinite(d.y) & np.isfinite(d.x1) & np.isfinite(d.TK)].reset_index(drop=True)
print(f"{len(d)} measurements | {d.Source.nunique()} distinct sources\n")

JANET_RMSE, JANET_MAE = 0.528, 0.322


def floor_at(xtol, ttol, drop_exact=True):
    q = d.copy()
    q["k"] = (q.SMILES_Solute + "|" + q.SMILES_Solvent1 + "|" + q.SMILES_Solvent2 + "|"
              + q.x1.round(xtol).astype(str) + "|" + (q.TK / ttol).round().astype(int).astype(str))
    if drop_exact:                                   # drop identical (copied) rows so we measure real disagreement
        q = q.drop_duplicates(subset=["k", "y"])
    g = q.groupby("k")
    n = g.y.transform("size")
    rep = q[n >= 2].copy()
    rep["mu"] = rep.groupby("k").y.transform("mean")
    dev = (rep.y - rep.mu).values                    # deviation of each measurement from its condition-mean
    # inter-source subset: conditions measured by >= 2 distinct sources
    nsrc = rep.groupby("k").Source.transform("nunique")
    inter = rep[nsrc >= 2]
    dev_i = (inter.y - inter.groupby("k").y.transform("mean")).values
    return dict(xtol=xtol, ttol=ttol,
                n_cond=rep.k.nunique(), n_meas=len(rep),
                rmse=np.sqrt(np.mean(dev ** 2)), mae=np.mean(np.abs(dev)),
                n_cond_i=inter.k.nunique(), n_meas_i=len(inter),
                rmse_i=np.sqrt(np.mean(dev_i ** 2)) if len(dev_i) else np.nan,
                mae_i=np.mean(np.abs(dev_i)) if len(dev_i) else np.nan, dev=dev, dev_i=dev_i)


print("=== within-condition disagreement (the noise floor) — sensitivity to matching tolerance ===")
print(f"{'x1 tol':>7} {'T tol(K)':>8} | {'#cond':>6} {'#meas':>7} | {'RMSE':>6} {'MAE':>6} | "
      f"{'inter-source: #cond':>19} {'RMSE':>6} {'MAE':>6}")
runs = []
for xt, tt in [(2, 0.5), (2, 1.0), (3, 0.5), (2, 2.0)]:
    r = floor_at(xt, tt); runs.append(r)
    print(f"{10.0**-xt:>7.2f} {tt:>8.1f} | {r['n_cond']:>6} {r['n_meas']:>7} | {r['rmse']:>6.3f} {r['mae']:>6.3f} | "
          f"{r['n_cond_i']:>19} {r['rmse_i']:>6.3f} {r['mae_i']:>6.3f}")

base = runs[0]
print(f"\n=== verdict (primary tolerance: x1 +/-0.01, T +/-0.5 K) ===")
print(f"  irreducible noise floor:  RMSE ~ {base['rmse']:.3f}   MAE ~ {base['mae']:.3f}   (all replicates)")
print(f"  inter-source:             RMSE ~ {base['rmse_i']:.3f}   MAE ~ {base['mae_i']:.3f}   "
      f"({base['n_cond_i']} conditions across >=2 papers)")
print(f"  JA-NET (leave-system-out): RMSE {JANET_RMSE:.3f}   MAE {JANET_MAE:.3f}")
gap_r = JANET_RMSE / base["rmse_i"] if base["rmse_i"] else np.nan
print(f"  JA-NET RMSE / inter-source noise-floor RMSE = {gap_r:.2f}x")

# figure: distribution of replicate disagreements vs the model error
plt.figure(figsize=(7, 4.6))
plt.hist(np.abs(base["dev_i"]), bins=60, color="#4a7ba6", alpha=0.85, density=True)
plt.axvline(base["mae_i"], color="#1f6f4a", lw=2, label=f"noise-floor MAE {base['mae_i']:.3f}")
plt.axvline(JANET_MAE, color="crimson", lw=2, ls="--", label=f"JA-NET MAE {JANET_MAE:.3f}")
plt.xlabel("|measurement - condition mean|  (logS, inter-source replicates)")
plt.ylabel("density"); plt.xlim(0, 2)
plt.title("Prediction error vs. label-noise floor (MixtureSolDB)")
plt.legend(); plt.tight_layout(); plt.savefig("figures/noise_floor.png", dpi=150)
print("\nsaved figures/noise_floor.png")
