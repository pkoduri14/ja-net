"""Repeated disjoint-split evaluation + paired significance (error bars across data splits (not just seeds)).
5 repeated system/solute-disjoint splits (mean+/-std) with cluster-bootstrap CIs and a Wilcoxon test, JA-NET vs baselines."""
import sys, numpy as np
sys.stdout.reconfigure(line_buffering=True)
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import wilcoxon
import janet as J

D = J.load_features()
print(f"{len(D.y)} rows | {len(set(D.sysid))} systems | {len(set(D.soluteid))} solutes\n")
rmse = lambda a, b: np.sqrt(mean_squared_error(a, b))


def repeated(name, groups, n_splits, seeds=(0, 1, 2)):
    rows, split0 = {}, None
    for rs in range(n_splits):
        tr, te, preds = J.evaluate(D, name, groups, seeds=seeds, rs=rs, quiet=True)
        for nm, p in preds.items():
            rows.setdefault(nm, []).append((mean_absolute_error(D.y[te], p), rmse(D.y[te], p)))
        if rs == 0: split0 = (te, preds, groups[te])
        print(f"  [{name} rs={rs}] " + "  ".join(f"{nm.split()[0]} {mean_absolute_error(D.y[te], p):.3f}"
                                                 for nm, p in preds.items()))
    print(f"\n=== {name}: {n_splits} repeated disjoint splits (mean +/- std across splits) ===")
    for nm, vals in rows.items():
        v = np.array(vals)
        print(f"  {nm:20s} MAE {v[:, 0].mean():.3f} +/- {v[:, 0].std():.3f}   "
              f"RMSE {v[:, 1].mean():.3f} +/- {v[:, 1].std():.3f}")
    print()
    return rows, split0


def significance(te, preds, te_groups, ref="CONCAT-MLP"):
    ja_key = [k for k in preds if k.startswith("JA-NET")][0]
    yte = D.y[te]; uniq = np.unique(te_groups)
    sysMAE = lambda p: np.array([np.abs(p[te_groups == s] - yte[te_groups == s]).mean() for s in uniq])
    scores = {k: sysMAE(p) for k, p in preds.items()}
    a = scores[ja_key]
    print(f"=== paired significance, leave-system-out split rs=0 ({len(uniq)} test systems) ===")
    for bname, b in scores.items():
        if bname == ja_key: continue
        diff = b - a                                   # >0 => JA-NET better on that system
        w = wilcoxon(a, b)
        rng = np.random.RandomState(0); n = len(diff); boot = np.empty(10000)
        for i in range(10000): boot[i] = diff[rng.randint(0, n, n)].mean()
        lo, hi = np.percentile(boot, [2.5, 97.5])
        win = np.mean(diff > 0)
        print(f"  JA-NET vs {bname:12s}: per-system MAE {a.mean():.3f} vs {b.mean():.3f} | "
              f"delta {diff.mean():+.3f} [95%CI {lo:+.3f},{hi:+.3f}] | Wilcoxon p={w.pvalue:.2e} | "
              f"JA-NET better on {win:.0%} of systems")
    print()


if __name__ == "__main__":
    sys_rows, split0 = repeated("LEAVE-SYSTEM-OUT", D.sysid, n_splits=5)
    significance(*split0)
    solute_rows, _ = repeated("LEAVE-SOLUTE-OUT", D.soluteid, n_splits=3)
    print("done.")
