"""Data-leakage / duplication / featurization audit (standard data-leakage checks).
Solute leakage across splits, near-duplicate Tanimoto, order/salt duplicates, and 128-bit Morgan collisions."""
import sys, collections, numpy as np
sys.stdout.reconfigure(line_buffering=True)
from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
from sklearn.model_selection import GroupShuffleSplit
import janet as J

D = J.load_features()
idx = np.arange(len(D.y))
first = {s: np.where(D.sysid == s)[0][0] for s in set(D.sysid)}   # a representative row per system

# ---- A. leave-system-out solute leakage ----
tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=0).split(idx, groups=D.sysid))
tr_sol = set(D.soluteid[tr]); te_sys = set(D.sysid[te])
sys_sol_in_tr = [D.soluteid[first[s]] in tr_sol for s in te_sys]
row_sol_in_tr = np.mean([D.soluteid[i] in tr_sol for i in te])
print("== A. leave-BINARY-SYSTEM-out: is it really solute generalization? ==")
print(f"  test systems: {len(te_sys)} | whose SOLUTE also appears in train: {np.mean(sys_sol_in_tr):.0%}")
print(f"  test ROWS whose solute is seen in train: {row_sol_in_tr:.0%}")


# ---- B. leave-solute-out near-duplicate leakage (Tanimoto ECFP4/2048) ----
uniq_sol = sorted(set(D.soluteid))
fps = {}
for s in uniq_sol:
    m = Chem.MolFromSmiles(s)
    fps[s] = rdMolDescriptors.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) if m else None
# real leave-solute-out split on rows, then take unique held-out solutes
trs, tes = next(GroupShuffleSplit(1, test_size=0.2, random_state=0).split(idx, groups=D.soluteid))
train_sol = sorted(set(D.soluteid[trs])); test_sol = sorted(set(D.soluteid[tes]))
tr_fps = [fps[s] for s in train_sol if fps[s] is not None]
maxsim = []
for s in test_sol:
    if fps[s] is None: continue
    sims = DataStructs.BulkTanimotoSimilarity(fps[s], tr_fps)
    maxsim.append(max(sims))
maxsim = np.array(maxsim)
print("== B. leave-SOLUTE-out: are held-out solutes genuinely novel? (max Tanimoto to any train solute) ==")
print(f"  held-out solutes: {len(maxsim)} | median max-Tanimoto to train: {np.median(maxsim):.2f}")
for thr in (0.9, 0.7, 0.5, 0.4):
    print(f"    with a train neighbor >= {thr:.1f} Tanimoto: {np.mean(maxsim >= thr):.0%}")


# ---- C. duplicate / salt systems ----
def unordered(s):
    a = s.split("|"); return a[0] + "||" + "++".join(sorted(a[1:]))
order_dups = sum(v - 1 for v in collections.Counter(unordered(s) for s in set(D.sysid)).values() if v > 1)
rem = SaltRemover()
def parent(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None: return smi
    frags = Chem.GetMolFrags(m, asMols=True)
    if not frags: return smi
    big = max(frags, key=lambda x: x.GetNumHeavyAtoms())
    return Chem.MolToSmiles(big)
par = {s: parent(s) for s in uniq_sol}
salt_groups = collections.Counter(par.values())
salt_dups = sum(v - 1 for v in salt_groups.values() if v > 1)
print("== C. duplicate systems ==")
print(f"  systems that are the same solute + UNORDERED solvent pair as another (order duplicates): {order_dups}")
print(f"  distinct solute SMILES: {len(uniq_sol)} | that share a parent (salt/free-base) with another: {salt_dups}")


# ---- D. 128-bit Morgan collisions ----
fp128 = {}
coll = collections.defaultdict(set)
for s in uniq_sol:
    m = Chem.MolFromSmiles(s)
    if m is None: continue
    arr = np.zeros(128, np.int8)
    DataStructs.ConvertToNumpyArray(rdMolDescriptors.GetMorganFingerprintAsBitVect(m, 2, nBits=128), arr)
    coll[arr.tobytes()].add(s)
ncoll = sum(len(v) for v in coll.values() if len(v) > 1)
print("== D. featurization: 128-bit Morgan collisions ==")
print(f"  distinct solutes: {len(uniq_sol)} | sharing an IDENTICAL 128-bit Morgan fp with another solute: {ncoll} "
      f"({ncoll/len(uniq_sol):.0%})")

