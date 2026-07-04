# JA-NET: mixed-solvent solubility from molecular structure

A Jouyban-Acree structured neural network that predicts a solute's solubility across a binary-solvent composition curve from molecular structure alone (solute + two solvent SMILES).

The model embeds the Jouyban-Acree cosolvency equation as its output layer, so composition enters only through the physical form:

```
logS = x1*e1 + x2*e2 + x1*x2*(C0 + C1*d + C2*d^2),   d = x1 - x2
```

The pure-solvent endpoints (`e1`, `e2`) and the cosolvency coefficients (`C0, C1, C2`) are predicted from structure; the equation assembles the curve. Because composition enters only through this form, the curve shape transfers to solute-solvent systems that were never measured.

## Installation

```bash
git clone https://github.com/pkoduri14/ja-net.git
cd ja-net
pip install -e .
```

Dependencies are pinned in `requirements.txt` (pip) and `environment.yml` (conda).

## Data

The datasets are not distributed with this repository. See [`data/README.md`](data/README.md) to obtain and place them in `data/`:

- **MixtureSolDB** - training data (~174k binary-solvent solubility measurements)
- **Bao 2024 27k benchmark** - for the apples-to-apples comparison

The first run builds a cached feature file in `data/`.

## Usage

Minimal example (also in [`notebooks/ja_net.ipynb`](notebooks/ja_net.ipynb)):

```python
from janet import load_features, evaluate

D = load_features()
evaluate(D, "LEAVE-BINARY-SYSTEM-OUT", D.sysid)   # JA-NET vs. baselines on unseen solvent systems
```

Each analysis is a script in `experiments/`, run from the repository root:

```bash
python experiments/rigor_eval.py        # repeated disjoint-split evaluation + significance
python experiments/replicate_bao.py     # comparison on the 27k benchmark
python experiments/interpretability.py  # recovering the JA coefficients from structure
python experiments/discovery.py         # ranking systems by predicted cosolvency synergy
```

### Predict on your own systems

```bash
python screen.py candidates.csv
```

`candidates.csv` needs columns `solute_smiles`, `solvent1_smiles`, `solvent2_smiles` (and optionally `name`, `temperature_K`); see `example_candidates.csv`.
The script writes `candidates_predictions.csv`, ranked by predicted cosolvency synergy, with the recovered Jouyban-Acree coefficients and an ensemble-uncertainty estimate. It runs from the shipped model in `models/`, so no training data necessary.

## Results

Leave-binary-system-out (systems held out entirely; 5 repeated disjoint splits):

| Model | MAE | RMSE |
|-------|-----|------|
| JA-NET (3-seed) | 0.32 | 0.53 |
| Gradient boosting | 0.38 | 0.55 |
| Concatenation MLP | 0.41 | 0.61 |

The model's advantage is the cosolvency **curve**: it generalizes to unseen solvent systems for a known solute. For brand-new solute chemistry (leave-solute-out) or whole unseen solvent combinations, gradient boosting is competitive; there the single-solvent endpoints, not the curve shape, are the limiting factor.

## Repository layout

```
janet/          model, featurization, training (importable package)
experiments/    evaluation, ablations, uncertainty, interpretability, discovery
notebooks/      demo notebook
tests/          smoke test
results/        predicted cosolvency-synergy targets
data/           dataset instructions (data not included)
screen.py       predict cosolvency synergy for your own solute-solvent systems
```

## License

MIT (see [`LICENSE`](LICENSE))
