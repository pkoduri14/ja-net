"""JA-NET: a Jouyban-Acree-structured neural network for mixed-solvent solubility from molecular structure."""
from .model import (
    load_features, JANet, ConcatMLP, train_ja, train_concat, evaluate,
    demo_cosolvency_curve, _feat, _val_split, C_SCALE,
)

__all__ = ["load_features", "JANet", "ConcatMLP", "train_ja", "train_concat",
           "evaluate", "demo_cosolvency_curve", "_feat", "_val_split", "C_SCALE"]
__version__ = "0.1.0"
