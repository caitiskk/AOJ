# src/chiari/engine/__init__.py
"""训练与评估编排。"""
from chiari.engine.trainer import cross_validation, train_one_fold
from chiari.engine.evaluate import evaluate_joint_level, summarize_folds, METRIC_KEYS

__all__ = ["cross_validation", "train_one_fold",
           "evaluate_joint_level", "summarize_folds", "METRIC_KEYS"]