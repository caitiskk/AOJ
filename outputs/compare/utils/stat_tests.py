# -*- coding: utf-8 -*-
"""
统计显著性检验模块：包含 DeLong's Test、配对 McNemar's Test 与 Paired Bootstrap Test
"""
import numpy as np
from scipy import stats


def compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + 1 + j)
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def fast_delong(predictions_sorted_transposed, label_1_count):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])

    total_auc = (np.sum(tz[:, :m], axis=1) - m * (m + 1) / 2) / (m * n)
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m

    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return total_auc, delongcov


def delong_roc_test(ground_truth, preds_A, preds_B):
    """DeLong's Test：检验两个模型在相同样本上的 ROC-AUC 差异双尾 p 值"""
    ground_truth = np.asarray(ground_truth).ravel()
    preds_A = np.asarray(preds_A).ravel()
    preds_B = np.asarray(preds_B).ravel()

    order = np.argsort(ground_truth)[::-1]
    y = ground_truth[order]
    m = int(np.sum(y == 1))

    preds = np.vstack([preds_A[order], preds_B[order]])
    aucs, sigma = fast_delong(preds, m)

    diff = aucs[0] - aucs[1]
    var = sigma[0, 0] + sigma[1, 1] - 2 * sigma[0, 1]
    if var <= 0:
        return 1.0
    z = diff / np.sqrt(var)
    return float(2 * (1 - stats.norm.cdf(np.abs(z))))


def mcnemar_test_paired(y_true, y_pred_A, y_pred_B, subset_label=None):
    """
    配对样本 McNemar 检验：
    - subset_label=1: 检验 Sensitivity 差异
    - subset_label=0: 检验 Specificity 差异
    - subset_label=None: 检验 Overall Accuracy (ACC) 差异
    """
    y_true = np.asarray(y_true).ravel()
    pred_A = np.asarray(y_pred_A).ravel()
    pred_B = np.asarray(y_pred_B).ravel()

    if subset_label is not None:
        mask = (y_true == subset_label)
        y_true = y_true[mask]
        pred_A = pred_A[mask]
        pred_B = pred_B[mask]

    correct_A = (pred_A == y_true)
    correct_B = (pred_B == y_true)

    b = int(np.sum(correct_A & (~correct_B)))
    c = int(np.sum((~correct_A) & correct_B))

    if b + c == 0:
        return 1.0

    if b + c < 25:
        res = stats.binomtest(min(b, c), b + c, 0.5, alternative='two-sided')
        return float(res.pvalue)
    else:
        chi2 = (np.abs(b - c) - 1.0) ** 2 / (b + c)
        return float(1 - stats.chi2.cdf(chi2, df=1))


def paired_bootstrap_test(y_true, vals_A, vals_B, metric_fn, n_bootstraps=2000, seed=42):
    """配对 Bootstrap 检验：用于检验 AU-PRC 和 F1-score 的显著性差异"""
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true).ravel()
    vals_A = np.asarray(vals_A).ravel()
    vals_B = np.asarray(vals_B).ravel()
    N = len(y_true)

    obs_diff = metric_fn(y_true, vals_A) - metric_fn(y_true, vals_B)
    boot_diffs = []

    for _ in range(n_bootstraps):
        idx = rng.randint(0, N, N)
        if len(np.unique(y_true[idx])) < 2:
            continue
        diff = metric_fn(y_true[idx], vals_A[idx]) - metric_fn(y_true[idx], vals_B[idx])
        boot_diffs.append(diff)

    boot_diffs = np.asarray(boot_diffs)
    centered_diffs = boot_diffs - np.mean(boot_diffs)
    return float(np.mean(np.abs(centered_diffs) >= np.abs(obs_diff)))


def format_pval(p):
    """格式化显著性 p 值字符串"""
    if p == "-":
        return "-"
    return f"{p:.4f}" if p >= 0.001 else "<0.001"
