# -*- coding: utf-8 -*-
"""
5-Fold 交叉验证指标计算与统计对比报表生成模块
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
)
from compare.utils.data_loader import extract_oof_and_fold_data
from compare.utils.stat_tests import (
    delong_roc_test,
    mcnemar_test_paired,
    paired_bootstrap_test,
    format_pval,
)


def evaluate_model_5fold(pkl_path, fixed_threshold=0.25):
    """计算单个模型的 5 折指标及汇总 OOF 预测"""
    parsed = extract_oof_and_fold_data(
        pkl_path,  fixed_threshold=fixed_threshold
    )

    fold_metrics = {
        'Thresh': [], 'AUC-ROC': [], 'AU-PRC': [],
        'F1': [], 'Specificity': [], 'Sensitivity': [], 'ACC': []
    }

    for fold_idx, (y_true, y_prob, y_pred, thresh) in parsed['aligned_folds'].items():
        auc_roc = roc_auc_score(y_true, y_prob)
        au_prc = average_precision_score(y_true, y_prob)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        acc = accuracy_score(y_true, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        fold_metrics['Thresh'].append(thresh)
        fold_metrics['AUC-ROC'].append(auc_roc)
        fold_metrics['AU-PRC'].append(au_prc)
        fold_metrics['F1'].append(f1)
        fold_metrics['Specificity'].append(spec)
        fold_metrics['Sensitivity'].append(sens)
        fold_metrics['ACC'].append(acc)

    return {
        'fold_metrics': fold_metrics,
        'oof_labels': parsed['oof_labels'],
        'oof_probs': parsed['oof_probs'],
        'oof_preds': parsed['oof_preds'],
        'oof_ids': parsed['oof_ids'],
    }


def generate_comparison_tables(
    model_files,
    target_model='Ours (MedicalGMU)',
    fixed_threshold=0.25,
    n_bootstraps=2000,
    seed=42,
    save_dir=None,
):
    """生成多模型 5-Fold 均值±标准差汇总表与配对显著性 p 值表"""
    parsed_models = {
        name: evaluate_model_5fold(path, fixed_threshold=fixed_threshold)
        for name, path in model_files.items()
    }

    ref = parsed_models[target_model]
    y_true_ref = ref['oof_labels']

    summary_rows = []
    pvalue_rows = []
    metric_names = ['Thresh', 'AUC-ROC', 'AU-PRC', 'F1', 'Specificity', 'Sensitivity', 'ACC']

    for m_name in model_files.keys():
        curr = parsed_models[m_name]
        fm = curr['fold_metrics']

        assert np.array_equal(y_true_ref, curr['oof_labels']), (
            f"[错误] 模型 {m_name} 与 {target_model} 的样本标签顺序不一致，无法直接进行配对检验！"
        )

        # 1. 组装 5-Fold Mean ± Std
        row_metric = {'Model': m_name}
        for k in metric_names:
            mean_val = np.mean(fm[k])
            std_val = np.std(fm[k])
            if k == 'Thresh':
                row_metric[k] = f"{mean_val:.2f}"
            else:
                row_metric[k] = f"{mean_val:.3f} ± {std_val:.3f}"
        summary_rows.append(row_metric)

        # 2. 计算与基准模型的配对 p 值
        if m_name == target_model:
            pvalue_rows.append({
                'Model': m_name,
                'AUC-ROC p-val': '-', 'AU-PRC p-val': '-', 'F1 p-val': '-',
                'Spec p-val': '-', 'Sens p-val': '-', 'ACC p-val': '-'
            })
        else:
            p_auc = delong_roc_test(y_true_ref, ref['oof_probs'], curr['oof_probs'])
            p_prc = paired_bootstrap_test(
                y_true_ref, ref['oof_probs'], curr['oof_probs'],
                metric_fn=average_precision_score, n_bootstraps=n_bootstraps, seed=seed
            )
            p_f1 = paired_bootstrap_test(
                y_true_ref, ref['oof_preds'], curr['oof_preds'],
                metric_fn=lambda y, p: f1_score(y, p, zero_division=0), n_bootstraps=n_bootstraps, seed=seed
            )
            p_spec = mcnemar_test_paired(y_true_ref, ref['oof_preds'], curr['oof_preds'], subset_label=0)
            p_sens = mcnemar_test_paired(y_true_ref, ref['oof_preds'], curr['oof_preds'], subset_label=1)
            p_acc  = mcnemar_test_paired(y_true_ref, ref['oof_preds'], curr['oof_preds'], subset_label=None)

            pvalue_rows.append({
                'Model': m_name,
                'AUC-ROC p-val': format_pval(p_auc),
                'AU-PRC p-val':  format_pval(p_prc),
                'F1 p-val':      format_pval(p_f1),
                'Spec p-val':    format_pval(p_spec),
                'Sens p-val':    format_pval(p_sens),
                'ACC p-val':     format_pval(p_acc)
            })

    df_metrics = pd.DataFrame(summary_rows)
    df_pvals = pd.DataFrame(pvalue_rows)

    print("\n" + "#" * 105)
    print(df_metrics.to_string(index=False))
    print("-" * 105)
    print(df_pvals.to_string(index=False))

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        df_metrics.to_csv(os.path.join(save_dir, f"metrics.csv"), index=False, encoding='utf-8-sig')
        df_pvals.to_csv(os.path.join(save_dir, f"pvalues.csv"), index=False, encoding='utf-8-sig')

    return df_metrics, df_pvals
