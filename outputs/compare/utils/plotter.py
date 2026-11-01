# -*- coding: utf-8 -*-
"""
可视化绘图模块：支持多模型 5-Fold ROC 曲线、PR 曲线及约登阈值混淆矩阵绘制
"""
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
)
from compare.config import MODEL_COLORS, CLASS_NAMES
from compare.utils.data_loader import to_numpy, load_fold_dict, extract_oof_and_fold_data


def _is_ours_model(model_name):
    return 'MedicalGMU' in model_name or model_name == 'Ours'


def plot_roc_curves_multi(model_files, save_path='roc_multi.svg', model_colors=None, show=False):
    """在同一张图内绘制多个模型的 5-Fold ROC 曲线及阴影置信区间"""
    if model_colors is None:
        model_colors = MODEL_COLORS

    fig, ax = plt.subplots(figsize=(6.2, 5.5))
    mean_fpr = np.linspace(0, 1, 100)
    baseline_linestyles = ['--', '-.', ':']

    for m_idx, (model_name, pkl_path) in enumerate(model_files.items()):
        fold_results = load_fold_dict(pkl_path)
        main_color = model_colors[m_idx % len(model_colors)]

        tprs, fold_aucs = [], []
        for fold_idx in sorted(fold_results.keys()):
            labels, probs, *_ = fold_results[fold_idx]
            y_true = to_numpy(labels).ravel()
            y_prob = to_numpy(probs).ravel()

            fpr, tpr, _ = roc_curve(y_true, y_prob)
            fold_aucs.append(roc_auc_score(y_true, y_prob))

            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = np.mean(fold_aucs)
        std_auc = np.std(fold_aucs)

        if _is_ours_model(model_name):
            linestyle, lw, display_name = '-', 2.4, 'Ours'
        else:
            linestyle = baseline_linestyles[m_idx % len(baseline_linestyles)]
            lw, display_name = 1.8, model_name

        ax.plot(
            mean_fpr, mean_tpr,
            color=main_color, lw=lw, linestyle=linestyle,
            label=f'{display_name} (AUC = {mean_auc:.3f} ± {std_auc:.3f})'
        )

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1.0)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0.0)
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color=main_color, alpha=0.12)

    ax.plot([0, 1], [0, 1], linestyle=':', color='#7F8C8D', lw=1.0)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold', labelpad=8)
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold', labelpad=8)
    ax.tick_params(axis='both', which='major', labelsize=10, width=1.0)
    ax.grid(True, linestyle=':', alpha=0.35)
    ax.legend(loc='lower right', fontsize=8.8, frameon=True, edgecolor='#E0E0E0', facecolor='white')

    plt.tight_layout()
    if os.path.dirname(save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fmt = 'svg' if str(save_path).endswith('.svg') else 'png'
    plt.savefig(save_path, format=fmt, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)
    print(f"Saved multi-model ROC curve to {save_path}")


def plot_pr_curves_multi(model_files, save_path='pr_multi.svg', model_colors=None, show=False):
    """在同一张图内绘制多个模型的 5-Fold PR (Precision-Recall) 曲线"""
    if model_colors is None:
        model_colors = MODEL_COLORS

    fig, ax = plt.subplots(figsize=(6.2, 5.5))
    mean_recall = np.linspace(0, 1, 100)
    baseline_linestyles = ['--', '-.', ':']

    for m_idx, (model_name, pkl_path) in enumerate(model_files.items()):
        fold_results = load_fold_dict(pkl_path)
        main_color = model_colors[m_idx % len(model_colors)]

        precisions, fold_aps = [], []
        for fold_idx in sorted(fold_results.keys()):
            labels, probs, *_ = fold_results[fold_idx]
            y_true = to_numpy(labels).ravel()
            y_prob = to_numpy(probs).ravel()

            prec, rec, _ = precision_recall_curve(y_true, y_prob)
            fold_aps.append(average_precision_score(y_true, y_prob))

            # recall 为降序排列，翻转后进行线性插值
            interp_prec = np.interp(mean_recall, rec[::-1], prec[::-1])
            precisions.append(interp_prec)

        mean_prec = np.mean(precisions, axis=0)
        mean_ap = np.mean(fold_aps)
        std_ap = np.std(fold_aps)

        if _is_ours_model(model_name):
            linestyle, lw, display_name = '-', 2.4, 'Ours'
        else:
            linestyle = baseline_linestyles[m_idx % len(baseline_linestyles)]
            lw, display_name = 1.8, model_name

        ax.plot(
            mean_recall, mean_prec,
            color=main_color, lw=lw, linestyle=linestyle,
            label=f'{display_name} (PR-AUC = {mean_ap:.3f} ± {std_ap:.3f})'
        )

        std_prec = np.std(precisions, axis=0)
        prec_upper = np.minimum(mean_prec + std_prec, 1.0)
        prec_lower = np.maximum(mean_prec - std_prec, 0.0)
        ax.fill_between(mean_recall, prec_lower, prec_upper, color=main_color, alpha=0.12)

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('Recall', fontsize=12, fontweight='bold', labelpad=8)
    ax.set_ylabel('Precision', fontsize=12, fontweight='bold', labelpad=8)
    ax.tick_params(axis='both', which='major', labelsize=10, width=1.0)
    ax.grid(True, linestyle=':', alpha=0.35)
    ax.legend(loc='lower left', fontsize=8.8, frameon=True, edgecolor='#E0E0E0', facecolor='white')

    plt.tight_layout()
    if os.path.dirname(save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fmt = 'svg' if str(save_path).endswith('.svg') else 'png'
    plt.savefig(save_path, format=fmt, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)
    print(f"Saved multi-model PR curve to {save_path}")


def _draw_cm_on_ax(ax, cm, title=None, show_percent=True, fontsize=12):
    """内部辅助函数：在指定 ax 上渲染符合期刊规范的混淆矩阵"""
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues, alpha=0.85)

    tick_marks = np.arange(len(CLASS_NAMES))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(CLASS_NAMES, fontsize=10.5)
    ax.set_yticklabels(CLASS_NAMES, fontsize=10.5, rotation=90, va="center")

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_perc = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0) * 100

    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            text_color = "white" if cm[i, j] > thresh else "#1F2D3D"
            cell_text = f"{cm[i, j]:d}\n({cm_perc[i, j]:.1f}%)" if show_percent else f"{cm[i, j]:d}"
            ax.text(
                j, i, cell_text,
                ha="center", va="center",
                color=text_color, fontsize=fontsize, fontweight='bold'
            )

    if title:
        ax.set_title(title, fontsize=11.5, fontweight='bold', pad=10)
    ax.grid(False)
    return im


def plot_single_cm(
    pkl_path,
    model_name='Ours (MedicalGMU)',
    show_title=True,
    show_percent=True,
    save_path='cm_single.svg',
    show=False,
):
    """绘制 5-Fold 合并混淆矩阵"""
    parsed = extract_oof_and_fold_data(pkl_path)
    cm = confusion_matrix(parsed['oof_labels'], parsed['oof_preds'], labels=[0, 1])
    thresh_list = parsed['fold_thresholds']

    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    t_mean, t_std = np.mean(thresh_list), np.std(thresh_list)
    title_str = None
    if show_title:
        title_str = (
             f"{model_name} (Thresh = {t_mean:.3f})"
        )

    im = _draw_cm_on_ax(ax, cm, title=title_str, show_percent=show_percent, fontsize=13)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)

    ax.set_ylabel('True Label', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold', labelpad=10)

    plt.tight_layout()
    if os.path.dirname(save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fmt = 'svg' if str(save_path).endswith('.svg') else 'png'
    plt.savefig(save_path, format=fmt, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)
    print(f"Saved single  Confusion Matrix to {save_path}")


def plot_multi_cm(
    model_files,
    show_percent=True,
    save_path='cm_4models.svg',
    show=False,
):
    """在同一张图 (1xN 网格) 内对比绘制多个模型的 5-Fold 混淆矩阵"""
    n_models = len(model_files)
    fig, axes = plt.subplots(1, n_models, figsize=(4.6 * n_models, 4.5))
    if n_models == 1:
        axes = [axes]

    for idx, (m_name, pkl_path) in enumerate(model_files.items()):
        ax = axes[idx]
        parsed = extract_oof_and_fold_data(pkl_path)
        cm = confusion_matrix(parsed['oof_labels'], parsed['oof_preds'], labels=[0, 1])
        thresh_list = parsed['fold_thresholds']

        t_mean, t_std = np.mean(thresh_list), np.std(thresh_list)
        sub_title = (
                f"{m_name}\n(Thresh: {t_mean:.2f})"
        )

        im = _draw_cm_on_ax(ax, cm, title=sub_title, show_percent=show_percent, fontsize=12)
        ax.set_xlabel('Predicted Label', fontsize=11, fontweight='bold', labelpad=8)
        ax.set_ylabel('True Label' if idx == 0 else '', fontsize=11, fontweight='bold', labelpad=8)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=9)

    plt.tight_layout()
    if os.path.dirname(save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fmt = 'svg' if str(save_path).endswith('.svg') else 'png'
    plt.savefig(save_path, format=fmt, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)
    print(f"Saved multi-model  Confusion Matrices to {save_path}")


def export_all_cms(
    model_files,
    save_dir='./outputs/cm_figures',
    file_ext='svg',
    show_title=False,
    show_percent=False,
    show=False,
):
    """循环生成并保存所有模型各自独立的约登阈值混淆矩阵矢量图/位图"""
    os.makedirs(save_dir, exist_ok=True)
    for m_name, pkl_path in model_files.items():
        clean_name = re.sub(r'[^a-zA-Z0-9_-]+', '_', m_name).strip('_')
        save_path = os.path.join(save_dir, f"cm_{clean_name}.{file_ext}")
        plot_single_cm(
            pkl_path=pkl_path,
            model_name=m_name,
            show_title=show_title,
            show_percent=show_percent,
            save_path=save_path,
            show=show,
        )
