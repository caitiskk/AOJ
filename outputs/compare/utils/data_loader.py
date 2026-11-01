# -*- coding: utf-8 -*-
"""
数据加载与预处理模块：统一读取 5fold_results.pkl、转换张量并按样本 ID 严格对齐
"""
import pickle
import torch
import numpy as np
from sklearn.metrics import roc_curve


def to_numpy(x):
    """将 PyTorch Tensor 或列表安全转换为 NumPy 数组"""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def load_fold_dict(pkl_path):
    """从 5fold_results.pkl 中提取 fold_results 字典"""
    try:
        data = torch.load(pkl_path, map_location='cpu', weights_only=False)
    except Exception:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

    if isinstance(data, dict) and 'fold_results' in data:
        return data['fold_results']
    return data





def extract_oof_and_fold_data(pkl_path,  fixed_threshold=0.25):
    """
    统一读取单模型 5 折数据，按 (fold_idx, joint_id) 严格排序对齐，返回逐折数据与汇总 OOF 预测
    """
    fold_dict = load_fold_dict(pkl_path)

    oof_labels, oof_probs, oof_preds, oof_ids = [], [], [], []
    fold_thresholds = []
    aligned_folds = {}

    for fold_idx in sorted(fold_dict.keys()):
        item = fold_dict[fold_idx]
        y_true = to_numpy(item[0]).ravel().astype(int)
        y_prob = to_numpy(item[1]).ravel().astype(float)

        # 若包含 joint_ids（第 3 项），按 joint_id 排序以保证跨模型 100% 配对对齐
        if len(item) >= 3 and item[2] is not None:
            jids = [f"F{fold_idx}_{jid}" for jid in item[2]]
            sort_idx = np.argsort(jids)
            y_true = y_true[sort_idx]
            y_prob = y_prob[sort_idx]
            jids = [jids[i] for i in sort_idx]
        else:
            jids = [f"F{fold_idx}_{i}" for i in range(len(y_true))]

        thresh = float(fixed_threshold)

        y_pred = (y_prob >= thresh).astype(int)

        fold_thresholds.append(thresh)
        aligned_folds[fold_idx] = (y_true, y_prob, y_pred, thresh)

        oof_labels.append(y_true)
        oof_probs.append(y_prob)
        oof_preds.append(y_pred)
        oof_ids.extend(jids)

    all_labels = np.concatenate(oof_labels)
    all_probs = np.concatenate(oof_probs)


    all_preds = np.concatenate(oof_preds)

    return {
        'aligned_folds': aligned_folds,
        'oof_labels': all_labels,
        'oof_probs': all_probs,
        'oof_preds': all_preds,
        'oof_ids': oof_ids,
        'fold_thresholds': fold_thresholds,
    }
