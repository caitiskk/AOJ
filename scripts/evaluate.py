# scripts/evaluate.py
"""5折统一评估。
  默认        读 fold_predictions.pkl → 直接出图（改图不重跑模型）
  --recompute 从 checkpoint 重建一切重新推理（权威评估，写回 pkl）
用法：python scripts/evaluate.py --config configs/default.yaml [--recompute] [--gpu 1]
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from chiari.config import load_config
from chiari.data.dataset import ChiariXrayDataset, collate_fn
from chiari.engine.evaluate import evaluate_joint_level, summarize_folds, METRIC_KEYS
from chiari.utils.checkpoint import ckpt_path, load_fold_model
from chiari.viz.curves import plot_roc_curves, plot_pr_curves
from chiari.viz.confusion import plot_confusion_matrix_total
from chiari.config import adapt_config


def recompute_predictions(cfg):
    device = cfg['device']
    fold_results, all_fold_metrics = {}, []
    labels_total, preds_total = [], []

    for fold in range(1, cfg['training']['n_splits'] + 1):
        path = ckpt_path(cfg['paths']['save_dir'], fold)
        model, _ = load_fold_model(path, device)

        val_ds = ChiariXrayDataset.from_checkpoint(path)
        val_loader = DataLoader(val_ds, batch_size=cfg['training']['batch_size'],
                                shuffle=False, num_workers=4,
                                collate_fn=collate_fn, pin_memory=True)

        metrics, labels, preds, probs, joint_ids = evaluate_joint_level(
            model, val_loader, device,
            search_threshold=False, best_t=cfg['eval']['threshold'])

        # 关键修复：确保所有数据脱钩转为 numpy，彻底摆脱 Tensor/GPU 绑定
        labels_np = np.asarray(labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels)
        probs_np = np.asarray(probs.cpu().numpy() if isinstance(probs, torch.Tensor) else probs)
        joints_np = np.asarray(joint_ids)

        fold_results[fold - 1] = (labels_np, probs_np, joints_np)
        all_fold_metrics.append(metrics)
        labels_total.extend(labels_np)
        preds_total.extend(preds)

    # 确保目录存在，并统一使用 pickle 保存
    save_path = Path(cfg['paths']['predictions_dir']) / 'fold_predictions.pkl'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(fold_results, f)

    return fold_results, all_fold_metrics, labels_total, preds_total

# def recompute_predictions(cfg):

#     device = cfg['device']
#     fold_results, all_fold_metrics = {}, []
#     labels_total, preds_total = [], []

#     for fold in range(1, cfg['training']['n_splits'] + 1):
#         path = ckpt_path(cfg['paths']['save_dir'], fold)
#         model, _ = load_fold_model(path, device)

#         val_ds = ChiariXrayDataset.from_checkpoint(path)     # ← 折切分 + scaler 全还原
#         val_loader = DataLoader(val_ds, batch_size=cfg['training']['batch_size'],
#                                 shuffle=False, num_workers=4,
#                                 collate_fn=collate_fn, pin_memory=True)

#         metrics, labels, preds, probs, joint_ids = evaluate_joint_level(
#             model, val_loader, device,
#             search_threshold=False, best_t=cfg['eval']['threshold'])

#         fold_results[fold - 1] = (labels, probs, joint_ids)   # key 0 起，绘图函数契约
#         all_fold_metrics.append(metrics)
#         labels_total.extend(labels); preds_total.extend(preds)

#     # 回存，供模式A下次秒开
#     with open(Path(cfg['paths']['predictions_dir']) / 'fold_predictions.pkl', 'wb') as f:
#         pickle.dump(fold_results, f)
#     return fold_results, all_fold_metrics, labels_total, preds_total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config, gpu=args.gpu)
    pred_pkl = Path(cfg['paths']['predictions_dir']) / 'fold_predictions.pkl'
    threshold = cfg['eval']['threshold']

    # # ── 取数：权威重算 或 快速读档 ──
    # if args.recompute or not pred_pkl.exists():
    #     fold_results, all_fold_metrics, labels_t, preds_t = recompute_predictions(cfg)
    # else:
    #     fold_results = torch.load(pred_pkl, weights_only=False) 
    #     # with open(pred_pkl, 'rb') as f:
    #     #     fold_results = pickle.load(f)
    #     all_fold_metrics = None   
    #     labels_t = np.concatenate([l for l, p, _ in fold_results.values()])
    #     preds_t = (np.concatenate([p for l, p, _ in fold_results.values()])
    #                >= threshold).astype(int)   # 与 evaluate_joint_level 的阈值化一致
# ── 取数：权威重算 或 快速读档 ──
    if args.recompute or not pred_pkl.exists():
        fold_results, all_fold_metrics, labels_t, preds_t = recompute_predictions(cfg)
    else:
        # 恢复使用 pickle.load，不要用 torch.load 读 pickle 文件
        with open(pred_pkl, 'rb') as f:
            fold_results = pickle.load(f)
        all_fold_metrics = None   
        labels_t = np.concatenate([l for l, p, _ in fold_results.values()])
        preds_t = (np.concatenate([p for l, p, _ in fold_results.values()])
                   >= threshold).astype(int)

    # ── 表：每折指标 + Mean±Std（原汇总打印段）──
    if all_fold_metrics:
        summary = summarize_folds(all_fold_metrics)
        summary.to_csv(Path(cfg['paths']['tables_dir']) / 'metrics.csv')
        with open(Path(cfg['paths']['predictions_dir']) / '5fold_results.pkl', 'wb') as f:
            pickle.dump({'threshold': threshold,
                         'fold_metrics': all_fold_metrics,
                         'fold_results': fold_results,
                         'mean': {k: float(np.mean([m[k] for m in all_fold_metrics]))
                                  for k in METRIC_KEYS},
                         'std': {k: float(np.std([m[k] for m in all_fold_metrics]))
                                 for k in METRIC_KEYS}}, f)

    # ── 图：路径由 cfg 注入，viz 函数零 config 依赖 ──
    figs = cfg['paths']['figures_dir']
    plot_roc_curves(fold_results, save_path=f'{figs}/roc_5fold.svg')
    plot_pr_curves(fold_results, save_path=f'{figs}/pr_5fold.svg')
    plot_confusion_matrix_total(np.asarray(labels_t), np.asarray(preds_t),
                                save_path=f'{figs}/cm_total.png')


if __name__ == "__main__":
    main()