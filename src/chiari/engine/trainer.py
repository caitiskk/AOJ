# src/chiari/engine/trainer.py
"""训练编排：cross_validation（5折循环）+ train_one_fold（单折）。"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from chiari.data.dataset import ChiariXrayDataset, collate_fn
from chiari.data.transforms import get_transforms
from chiari.models import build_model
from chiari.engine.evaluate import evaluate_joint_level
from chiari.utils.seed import set_seed
from chiari.utils.checkpoint import load_fold_model   


logger = logging.getLogger(__name__)

def train_one_fold(cfg, fold, train_pids, val_pids):
    """单折完整生命周期"""
    t, device = cfg['training'], cfg['device']
    save_dir = Path(cfg['paths']['save_dir'])
    ckpt_path = save_dir / f'{fold + 1}_medicalgmu_best.pt'

    # ── 4. Dataset & DataLoader──
    train_ds = ChiariXrayDataset(
        cfg['data']['pkl_path'], cfg['data']['image_root'], train_pids,
        transform=get_transforms('train', img_size=t['img_size']), fit_scaler=True,
    )
    val_ds = ChiariXrayDataset(
        cfg['data']['pkl_path'], cfg['data']['image_root'], val_pids,
        transform=get_transforms('val', img_size=t['img_size']),
        scaler=train_ds.scaler, fit_scaler=False,
    )
    train_loader = DataLoader(train_ds, batch_size=t['batch_size'], shuffle=True,
                              num_workers=4, collate_fn=collate_fn,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=t['batch_size'], shuffle=False,
                            num_workers=4, collate_fn=collate_fn, pin_memory=True)

    # ── 5. 模型 + 分层优化器（8/16 等硬编码 → cfg['model']）──
    model = build_model(cfg, pretrained=True).to(device)
	# 维度一致性断言：避免更改临床数据维度之后报错
    assert cfg['model']['num_cont'] == len(train_ds.cont_cols), (
        f"cfg.model.num_cont={cfg['model']['num_cont']} != "
        f"数据实际连续特征数 {len(train_ds.cont_cols)}（cont_cols={train_ds.cont_cols}）")
    model_name = cfg['model']
    # backbone_params = [p for n, p in model.named_parameters() if 'resnet' in n]
    if model_name == 'image_only':
         backbone_params = [p for n, p in model.resnet.named_parameters() if "fc" not in n]
         other_params   = model.resnet.fc.parameters()
    else:
        backbone_params = [p for n, p in model.named_parameters() if 'resnet' in n]
        other_params    = [p for n, p in model.named_parameters() if 'resnet' not in n]
    optimizer = torch.optim.AdamW(
        [{'params': backbone_params, 'lr': t['lr_backbone']},
         {'params': other_params,   'lr': t['lr_head']}],
        weight_decay=t['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10)  
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, t['pos_weight']]).to(device))

    # ── 6. 训练循环──
    best_auc, best_epoch, no_improve = 0, 0, 0
    for epoch in range(t['epochs']):
        model.train()
        epoch_loss = 0
        for imgs, labels, cat_batch, cont_batch, _, _ in train_loader:
            imgs, labels_f = imgs.to(device), labels.to(device)
            cat_batch = {k: v.to(device) for k, v in cat_batch.items()}
            cont_batch = cont_batch.to(device)
            optimizer.zero_grad()
            logits = model(imgs, cat_batch, cont_batch)
            loss = criterion(logits, labels_f)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        #  调用：每 epoch 验证，用于选模型 / 调 scheduler
        val_metrics, *_ = evaluate_joint_level(
            model, val_loader, device,
            search_threshold=False, best_t=t.get('threshold', 0.25),
        )
        scheduler.step(val_metrics['auc'])
        logger.info("Fold %d Epoch %03d | loss %.4f | AUC %.3f | Acc %.3f | "
                    "Sen %.3f | Spe %.3f | Thresh %.2f",
                    fold + 1, epoch, epoch_loss / len(train_loader),
                    val_metrics['auc'], val_metrics['accuracy'],
                    val_metrics['sensitivity'], val_metrics['specificity'],
                    val_metrics['threshold'])

        if val_auc := val_metrics['auc'] > best_auc:   # 见下方注
            best_auc, best_epoch = val_metrics['auc'], epoch
            no_improve = 0
            torch.save({
                'model_state': model.state_dict(),
                'scaler': train_ds.scaler,
                'cat_cols': train_ds.cat_cols,
                'cont_cols': train_ds.cont_cols,
                'threshold': val_metrics['threshold'],
                'metrics': val_metrics,
                'config': cfg,                    # ← 存的是 cfg dict，from_checkpoint 依赖它
                'train_patient_ids': train_pids,
                'val_patient_ids': val_pids,
            }, ckpt_path)
        else:
            no_improve += 1
        if no_improve >= t['patience']:
            logger.info("早停于 Epoch %d (Best: %d, AUC=%.3f)", epoch, best_epoch, best_auc)
            break

    #  调用：加载最优权重，重跑一次验证
    #   得到与 checkpoint 严格对应的，落盘给画图/表格用
    # model.load_state_dict(torch.load(ckpt_path, map_location=device)['model_state'])
    model, ckpt = load_fold_model(ckpt_path, device)

    metrics, labels, preds, probs, joint_ids = evaluate_joint_level(
        model, val_loader, device, best_t=t.get('threshold', 0.25))

    return {'fold': fold + 1, 'metrics': metrics,
            'labels': labels, 'probs': probs, 'joint_ids': joint_ids}


def cross_validation(cfg):
    """5折交叉验证编排。"""
    t = cfg['training']

    #  第 2 步：以影像文件夹为准取病人ID
    all_pids = sorted({p.name for p in Path(cfg['data']['image_root']).iterdir() if p.is_dir()})
    logger.info("总病人: %d", len(all_pids))

    kf = KFold(n_splits=t['n_splits'], shuffle=True, random_state=cfg['run']['seed'])
    fold_results = {}
    for fold, (train_idx, val_idx) in enumerate(kf.split(all_pids)):
        set_seed(cfg['run']['seed'])                       # 每折重置，保证折间可比
        train_pids = [all_pids[i] for i in train_idx]
        val_pids   = [all_pids[i] for i in val_idx]
        logger.info("Fold %d | Train: %d, Val: %d", fold + 1, len(train_pids), len(val_pids))

        fold_results[fold] = train_one_fold(cfg, fold, train_pids, val_pids)

    # ── 汇总落盘──
    pred_dir = Path(cfg['paths']['predictions_dir']); pred_dir.mkdir(parents=True, exist_ok=True)
    # 1. 预测：格式与 plot_roc_curves 的一致
    torch.save({f: (r['labels'], r['probs'], r['joint_ids'])
                for f, r in fold_results.items()},
               pred_dir / 'fold_predictions.pkl')
    # 2. 指标表：论文 Table 直接来源
    rows = [{'fold': r['fold'], **r['metrics']} for r in fold_results.values()]
    pd.DataFrame(rows).to_csv(pred_dir.parent / 'tables' / 'metrics.csv', index=False)
    return fold_results
