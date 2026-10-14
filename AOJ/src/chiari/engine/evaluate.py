# src/chiari/engine/evaluate.py
"""评估：模型 + DataLoader → 指标 + 预测。纯函数、无状态、不读 CONFIG。"""
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, confusion_matrix
from collections import defaultdict
from sklearn.metrics import (
	accuracy_score, precision_score, recall_score, 
	f1_score, roc_auc_score, confusion_matrix, average_precision_score,
)

import pandas as pd

METRIC_KEYS = ['auc', 'ap', 'accuracy', 'sensitivity', 'specificity',
               'precision', 'ppv', 'npv', 'f1']

def summarize_folds(all_fold_metrics, keys=METRIC_KEYS):
    """5折指标 → DataFrame（每折一列 + Mean + Std）。论文 Table 直接来源。"""
    df = pd.DataFrame(all_fold_metrics)[keys].T
    df.columns = [f'F{i+1}' for i in range(df.shape[1])]
    df['Mean'] = df.mean(axis=1)
    df['Std'] = df.std(axis=1)
    return df



@torch.no_grad()
def evaluate_joint_level(model, dataloader, device, search_threshold=False,best_t=0.25):
	"""
	关节级评估：同一个 joint_id（如 001_left）的多张影像概率取平均。
	
	Args:
		search_threshold: 是否基于约登指数搜索最优阈值。False 则用 0.5。
	"""
	model.eval()
	joint_probs = defaultdict(list)   # {joint_id: [prob_img1, prob_img2, ...]}
	joint_labels = {}                # {joint_id: label}

	for imgs, labels, cat_batch, cont_batch, _, joint_ids in dataloader:
		imgs = imgs.to(device)
		cat_batch = {k: v.to(device) for k, v in cat_batch.items()}
		cont_batch = cont_batch.to(device)

		logits = model(imgs, cat_batch, cont_batch)

		# ========== 兼容两种输出格式 ==========
		if logits.dim() == 1 or logits.size(-1) == 1:
			# 单输出 [B] 或 [B,1] → sigmoid
			probs = torch.sigmoid(logits).cpu().numpy()
			if probs.ndim == 2:
				probs = probs.squeeze(-1)
		else:
			# 双输出 [B, 2] → softmax 取第1类（III型）
			probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

		# 收集每个关节的概率
		for i in range(len(joint_ids)):
			jid = joint_ids[i]
			joint_probs[jid].append(float(probs[i]))
			# 同一个关节的 label 肯定相同，重复赋值无妨
			joint_labels[jid] = int(labels[i].item())
			

	# ========== 聚合：每个关节平均概率 ==========
	final_labels, final_probs = [], []
	final_joint_ids = []

	for jid in sorted(joint_probs.keys()):
		avg_prob = np.mean(joint_probs[jid])
		final_probs.append(avg_prob)
		final_labels.append(joint_labels[jid])
		final_joint_ids.append(jid)
		

	final_labels = np.array(final_labels)
	final_probs = np.array(final_probs)



	# ========== 阈值选择 ==========
	if search_threshold:
		# 基于约登指数搜索最优阈值
		# best_youden, best_thresh = -1, 0.5
		# for t in np.arange(0.05, 1.0, 0.05):
		# 	preds = (final_probs >= t).astype(int)
		# 	try:
		# 		tn, fp, fn, tp = confusion_matrix(final_labels, preds).ravel()
		# 		sens = tp / (tp + fn) if (tp + fn) > 0 else 0
		# 		spec = tn / (tn + fp) if (tn + fp) > 0 else 0
		# 		youden = sens + spec - 1
		# 		if youden > best_youden:
		# 			best_youden, best_thresh = youden, t
		# 	except ValueError:
		# 		continue
		# threshold = best_thresh
		fpr, tpr, thresholds = roc_curve(final_labels, final_probs)
		tnr = 1 - fpr
		youden_j = tpr + tnr - 1
		
		best_idx = np.argmax(youden_j)
		threshold = thresholds[best_idx]
		best_youden = youden_j[best_idx]
	else:
		threshold = best_t

	final_preds = (final_probs >= threshold).astype(int)
	fp_joints = []   # 假阳性：预测1，实际0
	fn_joints = []   # 假阴性：预测0，实际1
	for idx, jid in enumerate(sorted(joint_probs.keys())):
		pred = final_preds[idx]
		label = final_labels[idx]
		prob = final_probs[idx]
		
		if pred == 1 and label == 0:
			fp_joints.append(jid
				# {
				# 'joint_id': jid,
				# 'prob': round(float(prob), 4),
				# 'threshold': round(float(threshold), 4)
			# }
			)
		elif pred == 0 and label == 1:
			fn_joints.append(jid
				# {
				# 'joint_id': jid,
				# 'prob': round(float(prob), 4),
				# 'threshold': round(float(threshold), 4)
			# }
			)
	# print(f"误诊为III型的关节: {fp_joints}")
	# print(f"漏诊的III型关节: {fn_joints}")

	# ========== 指标计算 ==========
	acc = accuracy_score(final_labels, final_preds)
	prec = precision_score(final_labels, final_preds, zero_division=0)
	rec = recall_score(final_labels, final_preds, zero_division=0)
	f1 = f1_score(final_labels, final_preds, zero_division=0)

	try:
		auc = roc_auc_score(final_labels, final_probs)
	except ValueError:
		auc = 0.5  # 只有一个类别时无法计算
	try:
		ap = average_precision_score(final_labels, final_probs)  # ← PR曲线下面积
	except ValueError:
		ap = 0.0


	# 医学指标：敏感性/特异性
	try:
		tn, fp, fn, tp = confusion_matrix(final_labels, final_preds).ravel()
		sens = tp / (tp + fn) if (tp + fn) > 0 else 0
		spec = tn / (tn + fp) if (tn + fp) > 0 else 0
		ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
		npv = tn / (tn + fn) if (tn + fn) > 0 else 0
	except ValueError:
		# sens, spec = 0, 0
		tn = fp = fn = tp = 0
		sens = spec = ppv = npv = 0

	metrics = {
		'auc': float(auc),
		'ap': float(ap), 
		'threshold': float(threshold),
		'accuracy': float(acc),
		'precision': float(prec),
		'recall': float(rec),
		'f1': float(f1),
		'sensitivity': float(sens),
		'specificity': float(spec),
		 'ppv': float(ppv),         # ← 新增
		'npv': float(npv),         # ← 新增
		'tp': int(tp),             # ← 新增
		'fp': int(fp),             # ← 新增
		'tn': int(tn),             # ← 新增
		'fn': int(fn),             # ← 新增
		'num_joints': len(final_labels),
		'num_positive': int(final_labels.sum()),
	}

	return metrics, final_labels.tolist(), final_preds.tolist(), final_probs.tolist(), final_joint_ids

@torch.no_grad()
def predict_single_joint(model, imgs, cat_batch, cont_batch, device, threshold=0.25):
    """
    单关节多切片综合形态推理。
    
    Args:
        model: 已加载权重的模型
        imgs: [N, C, H, W] 图像 Tensor
        cat_batch: dict {col_name: Tensor [N]}
        cont_batch: Tensor [N, num_cont]
        device: 计算设备
        threshold: 分类阈值
    """
    model.eval()
    imgs = imgs.to(device)
    
    # 支持模态缺失或纯图像模型 (image_only)
    if cat_batch is not None:
        cat_batch = {k: v.to(device) for k, v in cat_batch.items()}
    if cont_batch is not None:
        cont_batch = cont_batch.to(device)

    # 1. 前向传播
    # 根据模型接口，如果模型支持纯图像或只接受 imgs，做相应适配
    try:
        logits = model(imgs, cat_batch, cont_batch)
    except TypeError:
        logits = model(imgs)

    # 2. 兼容单输出与双输出
    if logits.dim() == 1 or logits.size(-1) == 1:
        probs = torch.sigmoid(logits).cpu().numpy()
        if probs.ndim == 2:
            probs = probs.squeeze(-1)
        elif probs.ndim == 0:
            probs = np.array([probs.item()])
    else:
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

    # 3. 概率聚合
    img_probs = [float(p) for p in probs]
    avg_prob = float(np.mean(img_probs))
    pred_label = int(avg_prob >= threshold)

    return {
        'pred_label': pred_label,
        'avg_prob': avg_prob,
        'img_probs': img_probs,
        'num_images': len(img_probs),
    }