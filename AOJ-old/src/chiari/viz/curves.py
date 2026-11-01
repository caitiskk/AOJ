# src/chiari/viz/curves.py
"""5折 ROC / PR 曲线。
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_curve, roc_auc_score,          #
                             precision_recall_curve, average_precision_score)

from chiari.viz.style import COLOR_PALETTE, COLOR_MEAN, COLOR_SHADE  # ← import 即应用样式

def plot_roc_curves(fold_results, save_path='roc_5fold.svg'):
	"""绘制符合期刊标准的5折ROC + 平均ROC"""
	fig, ax = plt.subplots(figsize=(6, 5.5))  # 黄金学术比例
	mean_fpr = np.linspace(0, 1, 100)
	tprs = []

	# 1. 绘制各折曲线
	for idx, (fold_idx, (labels, probs, joint_ids)) in enumerate(fold_results.items()):
		fpr, tpr, _ = roc_curve(labels, probs)
		auc = roc_auc_score(labels, probs)
		color = COLOR_PALETTE[idx % len(COLOR_PALETTE)]
		ax.plot(fpr, tpr, lw=1.2, alpha=0.5, linestyle='--',
				color=color, label=f'Fold {fold_idx+1} (AUC = {auc:.3f})')
		
		interp_tpr = np.interp(mean_fpr, fpr, tpr)
		interp_tpr[0] = 0.0
		tprs.append(interp_tpr)

	# 2. 绘制均值与标准差阴影
	mean_tpr = np.mean(tprs, axis=0)
	mean_tpr[-1] = 1.0
	mean_auc = np.mean([roc_auc_score(l, p) for l, p, _ in fold_results.values()])
	std_auc = np.std([roc_auc_score(l, p) for l, p, _ in fold_results.values()])
	
	ax.plot(mean_fpr, mean_tpr, color=COLOR_MEAN, lw=2.0,
			label=f'Mean (AUC = {mean_auc:.3f} ± {std_auc:.3f})')
	
	std_tpr = np.std(tprs, axis=0)
	tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
	tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
	ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color=COLOR_SHADE, alpha=0.2, label='± 1 SD')

	# 3. 细节美化
	ax.plot([0, 1], [0, 1], linestyle=':', color='#7F8C8D', lw=1.0) # 对角基准线改为细虚线
	ax.set_xlim([-0.02, 1.02])
	ax.set_ylim([-0.02, 1.02])
	
	ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold', labelpad=8)
	ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold', labelpad=8)
	# ax.set_title('Receiver Operating Characteristic (ROC)', fontsize=13, fontweight='bold', pad=12)
	
	ax.tick_params(axis='both', which='major', labelsize=10, width=1.0)
	ax.grid(True, linestyle=':', alpha=0.4) # 轻量化网格线
	
	ax.legend(loc='lower right', fontsize=9, frameon=True, edgecolor='#E0E0E0', facecolor='white')
	
	plt.tight_layout()
	plt.savefig(save_path, format='svg', bbox_inches='tight') # bbox_inches确保标签不被截断
	plt.show()
	print(f"Saved ROC to {save_path}")


def plot_pr_curves(fold_results, save_path='pr_5fold.svg'):
	"""绘制符合期刊标准的5折PR + 平均PR"""
	fig, ax = plt.subplots(figsize=(6, 5.5))
	mean_recall = np.linspace(0, 1, 100)
	precisions = []
	total_labels = []
	# 1. 绘制各折曲线
	for idx, (fold_idx, (labels, probs, joint_ids)) in enumerate(fold_results.items()):
		precision, recall, _ = precision_recall_curve(labels, probs)
		ap = average_precision_score(labels, probs)
		color = COLOR_PALETTE[idx % len(COLOR_PALETTE)]
		ax.plot(recall, precision, lw=1.2, alpha=0.5, linestyle='--',
				color=color, label=f'Fold {fold_idx+1} (AP = {ap:.3f})')
		
		total_labels.extend(labels)
		# PR曲线插值修复（确保recall单调递增，避免反转带来的异常插值）
		interp_prec = np.interp(mean_recall, recall[::-1], precision[::-1])
		precisions.append(interp_prec)

	# 2. 绘制均值与标准差阴影
	mean_prec = np.mean(precisions, axis=0)
	mean_ap = np.mean([average_precision_score(l, p) for l, p, _ in fold_results.values()])
	std_ap = np.std([average_precision_score(l, p) for l, p, _ in fold_results.values()])
	
	ax.plot(mean_recall, mean_prec, color=COLOR_MEAN, lw=2.0,
			label=f'Mean (AP = {mean_ap:.3f} ± {std_ap:.3f})')
	
	std_prec = np.std(precisions, axis=0)
	prec_upper = np.minimum(mean_prec + std_prec, 1)
	prec_lower = np.maximum(mean_prec - std_prec, 0)
	ax.fill_between(mean_recall, prec_lower, prec_upper, color=COLOR_SHADE, alpha=0.2, label='± 1 SD')

	pos_ratio = sum(total_labels) / len(total_labels) if len(total_labels) > 0 else 0.5
	ax.axhline(y=pos_ratio, color='#E74C3C', linestyle=':', lw=1.5, 
				label=f'Baseline (Pos Ratio = {pos_ratio:.2f})')
	# 3. 细节美化
	ax.set_xlim([-0.02, 1.02])
	ax.set_ylim([-0.02, 1.02])
	
	ax.set_xlabel('Recall', fontsize=12, fontweight='bold', labelpad=8)
	ax.set_ylabel('Precision', fontsize=12, fontweight='bold', labelpad=8)
	# ax.set_title('Precision-Recall (PR) Curve', fontsize=13, fontweight='bold', pad=12)
	
	ax.tick_params(axis='both', which='major', labelsize=10, width=1.0)
	ax.grid(True, linestyle=':', alpha=0.4)
	
	ax.legend(loc='lower left', fontsize=9, frameon=True, edgecolor='#E0E0E0', facecolor='white')
	
	plt.tight_layout()
	plt.savefig(save_path, format='svg', bbox_inches='tight')
	plt.show()
	print(f"Saved PR to {save_path}")
