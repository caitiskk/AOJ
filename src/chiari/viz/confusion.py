# src/chiari/viz/confusion.py
"""合并混淆矩阵（5折验证样本池化）。"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from chiari.viz.style import COLOR_MEAN

def plot_confusion_matrix_total(all_labels, all_preds, save_path='cm_total.png',class_names=('Non-III', 'III Type'), show=False):
	"""绘制符合期刊标准的合并混淆矩阵"""
	cm = confusion_matrix(all_labels, all_preds)
	
	fig, ax = plt.subplots(figsize=(5.5, 5))
	# 采用更内敛清爽的学术蓝单色渐变
	im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues, alpha=0.85)
	# ax.set_title('Confusion Matrix', fontsize=13, fontweight='bold', pad=15)
	
	# 侧边颜色条美化
	cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
	cbar.ax.tick_params(labelsize=10)
	
	tick_marks = np.arange(2)
	ax.set_xticks(tick_marks)
	ax.set_yticks(tick_marks)
	ax.set_xticklabels(class_names, fontsize=11)
	ax.set_yticklabels(class_names, fontsize=11, rotation=90, va="center") # 垂直旋转更具期刊规范

	# 矩阵中心数值字体及颜色自适应调整
	thresh = cm.max() / 2.
	for i in range(2):
		for j in range(2):
			ax.text(j, i, format(cm[i, j], 'd'),
					ha="center", va="center",
					color="white" if cm[i, j] > thresh else "#1F2D3D",
					fontsize=14, fontweight='bold')

	ax.set_ylabel('True Label', fontsize=12, fontweight='bold', labelpad=10)
	ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold', labelpad=10)
	
	# 移除混淆矩阵内部的网格，防止格子错位
	ax.grid(False)
	
	plt.tight_layout()
	plt.savefig(save_path, dpi=300, bbox_inches='tight')
	plt.show()
	print(f"Saved Confusion Matrix to {save_path}")