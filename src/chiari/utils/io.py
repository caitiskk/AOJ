# src/chiari/utils/io.py
"""结果落盘工具。"""
import logging
from pathlib import Path

import numpy as np
from PIL import Image
import cv2

logger = logging.getLogger(__name__)


def save_cam_result(combined_img, gt_label, pred_label, prob,
                    sample_name, fold_num, save_root, img_idx):
	"""
	保存 CAM 结果到指定结构：
	save_root/
	├── 00/                    # 预测为非III型
	│   └── {patient_id}_{side}/
	│       ├── fold{fold_num}_img1.jpg
	│       └── fold{fold_num}_img2.jpg
	└── 01/                    # 预测为III型
		└── {patient_id}_{side}/
			├── fold{fold_num}_img1.jpg
			└── fold{fold_num}_img2.jpg
	"""
	# 按预测标签分文件夹
	pred_folder = '01' if gt_label == 1 else '00'
	
	# 关节级文件夹名
	save_dir = Path(save_root) / pred_folder / sample_name
	save_dir.mkdir(parents=True, exist_ok=True)
	
	# 文件名：fold{num}_prob{prob}_img{idx}.jpg
	# 需要外部传入当前是该关节的第几张图
	ext = ".jpg"
	base_name  = f"fold{fold_num}_num{img_idx}_pred{pred_label}_prob{prob:.3f}"
	save_path = save_dir / f'{base_name}{ext}' 
	counter = 1
	while save_path.exists():
		save_path = save_dir / f"{base_name}_{counter}{ext}"
		counter += 1
	cv2.imwrite(str(save_path), cv2.cvtColor(combined_img, cv2.COLOR_RGB2BGR))
	return save_path