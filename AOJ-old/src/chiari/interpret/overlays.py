# src/chiari/interpret/overlays.py
"""CAM 可视化渲染：原图 + 多阈值彩色热力图拼接。

"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

__all__ = ['concatenate_images_with_gap', 'create_colorful_overlay',
           'create_multi_threshold_colorful']

def concatenate_images_with_gap(images, gap=5, gap_color=(255, 255, 255)):
	"""
	水平拼接多张图，中间留白色 gap
	"""
	# 统一高度（以第一张图为基准）
	target_h = images[0].shape[0]
	resized = []
	for img in images:
		if img.shape[0] != target_h:
			ratio = target_h / img.shape[0]
			new_w = int(img.shape[1] * ratio)
			img = cv2.resize(img, (new_w, target_h))
		resized.append(img)
	
	# 计算总宽度
	total_width = sum(img.shape[1] for img in resized) + gap * (len(resized) - 1)
	result = np.full((target_h, total_width, 3), gap_color, dtype=np.uint8)
	
	# 逐个粘贴
	x_offset = 0
	for img in resized:
		result[:, x_offset:x_offset + img.shape[1]] = img
		x_offset += img.shape[1] + gap
	
	return result


def create_colorful_overlay(
	img_np, 
	cam_resized, 
	threshold=0.05,
	alpha=0.7,
	colormap_style='turbo',    # 彩虹色：蓝→青→绿→黄→红
	smooth_sigma=2.0,
):
	"""
	彩色热力图叠加原图
	"""
	# 统一转成 0-1 float
	if img_np.max() > 1.5:
		img_np = img_np.astype(np.float32) / 255.0
	else:
		img_np = img_np.astype(np.float32)
	
	# 高斯平滑 CAM
	cam_smooth = cv2.GaussianBlur(cam_resized.astype(np.float32), (0, 0), smooth_sigma)
	if cam_smooth.max() > 0:
		cam_smooth = cam_smooth / cam_smooth.max()
	
	# 软 mask：低阈值让更多区域染色
	mask = np.clip((cam_smooth - threshold) / 0.2, 0, 1)
	mask = cv2.GaussianBlur(mask, (0, 0), 1.0)
	
	# 生成彩色热力图（0-1 范围）
	cmap = plt.get_cmap(colormap_style)
	heatmap = cmap(cam_smooth)[:, :, :3].astype(np.float32)
	
	# 叠加：原图压暗 + 彩色热力图
	mask_3ch = mask[:, :, None]
	img_darker = img_np * 0.4  # 原图压暗 60%，让彩色更突出
	
	overlay = img_darker * (1 - mask_3ch * alpha) + heatmap * (mask_3ch * alpha)
	overlay = np.clip(overlay, 0, 1)
	
	return (overlay * 255).astype(np.uint8), cam_smooth


def create_multi_threshold_colorful(
	img_np, 
	cam_resized,
	thresholds=[0.05, 0.2, 0.4],
	alpha=0.7,
	gap=5,
	colormap_style='turbo',
):
	"""
	多阈值彩色对比图
	"""
	# 原图
	if img_np.max() <= 1.0:
		original = (img_np * 255).astype(np.uint8)
	else:
		original = img_np.astype(np.uint8)
	
	if len(original.shape) == 2:
		original = np.stack([original] * 3, axis=-1)
	
	images = [original]
	for th in thresholds:
		overlay, _ = create_colorful_overlay(
			img_np, cam_resized, th, alpha, colormap_style
		)
		images.append(overlay)
	
	return concatenate_images_with_gap(images, gap=gap)