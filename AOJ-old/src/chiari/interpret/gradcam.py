# src/chiari/interpret/gradcam.py
"""手动 Grad-CAM：支持多输入模型（影像 + 临床表格），适配 CE 多分类。
"""
import torch
import torch.nn.functional as F

class ManualGradCAM:
	"""
	手动 Grad-CAM：支持多输入模型（影像 + 临床表格），适配 CE 多分类
	"""
	def __init__(self, model, target_layer='resnet.layer3'):
		self.model = model
		self.model.eval()
		self.target_layer = target_layer
		
		for param in self.model.parameters():
			param.requires_grad = True
		
		self.gradients = None
		self.activations = None
		self._register_hook()
	
	def _register_hook(self):
		target_module = None
		for name, module in self.model.named_modules():
			if name == self.target_layer:
				target_module = module
				break
		
		if target_module is None:
			raise ValueError(f"Layer {self.target_layer} not found")
		
		fwd_handle = target_module.register_forward_hook(self._forward_hook)
		bwd_handle = target_module.register_full_backward_hook(self._backward_hook)
		self._handles = [fwd_handle, bwd_handle] 
		print(f"[GradCAM] Hook registered on {self.target_layer}")
	
	def _forward_hook(self, module, input, output):
		self.activations = output.detach()
	
	def _backward_hook(self, module, grad_input, grad_output):
		self.gradients = grad_output[0].detach()
	
	def generate(self, img_tensor, cat_features, cont_features, target_class=None):
		"""
		生成 Grad-CAM（CE 多分类版本）
		
		Args:
			img_tensor: [1, C, H, W] 预处理后的影像
			cat_features: dict of [1] LongTensor 分类特征
			cont_features: [1, num_cont] FloatTensor 连续特征
			target_class: int, 目标类别索引。若为 None，则使用预测类别
		
		Returns:
			cam: [H, W] numpy 热力图，归一化到 0~1
			probs: [num_classes] numpy 各类别概率
			pred_class: int 预测类别
		"""
		device = next(self.model.parameters()).device
		
		img_tensor = img_tensor.to(device).requires_grad_(True)
		cat_features = {k: v.to(device) for k, v in cat_features.items()}
		cont_features = cont_features.to(device)
		
		self.model.zero_grad()
		logits = self.model(img_tensor, cat_features, cont_features)  # [1, num_classes]
		
		# ===== 1. CE 多分类：计算所有类别概率 =====
		probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()  # [num_classes]
		pred_class = int(torch.argmax(logits, dim=1).item())
		
		# 若未指定目标类别，默认使用预测类别
		if target_class is None:
			target_class = pred_class
		
		# ===== 2. 反向传播：对目标类别 score 求梯度 =====
		score = logits[0, target_class]
		score.backward()
		
		if self.gradients is None:
			raise RuntimeError("Backward hook failed to capture gradients")
		if self.activations is None:
			raise RuntimeError("Forward hook failed to capture activations")
		
		# ===== 3. Grad-CAM 核心计算=====
		weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)  # [1, C, 1, 1]
		cam = torch.sum(weights * self.activations, dim=1)  # [1, H, W]
		cam = F.relu(cam)
		
		cam = cam - cam.min()
		cam = cam / (cam.max() + 1e-8)
		
		cam = F.interpolate(
			cam.unsqueeze(0), 
			size=(img_tensor.size(2), img_tensor.size(3)),
			mode='bilinear',
			align_corners=False
		)
		
		return cam.squeeze().cpu().numpy(), probs, pred_class

	def remove_hooks(self):
		"""显式摘除 hook，防止 hook 堆积/显存泄漏。"""
		for h in self._handles:
			h.remove()