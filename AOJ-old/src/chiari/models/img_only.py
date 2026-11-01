# src/chiari/models/image_only.py —— arch='image_only'

from torchvision.models import resnet18


import torch
import torch.nn as nn
import torchvision.models as models

class AOJBIResNET(nn.Module):
	def __init__(self, pretrained_path = None):
		super(AOJBIResNET, self).__init__()
		self.resnet = models.resnet18(weights=None)
		num_ftrs= self.resnet.fc.in_features
		self.resnet.fc = nn.Sequential(
						nn.Linear(num_ftrs, 2),
		)
		# 读取权重
		if pretrained_path:
			self.load_smart_weights(pretrained_path)


	def load_smart_weights(self, path):
		# print(f"DEBUG:正在加载权重文件: {path}")
		state_dict = torch.load(path, map_location='cpu', weights_only=False)
		# 自动拆包 (针对 'state_dict' 等外壳)
		possible_keys = ['state_dict', 'model', 'net', 'model_state_dict']
		for key in possible_keys:
			if key in state_dict and isinstance(state_dict[key], dict):
				# print(f"DEBUG:发现权重被包裹在 '{key}' 中，正在拆包...")
				state_dict = state_dict[key]
				break
		model_state_dict = self.state_dict()
		new_state_dict = {}
		matched_layers = 0
		# print(f"DEBUG:开始匹配 (文件包含 {len(state_dict)} 个键)...")
		for k, v in state_dict.items():
			# 多重前缀清洗
			key_clean = k.replace('_orig_mod.', '')
			key_clean = key_clean.replace('module.', '')
			target_key = "resnet." + key_clean
			# 开始尝试匹配
			final_key = None
			if target_key in model_state_dict:
				final_key = target_key
			elif key_clean in model_state_dict:
				final_key = key_clean
			if final_key:
				if v.shape == model_state_dict[final_key].shape:
					new_state_dict[final_key] = v
					matched_layers += 1
				else:
					# 形状不匹配跳过 
					pass
		self.load_state_dict(new_state_dict, strict=False)
		# print(f"DEBUG: 权重加载完成！成功匹配: {matched_layers}")

		if matched_layers == 0:
				# print(f"DEBUG: 匹配数为 0")
				# print(f"DEBUG:文件里的第一条Key (清洗后): {key_clean}")
				# print(f"DEBUG:模型里的第一条Key (预期): {list(model_state_dict.keys())[0]}")
				raise RuntimeError("DEBUG:无法匹配任何权重，请检查上述 Key 的差异。")

	def load_local_weights(self, path):
		state_dict = torch.load(path, map_location='cpu')

		_ = self.load_state_dict(state_dict, strict=False)
		# print(f"DEBUG:本地权重加载结果:{msg}")

	def forward(self, x,cat_features, cont_features):
		return self.resnet(x)