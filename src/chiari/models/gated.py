# src/chiari/models/gated.py —— arch='gmu_standard'

from chiari.models.clinical_encoder import ClinicalEncoder
from chiari.models.fusion import StandardGMUFusion
from torchvision.models import resnet18
import torch
import torch.nn as nn

class MedicalGatedNet(nn.Module):
	def __init__(self, num_cont=8, clin_out=16, pretrained_path=None, dropout=0.4):
		super().__init__()

		# --- 影像塔 ---
		self.resnet = resnet18(weights='DEFAULT')
		self.resnet.fc = nn.Identity()
		if pretrained_path:
			self.load_smart_weights(pretrained_path)

		# --- 临床塔 ---
		self.clinical_encoder = ClinicalEncoder(cont_dim=num_cont, out_dim=clin_out)

		# --- 标准 GMU 融合层（融合空间保持为 512 维） ---
		self.gate_fusion = StandardGMUFusion(img_dim=512, clin_dim=clin_out, hidden_dim=512)

		# --- 分类头 ---
		self.classifier = nn.Sequential(
			nn.Linear(512, 2),
		)

		self._init_weights()

	def load_smart_weights(self, path):
		print(f"正在加载权重文件: {path}")
		state_dict = torch.load(path, map_location='cpu', weights_only=False)
		possible_keys = ['state_dict', 'model', 'net', 'model_state_dict']
		for key in possible_keys:
			if key in state_dict and isinstance(state_dict[key], dict):
				print(f"发现权重被包裹在 '{key}' 中，正在拆包...")
				state_dict = state_dict[key]
				break
		model_state_dict = self.state_dict()
		new_state_dict = {}
		matched_layers = 0
		print(f" 开始匹配 (文件包含 {len(state_dict)} 个键)...")
		for k, v in state_dict.items():
			key_clean = k.replace('_orig_mod.', '').replace('module.', '')
			target_key = "resnet." + key_clean
			final_key = None
			if target_key in model_state_dict:
				final_key = target_key
			elif key_clean in model_state_dict:
				final_key = key_clean
			if final_key:
				if v.shape == model_state_dict[final_key].shape:
					new_state_dict[final_key] = v
					matched_layers += 1
		self.load_state_dict(new_state_dict, strict=False)
		print(f"权重加载完成！成功匹配: {matched_layers}")
		if matched_layers == 0:
			print(f" 警告：匹配数为 0！")
			print(f" 文件里的第一条Key (清洗后): {key_clean}")
			print(f"  模型里的第一条Key (预期): {list(model_state_dict.keys())[0]}")
			raise RuntimeError("无法匹配任何权重，请检查上述 Key 的差异。")

	def _init_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Linear):
				nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
				if m.bias is not None:
					nn.init.zeros_(m.bias)
			elif isinstance(m, nn.Embedding):
				nn.init.normal_(m.weight, std=0.1)

	def forward(self, img, cat_features, cont_features):
		img_feat = self.resnet(img)  # [B, 512]
		clin_feat = self.clinical_encoder(cat_features, cont_features)  # [B, 16]

		fused = self.gate_fusion(img_feat, clin_feat)  # [B, 512]

		logits = self.classifier(fused)  # [B, 2]
		return logits