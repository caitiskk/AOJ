# src/chiari/models/multimodal.py
"""顶层多模态网络：X-ray(ResNet18) + 临床表格 → III 型二分类。"""
import torch.nn as nn

from chiari.models.clinical_encoder import ClinicalEncoder
from chiari.models.fusion import MedicalGMUFusion

from torchvision.models import resnet18
import torch
import torch.nn as nn

class MedicalGMUNet(nn.Module):
	def __init__(self, num_cont=7, clin_out=16,pretrained_path = None, dropout=0.4):
		super().__init__()

		# --- 影像塔：ResNet18，单通道 X-ray ---
		self.resnet = resnet18(weights='DEFAULT')
		num_ftrs= self.resnet.fc.in_features

		self.resnet.fc = nn.Identity()
		if pretrained_path:
			self.load_smart_weights(pretrained_path)

		# --- 临床塔 ---
		self.clinical_encoder = ClinicalEncoder(cont_dim=num_cont, out_dim=clin_out)

		# --- GMU 融合 ---
		self.gmu = MedicalGMUFusion(img_dim=512, clin_dim=clin_out)

		# --- 分类头 ---
		self.classifier = nn.Sequential(
			nn.Dropout(0.5),
			nn.Linear(512, 2),
			# nn.BatchNorm1d(256),
			# nn.ReLU(),
			# nn.Dropout(CONFIG['dropout']),
			# nn.Linear(256, 2)
		)

		self._init_weights()

	def load_smart_weights(self, path):
		# print(f"正在加载权重文件: {path}")
		state_dict = torch.load(path, map_location='cpu', weights_only=False)
		possible_keys = ['state_dict', 'model', 'net', 'model_state_dict']
		for key in possible_keys:
			if key in state_dict and isinstance(state_dict[key], dict):
				# print(f"发现权重被包裹在 '{key}' 中，正在拆包...")
				state_dict = state_dict[key]
				break
		model_state_dict = self.state_dict()
		new_state_dict = {}
		matched_layers = 0
		# print(f" 开始匹配 (文件包含 {len(state_dict)} 个键)...")
		for k, v in state_dict.items():
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
					pass

		self.load_state_dict(new_state_dict, strict=False)
		# print(f"权重加载完成！成功匹配: {matched_layers}")

		if matched_layers == 0:
				# print(f" 警告：匹配数为 0！")
				# print(f" 文件里的第一条Key (清洗后): {key_clean}")
				# print(f"  模型里的第一条Key (预期): {list(model_state_dict.keys())[0]}")
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
		# img: [B, 1, H, W]
		img_feat = self.resnet(img)  # [B, 512]

		clin_feat = self.clinical_encoder(cat_features, cont_features)  # [B, 16]
		fused = self.gmu(img_feat, clin_feat)  # [B, 512]

		logits = self.classifier(fused)  # [B]
		return logits

def build_model(cfg, pretrained=True):
    """从 config 构建模型 —— 全项目唯一的模型构造入口。

    pretrained=True : 训练起点，加载 cfg['data']['resnet_weight'] 的 ImageNet 权重
    pretrained=False: 从 checkpoint 重建（权重随后被 load_state_dict 覆盖，
                      跳过预训练加载省一次磁盘 IO）
    """
    m = cfg['model']
    return MedicalGMUNet(
        num_cont=m['num_cont'],
        clin_out=m['clin_out'],
        dropout=m.get('dropout', 0.4),
        pretrained_path=cfg['data']['resnet_weight'] if pretrained else None,
    )