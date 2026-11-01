# src/chiari/models/tabnet.py
import torch.nn as nn
from chiari.models.clinical_encoder import ClinicalEncoder


class TabNet(nn.Module):
	"""
	纯表格分类网络：ClinicalEncoder + 分类头
	"""
	def __init__(self, num_cont=8, clin_out=16, num_classes=2):
		super().__init__()
		self.clinical_encoder = ClinicalEncoder(cont_dim=num_cont, out_dim=clin_out)
		self.classifier = nn.Sequential(
			nn.Linear(clin_out, 32),
			nn.BatchNorm1d(32),
			nn.ReLU(),
			nn.Dropout(0.3),
			nn.Linear(32, num_classes)
		)
		self._init_weights()

	def _init_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Linear):
				nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
				if m.bias is not None:
					nn.init.zeros_(m.bias)
			elif isinstance(m, nn.Embedding):
				nn.init.normal_(m.weight, std=0.1)

	def forward(self,img, cat_features, cont_features):
		"""
		Args:
			cat_features: dict of [B] LongTensor
			cont_features: [B, num_cont] FloatTensor
		Returns:
			logits: [B, num_classes]
		"""
		clin_feat = self.clinical_encoder(cat_features, cont_features)  # [B, 16]
		return self.classifier(clin_feat)  # [B, 2]