# src/chiari/models/clinical_encoder.py
"""临床表格编码器：类别 Embedding + 连续特征投影 → 融合表示。"""
import torch.nn as nn
import torch

class ClinicalEncoder(nn.Module):
	"""
	临床表格编码器：
	- Categorical -> Embedding
	- Continuous -> StandardScaler后投影
	- 融合 -> 16-d 临床表示
	"""
	def __init__(self, cont_dim=8, out_dim=16):
		super().__init__()

		# 
		self.embeddings = nn.ModuleDict({
			'has_syrinx': nn.Embedding(2, 2),      # 2 classes
			'scale2_1': nn.Embedding(4, 3),         # 4 classes (0-3)
			'scale2_2': nn.Embedding(4, 3),
			'scale2_3': nn.Embedding(4, 3),
			'gender': nn.Embedding(2,2),
			'asa': nn.Embedding(4,3),		
		})
		cat_dim = 2 + 3 + 3 + 3  +3+2# 16

		self.cont_proj = nn.Sequential(
			nn.Linear(cont_dim, 16),
			nn.BatchNorm1d(16),
			nn.ReLU(),
			nn.Dropout(0.2)
		)

		self.fusion = nn.Sequential(
			nn.Linear(cat_dim + 16, out_dim),
			nn.LayerNorm(16),
			nn.Dropout(p=0.1)
		)

	def forward(self, cat_features, cont_features):
		# cat_features: dict of [B] LongTensor
		embeds = []
		for key, emb in self.embeddings.items():
			embeds.append(emb(cat_features[key]))
		cat_feat = torch.cat(embeds, dim=-1)  # [B, 11]

		cont_feat = self.cont_proj(cont_features)  # [B, 16]

		combined = torch.cat([cat_feat, cont_feat], dim=-1)  # [B, 27]
		return self.fusion(combined)  # [B, 16]

