# src/chiari/models/fusion.py
"""GMU 融合门控单元。"""
import torch.nn as nn
import torch


class ImprovedMedicalGMUFusion(nn.Module):
	"""
	进阶改良 GMU：
	1. 影像特征先降维对齐临床维度，再联合生成门控（兼顾影像质量感知与防过拟合）
	2. 采用瓶颈结构（Bottleneck Gate），避免 512x512 的参数爆炸
	3. 保留零初始化残差连接，确保训练初期影像特征稳定
	"""
	def __init__(self, img_dim=512, clin_dim=16, bottleneck_dim=32, gate_dropout=0.2):
		super().__init__()
		
		# 1. 临床特征升维映射 -> [B, img_dim]
		self.clin_proj = nn.Sequential(
			nn.Linear(clin_dim, 128),
			nn.ReLU(),
			nn.Linear(128, img_dim)
		)
		
		# 2. 影像特征降维压缩 -> [B, clin_dim]（强制与临床特征站在同一起跑线）
		self.img_compress = nn.Sequential(
			nn.Linear(img_dim, clin_dim),
			nn.LayerNorm(clin_dim),
			nn.ReLU()
		)
		
		# 3. 联合门控：对齐后的 [img_compressed, clin_feat] 共同决定权重
		self.gate = nn.Sequential(
			nn.Linear(clin_dim * 2, bottleneck_dim),
			nn.ReLU(),
			nn.Dropout(gate_dropout),  # 防止门控走捷径锁死
			nn.Linear(bottleneck_dim, img_dim),
			nn.Sigmoid()
		)
		
		fusion_mid_dim = 64  # 可选 64 或 128
		self.fusion = nn.Sequential(
			nn.Linear(img_dim, fusion_mid_dim, bias=False),
			nn.BatchNorm1d(fusion_mid_dim),
			nn.ReLU(inplace=True),
			nn.Linear(fusion_mid_dim, img_dim, bias=False),
			nn.BatchNorm1d(img_dim)  # 末尾不加 ReLU，允许残差有正有负
		)
		
		# 零初始化残差缩放因子
		self.fusion_scale = nn.Parameter(torch.zeros(1))

	def forward(self, img_feat, clin_feat):
		# 临床特征映射到高维: [B, img_dim]
		c = self.clin_proj(clin_feat)
		
		# 影像特征压缩到低维: [B, clin_dim]
		img_low = self.img_compress(img_feat)
		
		# 联合生成门控: [B, clin_dim * 2] -> [B, img_dim]
		joint_context = torch.cat([img_low, clin_feat], dim=-1)
		z = self.gate(joint_context)

		# GMU 加权融合
		fused = z * img_feat + (1.0 - z) * c
		out = self.fusion(fused)
		
		return self.fusion_scale * out + img_feat






class MedicalGMUFusion(nn.Module):
	"""
	改良 GMU：
	- gate 仅由临床特征生成（非对称，防止偷懒）
	- clin_proj 把临床映射到影像维度
	- 残差 +img_feat 保证影像信息不丢失
	"""
	def __init__(self, img_dim=512, clin_dim=16):
		super().__init__()
		self.clin_proj = nn.Sequential(
			nn.Linear(clin_dim, 64),
			nn.ReLU(),
			nn.Linear(64, img_dim)
		)
		# 门控：临床 → 影像维度权重 (0,1)
		self.gate = nn.Sequential(
			nn.Linear(clin_dim, img_dim),
			nn.Sigmoid()
		)
		# 融合后投影
		self.fusion = nn.Sequential(
			nn.Linear(img_dim, img_dim),
			nn.BatchNorm1d(img_dim),
			nn.ReLU()
		)
		# 更好的融合
		self.fusion_scale = nn.Parameter(torch.zeros(1))

	def forward(self, img_feat, clin_feat):
		# clin_feat: [B, clin_dim]
		c = self.clin_proj(clin_feat)   # [B, img_dim]
		z = self.gate(clin_feat)         # [B, img_dim]

		# GMU 加权：影像为主，临床补充/抑制
		fused = z * img_feat + (1.0 - z) * c
		# 残差：即使 gate≈0，影像特征仍能直通
		out = self.fusion(fused) 
		# 强化影像主导地位
		return self.fusion_scale * out+ img_feat



class StandardGMUFusion(nn.Module):
	"""
	严格遵循 ICLR 2017 论文公式的标准双模态门控多模态单元 (GMU)[cite: 3]
	公式实现:
	h_v = tanh(W_v * x_v)[cite: 3]
	h_c = tanh(W_c * x_c)[cite: 3]
	z   = sigmoid(W_z * [x_v, x_c])[cite: 3]
	h   = z * h_v + (1.0 - z) * h_c[cite: 3]
	"""
	def __init__(self, img_dim=512, clin_dim=16, hidden_dim=512):
		super().__init__()
		# 影像特征投影矩阵 (W_v)[cite: 3]
		self.W_v = nn.Linear(img_dim, hidden_dim, bias=False)
		
		# 临床特征投影矩阵 (W_c)[cite: 3]
		self.W_c = nn.Linear(clin_dim, hidden_dim, bias=False)
		
		# 门控特征投影矩阵 (W_z)：接收两路原始特征的拼接[cite: 3]
		self.W_z = nn.Linear(img_dim + clin_dim, hidden_dim)
		
		# 标准激活层[cite: 3]
		self.tanh = nn.Tanh()
		self.sigmoid = nn.Sigmoid()

	def forward(self, img_feat, clin_feat):
		# img_feat (x_v): [B, 512], clin_feat (x_c): [B, 16]

		# 1. 各模态分别计算经过 tanh 转换后的隐藏表征[cite: 3]
		h_v = self.tanh(self.W_v(img_feat))   # [B, hidden_dim]
		h_c = self.tanh(self.W_c(clin_feat))  # [B, hidden_dim]

		# 2. 门控向量 z 由最原始的双模态特征拼接共同决定[cite: 3]
		gate_input = torch.cat([img_feat, clin_feat], dim=-1)  # [B, 512 + 16]
		z = self.sigmoid(self.W_z(gate_input))                 # [B, hidden_dim]

		# 3. 在完全对齐的隐藏语义空间中执行标准加权融合[cite: 3]
		fused = z * h_v + (1.0 - z) * h_c   # [B, hidden_dim]

		return fused

