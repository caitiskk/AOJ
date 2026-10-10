# src/chiari/data/dataset.py
"""多模态数据集：X-ray 影像 + 临床表格特征。

契约（__getitem__ 返回 6 元组，collate_fn 与模型 forward 共同依赖）：
	img          FloatTensor [3, H, W]   已归一化
	label        int                      0=Non-III, 1=III
	cat_features dict[str, LongTensor]   各类别特征（embedding 输入）
	cont_features FloatTensor [n_cont]   已标准化
	patient_id   str
	joint_id     str
"""
import torch
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler


from PIL import Image

__all__ = ['ChiariXrayDataset', 'collate_fn']


class ChiariXrayDataset(Dataset):
	def __init__(self, pkl_path, image_root, patient_ids, transform=None,
				 scaler=None, fit_scaler=False):
		"""
		patient_ids: 当前Fold的病人ID列表（str）
		"""
		self.root_dir = Path(image_root)
		self.transform = transform
		self.samples = []

		# 加载临床表格
		df = pd.read_pickle(pkl_path)
		print(len(df))
		df['patient_id'] = df['patient_id'].astype(str).str.strip()
		df_indexed = df.set_index('patient_id')
		# print(df_indexed)
		# 特征列定义（与你上传的代码严格一致）
		self.cat_cols = ['has_syrinx', 'scale2_1', 'scale2_2', 'scale2_3','gender','asa']
		self.cont_cols = ['syrinx_num', 'syrinx_wide', 'CTH',
							'cisterna_flipped', 'scale1', 'scale3', 'joa_flipped','age']

		# Continuous 标准化（仅训练集fit）
		if fit_scaler:
			self.scaler = StandardScaler()
			df[self.cont_cols] = self.scaler.fit_transform(df[self.cont_cols])
		else:
			self.scaler = scaler
			if scaler is not None:
				df[self.cont_cols] = scaler.transform(df[self.cont_cols])

		# 遍历指定病人，解析影像文件夹
		for p_id in patient_ids:
			p_id_str = str(p_id).strip()
			if p_id_str not in df_indexed.index:
				continue

			row = df_indexed.loc[p_id_str]

			# Categorical: dict of int scalars
			cat_dict = {c: int(row[c]) for c in self.cat_cols}
			# Continuous: numpy array -> tensor
			cont_arr = row[self.cont_cols].values.astype(float)

			patient_folder = self.root_dir / p_id_str
			if not patient_folder.is_dir():
				continue

			# 遍历左右关节
			for side_folder in patient_folder.iterdir():
				if not side_folder.is_dir():
					continue
				folder_name = side_folder.name

				side = 'left' if '左' in folder_name else 'right'

				# 标签从文件夹名解析（3型/III型 = 1）
				if '3型' in folder_name or 'III型' in folder_name:
					label = 1
				else:
					label = 0

				# 遍历该关节下所有影像
				for img_file in side_folder.glob('*.*'):
					if img_file.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
						self.samples.append({
							'path': str(img_file),
							'label': label,
							'patient_id': p_id_str,
							'side': side,
							'cat_dict': cat_dict,      # dict of ints
							'cont_arr': cont_arr.copy()  # float array
						})

		print(f"[Dataset] 病人: {len(set(s['patient_id'] for s in self.samples))}, "
				f"影像: {len(self.samples)}, "
				f"正类(III型): {sum(s['label'] for s in self.samples)}")

	def __len__(self):
		return len(self.samples)

	def __getitem__(self, idx):
		s = self.samples[idx]
		img = Image.open(s['path']).convert('L')
		if self.transform:
			img = self.transform(img)

		label = torch.tensor(s['label'], dtype=torch.long)

		# Categorical: dict of tensors (for Embedding)
		cat_features = {
			k: torch.tensor(v, dtype=torch.long)
			for k, v in s['cat_dict'].items()
		}
		# Continuous: tensor (already scaled)
		cont_features = torch.tensor(s['cont_arr'], dtype=torch.float32)

		joint_id = f"{s['patient_id']}_{s['side']}"

		return img, label, cat_features, cont_features, s['patient_id'], joint_id

	@classmethod
	def from_checkpoint(cls, ckpt_path, transform=None):
		"""从 checkpoint 重建该折验证集（GradCAM / 推理 / 复现用）。

		读取训练时存入 ckpt 的 scaler / val_patient_ids / config，
		逐 bit 还原该折的验证集构造参数 —— 免去手工重算 KFold 的复现陷阱。
		"""
		import torch
		ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

		cfg = ckpt['config']                    # 训练时存入的完整配置 dict
		
		if transform is None:
			from chiari.data.transforms import get_transforms
			transform = get_transforms('val', img_size=cfg['training']['img_size'])

		return cls(
			pkl_path=cfg['data']['pkl_path'],
			image_root=cfg['data']['image_root'],
			patient_ids=ckpt['val_patient_ids'],   # ← 精确还原该折验证病人
			transform=transform,
			scaler=ckpt['scaler'],                 # ← 精确还原标准化
			fit_scaler=False,
		)


def collate_fn(batch):
	"""拼接 batch，处理 dict 型 categorical features"""
	imgs = torch.stack([b[0] for b in batch])
	labels = torch.stack([b[1] for b in batch])

	cat_keys = batch[0][2].keys()
	cat_batch = {k: torch.stack([b[2][k] for b in batch]) for k in cat_keys}

	cont_batch = torch.stack([b[3] for b in batch])
	patient_ids = [b[4] for b in batch]
	joint_ids = [b[5] for b in batch]

	return imgs, labels, cat_batch, cont_batch, patient_ids, joint_ids