# src/chiari/data/preprocessing.py
"""临床表格预处理：Excel → clinical.pkl"""
import logging
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)


def preprocess_and_save(excel_path, pkl_path, force=False):
	df = pd.read_excel(excel_path, header=1, nrows=280)
	df.columns = df.columns.str.strip()


	df_sorted = df.sort_values(by=['编号', '关节形态'], ascending=[True, False])
	patient_level =df.groupby('编号')['关节形态'].max().reset_index()
	# grouped = df.groupby('编号')
	# for patient_id, group_data in grouped:
	# 	print(f"=== 编号: {patient_id} 的分组 ===")
	# 	print(group_data)
	# 	print("----------------")
	# clinical_data = df.drop_duplicates(subset=['编号'], keep='first').copy()
	clinical_data = df_sorted.drop_duplicates(subset=['编号'], keep='first').copy()
	clinical_data = clinical_data.drop(columns= ['关节形态'])
	df = pd.merge(clinical_data, patient_level, on='编号')
	df['关节形态'] = df['关节形态'].apply(lambda x: 1 if x == 3 else 0)
	print(df.head())

	print(len(df))
	print(df.columns)
	data = pd.DataFrame()
	data['patient_id'] = df['编号'].astype(str).str.strip()
	# Categorical features (for Embedding)
	data['has_syrinx'] = df['合并脊髓空洞'].map({'是': 1, '有': 1, '否': 0, '无': 0}).fillna(0).astype(int)
	scale2_map = {'无痛': 0, '轻度疼痛': 1, '中度疼痛': 2, '重度疼痛': 3}
	data['scale2_1'] = df['疼痛分级'].map(scale2_map).fillna(0).astype(int)
	data['scale2_2'] = df['疼痛分级.1'].map(scale2_map).fillna(0).astype(int)
	data['scale2_3'] = df['疼痛分级.2'].map(scale2_map).fillna(0).astype(int)
	data['age']= df['年龄'].fillna(0).astype(int)
	data['gender']=df['性别'].map({'男': 1, '女': 0}).fillna(0).astype(int)
	asa_map = {'I': 0, 'II': 1, 'III': 2, 'IV': 3}
	data['asa']=df['ASA分级'].map(asa_map).fillna(0).astype(int)
	# Continuous features (for StandardScaler)
	data['syrinx_num'] = pd.to_numeric(df['脊髓空洞节段数量'], errors='coerce').fillna(0)
	data['syrinx_wide'] = pd.to_numeric(df['脊髓空洞最大宽度（只计算有空洞的）'], errors='coerce').fillna(0)
	data['CTH'] = pd.to_numeric(df['小脑扁桃体疝出程度'], errors='coerce').fillna(0)
	
	cisterna = pd.to_numeric(df['枕大池容积'], errors='coerce')
	# cisterna_max = cisterna.max()
	data['cisterna_flipped'] = (-cisterna).fillna(0)
	
	data['scale1'] = pd.to_numeric(df['量表1总分'], errors='coerce').fillna(0)
	data['scale3'] = pd.to_numeric(df['术前量表3总分'], errors='coerce').fillna(100)  # ADL，缺失=满分
	joa = pd.to_numeric(df['术前量表4总分'], errors='coerce')
	data['joa_flipped'] = (17 - joa).fillna(0)

	# 保存（标签从影像文件夹解析，这里不存label）
	data.to_pickle(pkl_path)
	print(f"[Preprocess] Saved clinical features to {pkl_path}")
	return data
