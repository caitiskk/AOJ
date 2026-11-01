"""
单关节形态预测入口。
用法示例：
    python scripts/predict_joint.py --config configs/default.yaml --fold 1 --pid 001 --side left
    python scripts/predict_joint.py --config configs/default.yaml --fold 1 --pid 042 --side right --threshold 0.25
"""
import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
import re
from chiari.config import load_config
from chiari.data.dataset import ChiariXrayDataset
from chiari.data.transforms import get_transforms
from chiari.engine.evaluate import predict_single_joint
from chiari.utils.checkpoint import ckpt_path, load_fold_model


def parse_args():
    parser = argparse.ArgumentParser(description="Chiari 单关节影像形态判定")
    parser.add_argument("--config", default="configs/default.yaml", help="配置文件路径")
    parser.add_argument("--fold", type=int, default=1, help="指定使用的模型折数 (1~5)")
    parser.add_argument("--pid", type=str, required=True, help="病人编号，例如 '001'")
    parser.add_argument("--side", type=str, choices=["left", "right", "L", "R"], required=True, help="关节侧别")
    parser.add_argument("--threshold", type=float, default=None, help="覆盖 yaml 中的阈值")
    parser.add_argument("--gpu", default=None, help="GPU 编号")
    return parser.parse_args()

def parse_side_target(side: str) -> str:
    """归一化传入的侧别，返回标准中文 '左' 或 '右'"""
    s = str(side).strip().lower()
    if s.startswith("l") or "左" in s:
        return "左"
    elif s.startswith("r") or "右" in s:
        return "右"
    else:
        raise ValueError(f"无法识别的侧别: '{side}'，仅支持 left/right/l/r/左/右")


def load_joint_images_by_dir(image_root, pid, side, img_size=224):
    """
    层级匹配检索：
      image_root / <pid> / <包含'左'或'右'的文件夹> / *.JPG
    """
    image_root = Path(image_root)
    pid_str = str(pid).strip()
    target_char = parse_side_target(side)  # '左' 或 '右'

    # 1. 定位病人文件夹 (例如 69)
    patient_dir = None
    if (image_root / pid_str).is_dir():
        patient_dir = image_root / pid_str
    else:
        # 兼容补零或去零匹配 (如 069 与 69)
        for p in image_root.iterdir():
            if p.is_dir() and p.name.lstrip("0") == pid_str.lstrip("0"):
                patient_dir = p
                break

    if patient_dir is None:
        raise FileNotFoundError(f"未在 {image_root} 目录下找到病人文件夹: '{pid_str}'")

    # 2. 遍历病人目录下的所有子文件夹，只要文件夹名中包含 '左' 或 '右' 就选中
    target_joint_dir = None
    for sub_dir in patient_dir.iterdir():
        if sub_dir.is_dir() and target_char in sub_dir.name:
            target_joint_dir = sub_dir
            break

    if target_joint_dir is None:
        existing = [d.name for d in patient_dir.iterdir() if d.is_dir()]
        raise FileNotFoundError(
            f"在病人目录 '{patient_dir.name}' 下未找到名字包含 '{target_char}' 的关节文件夹！\n"
            f"当前目录下的子文件夹有: {existing}"
        )

    # 3. 获取该文件夹下的所有图片文件
    valid_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    img_paths = [
        f for f in target_joint_dir.iterdir()
        if f.is_file() and f.suffix.lower() in valid_exts
    ]

    if not img_paths:
        raise FileNotFoundError(f"关节目录 '{target_joint_dir.name}' 中未找到有效格式的图像切片！")

    # 按文件名中的数字自然排序 (如 1.JPG, 2.JPG, 3.JPG, 4.JPG)
    def natural_sort_key(p: Path):
        num = re.findall(r"\d+", p.stem)
        return int(num[0]) if num else p.stem

    img_paths = sorted(img_paths, key=natural_sort_key)

    # 4. 转为 Tensor 并堆叠为 Batch [N, 3, H, W]
    transform = get_transforms("val", img_size=img_size)
    tensors = []
    for p in img_paths:
        img = Image.open(p).convert("RGB")
        tensors.append(transform(img))

    imgs_tensor = torch.stack(tensors, dim=0)
    return imgs_tensor, img_paths, target_joint_dir.name


import numpy as np
import pandas as pd
import torch


def load_patient_clinical_from_excel(excel_path, ckpt_dict, pid, num_imgs):
    """直接读取 Excel，使用与 preprocess_and_save 严格相同的逻辑清洗特征，

    并调用 checkpoint 内拟合好的 scaler 进行标准化。
    """
    # 1. 读取 Excel（与 preprocess_and_save 的 header 与行数完全一致）
    df = pd.read_excel(excel_path, header=1, nrows=280)
    df.columns = df.columns.str.strip()

    # 2. 复现原工程的排序去重与患者级合并逻辑
    df_sorted = df.sort_values(by=['编号', '关节形态'], ascending=[True, False])
    patient_level = df.groupby('编号')['关节形态'].max().reset_index()
    clinical_data = df_sorted.drop_duplicates(
        subset=['编号'], keep='first'
    ).copy()
    clinical_data = clinical_data.drop(columns=['关节形态'])
    df_merged = pd.merge(clinical_data, patient_level, on='编号')

    # 3. 严格复现 preprocess_and_save 的离散与连续特征映射
    data = pd.DataFrame()
    data['patient_id'] = df_merged['编号'].astype(str).str.strip()

    # ── 离散类别特征 (Categorical for Embedding) ──
    data['has_syrinx'] = (
        df_merged['合并脊髓空洞']
        .map({'是': 1, '有': 1, '否': 0, '无': 0})
        .fillna(0)
        .astype(int)
    )
    scale2_map = {'无痛': 0, '轻度疼痛': 1, '中度疼痛': 2, '重度疼痛': 3}
    data['scale2_1'] = (
        df_merged['疼痛分级'].map(scale2_map).fillna(0).astype(int)
    )
    data['scale2_2'] = (
        df_merged['疼痛分级.1'].map(scale2_map).fillna(0).astype(int)
    )
    data['scale2_3'] = (
        df_merged['疼痛分级.2'].map(scale2_map).fillna(0).astype(int)
    )
    data['age'] = df_merged['年龄'].fillna(0).astype(int)
    data['gender'] = (
        df_merged['性别'].map({'男': 1, '女': 0}).fillna(0).astype(int)
    )
    asa_map = {'I': 0, 'II': 1, 'III': 2, 'IV': 3}
    data['asa'] = df_merged['ASA分级'].map(asa_map).fillna(0).astype(int)

    # ── 连续特征 (Continuous for StandardScaler) ──
    data['syrinx_num'] = pd.to_numeric(
        df_merged['脊髓空洞节段数量'], errors='coerce'
    ).fillna(0)
    data['syrinx_wide'] = pd.to_numeric(
        df_merged['脊髓空洞最大宽度（只计算有空洞的）'], errors='coerce'
    ).fillna(0)
    data['CTH'] = pd.to_numeric(
        df_merged['小脑扁桃体疝出程度'], errors='coerce'
    ).fillna(0)

    cisterna = pd.to_numeric(df_merged['枕大池容积'], errors='coerce')
    data['cisterna_flipped'] = (-cisterna).fillna(0)

    data['scale1'] = pd.to_numeric(
        df_merged['量表1总分'], errors='coerce'
    ).fillna(0)
    data['scale3'] = pd.to_numeric(
        df_merged['术前量表3总分'], errors='coerce'
    ).fillna(100)
    joa = pd.to_numeric(df_merged['术前量表4总分'], errors='coerce')
    data['joa_flipped'] = (17 - joa).fillna(0)

    # 4. 匹配目标患者 ID（支持 69 与 069 容错匹配）
    pid_str = str(pid).strip()
    matched = data[data['patient_id'] == pid_str]
    if matched.empty:
      matched = data[data['patient_id'].str.lstrip('0') == pid_str.lstrip('0')]

    if matched.empty:
      raise ValueError(
          f"未在 Excel ({excel_path}) 中找到患者编号为 '{pid}' 的记录！\n"
          f'前10位患者编号示例: {data["patient_id"].head(10).tolist()}'
      )

    row = matched.iloc[0]

    # 5. 从 Checkpoint 中取出训练阶段保存的列名与 scaler
    cat_cols = ckpt_dict.get(
        'cat_cols',
        [
            'has_syrinx',
            'scale2_1',
            'scale2_2',
            'scale2_3',
            'age',
            'gender',
            'asa',
        ],
    )
    cont_cols = ckpt_dict.get(
        'cont_cols',
        [
            'syrinx_num',
            'syrinx_wide',
            'CTH',
            'cisterna_flipped',
            'scale1',
            'scale3',
            'joa_flipped',
        ],
    )
    scaler = ckpt_dict.get('scaler', None)

    # 6. 连续特征：必须使用训练折所保存的 scaler 做相同变换
    cont_vals = row[cont_cols].values.astype(np.float32).reshape(1, -1)
    if scaler is not None:
      cont_vals = scaler.transform(cont_vals)
    cont_batch = torch.tensor(cont_vals, dtype=torch.float32).expand(
        num_imgs, -1
    )

    # 7. 类别特征：转为 LongTensor 并沿切片数量 expand
    cat_batch = {}
    for col in cat_cols:
      val = int(row[col])
      cat_batch[col] = torch.tensor([val], dtype=torch.long).expand(num_imgs)

    return cat_batch, cont_batch

def main():
    args = parse_args()
    cfg = load_config(args.config, gpu=args.gpu)
    device = cfg["device"]

    # 1. 确定阈值与权重路径
    threshold = args.threshold if args.threshold is not None else cfg["eval"]["threshold"]
    ckpt_file = ckpt_path(cfg["paths"]["save_dir"], args.fold)
    if not Path(ckpt_file).exists():
        # 兼容另一种命名风格: 1_medicalgmu_best.pt
        alt_ckpt = Path(cfg["paths"]["save_dir"]) / f"{args.fold}_medicalgmu_best.pt"
        ckpt_file = alt_ckpt if alt_ckpt.exists() else ckpt_file

    print(f"\n[1/4] 载入 Checkpoint: {ckpt_file}")
    model, ckpt_dict = load_fold_model(ckpt_file, device)

    # 2. 借用 from_checkpoint 恢复 scaler 与特征列表定义
    print("[2/4] 从 Checkpoint 恢复数据变换与标准化器...")
    val_ds = ChiariXrayDataset.from_checkpoint(ckpt_file)

    # 3. 检索并加载该关节的切片
    print(f"[3/4] 检索病例 [{args.pid}] - 侧别 [{args.side}] 的切片...")
    img_size = cfg["training"].get("img_size", 224)
    imgs_tensor, img_paths, folder_name = load_joint_images_by_dir(
        image_root=cfg["data"]["image_root"],
        pid=args.pid,
        side=args.side,
        img_size=img_size,
    )
    print(f"      找到 {imgs_tensor.size(0)} 张切片: {[p.name for p in img_paths]}")

    # 4. 构建对齐好的临床输入特征
    cat_batch, cont_batch = None, None
    if cfg["model"].get("arch") != "image_only":
        cat_batch, cont_batch = load_patient_clinical_from_excel(
    excel_path=cfg['data']['excel_path'],
    ckpt_dict=ckpt_dict,
    pid=args.pid,
    num_imgs=imgs_tensor.size(0)
)

    # 5. 执行推理
    result = predict_single_joint(
        model=model,
        imgs=imgs_tensor,
        cat_batch=cat_batch,
        cont_batch=cont_batch,
        device=device,
        threshold=threshold,
    )

    # 6. 打印报告
    type_str = "III型 (病变/异常)" if result["pred_label"] == 1 else "正常/I-II型 (阴性)"
    print("\n" + "=" * 50)
    print(f"      关节形态诊断报告 ({args.pid}_{args.side})")
    print("=" * 50)
    print(f"使用模型       : Fold {args.fold}")
    print(f"切片输入数量   : {result['num_images']} 张")
    print(f"各切片预测概率 : {[round(p, 4) for p in result['img_probs']]}")
    print(f"关节平均概率   : {result['avg_prob']:.4f}")
    print(f"判断阈值       : {threshold}")
    print(f"综合形态判定   : 【 {type_str} 】")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()