#src/chiari/engine/predicter.py
# 该文件为预测单个关节结果
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, confusion_matrix
from collections import defaultdict
from sklearn.metrics import (
	accuracy_score, precision_score, recall_score, 
	f1_score, roc_auc_score, confusion_matrix, average_precision_score,
)

import pandas as pd

@torch.no_grad()
def predict_single_joint(
    model, 
    imgs, 
    cat_features=None, 
    cont_features=None, 
    device="cuda", 
    threshold=0.25
):
    """
    对单个关节的一组影像（多切片/多视图）进行联合推理与形态判定。
    
    Args:
        model: 训练好的 PyTorch 模型
        imgs: 图像 Tensor，shape 可以为:
              - [N, C, H, W] (N 张图作为 batch)
              - [1, C, H, W] 或 [C, H, W] (仅 1 张图)
        cat_features: 分类特征 dict {k: Tensor}。
                      如果输入是一维/未扩展维度，函数会自动扩展为 batch 大小 N。
        cont_features: 连续特征 Tensor。同样会自动对齐为 [N, D]。
        device: 计算设备 ('cuda' 或 'cpu')
        threshold: 分类阈值（默认使用训练或验证阶段得到的 best_t，如 0.25）
        
    Returns:
        dict: {
            'pred_label': int (0 或 1),
            'avg_prob': float (该关节的平均阳性概率),
            'img_probs': list[float] (每张输入切片的单独概率),
            'num_images': int (输入的切片数量)
        }
    """
    model.eval()
    
    # 1. 规范化图像维度为 [N, C, H, W]
    if isinstance(imgs, (list, tuple)):
        imgs = torch.stack(imgs, dim=0)
    elif imgs.dim() == 3:
        imgs = imgs.unsqueeze(0)  # [C, H, W] -> [1, C, H, W]
        
    imgs = imgs.to(device)
    num_imgs = imgs.size(0)

    # 2. 对齐临床/表格特征维度与 N 一致
    processed_cat = {}
    if cat_features is not None:
        for k, v in cat_features.items():
            v = v.to(device)
            # 若是标量或形状不匹配，扩展至 batch 大小
            if v.dim() == 0 or (v.dim() == 1 and v.size(0) == 1):
                v = v.view(1).expand(num_imgs)
            elif v.size(0) != num_imgs:
                v = v.expand(num_imgs, *v.shape[1:])
            processed_cat[k] = v

    processed_cont = cont_features
    if cont_features is not None:
        processed_cont = cont_features.to(device)
        if processed_cont.dim() == 1:
            processed_cont = processed_cont.unsqueeze(0).expand(num_imgs, -1)
        elif processed_cont.size(0) == 1 and num_imgs > 1:
            processed_cont = processed_cont.expand(num_imgs, -1)

    # 3. 模型前向传播
    logits = model(imgs, processed_cat, processed_cont)

    # 4. 兼容单输出与双输出
    if logits.dim() == 1 or logits.size(-1) == 1:
        probs = torch.sigmoid(logits).cpu().numpy()
        if probs.ndim == 2:
            probs = probs.squeeze(-1)
        elif probs.ndim == 0:  # 只有 1 张图时的标量处理
            probs = np.array([probs.item()])
    else:
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

    # 5. 计算单关节的汇总结果
    img_probs = [float(p) for p in probs]
    avg_prob = float(np.mean(img_probs))
    pred_label = int(avg_prob >= threshold)

    return {
        "pred_label": pred_label,
        "avg_prob": avg_prob,
        "img_probs": img_probs,
        "num_images": num_imgs,
    }