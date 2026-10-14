# src/chiari/utils/checkpoint.py
"""checkpoint 重建工具：文件名模板 + 模型重建的单一入口。"""
from pathlib import Path
import torch

from chiari.models import build_model

CKPT_TEMPLATE = '{fold}_medicalgmu_best.pt'   # ← 全项目唯一定义处，修掉 fold_ 前缀不一致


def ckpt_path(save_dir, fold):
    return Path(save_dir) / CKPT_TEMPLATE.format(fold=fold)


def load_fold_model(path, device):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    model = build_model(ckpt['config'], pretrained=False)   # ← 仅重建结构，不加载 ImageNet 权重
    model.load_state_dict(ckpt['model_state'])
    return model.to(device).eval(), ckpt