# src/chiari/utils/checkpoint.py
"""checkpoint 重建工具：文件名模板 + 模型重建的单一入口。"""
from pathlib import Path
import torch

from chiari.models import build_model
from chiari.config import adapt_config

CKPT_TEMPLATE = '{fold}_medicalgmu_best.pt'   # ← 全项目唯一定义处，修掉 fold_ 前缀不一致


def ckpt_path(save_dir, fold):
    return Path(save_dir) / CKPT_TEMPLATE.format(fold=fold)


# def load_fold_model(path, device):
#     ckpt = torch.load(path, map_location='cpu', weights_only=False)
#     model = build_model(ckpt['config'], pretrained=False)   # ← 仅重建结构，不加载 ImageNet 权重
#     model.load_state_dict(ckpt['model_state'])
#     return model.to(device).eval(), ckpt

def load_fold_model(path, device):
    """同时兼容新旧格式的模型载入函数。"""
    ckpt = torch.load(path, map_location='cpu', weights_only=False)

    # 1. 配置适配：无论旧 dict 还是新 yaml，均标准化为嵌套 dict
    cfg = adapt_config(ckpt.get('config', {}))
    ckpt['config'] = cfg  # 回写标准化配置，确保下游 Dataset.from_checkpoint 能正常解析

    # 2. 依据统一配置重建网络结构
    model = build_model(cfg, pretrained=False)

    # 3. 提取权重并清理可能由多卡/DataParallel 产生的 'module.' 前缀
    state_dict = ckpt.get('model_state', ckpt.get('state_dict', ckpt))
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    return model.to(device).eval(), ckpt
