# src/chiari/utils/seed.py
"""可复现性：统一设置随机种子。"""
import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42):
    """设置 Python / NumPy / Torch 全局种子。trainer 每折开始时调用一次。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
